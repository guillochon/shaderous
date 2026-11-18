// The block tree of the adaptive grid. It lives on the CPU: it is tiny (one record per block),
// changes only when the grid is re-meshed, and is uploaded to the GPU as flat tables.
//
// The domain is covered by nr^ndim root blocks. Every block holds BS^ndim cells and may be
// split into 2^ndim children of half the cell size. Leaves carry the solution; parents carry
// nothing (coarse data next to a fine region is restricted on the fly from the children).
// Face neighbors never differ by more than one level (2:1 balance). With `edgeBalance`, the
// same holds for blocks that share only an edge (3D) or a vertex (2D): constrained transport
// for magnetic fields defines each edge's field from the blocks around it (docs/numerics.md
// section 21, Block grid).

export const BS = 16; // cells per block side
export const NG = 2; // ghost layers
export const PAD = BS + 2 * NG; // stored cells per block side, in the atlas
export const BT_STRIDE = 32; // ints per block in the block table
export const RG_STRIDE = 12; // ints per block in the regrid table
/** A box of index-grid cells: origin and size, (x, y, z). */
/**
 * The cells of box (origin o, size s) of an m x m (x m) index grid, packed x fastest from the
 * start of their own array (the grid itself when the box is all of it).
 */
export function gridBox(grid: Uint32Array, m: number, o: readonly number[], s: readonly number[]): Uint32Array {
  if (s[0] * s[1] * s[2] === grid.length) return grid;
  const out = new Uint32Array(s[0] * s[1] * s[2]);
  for (let k = 0; k < s[2]; k++)
    for (let j = 0; j < s[1]; j++) {
      const from = ((o[2] + k) * m + o[1] + j) * m + o[0];
      out.set(grid.subarray(from, from + s[0]), (k * s[1] + j) * s[0]);
    }
  return out;
}

export interface GridBox {
  o: [number, number, number];
  s: [number, number, number];
}

// Block table layout (i32), per slot:
//   [0] level   [1..3] block coordinates at that level   [4] flags: 1 active, 2 leaf
//   [5] parent slot (-1 for roots)
//   [6..11] same-level face neighbor (-x, +x, -y, +y, -z, +z): slot, or
//           NB_COARSE where the region beyond is one level coarser, or -1 - (boundary type)
//           at a physical boundary (its kind, problems/boundaries.ts)
//   [12..19] child slots (octant = x | y << 1 | z << 2), -1 for leaves
//   [20..25] where [6..11] is NB_COARSE: slot of the coarser leaf beyond that face
export const NB_COARSE = -1;
export const FLAG_ACTIVE = 1;
export const FLAG_LEAF = 2;

// Regrid table layout (i32), per slot of the new tree:
//   [0] RG_* kind   [1] source slot in the old tree   [2] octant (prolongation)
//   [4..11] the eight old child slots (restriction)
export const RG_NONE = 0;
export const RG_COPY = 1;
export const RG_PROLONG = 2;
export const RG_RESTRICT = 3;

export interface Block {
  slot: number;
  level: number;
  c: number[]; // block coordinates at its level
  parent: Block | null;
  children: Block[] | null;
}

export type Neighbor = { kind: 'same'; block: Block } | { kind: 'coarse'; block: Block } | { kind: 'bc'; type: number };

export interface RegridOp {
  kind: number;
  src: number;
  octant: number;
  children: number[];
}

/**
 * One step of a re-mesh as adapt() took it, for replaying it (replayAdapt()): [what, level,
 * coordinates...] of the block it acted on, `what` one of the ADAPT_ constants. Made again in
 * order on the same tree, the steps give the same tree, slot for slot, and the same table.
 */
export type AdaptStep = number[];
/** refine(b) (with whatever balance asks for) */
export const ADAPT_REFINE = 0;
/** a parent merged to make room for a refinement */
export const ADAPT_EVICT = 1;
/** a parent merged because its children were no longer flagged */
export const ADAPT_MERGE = 2;

/**
 * A block may be merged to make room for another's refinement only when it is flagged less than
 * this share of the other's indicator: blocks flagged about equally would otherwise take the
 * slots from each other on alternate re-meshes, each time losing detail to the restriction.
 */
export const EVICT_MARGIN = 0.8;

export class BlockTree {
  readonly nchild: number;
  /** Deepest level refinement may reach; at most maxLevel. Lowered when the slot budget cannot hold more. */
  levelLimit: number;
  readonly nface: number;
  /** Every block, keyed by key(level, c). */
  readonly blocks = new Map<number, Block>();
  readonly slots: (Block | undefined)[];
  private free: number[] = [];
  /** Blocks per side at the deepest level: bounds every block coordinate. */
  private readonly span: number;
  /** all(), cached until the tree next changes (see changed()). */
  private sorted: Block[] | null = null;
  /**
   * The block table and index grid, kept between re-meshes and brought up to date
   * incrementally: a re-mesh at a fine level touches a small part of them, and rebuilding
   * them whole held the page for long enough to starve the GPU.
   */
  private tab: Int32Array | null = null;
  private grid: Uint32Array | null = null;
  /** Slots whose block-table rows may have changed since table() was last called. */
  private dirtyRows = new Set<number>();
  /** Blocks that may have become leaves since indexGrid() was last called. */
  private newLeaves: Block[] = [];
  private lo: number[];
  private hi: number[];
  /** Root blocks along each axis (the constructor's `lim`). */
  readonly lim: number[];
  /** Offsets to the blocks sharing only an edge (3D) or a vertex (2D), when edgeBalance is on. */
  private readonly diagonals: number[][];

  /**
   * @param ndim     2 or 3
   * @param nr       root blocks per side
   * @param maxLevel deepest level allowed (0 = roots only)
   * @param cap      number of slots in the atlas
   * @param bcLo     boundary type per axis at the lower faces (problems/boundaries.ts: 0 periodic)
   */
  constructor(
    readonly ndim: number,
    readonly nr: number,
    readonly maxLevel: number,
    readonly cap: number,
    bcLo: number[],
    bcHi: number[],
    readonly edgeBalance = false,
    /**
     * Root blocks along each axis, if fewer than nr along some (a rectangular grid: half along
     * a halved mirrored axis, src/domain.ts gridDims). Coordinates keep the cube of nr; the
     * blocks stop at the limit, past which lies the domain's upper boundary (bcHi).
     */
    lim?: readonly number[],
  ) {
    this.lo = bcLo;
    this.hi = bcHi;
    this.lim = [0, 1, 2].map((a) => lim?.[a] ?? nr);
    this.diagonals = [];
    if (edgeBalance) {
      // Two of the axes in use offset by one: the 4 vertex neighbors in 2D, the 12 edge neighbors in 3D.
      for (let a = 0; a < ndim; a++)
        for (let b = a + 1; b < ndim; b++)
          for (const sa of [-1, 1])
            for (const sb of [-1, 1]) {
              const d = [0, 0, 0];
              d[a] = sa;
              d[b] = sb;
              this.diagonals.push(d);
            }
    }
    this.nchild = 1 << ndim;
    this.levelLimit = maxLevel;
    this.nface = 2 * ndim;
    this.span = nr << maxLevel;
    this.slots = new Array(cap);
    for (let s = cap - 1; s >= 0; s--) this.free.push(s);
    const L = this.lim;
    const nroot = L[0] * L[1] * (ndim === 3 ? L[2] : 1);
    if (nroot > cap) throw new Error(`block budget ${cap} is smaller than the ${nroot} root blocks`);
    for (let k = 0; k < (ndim === 3 ? L[2] : 1); k++)
      for (let j = 0; j < L[1]; j++) for (let i = 0; i < L[0]; i++) this.create(0, [i, j, k], null);
  }

  // A number rather than a string: the tables are rebuilt from neighbor lookups on every re-mesh.
  private key(level: number, c0: number, c1: number, c2: number): number {
    return ((level * this.span + c2) * this.span + c1) * this.span + c0;
  }

  get bcLo(): number[] {
    return this.lo;
  }
  set bcLo(v: number[]) {
    this.lo = v;
    this.tab = null; // boundary faces are in every edge block's row
  }
  get bcHi(): number[] {
    return this.hi;
  }
  set bcHi(v: number[]) {
    this.hi = v;
    this.tab = null;
  }

  private changed() {
    this.sorted = null;
  }

  /**
   * A block was added or is about to go: its row, its parent's (leaf flag and children) and
   * its same-level face neighbors' (same-level or coarser neighbor) change. No finer block can
   * border it: 2:1 balance rules that out when it is created, and canCoarsen() when it goes.
   */
  private touch(b: Block) {
    this.dirtyRows.add(b.slot);
    if (b.parent) this.dirtyRows.add(b.parent.slot);
    for (let f = 0; f < this.nface; f++) {
      const nb = this.sameLevel(b, f);
      if (nb) this.dirtyRows.add(nb.slot);
    }
  }

  private create(level: number, c: number[], parent: Block | null): Block {
    const slot = this.free.pop()!;
    if (slot > this.highSlot) this.highSlot = slot;
    const b: Block = { slot, level, c, parent, children: null };
    this.slots[slot] = b;
    this.blocks.set(this.key(level, c[0], c[1], c[2] ?? 0), b);
    this.changed();
    this.touch(b);
    this.newLeaves.push(b);
    return b;
  }

  private remove(b: Block) {
    this.touch(b);
    this.blocks.delete(this.key(b.level, b.c[0], b.c[1], b.c[2] ?? 0));
    this.changed();
    this.slots[b.slot] = undefined;
    this.free.push(b.slot);
  }

  /**
   * The highest slot any block of this tree has held. Free slots are taken lowest first (the
   * stack starts with slot 0 on top, and a freed slot goes back on top), so the slots in use stay
   * packed low, and everything above this one has never been touched.
   */
  highSlot = -1;

  get freeSlots(): number {
    return this.free.length;
  }
  get count(): number {
    return this.cap - this.free.length;
  }

  /** Every block, coarsest first, and by slot within a level. */
  all(): Block[] {
    if (!this.sorted) {
      const byLevel: Block[][] = Array.from({ length: this.maxLevel + 1 }, () => []);
      for (const b of this.slots) if (b) byLevel[b.level].push(b);
      this.sorted = byLevel.flat();
    }
    return this.sorted.slice();
  }
  leaves(): Block[] {
    return this.all().filter((b) => !b.children);
  }

  /** The steps the last adapt() took (replayAdapt() takes them again). */
  lastSteps: AdaptStep[] = [];

  get(level: number, c: number[]): Block | undefined {
    return this.blocks.get(this.key(level, c[0], c[1], c[2] ?? 0));
  }

  /** The block of the same level beyond face f, if there is one (none at a closed boundary). */
  private sameLevel(b: Block, f: number): Block | undefined {
    const axis = f >> 1;
    const n = this.lim[axis] << b.level;
    let v = b.c[axis] + (f & 1 ? 1 : -1);
    if (v < 0 || v >= n) {
      if ((f & 1 ? this.hi[axis] : this.lo[axis]) !== 0) return undefined;
      v = (v + n) % n;
    }
    return this.blocks.get(this.key(b.level, axis === 0 ? v : b.c[0], axis === 1 ? v : b.c[1], axis === 2 ? v : (b.c[2] ?? 0)));
  }

  /** What lies beyond face f (0..5: -x, +x, -y, +y, -z, +z) of a block. */
  neighbor(b: Block, f: number): Neighbor {
    const axis = f >> 1;
    let c0 = b.c[0];
    let c1 = b.c[1];
    let c2 = b.c[2] ?? 0;
    const step = f & 1 ? 1 : -1;
    const n = this.lim[axis] << b.level;
    let v = (axis === 0 ? c0 : axis === 1 ? c1 : c2) + step;
    if (v < 0 || v >= n) {
      const type = f & 1 ? this.bcHi[axis] : this.bcLo[axis];
      if (type !== 0) return { kind: 'bc', type };
      v = (v + n) % n;
    }
    if (axis === 0) c0 = v;
    else if (axis === 1) c1 = v;
    else c2 = v;
    const same = this.blocks.get(this.key(b.level, c0, c1, c2));
    if (same) return { kind: 'same', block: same };
    const coarse = b.level > 0 ? this.blocks.get(this.key(b.level - 1, c0 >> 1, c1 >> 1, c2 >> 1)) : undefined;
    if (!coarse) throw new Error(`2:1 balance broken at level ${b.level} block ${b.c} face ${f}`);
    return { kind: 'coarse', block: coarse };
  }

  /**
   * The leaf (or, if the region is refined further, the same-level block) covering the block
   * at offset d from b: at b's level, or the coarser leaf around it. Null beyond a closed
   * boundary of the domain.
   */
  atOffset(b: Block, d: number[]): Block | null {
    const c = [b.c[0], b.c[1], b.c[2] ?? 0];
    for (let a = 0; a < 3; a++) {
      if (!d[a]) continue;
      const n = this.lim[a] << b.level;
      let v = c[a] + d[a];
      if (v < 0 || v >= n) {
        if ((d[a] > 0 ? this.hi[a] : this.lo[a]) !== 0) return null;
        v = (v + n) % n;
      }
      c[a] = v;
    }
    for (let l = b.level; l >= 0; l--) {
      const s = b.level - l;
      const q = this.blocks.get(this.key(l, c[0] >> s, c[1] >> s, c[2] >> s));
      if (q) return q;
    }
    return null;
  }

  /** Coarser blocks that must be split before b is (face neighbors, and with edgeBalance the diagonal ones). */
  private coarserAround(b: Block): Block[] {
    const out: Block[] = [];
    for (let f = 0; f < this.nface; f++) {
      const nb = this.neighbor(b, f);
      if (nb.kind === 'coarse') out.push(nb.block);
    }
    for (const d of this.diagonals) {
      const q = this.atOffset(b, d);
      if (q && q.level < b.level) out.push(q);
    }
    return out;
  }

  /** Whether refining `b` (and whatever balance drags in) fits in the free slots. */
  private refineCost(b: Block, seen: Set<Block>): number {
    if (seen.has(b) || b.children) return 0;
    seen.add(b);
    let cost = this.nchild;
    for (const q of this.coarserAround(b)) cost += this.refineCost(q, seen);
    return cost;
  }

  /** Slots it takes to refine every block of `list`, with whatever balance drags in. */
  costToRefine(list: Block[]): number {
    const seen = new Set<Block>();
    let cost = 0;
    for (const b of list) cost += this.refineCost(b, seen);
    return cost;
  }

  /**
   * Split a leaf, first splitting any coarser face neighbor so balance holds. Returns the
   * blocks that were split (empty if it does not fit or is already at the deepest level).
   */
  refine(b: Block, forbidden?: Set<Block>): Block[] {
    if (b.children || b.level >= this.levelLimit) return [];
    const need = new Set<Block>();
    if (this.refineCost(b, need) > this.free.length) return [];
    if (forbidden && [...need].some((q) => forbidden.has(q))) return [];
    const done: Block[] = [];
    const split = (q: Block) => {
      if (q.children) return;
      for (const nb of this.coarserAround(q)) split(nb);
      q.children = [];
      this.changed();
      for (let o = 0; o < this.nchild; o++) {
        const c = [2 * q.c[0] + (o & 1), 2 * q.c[1] + ((o >> 1) & 1), this.ndim === 3 ? 2 * q.c[2] + ((o >> 2) & 1) : 0];
        q.children.push(this.create(q.level + 1, c, q));
      }
      done.push(q);
    };
    split(b);
    return done;
  }

  /** Whether a parent's children can be merged without breaking balance. */
  canCoarsen(p: Block): boolean {
    if (!p.children || p.children.some((ch) => ch.children)) return false;
    for (const ch of p.children) {
      for (let f = 0; f < this.nface; f++) {
        const nb = this.neighbor(ch, f);
        if (nb.kind === 'same' && nb.block.children) return false; // a finer leaf would end up two levels away
      }
      for (const d of this.diagonals) {
        const q = this.atOffset(ch, d);
        if (q && q.level === ch.level && q.children) return false;
      }
    }
    return true;
  }

  coarsen(p: Block) {
    for (const ch of p.children!) this.remove(ch);
    p.children = null;
    this.changed();
    this.newLeaves.push(p);
  }

  /** Whether the block is still in the tree. */
  private live(b: Block): boolean {
    return this.blocks.get(this.key(b.level, b.c[0], b.c[1], b.c[2] ?? 0)) === b;
  }

  /**
   * Parents that could be merged to make room, least flagged first: every child a leaf, scored
   * by the most strongly flagged child (a negative indicator, which needs no buffer, counts as
   * least of all).
   */
  private evictable(leaves: Block[], ind: Map<Block, number>): { p: Block; score: number }[] {
    const parents = new Set(leaves.map((b) => b.parent).filter((p): p is Block => !!p));
    const out: { p: Block; score: number }[] = [];
    for (const p of parents) {
      if (!p.children || p.children.some((ch) => ch.children)) continue;
      out.push({ p, score: Math.max(...p.children.map((ch) => ind.get(ch) ?? 0)) });
    }
    return out.sort((a, b) => a.score - b.score);
  }

  /** Whether two blocks' boxes, the second grown by its own size on every side, overlap. */
  private near(a: Block, q: Block): boolean {
    const sa = 1 << (this.maxLevel - a.level);
    const sq = 1 << (this.maxLevel - q.level);
    for (let i = 0; i < this.ndim; i++) {
      const lo = a.c[i] * sa;
      const qlo = (q.c[i] - 1) * sq;
      if (lo + sa <= qlo || qlo + 3 * sq <= lo) return false;
    }
    return true;
  }

  /**
   * Parents to merge so that refining `b` fits: the least flagged first, each flagged less
   * than EVICT_MARGIN times `level` (b's indicator) and away from b and the blocks its refinement splits (merging
   * those would only be undone by balance). Null if they cannot free enough; nothing is merged here.
   */
  private makeRoom(b: Block, level: number, evictable: { p: Block; score: number }[], touched: Set<Block>): Block[] | null {
    if (b.children || b.level >= this.levelLimit) return null;
    const need = new Set<Block>();
    let short = this.refineCost(b, need) - this.free.length;
    if (short <= 0) return null;
    const nearby = [...need];
    const out: Block[] = [];
    for (const { p, score } of evictable) {
      if (score >= EVICT_MARGIN * level) break;
      if (touched.has(p) || !this.live(p) || !p.children || p.children.some((ch) => ch.children || touched.has(ch))) continue;
      if (need.has(p) || nearby.some((q) => this.near(p, q)) || !this.canCoarsen(p)) continue;
      out.push(p);
      short -= this.nchild;
      if (short <= 0) return out;
    }
    return null;
  }

  /**
   * Re-mesh from a refinement indicator per leaf. Leaves above `hi` are split; a parent whose
   * children are all below `lo`, and have no face neighbor above `hi`, is merged. A negative
   * indicator marks a leaf that needs no buffer either (the inside of an obstacle): it merges
   * whatever its neighbors.
   * Returns, per slot of the new tree, how to fill it from the old tree, or null if nothing changed.
   */
  adapt(indicator: (b: Block) => number, hi: number, lo: number): RegridOp[] | null {
    const before = this.leaves();
    const ind = new Map<Block, number>();
    for (const b of before) ind.set(b, indicator(b));
    const touched = new Set<Block>();
    const run = this.adaptRun(touched);

    // Refine, most strongly flagged first so they get the slots if the budget runs out. When it
    // has, room is made by merging the parents whose children are least flagged (closest to
    // being merged anyway), as long as they are flagged well below the block that needs it.
    const wanted = before.filter((b) => ind.get(b)! > hi && b.level < this.levelLimit).sort((a, b) => ind.get(b)! - ind.get(a)!);
    let evictable: { p: Block; score: number }[] | null = null;
    // Blocks merged to make room hold their data only through this re-mesh's restriction: none
    // may be split again in it (balance would ask that of a later refinement's neighbor), as its
    // new children would be filled from the old parent's slot, which holds no data.
    for (const b of wanted) {
      if (b.children || !this.live(b)) continue; // split by balance already, or merged to make room
      if (!run.step(ADAPT_REFINE, b)) {
        evictable ??= this.evictable(before, ind);
        const room = this.makeRoom(b, ind.get(b)!, evictable, touched);
        if (room) {
          for (const p of room) run.step(ADAPT_EVICT, p);
          run.step(ADAPT_REFINE, b);
        }
      }
    }

    // Merge. A leaf next to a strongly flagged one stays, as a buffer ahead of moving features.
    const hot = (b: Block) => {
      if (ind.get(b)! > lo) return true;
      if (ind.get(b)! < 0) return false;
      for (let f = 0; f < this.nface; f++) {
        const nb = this.neighbor(b, f);
        if (nb.kind !== 'bc' && (ind.get(nb.block) ?? 0) > hi) return true;
      }
      return false;
    };
    const parents = [...new Set(before.map((b) => b.parent).filter((p): p is Block => !!p))];
    for (const p of parents) {
      if (touched.has(p) || !this.live(p) || !p.children || p.children.some((ch) => touched.has(ch) || ch.children)) continue;
      if (p.children.some(hot) || !this.canCoarsen(p)) continue;
      run.step(ADAPT_MERGE, p);
    }
    this.lastSteps = run.steps;
    return run.table();
  }

  /**
   * Take the steps of an earlier adapt() again (its lastSteps), on the tree as it was then: the
   * same re-mesh, decided without its indicators. For replaying a run (ui/session.ts).
   */
  replayAdapt(steps: AdaptStep[]): RegridOp[] | null {
    const run = this.adaptRun(new Set());
    for (const [what, level, ...c] of steps) {
      const b = this.get(level, c);
      if (!b) throw new Error(`a re-mesh to replay acts on a block at level ${level}, [${c.join(', ')}] that the mesh does not have`);
      run.step(what, b);
    }
    this.lastSteps = run.steps;
    return run.table();
  }

  /**
   * The bookkeeping of a re-mesh: each step taken, recorded, and the way each slot of the new
   * tree is filled from the old one. step() returns whether it changed the tree (a refinement
   * may find no room).
   */
  private adaptRun(touched: Set<Block>) {
    const ops = new Map<number, RegridOp>();
    const steps: AdaptStep[] = [];
    // Blocks merged to make room hold their data only through this re-mesh's restriction: none
    // may be split again in it (balance would ask that of a later refinement's neighbor), as its
    // new children would be filled from the old parent's slot, which holds no data.
    const evicted = new Set<Block>();
    const merge = (p: Block) => {
      const children = p.children!.map((ch) => ch.slot);
      this.coarsen(p);
      touched.add(p);
      ops.set(p.slot, { kind: RG_RESTRICT, src: p.slot, octant: 0, children });
    };
    return {
      steps,
      step: (what: number, b: Block): boolean => {
        steps.push([what, b.level, ...b.c.slice(0, this.ndim)]);
        if (what === ADAPT_REFINE) {
          const split = this.refine(b, evicted);
          for (const q of split) {
            touched.add(q);
            q.children!.forEach((ch, o) => {
              touched.add(ch);
              ops.set(ch.slot, { kind: RG_PROLONG, src: q.slot, octant: o, children: [] });
            });
          }
          return split.length > 0;
        }
        merge(b);
        if (what === ADAPT_EVICT) evicted.add(b);
        return true;
      },
      table: (): RegridOp[] | null => {
        if (ops.size === 0) return null;
        const out: RegridOp[] = new Array(this.cap);
        for (const leaf of this.leaves()) out[leaf.slot] = ops.get(leaf.slot) ?? { kind: RG_COPY, src: leaf.slot, octant: 0, children: [] };
        return out;
      },
    };
  }

  /**
   * The block table uploaded to the GPU. The same array is returned each time, updated in
   * place: upload it before the tree next changes, and do not modify it.
   */
  table(): Int32Array {
    let t = this.tab;
    if (!t) {
      t = this.tab = new Int32Array(this.cap * BT_STRIDE);
      for (let s = 0; s < this.cap; s++) this.writeRow(t, s);
    } else for (const s of this.dirtyRows) this.writeRow(t, s);
    this.dirtyRows.clear();
    return t;
  }

  private writeRow(t: Int32Array, slot: number) {
    const o = slot * BT_STRIDE;
    t.fill(-1, o, o + BT_STRIDE);
    t[o + 4] = 0;
    const b = this.slots[slot];
    if (b) {
      t[o] = b.level;
      t.set(b.c, o + 1);
      t[o + 4] = FLAG_ACTIVE | (b.children ? 0 : FLAG_LEAF);
      t[o + 5] = b.parent ? b.parent.slot : -1;
      for (let f = 0; f < 6; f++) {
        if (f >= this.nface) {
          t[o + 6 + f] = b.slot; // 2D: the single layer is its own neighbor in z
          continue;
        }
        const nb = this.neighbor(b, f);
        if (nb.kind === 'same') t[o + 6 + f] = nb.block.slot;
        else if (nb.kind === 'coarse') {
          t[o + 6 + f] = NB_COARSE;
          t[o + 20 + f] = nb.block.slot;
        } else t[o + 6 + f] = -1 - nb.type;
      }
      if (b.children) b.children.forEach((ch, i) => (t[o + 12 + i] = ch.slot));
    }
  }

  static regridTable(ops: RegridOp[], cap: number): Int32Array {
    const t = new Int32Array(cap * RG_STRIDE);
    ops.forEach((op, slot) => {
      if (!op) return;
      const o = slot * RG_STRIDE;
      t[o] = op.kind;
      t[o + 1] = op.src;
      t[o + 2] = op.octant;
      op.children.forEach((c, i) => (t[o + 4 + i] = c));
    });
    return t;
  }

  /** Slot lists the passes are dispatched over: every block (coarsest first), and the leaves. */
  lists(): { all: Uint32Array; leaves: Uint32Array; byLevel: Uint32Array[] } {
    const all = this.all();
    const out = new Uint32Array(all.length);
    const leaves: number[] = [];
    const counts = new Array(this.maxLevel + 1).fill(0);
    all.forEach((b, i) => {
      out[i] = b.slot;
      counts[b.level]++;
      if (!b.children) leaves.push(b.slot);
    });
    // all() is coarsest first, so each level is a run of it.
    const byLevel: Uint32Array[] = [];
    let at = 0;
    for (const n of counts) {
      byLevel.push(out.slice(at, at + n));
      at += n;
    }
    return { all: out, leaves: Uint32Array.from(leaves), byLevel };
  }

  /**
   * Leaf slot covering each block-sized cell of the deepest level: a lookup from position to
   * block, (nr << maxLevel) per side. Like table(), the same array updated in place. Given
   * `boxes`, adds to it the boxes (in cells) that changed since the last call, so only those
   * need uploading: the whole grid on the first call, else a few slabs around the re-mesh.
   */
  indexGrid(boxes?: GridBox[]): Uint32Array {
    const m = this.nr << this.maxLevel;
    const d3 = this.ndim === 3;
    const mz = d3 ? m : 1;
    // Every cell lies in exactly one leaf, so filling the new leaves covers whatever changed.
    let g = this.grid;
    if (!g) {
      g = this.grid = new Uint32Array(m * m * mz);
      for (const b of this.blocks.values()) if (!b.children) this.fillLeaf(g, b, m);
      this.newLeaves = [];
      boxes?.push({ o: [0, 0, 0], s: [m, m, mz] });
      return g;
    }
    // The extent filled along the inner axes (x, and y in 3D) of each row of the outer axis
    // (y in 2D, z in 3D), merged into boxes below.
    const lo0 = new Int32Array(m).fill(m);
    const hi0 = new Int32Array(m);
    const lo1 = new Int32Array(m).fill(m);
    const hi1 = new Int32Array(m);
    for (const b of this.newLeaves) {
      if (b.children || this.slots[b.slot] !== b) continue;
      this.fillLeaf(g, b, m);
      const span = 1 << (this.maxLevel - b.level);
      const x0 = b.c[0] * span;
      const y0 = b.c[1] * span;
      const o0 = (d3 ? b.c[2] : b.c[1]) * span;
      for (let o = o0; o < o0 + span; o++) {
        lo0[o] = Math.min(lo0[o], x0);
        hi0[o] = Math.max(hi0[o], x0 + span);
        if (d3) {
          lo1[o] = Math.min(lo1[o], y0);
          hi1[o] = Math.max(hi1[o], y0 + span);
        }
      }
    }
    this.newLeaves = [];
    if (!boxes) return g;
    // Runs of changed rows, bridging gaps of a few rows: each box costs a copy call, and an
    // unchanged row inside one costs only its bytes.
    const GAP = 4;
    for (let o = 0; o < m; ) {
      if (hi0[o] === 0) {
        o++;
        continue;
      }
      let a0 = m, b0 = 0, a1 = m, b1 = 0;
      let end = o;
      for (let k = o; k < m && k <= end + GAP; k++) {
        if (hi0[k] === 0) continue;
        a0 = Math.min(a0, lo0[k]);
        b0 = Math.max(b0, hi0[k]);
        a1 = Math.min(a1, lo1[k]);
        b1 = Math.max(b1, hi1[k]);
        end = k;
      }
      boxes.push(d3 ? { o: [a0, a1, o], s: [b0 - a0, b1 - a1, end + 1 - o] } : { o: [a0, o, 0], s: [b0 - a0, end + 1 - o, 1] });
      o = end + 1;
    }
    return g;
  }

  private fillLeaf(g: Uint32Array, b: Block, m: number) {
    const span = 1 << (this.maxLevel - b.level);
    const z0 = this.ndim === 3 ? b.c[2] * span : 0;
    for (let k = z0; k < z0 + (this.ndim === 3 ? span : 1); k++)
      for (let j = b.c[1] * span; j < (b.c[1] + 1) * span; j++)
        g.fill(b.slot, (k * m + j) * m + b.c[0] * span, (k * m + j) * m + (b.c[0] + 1) * span);
  }

  /** Leaves, cells and the finest level in use. */
  stats() {
    const leaves = this.leaves();
    return {
      blocks: this.count,
      leaves: leaves.length,
      cells: leaves.length * BS ** this.ndim,
      deepest: leaves.reduce((m, b) => Math.max(m, b.level), 0),
    };
  }
}
