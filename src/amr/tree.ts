// The block tree of the adaptive grid. It lives on the CPU: it is tiny (one record per block),
// changes only when the grid is re-meshed, and is uploaded to the GPU as flat tables.
//
// The domain is covered by nr^ndim root blocks. Every block holds BS^ndim cells and may be
// split into 2^ndim children of half the cell size. Leaves carry the solution; parents carry
// nothing (coarse data next to a fine region is restricted on the fly from the children).
// Face neighbors never differ by more than one level (2:1 balance).

export const BS = 16; // cells per block side
export const NG = 2; // ghost layers
export const PAD = BS + 2 * NG; // stored cells per block side, in the atlas
export const BT_STRIDE = 32; // ints per block in the block table
export const RG_STRIDE = 12; // ints per block in the regrid table

// Block table layout (i32), per slot:
//   [0] level   [1..3] block coordinates at that level   [4] flags: 1 active, 2 leaf
//   [5] parent slot (-1 for roots)
//   [6..11] same-level face neighbor (-x, +x, -y, +y, -z, +z): slot, or
//           NB_COARSE where the region beyond is one level coarser, or -1 - (boundary type)
//           at a physical boundary (outflow 1, reflecting 2, outflow-only 3)
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

  /**
   * @param ndim     2 or 3
   * @param nr       root blocks per side
   * @param maxLevel deepest level allowed (0 = roots only)
   * @param cap      number of slots in the atlas
   * @param bcLo     boundary type per axis at the lower faces: 0 periodic, 1 outflow, 2 reflecting, 3 outflow only
   */
  constructor(
    readonly ndim: number,
    readonly nr: number,
    readonly maxLevel: number,
    readonly cap: number,
    bcLo: number[],
    bcHi: number[],
  ) {
    this.lo = bcLo;
    this.hi = bcHi;
    this.nchild = 1 << ndim;
    this.levelLimit = maxLevel;
    this.nface = 2 * ndim;
    this.span = nr << maxLevel;
    this.slots = new Array(cap);
    for (let s = cap - 1; s >= 0; s--) this.free.push(s);
    const nroot = nr ** ndim;
    if (nroot > cap) throw new Error(`block budget ${cap} is smaller than the ${nroot} root blocks`);
    for (let k = 0; k < (ndim === 3 ? nr : 1); k++)
      for (let j = 0; j < nr; j++) for (let i = 0; i < nr; i++) this.create(0, [i, j, k], null);
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

  get(level: number, c: number[]): Block | undefined {
    return this.blocks.get(this.key(level, c[0], c[1], c[2] ?? 0));
  }

  /** The block of the same level beyond face f, if there is one (none at a closed boundary). */
  private sameLevel(b: Block, f: number): Block | undefined {
    const axis = f >> 1;
    const n = this.nr << b.level;
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
    const n = this.nr << b.level;
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

  /** Whether refining `b` (and whatever balance drags in) fits in the free slots. */
  private refineCost(b: Block, seen: Set<Block>): number {
    if (seen.has(b) || b.children) return 0;
    seen.add(b);
    let cost = this.nchild;
    for (let f = 0; f < this.nface; f++) {
      const nb = this.neighbor(b, f);
      if (nb.kind === 'coarse') cost += this.refineCost(nb.block, seen);
    }
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
      for (let f = 0; f < this.nface; f++) {
        const nb = this.neighbor(q, f);
        if (nb.kind === 'coarse') split(nb.block);
      }
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
    }
    return true;
  }

  coarsen(p: Block) {
    for (const ch of p.children!) this.remove(ch);
    p.children = null;
    this.changed();
    this.newLeaves.push(p);
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
    const ops = new Map<number, RegridOp>();
    const touched = new Set<Block>();

    // Refine, most strongly flagged first so they get the slots if the budget runs out.
    const wanted = before.filter((b) => ind.get(b)! > hi && b.level < this.levelLimit).sort((a, b) => ind.get(b)! - ind.get(a)!);
    for (const b of wanted) {
      for (const q of this.refine(b)) {
        touched.add(q);
        q.children!.forEach((ch, o) => {
          touched.add(ch);
          ops.set(ch.slot, { kind: RG_PROLONG, src: q.slot, octant: o, children: [] });
        });
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
      if (touched.has(p) || !p.children || p.children.some((ch) => touched.has(ch) || ch.children)) continue;
      if (p.children.some(hot) || !this.canCoarsen(p)) continue;
      const children = p.children.map((ch) => ch.slot);
      this.coarsen(p);
      touched.add(p);
      ops.set(p.slot, { kind: RG_RESTRICT, src: p.slot, octant: 0, children });
    }

    if (ops.size === 0) return null;
    const out: RegridOp[] = new Array(this.cap);
    for (const b of this.leaves()) out[b.slot] = ops.get(b.slot) ?? { kind: RG_COPY, src: b.slot, octant: 0, children: [] };
    return out;
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
   * block, (nr << maxLevel) per side. Like table(), the same array updated in place.
   */
  indexGrid(): Uint32Array {
    const m = this.nr << this.maxLevel;
    const mz = this.ndim === 3 ? m : 1;
    // Every cell lies in exactly one leaf, so filling the new leaves covers whatever changed.
    let g = this.grid;
    let fill: Iterable<Block> = this.newLeaves;
    if (!g) {
      g = this.grid = new Uint32Array(m * m * mz);
      fill = this.blocks.values();
    }
    for (const b of fill) {
      if (b.children || this.slots[b.slot] !== b) continue;
      const span = 1 << (this.maxLevel - b.level);
      const z0 = this.ndim === 3 ? b.c[2] * span : 0;
      for (let k = z0; k < z0 + (this.ndim === 3 ? span : 1); k++)
        for (let j = b.c[1] * span; j < (b.c[1] + 1) * span; j++)
          g.fill(b.slot, (k * m + j) * m + b.c[0] * span, (k * m + j) * m + (b.c[0] + 1) * span);
    }
    this.newLeaves = [];
    return g;
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
