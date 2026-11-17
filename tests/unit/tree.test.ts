import { describe, expect, it } from 'vitest';
import { BlockTree, BT_STRIDE, gridBox, FLAG_LEAF, NB_COARSE, RG_COPY, RG_PROLONG, RG_RESTRICT, type Block, type GridBox } from '../../src/amr/tree';

const periodic = [0, 0, 0];
const outflow = [1, 1, 1];

/** Every leaf's face neighbors are at most one level away. */
function balanced(t: BlockTree) {
  for (const b of t.leaves()) for (let f = 0; f < t.nface; f++) t.neighbor(b, f); // throws if a region is two levels coarser
  for (const b of t.all()) {
    if (!b.children) continue;
    for (const ch of b.children) expect(ch.parent).toBe(b);
  }
}

/** The leaves tile the domain exactly once. */
function covers(t: BlockTree) {
  const g = t.indexGrid();
  const m = t.nr << t.maxLevel;
  const count = new Map<number, number>();
  for (const s of g) count.set(s, (count.get(s) ?? 0) + 1);
  for (const b of t.leaves()) expect(count.get(b.slot)).toBe((1 << (t.maxLevel - b.level)) ** t.ndim);
  expect(g.length).toBe(m ** t.ndim);
}

describe('block tree', () => {
  it('replays a re-mesh from its steps alone, slot for slot (session logs)', () => {
    // One tree re-meshes from indicators; its twin only from the steps the first recorded. A small
    // budget (cap) makes some refinements evict, and edge balance (MHD) cascades.
    for (const [ndim, cap, edge] of [[2, 100, false], [2, 400, true], [3, 300, true]] as const) {
      const make = () => new BlockTree(ndim, 2, ndim === 2 ? 4 : 3, cap, [...periodic], [...periodic], edge);
      const a = make();
      const b = make();
      let seed = 12345;
      const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
      let evictions = 0;
      for (let round = 0; round < 25; round++) {
        // Flagged near a point that moves, and noise.
        const p = [0.3 + 0.02 * round, 0.5, 0.5];
        const ind = (blk: { level: number; c: number[] }) => {
          const size = 1 / (2 << blk.level);
          let d2 = 0;
          for (let i = 0; i < ndim; i++) {
            const lo = blk.c[i] * size;
            d2 += Math.max(0, lo - p[i], p[i] - lo - size) ** 2;
          }
          return Math.exp(-d2 / 0.003) + 0.3 * rand();
        };
        const opsA = a.adapt(ind, 0.6, 0.2);
        evictions += a.lastSteps.filter((st) => st[0] === 1).length;
        const opsB = b.replayAdapt(a.lastSteps);
        expect(opsB).toEqual(opsA);
        expect(b.lastSteps).toEqual(a.lastSteps);
        expect(Array.from(b.table())).toEqual(Array.from(a.table()));
      }
      expect(a.count).toBeGreaterThan(ndim === 2 ? 40 : 60); // it did refine
      if (cap === 100) expect(evictions).toBeGreaterThan(0);
      balanced(b);
    }
    // A step on a block the tree does not have is an error, not a silent other mesh.
    const t = new BlockTree(2, 2, 3, 100, periodic, periodic);
    expect(() => t.replayAdapt([[0, 3, 5, 5]])).toThrow(/does not have/);
  });

  it('keeps its block table and index grid up to date across re-meshes', () => {
    // Two trees take the same re-meshes: one is read after every one (updated in place), the
    // other only at the end (built whole).
    for (const [ndim, bc] of [[2, periodic], [2, [1, 2, 0]], [3, periodic], [3, outflow]] as const) {
      const make = () => new BlockTree(ndim, 2, ndim === 2 ? 4 : 3, 3000, [...bc], [...bc]);
      const a = make();
      const b = make();
      let seed = 7;
      const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (let round = 0; round < 10; round++) {
        const ind = Array.from({ length: 3000 }, () => rnd() * 1.2);
        a.adapt((x) => ind[x.slot], 0.8, 0.3);
        b.adapt((x) => ind[x.slot], 0.8, 0.3);
        a.table();
        a.indexGrid();
      }
      expect(a.table()).toEqual(b.table());
      expect(a.indexGrid()).toEqual(b.indexGrid());
      covers(a);
      // A change of boundaries reaches the table read before it (b's is built only now).
      const flipped = bc.map((x) => (x ? 0 : 1));
      a.bcLo = flipped;
      const b2 = make();
      b2.bcLo = flipped;
      let s2 = 7;
      const rnd2 = () => (s2 = (s2 * 16807) % 2147483647) / 2147483647;
      for (let round = 0; round < 10; round++) {
        const ind = Array.from({ length: 3000 }, () => rnd2() * 1.2);
        // Same re-meshes as a's, made under the old boundaries; only the table read differs.
        b2.bcLo = [...bc];
        b2.adapt((x) => ind[x.slot], 0.8, 0.3);
      }
      b2.bcLo = flipped;
      expect(a.table()).toEqual(b2.table());
    }
  });

  it('reports the index-grid boxes that changed', () => {
    // A copy updated only inside the reported boxes, as the GPU texture is, stays equal.
    for (const ndim of [2, 3] as const) {
      const t = new BlockTree(ndim, 2, ndim === 2 ? 4 : 3, 3000, periodic, periodic);
      const m = t.nr << t.maxLevel;
      const copy = new Uint32Array(m * m * (ndim === 3 ? m : 1));
      let seed = 11;
      const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      let partial = 0;
      for (let round = 0; round < 12; round++) {
        if (round) {
          const ind = Array.from({ length: 3000 }, () => rnd() * 1.2);
          t.adapt((x) => ind[x.slot], 0.8, 0.3);
        }
        const boxes: GridBox[] = [];
        const g = t.indexGrid(boxes);
        let cells = 0;
        for (const { o, s } of boxes) {
          cells += s[0] * s[1] * s[2];
          // As the GPU texture is written: each box packed on its own (gridBox), from its start.
          const data = gridBox(g, m, o, s);
          expect(data.length).toBe(s[0] * s[1] * s[2]);
          for (let k = 0; k < s[2]; k++)
            for (let j = 0; j < s[1]; j++) {
              const at = ((o[2] + k) * m + o[1] + j) * m + o[0];
              copy.set(data.subarray((k * s[1] + j) * s[0], (k * s[1] + j + 1) * s[0]), at);
            }
        }
        if (round && cells < g.length) partial++;
        expect(copy).toEqual(g);
      }
      expect(partial).toBeGreaterThan(0); // not every re-mesh re-sends the whole grid
    }
  });

  it('starts as a uniform root grid', () => {
    const t = new BlockTree(3, 2, 2, 64, periodic, periodic);
    expect(t.count).toBe(8);
    expect(t.leaves()).toHaveLength(8);
    covers(t);
    const b = t.get(0, [0, 0, 0])!;
    const nb = t.neighbor(b, 0);
    expect(nb.kind).toBe('same');
    expect(nb.kind === 'same' && nb.block.c).toEqual([1, 0, 0]); // wrapped
  });

  it('reports physical boundaries', () => {
    const t = new BlockTree(2, 2, 1, 16, outflow, [2, 3, 0]);
    const b = t.get(0, [1, 1, 0])!;
    expect(t.neighbor(b, 0).kind).toBe('same');
    expect(t.neighbor(b, 1)).toEqual({ kind: 'bc', type: 2 });
    expect(t.neighbor(b, 3)).toEqual({ kind: 'bc', type: 3 });
    const tab = t.table();
    expect(tab[b.slot * BT_STRIDE + 6 + 1]).toBe(-3);
    expect(tab[b.slot * BT_STRIDE + 6 + 3]).toBe(-4);
    expect(tab[b.slot * BT_STRIDE + 6 + 4]).toBe(b.slot); // 2D: its own neighbor in z
  });

  it('refines and keeps 2:1 balance by splitting coarser neighbors', () => {
    const t = new BlockTree(2, 4, 3, 400, outflow, outflow);
    let b: Block = t.get(0, [1, 1, 0])!;
    for (let l = 0; l < 3; l++) {
      expect(t.refine(b).length).toBeGreaterThan(0);
      b = b.children![3];
      balanced(t);
      covers(t);
    }
    expect(t.stats().deepest).toBe(3);
    // The block next to the level-1 parent had to be split when its child went to level 2.
    expect(t.get(0, [2, 1, 0])!.children).not.toBeNull();
  });

  it('refuses to refine past the deepest level or the block budget', () => {
    const t = new BlockTree(3, 1, 1, 9, periodic, periodic);
    const root = t.get(0, [0, 0, 0])!;
    expect(t.refine(root)).toHaveLength(1);
    expect(t.freeSlots).toBe(0);
    expect(t.refine(root.children![0])).toHaveLength(0); // deepest level
    const u = new BlockTree(3, 2, 2, 12, periodic, periodic);
    expect(u.refine(u.get(0, [0, 0, 0])!)).toHaveLength(0); // 8 roots + 8 children > 12
    expect(u.count).toBe(8);
  });

  it('records coarse neighbors and children in the block table', () => {
    const t = new BlockTree(2, 2, 1, 16, outflow, outflow);
    const p = t.get(0, [0, 0, 0])!;
    t.refine(p);
    const tab = t.table();
    const ch = p.children![1]; // x = 1, y = 0: its +x neighbor is the coarse block (1, 0)
    const o = ch.slot * BT_STRIDE;
    expect(tab[o]).toBe(1);
    expect(tab[o + 4] & FLAG_LEAF).toBe(FLAG_LEAF);
    expect(tab[o + 5]).toBe(p.slot);
    expect(tab[o + 6 + 1]).toBe(NB_COARSE);
    expect(tab[o + 20 + 1]).toBe(t.get(0, [1, 0, 0])!.slot);
    expect(tab[o + 6 + 0]).toBe(p.children![0].slot);
    expect(tab[p.slot * BT_STRIDE + 4] & FLAG_LEAF).toBe(0);
    expect(Array.from(tab.subarray(p.slot * BT_STRIDE + 12, p.slot * BT_STRIDE + 16))).toEqual(p.children!.map((c) => c.slot));
    // The coarse block sees the parent as its same-level neighbor.
    expect(tab[t.get(0, [1, 0, 0])!.slot * BT_STRIDE + 6 + 0]).toBe(p.slot);
  });

  it('adapts: splits flagged leaves, merges quiet ones, and says how to fill each block', () => {
    const t = new BlockTree(2, 4, 2, 200, periodic, periodic);
    const target = t.get(0, [2, 2, 0])!;
    let ops = t.adapt((b) => (b === target ? 1 : 0), 0.5, 0.1)!;
    expect(ops).not.toBeNull();
    balanced(t);
    covers(t);
    for (const ch of target.children!) expect(ops[ch.slot].kind).toBe(RG_PROLONG);
    expect(ops[target.children![2].slot]).toMatchObject({ src: target.slot, octant: 2 });
    expect(ops[t.get(0, [0, 0, 0])!.slot].kind).toBe(RG_COPY);
    expect(ops[target.slot]).toBeUndefined(); // parents hold no data

    // Nothing flagged any more: the children merge back.
    const kids = target.children!.map((c) => c.slot);
    ops = t.adapt(() => 0, 0.5, 0.1)!;
    expect(target.children).toBeNull();
    expect(ops[target.slot]).toMatchObject({ kind: RG_RESTRICT, children: kids });
    expect(t.count).toBe(16);
    expect(t.adapt(() => 0, 0.5, 0.1)).toBeNull();
  });

  it('does not merge blocks next to a strongly flagged one, or where balance forbids it', () => {
    const t = new BlockTree(2, 4, 2, 400, periodic, periodic);
    const a = t.get(0, [1, 1, 0])!;
    t.refine(a);
    t.refine(a.children![3]); // forces neighbors of a to split too
    balanced(t);
    const deep = a.children![3].children!;
    // Quiet everywhere except the deepest blocks: their parents' siblings must stay refined.
    t.adapt((b) => (deep.includes(b) ? 1 : 0), 2, 0.1);
    balanced(t);
    covers(t);
    expect(a.children).not.toBeNull();
    expect(a.children![3].children).not.toBeNull();
  });

  it('makes room in a full budget by merging the least flagged blocks, never more flagged ones', () => {
    // 16 roots and one refined corner fill the 20 slots.
    const make = () => {
      const t = new BlockTree(2, 4, 1, 20, outflow, outflow);
      const a = t.get(0, [0, 0, 0])!;
      t.refine(a);
      expect(t.freeSlots).toBe(0);
      return { t, a, b: t.get(0, [3, 3, 0])! };
    };
    {
      const { t, a, b } = make();
      const old = a.children!.map((ch) => ch.slot);
      // a's children are flagged, but under the threshold to refine and over the one to merge:
      // without the eviction they would stay, and b could not be refined.
      const ops = t.adapt((q) => (q === b ? 2 : 0.3), 0.5, 0.1)!;
      expect(a.children).toBeNull();
      expect(b.children).not.toBeNull();
      expect(ops[a.slot]).toMatchObject({ kind: RG_RESTRICT, children: old });
      // b's children take the slots a's children had, filled from b.
      expect(b.children!.map((ch) => ch.slot).sort()).toEqual([...old].sort());
      for (const ch of b.children!) expect(ops[ch.slot]).toMatchObject({ kind: RG_PROLONG, src: b.slot });
      balanced(t);
      covers(t);
    }
    {
      // a's children flagged more strongly than b: they keep their slots, and b stays as it is.
      const { t, a, b } = make();
      t.adapt((q) => (q === b ? 2 : a.children!.includes(q) ? 3 : 0.3), 0.5, 0.1);
      expect(a.children).not.toBeNull();
      expect(b.children).toBeNull();
    }
  });

  it('fills every new block from data the old mesh holds, through many re-meshes of a full budget', () => {
    let seed = 7;
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (const [ndim, cap] of [[2, 60], [3, 120]] as const) {
      const t = new BlockTree(ndim, 4, 3, cap, outflow, outflow);
      for (let round = 0; round < 60; round++) {
        const leaves = new Set(t.leaves().map((b) => b.slot));
        // A moving hot spot and noise: a budget this small fills, and has to make room.
        const cx = rand() * 4, cy = rand() * 4;
        const ind = new Map<Block, number>();
        const ops = t.adapt((b) => {
          if (!ind.has(b)) {
            const s = 1 / (1 << b.level);
            const d = Math.hypot((b.c[0] + 0.5) * s - cx, (b.c[1] + 0.5) * s - cy);
            ind.set(b, 2 / (0.3 + d) + rand());
          }
          return ind.get(b)!;
        }, 1.5, 0.5);
        balanced(t);
        covers(t);
        if (!ops) continue;
        for (const b of t.leaves()) {
          const op = ops[b.slot];
          if (op.kind === RG_PROLONG || op.kind === RG_COPY) expect(leaves.has(op.src)).toBe(true);
          if (op.kind === RG_RESTRICT) for (const ch of op.children) expect(leaves.has(ch)).toBe(true);
        }
      }
    }
  });

  it('lists blocks coarsest first and leaves separately', () => {
    const t = new BlockTree(3, 2, 1, 64, periodic, periodic);
    t.refine(t.get(0, [1, 1, 1])!);
    const { all, leaves } = t.lists();
    expect(all).toHaveLength(16);
    expect(leaves).toHaveLength(15);
    const levels = Array.from(all, (s) => t.slots[s]!.level);
    expect(levels).toEqual([...levels].sort((x, y) => x - y));
  });
});

describe('edge balance (magnetic fields)', () => {
  /** Offsets to the blocks sharing an edge (3D) or a vertex (2D). */
  const diagonals = (ndim: number) => {
    const out: number[][] = [];
    for (let a = 0; a < ndim; a++)
      for (let b = a + 1; b < ndim; b++)
        for (const sa of [-1, 1])
          for (const sb of [-1, 1]) {
            const d = [0, 0, 0];
            d[a] = sa;
            d[b] = sb;
            out.push(d);
          }
    return out;
  };
  /** The largest level difference between a leaf and a coarser leaf it shares only an edge or vertex with. */
  const worst = (t: BlockTree) => {
    let w = 0;
    for (const b of t.leaves())
      for (const d of diagonals(t.ndim)) {
        const q = t.atOffset(b, d);
        if (q && !q.children) w = Math.max(w, b.level - q.level);
      }
    return w;
  };

  it('keeps blocks that share an edge or vertex within one level, through refining and merging', () => {
    for (const [ndim, bc] of [[2, periodic], [2, outflow], [3, periodic], [3, outflow]] as const) {
      const t = new BlockTree(ndim, 2, ndim === 2 ? 4 : 3, 4000, [...bc], [...bc], true);
      let seed = 3;
      const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      for (let round = 0; round < 12; round++) {
        const ind = Array.from({ length: 4000 }, () => rnd() * 1.2);
        t.adapt((x) => ind[x.slot], 0.85, 0.3);
        balanced(t);
        covers(t);
        expect(worst(t)).toBeLessThanOrEqual(1);
      }
    }
  });

  it('is needed: without it a single deep refinement leaves diagonal blocks two levels apart', () => {
    for (const ndim of [2, 3] as const) {
      const without = new BlockTree(ndim, 2, 3, 4000, periodic, periodic);
      const withIt = new BlockTree(ndim, 2, 3, 4000, periodic, periodic, true);
      for (const t of [without, withIt]) {
        // Refine toward one corner of a root block, three times.
        let b = t.leaves()[0];
        for (let l = 0; l < 3; l++) {
          t.refine(b);
          b = b.children![0];
        }
      }
      expect(worst(without)).toBe(2);
      expect(worst(withIt)).toBe(1);
    }
  });
});
