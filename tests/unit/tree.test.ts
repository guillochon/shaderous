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
