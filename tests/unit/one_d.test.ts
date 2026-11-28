// 1D runs (plans/one-dimensional.md): the grid's shape, the block tree, the profile lock, links,
// and what a 1D run refuses.
import { describe, expect, it } from 'vitest';
import { domainHi, domainLo, gridDims, rectangular } from '../../src/domain';
import { BlockTree, BT_STRIDE, RG_RESTRICT } from '../../src/amr/tree';
import { clipSegment, decodeProfileLines, encodeProfileLines, lockLine, type Vec3 } from '../../src/ui/profile';
import { decodeLink, encodeLink, viewDefaults } from '../../src/ui/link';
import { configFor, lineGravity, onBlocks, oneDimUnsupported, refusal } from '../../src/sim';
import { mapWindow } from '../../src/render/renderer';
import { problemById } from '../../src/problems';
import { defaultOpacity } from '../../src/render/transfer';
import { factsOf, OPTIONS, unavailable } from '../../src/setup/options';
import { curvedView, sphFromView, sphGrid, sphToView, sphViewParam } from '../../src/spherical';
import type { SetupSpec } from '../../src/setup/spec';

const shape = (n: number, extra: Partial<{ symmetry: boolean[]; amrLevels: number }> = {}) => ({ n, ndim: 1 as const, problem: { box: 1 } as { box: number; mhd?: unknown }, ...extra });

describe('1D grid', () => {
  it('is a row of n cells, over the whole box along y and z', () => {
    expect(gridDims(shape(1024))).toEqual([1024, 1, 1]);
    expect(rectangular(shape(1024))).toBe(false);
    expect(domainLo(shape(1024))).toEqual([-0.5, -0.5, -0.5]);
    expect(domainHi(shape(1024))).toEqual([0.5, 0.5, 0.5]);
  });

  it('mirrored along x runs from 0 over the box with all n cells', () => {
    const s = shape(512, { symmetry: [true, false, false] });
    expect(gridDims(s)).toEqual([512, 1, 1]);
    expect(domainLo(s)[0]).toBe(0);
  });
});

describe('1D block tree', () => {
  /** A tree refined toward x = 0.37 until its deepest level. */
  function refined(levels: number, nr = 2) {
    const t = new BlockTree(1, nr, levels, 2000, [1, 0, 0], [1, 0, 0], false, [nr, 1, 1]);
    for (let round = 0; round < levels + 2; round++) {
      t.adapt((b) => {
        const size = 1 / (nr << b.level);
        return 0.37 >= b.c[0] * size && 0.37 < (b.c[0] + 1) * size ? 1 : 0;
      }, 0.6, 0.2);
    }
    return t;
  }

  it('splits each block in two along x, and keeps 2:1 balance', () => {
    const t = refined(5);
    expect(t.nchild).toBe(2);
    expect(t.nface).toBe(2);
    expect(t.stats().deepest).toBe(5);
    for (const b of t.leaves()) {
      expect(b.c[1]).toBe(0);
      for (let f = 0; f < 2; f++) t.neighbor(b, f); // throws if a region is two levels coarser
    }
  });

  it('indexes the finest blocks in a row, each leaf over its span once', () => {
    const t = refined(4);
    const g = t.indexGrid();
    const m = t.nr << t.maxLevel;
    expect(g.length).toBe(m);
    const count = new Map<number, number>();
    for (const s of g) count.set(s, (count.get(s) ?? 0) + 1);
    for (const b of t.leaves()) expect(count.get(b.slot)).toBe(1 << (t.maxLevel - b.level));
  });

  it('lays each block out as its own neighbor along y, each child for both octants along y', () => {
    const t = refined(3);
    const tab = t.table();
    for (const b of t.all()) {
      const o = b.slot * BT_STRIDE;
      for (let f = 2; f < 6; f++) expect(tab[o + 6 + f]).toBe(b.slot);
      if (b.children) {
        expect([tab[o + 12], tab[o + 13]]).toEqual(b.children.map((c) => c.slot));
        expect([tab[o + 14], tab[o + 15]]).toEqual(b.children.map((c) => c.slot));
      }
    }
  });

  it('restricts a merged parent from its two children, standing for both octants along y', () => {
    const t = refined(3);
    // Unflag everything: the refined blocks merge back.
    const ops = t.adapt(() => 0, 0.6, 0.2);
    const restricts = (ops ?? []).filter((op) => op && op.kind === RG_RESTRICT);
    expect(restricts.length).toBeGreaterThan(0);
    for (const op of restricts) {
      expect(op.children.length).toBe(4);
      expect(op.children.slice(2)).toEqual(op.children.slice(0, 2));
    }
  });
});

describe('1D profile lines', () => {
  const a: Vec3 = [0.2, 0.3, 0];
  const b: Vec3 = [0.8, 0.7, 0];
  it('are horizontal in a Cartesian run, about the end that stayed', () => {
    expect(lockLine(a, b, 1, 1, false)).toEqual([a, [0.8, 0.3, 0]]);
    expect(lockLine(a, b, 0, 1, false)).toEqual([[0.2, 0.7, 0], b]);
  });
  it('are radial in a curved run, from the center to the outer end at its angle', () => {
    expect(lockLine(a, b, 1, 1, true)).toEqual([[0, 0.7, 0], [0.8, 0.7, 0]]);
    expect(lockLine(a, b, 0, 1, true)).toEqual([[0, 0.3, 0], [0.2, 0.3, 0]]);
  });
  it('are as drawn in 2D and 3D', () => {
    expect(lockLine(a, b, 1, 2, false)).toEqual([a, b]);
    expect(lockLine(a, b, 1, 3, true)).toEqual([a, b]);
  });
  it('travel in links as their ends along x', () => {
    const s = encodeProfileLines([{ a: [0.1, 0.5, 0], b: [0.9, 0.5, 0] }], 1);
    expect(s).toBe('0.1,0.9');
    expect(decodeProfileLines(s, 1)).toEqual([[[0.1, 0.5, 0], [0.9, 0.5, 0]]]);
    // (Ends with a y too are read, keeping it.)
    expect(decodeProfileLines('0.1,0.2,0.9,0.2', 1)).toEqual([[[0.1, 0.2, 0], [0.9, 0.2, 0]]]);
    expect(decodeProfileLines('0.1,0.2,0.9', 1)).toBeNull();
  });
  it('are clipped to the box along x', () => {
    expect(clipSegment([-0.5], [0.5], 1)).toEqual([[0, 0.5, 0], [0.5, 0.5, 0]]);
  });
});

describe('1D grids up to 262144 cells', () => {
  it('index a leaf a finest block, 16384 of them at 262144 cells', () => {
    // 262144 cells, 7 levels: 128 root blocks, 16384 finest blocks.
    const t = new BlockTree(1, 128, 7, 40000, [1, 0, 0], [1, 0, 0], false, [128, 1, 1]);
    t.adapt((b) => (b.c[0] === 0 ? 1 : 0), 0.6, 0.2);
    expect(t.indexGrid().length).toBe(16384);
  });
  it('are offered by the option table and kept by links', () => {
    const n = OPTIONS.find((o) => o.path === 'domain.n')!;
    expect(n.control.type === 'number' && n.control.max).toBe(262144);
    const l = decodeLink(new URLSearchParams('problem=sod-x&dim=1&n=262144&amr=7'));
    expect([l.ndim, l.n, l.amr]).toEqual([1, 262144, 7]);
  });
});

describe('uniform 1D grids past a texture\'s side', () => {
  it('run on blocks with no levels', () => {
    const c = (n: number, amrLevels = 0) => configFor(problemById('sod-x'), n, { ndim: 1, amrLevels });
    expect(onBlocks(c(2048))).toBe(false);
    expect(onBlocks(c(4096))).toBe(true);
    expect(onBlocks(c(262144))).toBe(true);
    expect(onBlocks(configFor(problemById('sod-x'), 4096, { ndim: 2 }))).toBe(false);
  });
});

describe('a 1D map over what is in view', () => {
  const lay = { hw: 0.5, hh: 0.5, shift: 0 };
  const view = (zoom: number, cx: number) => ({ zoom, cx, cy: 0.5, mirrorImages: false });
  const sod = (n: number) => configFor(problemById('sod-x'), n, { ndim: 1, amrLevels: n > 2048 ? 3 : 0 });
  it('is the whole box for a run its map holds, or a view that sees most of it', () => {
    expect(mapWindow(sod(4096), view(64, 0.5), 8192, lay)).toBeNull();
    expect(mapWindow(sod(262144), view(1, 0.5), 8192, lay)).toBeNull();
  });
  it('is what is in view, a little wider, zoomed in', () => {
    const w = mapWindow(sod(262144), view(100, 0.3), 8192, lay)!;
    expect(w[0]).toBeLessThan(0.3 - 0.005);
    expect(w[1]).toBeGreaterThan(0.3 + 0.005);
    expect(w[1] - w[0]).toBeLessThan(0.0105 + 4 / 262144);
  });
  it('holds every cell in view at a texel a cell once zoomed in far enough', () => {
    const w = mapWindow(sod(262144), view(64, 0.5), 8192, lay)!;
    expect((w[1] - w[0]) * 262144).toBeLessThanOrEqual(8192);
  });
  it('is a range of radii in a curved run', () => {
    const c = configFor(problemById('noh-spherical'), 65536, { ndim: 1 });
    const w = mapWindow(c, view(300, 0.05), 8192, lay)!;
    expect(w[0]).toBeGreaterThan(0.08);
    expect(w[1]).toBeLessThan(0.12);
  });
});

describe('1D links', () => {
  it('carry dim=1 both ways', () => {
    const problem = problemById('sod-x');
    const cfg = configFor(problem, 1024, { ndim: 1 });
    const q = encodeLink({ cfg, view: viewDefaults(problem, 1), auto: true, colormap: 'inferno', opacity: defaultOpacity() });
    expect(q.get('dim')).toBe('1');
    const l = decodeLink(q);
    expect([l.problem, l.ndim, l.n]).toEqual(['sod-x', 1, 1024]);
  });
  it('open on the slice view', () => {
    expect(viewDefaults(problemById('sod-x'), 1).mode).toBe('slice');
  });
});

describe('1D configuration', () => {
  it('makes y and z periodic, but a spherical run keeps its poles', () => {
    const c = configFor(problemById('sod-x'), 1024, { ndim: 1 });
    expect([c.bcLo[1], c.bcHi[1], c.bcLo[2], c.bcHi[2]]).toEqual([0, 0, 0, 0]);
    const s = configFor(problemById('noh-spherical'), 1024, { ndim: 1 });
    expect(s.bcLo[1]).toBe(problemById('noh-spherical').bcLo[1]);
  });
  it('takes the presets that say they make sense in 1D', () => {
    for (const id of ['sod-x', 'riemann-leblanc', 'noh-planar', 'noh-cylindrical', 'noh-spherical', 'jeans']) {
      const p = problemById(id);
      expect(p.supports1d).toBe(true);
      expect(refusal(configFor(p, 1024, { ndim: 1 }))).toBeNull();
      expect(refusal(configFor(p, 4096, { ndim: 1, amrLevels: 3 }))).toBeNull();
    }
  });
  it('refuses what 1D does not take, naming the phase that brings it', () => {
    const c = configFor(problemById('sod-x'), 1024, { ndim: 1 });
    expect(oneDimUnsupported({ ...c, viscosity: 0.1 })).toMatch(/phase 7/);
    expect(oneDimUnsupported({ ...c, cooling: 'power' } as never)).toMatch(/phase 6/);
    expect(oneDimUnsupported({ ...c, ndim: 2 })).toBeNull();
  });
  it('solves its self-gravity as its geometry asks', () => {
    expect(lineGravity(configFor(problemById('jeans'), 512, { ndim: 1 }))).toBe(2);
    expect(lineGravity(configFor(problemById('noh-spherical'), 512, { ndim: 1 }))).toBe(0);
    expect(lineGravity(configFor(problemById('noh-cylindrical'), 512, { ndim: 1 }))).toBe(3);
  });
});

describe('1D in the option table', () => {
  const dims = OPTIONS.find((o) => o.path === 'domain.ndim')!;
  const one = dims.control.type === 'select' ? dims.control.choices!.find((c) => c.value === 1)! : undefined;
  const doc = (extra: Partial<SetupSpec> = {}): SetupSpec => ({ format: 'shaderous-setup', version: 1, domain: { supports1d: true, ndim: 1 }, physics: { eos: 'ideal' }, ...extra }) as SetupSpec;
  it('offers 1D to a setup that makes sense in it', () => {
    expect(one).toBeDefined();
    expect(unavailable(one!, factsOf(doc()))).toEqual([]);
  });
  it('withholds it from one that does not say so, or that has what 1D does not take', () => {
    expect(unavailable(one!, factsOf(doc({ domain: { ndim: 1 } })))).not.toEqual([]);
    expect(unavailable(one!, factsOf(doc({ physics: { eos: 'ideal', radiation: 'm1' } } as never)))).not.toEqual([]);
  });
});

describe('the 1D axisymmetric view', () => {
  const c = { problem: { box: 1, geometry: 'cylindrical' }, n: 256, ndim: 1 };
  it('is the half-disk with no inner radius', () => {
    expect(curvedView(c)).toBe(true);
    expect(sphViewParam(c)).toBeGreaterThan(0);
    expect(sphViewParam(c)).toBeLessThan(1e-20);
    expect(sphGrid(c).rMin).toBe(0);
  });
  it('maps the view to (R, angle) and back', () => {
    for (const [fx, fy] of [[0.1, 0.2], [0.5, 0.5], [0.9, 0.95]]) {
      const [a, b] = sphToView(c, fx, fy);
      const back = sphFromView(c, a, b)!;
      expect(back[0]).toBeCloseTo(fx, 6);
      expect(back[1]).toBeCloseTo(fy, 6);
    }
  });
});
