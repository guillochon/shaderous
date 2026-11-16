// Presets against the hand-written problems they replace: every preset file, compiled, must
// give the same configuration and start bit for bit as its old object. The old objects live in
// tests/legacy until the last preset has passed.
import { beforeAll, describe, expect, it } from 'vitest';
import { makeEos } from '../../src/eos/eos';
import { loadHelmholtz } from '../../src/eos/helmholtz';
import { PRESET_IDS, preset } from '../../src/presets/index';
import { settlingOf, type Settling } from '../../src/problems/settle';
import type { Prim, Problem } from '../../src/problems/types';
import { problemUnits } from '../../src/problems/types';
import { configFor } from '../../src/sim';
import { LEGACY } from '../legacy/index';

const same = (a: number, b: number) => Object.is(Math.fround(a), Math.fround(b)) || Object.is(a, b);

/** Differences between two plain values, by path; functions are skipped. */
function diff(a: unknown, b: unknown, path: string, out: string[], skip: (path: string) => boolean = () => false): void {
  if (skip(path)) return;
  if (typeof a === 'function' || typeof b === 'function') {
    if (typeof a !== typeof b) out.push(`${path}: ${typeof a} vs ${typeof b}`);
    return;
  }
  if (typeof a === 'number' && typeof b === 'number') {
    if (!Object.is(a, b)) out.push(`${path}: ${a} vs ${b}`);
    return;
  }
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') {
    if (a !== b) out.push(`${path}: ${String(a)} vs ${String(b)}`);
    return;
  }
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    const [x, y] = [(a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]];
    if (x === undefined && y === undefined) continue;
    diff(x, y, path ? `${path}.${k}` : k, out, skip);
  }
}

function primDiff(a: Prim, b: Prim): string {
  for (const k of ['rho', 'vx', 'vy', 'vz', 'p', 's'] as const) if (!same(a[k], b[k])) return `${k}: ${a[k]} vs ${b[k]}`;
  if (!!a.solid !== !!b.solid) return 'solid';
  if (!!a.fixed !== !!b.fixed) return 'fixed';
  if (!!a.X !== !!b.X) return 'X present';
  if (a.X && b.X && (a.X.length !== b.X.length || a.X.some((v, i) => !same(v, b.X![i])))) return `X: ${a.X} vs ${b.X}`;
  return '';
}

/** Sample points: cell centers of an n^ndim grid on a stride, and a few off-lattice ones. */
function points(box: number, n: number, ndim: 2 | 3): [number, number, number][] {
  const out: [number, number, number][] = [];
  const dx = box / n;
  const at = (i: number) => (i + 0.5) * dx - box / 2;
  const step = n > 32 ? 3 : 1;
  const ks = ndim === 2 ? [-1] : Array.from({ length: Math.ceil(n / (step * 2)) }, (_, k) => k * step * 2);
  for (const k of ks) for (let j = 0; j < n; j += step) for (let i = 0; i < n; i += step) out.push([at(i), at(j), k < 0 ? 0 : at(k)]);
  let seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5) * box;
  for (let q = 0; q < 300; q++) out.push([rnd(), rnd(), ndim === 2 ? 0 : rnd()]);
  return out;
}

/** The settling of a problem as the solvers ask for it. */
const settling = (p: Problem, params: Record<string, number>, eos: ReturnType<typeof makeEos>, G: number): Settling | null =>
  p.settle ? p.settle(params, eos) : p.star ? settlingOf(p.star, G) : null;

/** Presets whose values are known to differ from the old object in a way that changes nothing, and why. */
const SKIP: Record<string, (path: string) => boolean> = {};

function compare(id: string, params?: Record<string, number>, light = false) {
  const old = LEGACY[id];
  const neu = preset(id).problem;
  const problems: string[] = [];
  const dims: (2 | 3)[] = old.supports2d ? [2, 3] : [3];
  for (const ndim of dims) {
    const over = { ndim, amrLevels: old.grid?.amr ?? 0, ...(params ? {} : {}) };
    const a = configFor(old, 64, over);
    const b = configFor(neu, 64, { ...preset(id).overrides, ...over });
    if (params) {
      Object.assign(a.params, params);
      Object.assign(b.params, params);
    }
    const out: string[] = [];
    // The configuration, field by field. The problems are compared below.
    diff({ ...a, problem: null }, { ...b, problem: null }, 'cfg', out);
    if (!params) {
      // Everything the problem holds as data. `star` became `settle`, compared as settlings.
      diff(old, neu, 'problem', out, (path) =>
        ['problem.setup', 'problem.star', 'problem.settle', 'problem.dir', 'problem.left', 'problem.right'].includes(path) || /^problem\.params\.\d+\.unit/.test(path) || (SKIP[id]?.(path) ?? false));
      const [pa, pb] = [old.params ?? [], neu.params ?? []];
      pa.forEach((q, i) => {
        if (q.unit && (!pb[i]?.unit || q.unit.label !== pb[i].unit!.label || q.unit.factor(a.params) !== pb[i].unit!.factor(b.params))) out.push(`param ${q.id}: unit`);
      });
    }
    diff(problemUnits(old, a.params), problemUnits(neu, b.params), 'units', out);
    diff(old.gext?.(ndim), neu.gext?.(ndim), 'gext', out);
    diff(old.burn?.(a.params), neu.burn?.(b.params), 'burn', out);
    diff(old.debris?.(a.params, a.G), neu.debris?.(b.params, b.G), 'debris', out);
    for (const dx of [old.box / 64, old.box / 512]) diff(old.masses?.(a.params, a.G, dx, ndim), neu.masses?.(b.params, b.G, dx, ndim), `masses(dx=${dx})`, out);

    const eos = makeEos(a.eos, a.eosParams);
    for (const n of light ? [32] : [32, 128]) {
      const fa = old.init(n, eos, ndim, a.params);
      const fb = neu.init(n, eos, ndim, b.params);
      let bad = 0;
      for (const [x, y, z] of points(old.box, n, ndim)) {
        const d = primDiff(fa(x, y, z), fb(x, y, z));
        if (d && bad++ < 3) out.push(`init n=${n} ${ndim}D at (${x}, ${y}, ${z}): ${d}`);
      }
      if (bad > 3) out.push(`init n=${n} ${ndim}D: ${bad} points differ`);
    }

    if (ndim === 3) {
      const [sa, sb] = [settling(old, a.params, eos, a.G), settling(neu, b.params, eos, b.G)];
      if (!!sa !== !!sb) out.push('settling: one has none');
      if (sa && sb) {
        // A lone star's nominal radius only ranks stars against each other.
        diff(sa, sb, 'settle', out, (path) => /^settle\.stars\.\d+\.(R|barotrope)$/.test(path) && (sa.stars.length === 1 || path.endsWith('barotrope')));
        sa.stars.forEach((st, i) => {
          for (const rho of [1e-6, 1e-3, 0.3, 1, 7]) if (!same(st.barotrope.pressure(rho), sb.stars[i].barotrope.pressure(rho))) out.push(`settle: star ${i} pressure(${rho})`);
          for (const h of [-1, 1e-3, 0.5, 3]) if (!same(st.barotrope.rhoOfH(h), sb.stars[i].barotrope.rhoOfH(h))) out.push(`settle: star ${i} rhoOfH(${h})`);
          diff(st.barotrope.gpu, sb.stars[i].barotrope.gpu, `settle.stars.${i}.gpu`, out);
        });
      }
      const [ha, hb] = [old.heat?.(a.params, eos), neu.heat?.(b.params, eos)];
      if (!!ha !== !!hb) out.push('heat: one has none');
      if (ha && hb) for (const [x, y, z] of points(old.box, 32, 3)) for (const rho of [1e-7, 1e-2, 1]) if (!Object.is(ha(x, y, z, rho), hb(x, y, z, rho))) out.push(`heat at (${x}, ${y}, ${z})`);
    }

    const [ma, mb] = [old.mhd, neu.mhd];
    if (!!ma !== !!mb) out.push('mhd: one has none');
    if (ma && mb) {
      if (!!ma.potential !== !!mb.potential || !!ma.face !== !!mb.face) out.push('mhd: potential or face in one only');
      const pts = points(old.box, 32, ndim).filter((_, i) => i % (light ? 29 : 7) === 0);
      if (ma.potential && mb.potential) {
        const [A, B] = [ma.potential(a.params, ndim), mb.potential(b.params, ndim)];
        let bad = 0;
        for (const [x, y, z] of pts) {
          const [u, v] = [A(x, y, z), B(x, y, z)];
          if (u.some((c, i) => !Object.is(c, v[i])) && bad++ < 2) out.push(`potential ${ndim}D at (${x}, ${y}, ${z}): ${u} vs ${v}`);
        }
      }
      if (ma.face && mb.face) {
        const [A, B] = [ma.face(a.params, ndim), mb.face(b.params, ndim)];
        for (const [x, y, z] of pts) for (let axis = 0; axis < 3; axis++) if (!Object.is(A(axis, x, y, z), B(axis, x, y, z))) out.push(`face ${axis} ${ndim}D at (${x}, ${y}, ${z})`);
      }
    }
    problems.push(...out.slice(0, 12).map((s) => `${ndim}D ${s}`));
  }
  return problems;
}

/** A second set of values for a preset's parameters: each a step from its default, inside its range. */
function otherParams(p: Problem): Record<string, number> {
  const out: Record<string, number> = {};
  for (const q of p.params ?? []) {
    if (q.kind === 'checkbox') out[q.id] = 1 - q.value;
    else {
      const next = q.factor ? q.value / q.factor : q.value + (q.step ?? 0.1 * (q.max - q.min));
      out[q.id] = next >= q.min && next <= q.max ? next : q.factor ? q.value * q.factor : q.value - (q.step ?? 0);
    }
  }
  return out;
}

describe('presets compiled from their files', () => {
  beforeAll(async () => {
    await loadHelmholtz();
  });

  it('are all there, each with the old problem it replaces', () => {
    expect([...PRESET_IDS].sort()).toEqual(Object.keys(LEGACY).sort());
  });

  for (const id of PRESET_IDS) {
    if (!LEGACY[id]) continue;
    it(`${id}: the same configuration and initial state`, () => {
      expect(compare(id)).toEqual([]);
    });
    if (LEGACY[id].params?.length)
      it(`${id}: the same with other parameter values`, () => {
        expect(compare(id, otherParams(LEGACY[id]), true)).toEqual([]);
      });
  }
});
