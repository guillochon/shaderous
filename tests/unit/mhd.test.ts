import { describe, expect, it } from 'vitest';
import { cellField, initialFaces } from '../../src/mhd/mhd';
import type { SimConfig } from '../../src/sim';
import { orszagTang } from '../../src/problems/mhd';
import { all, OUTFLOW, PERIODIC, type Problem } from '../../src/problems/types';

const cfgOf = (problem: Problem, ndim: 2 | 3, periodic = true) =>
  ({ problem, ndim, params: {}, bcLo: all(periodic ? PERIODIC : OUTFLOW), bcHi: all(periodic ? PERIODIC : OUTFLOW) }) as unknown as SimConfig;

/** Largest |div b| dx over all cells, from the faces, and the largest face value. */
function divergence(f: Float32Array, n: number, nz: number, periodic: boolean) {
  const fx = n + 1;
  const up = (m: number, len: number) => (periodic && m + 1 === len ? 0 : m + 1);
  const at = (a: number, i: number, j: number, k: number) => f[4 * ((k * fx + j) * fx + i) + a];
  let div = 0;
  let big = 0;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const kz = nz === 1 ? 0 : up(k, nz);
        const d = at(0, up(i, n), j, k) - at(0, i, j, k) + at(1, i, up(j, n), k) - at(1, i, j, k) + at(2, i, j, kz) - at(2, i, j, k);
        div = Math.max(div, Math.abs(d));
        big = Math.max(big, Math.abs(at(0, i, j, k)), Math.abs(at(1, i, j, k)), Math.abs(at(2, i, j, k)));
      }
  return { div, big };
}

describe('initial magnetic fields (mhd/mhd.ts)', () => {
  it('a vector potential gives a divergence-free face field in 3D, with an outflow box', () => {
    // A tangled field: every component of A depends on all three coordinates.
    const p: Problem = {
      ...orszagTang,
      mhd: {
        potential: () => (x, y, z) => [Math.sin(3 * y + z) * Math.cos(2 * x), Math.cos(5 * z - x) * y, Math.sin(4 * x * y + z)],
      },
    };
    const n = 12;
    const f = initialFaces(cfgOf(p, 3, false), n, n, 1 / n);
    const { div, big } = divergence(f, n, n, false);
    expect(big).toBeGreaterThan(0.5);
    expect(div / big).toBeLessThan(1e-6);
  });

  it('the Orszag-Tang field is divergence-free on a periodic 2D grid and matches the analytic field', () => {
    const n = 64;
    const c = cfgOf(orszagTang, 2);
    const f = initialFaces(c, n, 1, 1 / n);
    const { div, big } = divergence(f, n, 1, true);
    expect(div / big).toBeLessThan(1e-6);
    // Cell-centered field against B = (-sin 2 pi y, sin 4 pi x) / sqrt(4 pi), to second order.
    const s = 1 / Math.sqrt(4 * Math.PI);
    let err = 0;
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const B = cellField(c, f, n, 1, i, j, 0);
        const X = (i + 0.5) / n;
        const Y = (j + 0.5) / n;
        err = Math.max(err, Math.abs(B[0] + s * Math.sin(2 * Math.PI * Y)), Math.abs(B[1] - s * Math.sin(4 * Math.PI * X)), Math.abs(B[2]));
      }
    expect(err).toBeLessThan(5e-3 * s);
  });
});

describe('initial magnetic fields on blocks (mhd/amr_mhd.ts)', () => {
  it('gives a coarse block faces that are exactly the means of its children faces', async () => {
    const { blockFaces } = await import('../../src/mhd/amr_mhd');
    const BS = 16;
    const p: Problem = {
      ...orszagTang,
      box: 2,
      mhd: { potential: () => (x, y, z) => [Math.sin(3 * y + z) * Math.cos(2 * x), Math.cos(5 * z - x) * y, Math.sin(4 * x * y + z)] },
    };
    for (const ndim of [2, 3] as const) {
      const c = { ...cfgOf(p, ndim), ndim } as SimConfig;
      const dx0 = 2 / BS; // one root block across the box
      const coarse = blockFaces(c, { slot: 0, level: 0, c: [0, 0, 0], parent: null, children: null }, dx0, 1);
      const W = BS + 1;
      const D = ndim === 3 ? W : 1;
      const at = (f: Float32Array, a: number, i: number, j: number, k: number) => f[4 * ((k * W + j) * W + i) + a];
      const kids = new Map<string, Float32Array>();
      for (let o = 0; o < (ndim === 3 ? 8 : 4); o++) {
        const cc = [o & 1, (o >> 1) & 1, (o >> 2) & 1];
        kids.set(cc.join(), blockFaces(c, { slot: 1, level: 1, c: cc, parent: null, children: null }, dx0, 1));
      }
      // A fine face at global fine index g (normal a), from the child that holds it.
      const fine = (a: number, g: number[]) => {
        const cc = g.map((v, ax) => Math.min(ax === a ? (v === 2 * BS ? 1 : v >> 4) : v >> 4, 1));
        if (ndim === 2) cc[2] = 0;
        const l = g.map((v, ax) => v - cc[ax] * BS);
        return at(kids.get(cc.join())!, a, l[0], l[1], ndim === 3 ? l[2] : 0);
      };
      let worst = 0;
      let big = 0;
      for (let k = 0; k < (ndim === 3 ? BS : 1); k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++)
            for (let a = 0; a < ndim; a++) {
              const g = [i, j, k];
              const v = at(coarse, a, i, j, k);
              const t = [0, 1, 2].filter((x) => x !== a && (ndim === 3 || x !== 2));
              let sum = 0;
              let n = 0;
              for (let q = 0; q < (t.length > 1 ? 2 : 1); q++)
                for (let r = 0; r < 2; r++) {
                  const gf = g.map((x) => 2 * x);
                  gf[t[0]] += r;
                  if (t.length > 1) gf[t[1]] += q;
                  if (ndim === 2) gf[2] = 0;
                  sum += fine(a, gf);
                  n++;
                }
              worst = Math.max(worst, Math.abs(sum / n - v));
              big = Math.max(big, Math.abs(v));
            }
      void D;
      expect(big).toBeGreaterThan(0.3);
      expect(worst / big).toBeLessThan(1e-6);
    }
  });
});
