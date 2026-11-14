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
