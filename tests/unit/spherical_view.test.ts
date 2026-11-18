import { describe, expect, it } from 'vitest';
import { sphFromView, sphGrid, sphToView, sphViewParam } from '../../src/spherical';

const shape = (log: boolean) => ({ problem: { box: 0.5, geometry: 'spherical', sphere: { rMin: 0.005, log } }, n: 64, ndim: 2 });

describe('spherical slice view: the (R, z) half-plane', () => {
  for (const log of [false, true]) {
    const c = shape(log);
    const g = sphGrid(c);
    it(`maps cells to (R, z) and back (${log ? 'logarithmic' : 'uniform'} r)`, () => {
      for (const [i, j] of [[0, 0], [10, 31], [63, 63], [40, 5]]) {
        const fx = (i + 0.5) / g.n;
        const fy = (j + 0.5) / g.n;
        const [a, b] = sphToView(c, fx, fy);
        // R = r sin theta, z = r cos theta, in units of the square's side (2 r_max).
        expect(2 * a * g.rMax).toBeCloseTo(g.r(i) * Math.sin(g.theta(j)), 10);
        expect((2 * b - 1) * g.rMax).toBeCloseTo(g.r(i) * Math.cos(g.theta(j)), 10);
        const back = sphFromView(c, a, b)!;
        expect(back[0]).toBeCloseTo(fx, 10);
        expect(back[1]).toBeCloseTo(fy, 10);
      }
    });
    it(`is off the grid inside r_min, beyond r_max and at R < 0 (${log ? 'logarithmic' : 'uniform'} r)`, () => {
      expect(sphFromView(c, 0, 0.5)).toBeNull();
      expect(sphFromView(c, 0.51, 0.5)).toBeNull();
      expect(sphFromView(c, -0.01, 0.6)).toBeNull();
      expect(sphFromView(c, 0.3, 0.5)).not.toBeNull();
      expect(sphViewParam(c)).toBeCloseTo((log ? -1 : 1) * (0.005 / 0.505), 12);
    });
  }
  it('leaves other runs alone', () => {
    expect(sphViewParam({ problem: { box: 1, geometry: 'cylindrical' }, n: 64, ndim: 2 })).toBe(0);
    expect(sphViewParam({ problem: { box: 1 }, n: 64, ndim: 3 })).toBe(0);
  });
});
