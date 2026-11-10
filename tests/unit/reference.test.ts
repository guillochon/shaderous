import { describe, expect, it } from 'vitest';
import { exactRiemann } from '../../src/reference/riemann_exact';
import { LIMITERS, RefHydro, type RefConfig } from '../../src/reference/hydro';
import { defaultEosParams, makeEos } from '../../src/eos/eos';

const base = (over: Partial<RefConfig> = {}): RefConfig => ({
  nx: 128, ny: 1, nz: 1, dx: 1 / 128,
  eos: makeEos('ideal', { ...defaultEosParams(), gamma: 1.4 }),
  bcLo: [1, 0, 0], bcHi: [1, 0, 0],
  cfl: 0.8, rhoFloor: 1e-10, pFloor: 1e-12, etaDual: 1e-3, shockThresh: 0.5,
  riemann: 'hllc', limiter: 'mc',
  ...over,
});
const tube = (l: number[], r: number[]) => (x: number) => {
  const [rho, vx, p] = x < 0 ? l : r;
  return { rho, vx, vy: 0, vz: 0, p, s: x < 0 ? 1 : 0 };
};
const sum = (a: Float64Array) => a.reduce((s, v) => s + v, 0);

describe('exact Riemann solver', () => {
  it('reproduces the Sod star state (Toro, table 4.2)', () => {
    const sol = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
    const star = sol(0.5);
    expect(star.p).toBeCloseTo(0.30313, 4);
    expect(star.u).toBeCloseTo(0.92745, 4);
    expect(sol(-2).rho).toBe(1);
    expect(sol(2).rho).toBe(0.125);
    expect(sol(1.0).rho).toBeCloseTo(0.26557, 4); // between contact and shock
  });

  it('reproduces the strong-shock star state (Toro test 3)', () => {
    const star = exactRiemann({ rho: 1, u: 0, p: 1000 }, { rho: 1, u: 0, p: 0.01 }, 1.4)(19);
    expect(star.p).toBeCloseTo(460.894, 2);
    expect(star.u).toBeCloseTo(19.5975, 3);
  });
});

describe('limiters', () => {
  it('vanish at extrema, keep the sign, and are exact for linear data', () => {
    for (const lim of Object.values(LIMITERS)) {
      expect(lim(1, -1)).toBe(0);
      expect(lim(0, 3)).toBe(0);
      for (const [a, b] of [[1, 2], [2, 1], [-1, -4], [0.3, 0.3]]) {
        const s = lim(a, b);
        expect(Math.sign(s)).toBe(Math.sign(a));
        expect(Math.abs(s)).toBeLessThanOrEqual(2 * Math.min(Math.abs(a), Math.abs(b)) + 1e-15);
      }
      expect(lim(0.3, 0.3)).toBeCloseTo(0.3, 14);
    }
  });
});

describe('float64 reference scheme', () => {
  it('converges on the Sod tube', () => {
    const exact = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
    const err = (n: number) => {
      const h = new RefHydro(base({ nx: n, dx: 1 / n }));
      h.init(tube([1, 0, 1], [0.125, 0, 0.1]));
      h.run(0.2);
      expect(h.t).toBeCloseTo(0.2, 12);
      let e = 0;
      for (let i = 0; i < n; i++) e += Math.abs(h.U[0][i] - exact(((i + 0.5) / n - 0.5) / 0.2).rho);
      return e / n;
    };
    const e64 = err(64);
    const e128 = err(128);
    expect(e128).toBeLessThan(0.005);
    expect(e128).toBeLessThan(0.7 * e64);
  });

  it('is second order on a smooth advected profile', () => {
    const err = (n: number) => {
      const h = new RefHydro(base({ nx: n, dx: 1 / n, bcLo: [0, 0, 0], bcHi: [0, 0, 0] }));
      const rho0 = (x: number) => 1.5 + 0.2 * Math.sin(2 * Math.PI * x);
      h.init((x) => ({ rho: rho0(x), vx: 1, vy: 0, vz: 0, p: 1, s: 0 }));
      h.run(1);
      let e = 0;
      for (let i = 0; i < n; i++) e += Math.abs(h.U[0][i] - rho0((i + 0.5) / n - 0.5));
      return e / n;
    };
    expect(Math.log2(err(32) / err(64))).toBeGreaterThan(1.7);
  });

  it('conserves mass, momentum and energy to round-off in a periodic box', () => {
    const h = new RefHydro(base({ nx: 16, ny: 16, nz: 4, dx: 1 / 16, bcLo: [0, 0, 0], bcHi: [0, 0, 0] }));
    h.init((x, y, z) => ({
      rho: 1.5 + Math.exp(-(x * x + y * y) / 0.02), vx: 1, vy: 0.5, vz: 0.1 * Math.sin(8 * Math.PI * z),
      p: 1 + 0.5 * Math.exp(-(x * x + y * y) / 0.01), s: 0,
    }));
    const before = [0, 1, 2, 4].map((i) => sum(h.U[i]));
    for (let i = 0; i < 40; i++) h.step();
    [0, 1, 2, 4].forEach((i, k) => expect(Math.abs(sum(h.U[i]) - before[k]) / Math.abs(before[k])).toBeLessThan(1e-12));
  });

  it('keeps density and pressure positive through a pressure ratio of 1e5 without floors', () => {
    const h = new RefHydro(base());
    h.init(tube([1, 0, 1000], [1, 0, 0.01]));
    h.run(0.012);
    for (let i = 0; i < h.ncell; i++) {
      expect(h.U[0][i]).toBeGreaterThan(0.5);
      expect(0.4 * h.U[5][i]).toBeGreaterThan(0.009);
    }
  });

  it('gives the same answer along x, y and z', () => {
    const run = (axis: number) => {
      const dims = [1, 1, 1];
      dims[axis] = 48;
      const bc: [number, number, number] = [0, 0, 0];
      bc[axis] = 1;
      const h = new RefHydro(base({ nx: dims[0], ny: dims[1], nz: dims[2], dx: 1 / 48, bcLo: bc, bcHi: bc }));
      h.init((x, y, z) => {
        const s = [x, y, z][axis];
        const v = [0, 0, 0];
        v[axis] = 0.3;
        return { rho: s < 0 ? 1 : 0.125, vx: v[0], vy: v[1], vz: v[2], p: s < 0 ? 1 : 0.1, s: 0 };
      });
      h.run(0.15);
      return Array.from(h.U[0]);
    };
    const x = run(0);
    expect(run(1)).toEqual(x);
    expect(run(2)).toEqual(x);
  });

  it('reflecting walls keep a symmetric flow symmetric and the box closed', () => {
    const h = new RefHydro(base({ nx: 32, dx: 1 / 32, bcLo: [2, 0, 0], bcHi: [2, 0, 0] }));
    h.init((x) => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: Math.abs(x) < 0.1 ? 10 : 0.1, s: 0 }));
    const m0 = sum(h.U[0]);
    h.run(0.5);
    expect(Math.abs(sum(h.U[0]) - m0) / m0).toBeLessThan(1e-12);
    for (let i = 0; i < 16; i++) expect(h.U[0][i]).toBeCloseTo(h.U[0][31 - i], 10);
  });

  it('barotropic gas slaves the internal energy to the density', () => {
    const h = new RefHydro(base({ eos: makeEos('barotropic', { ...defaultEosParams(), gamma: 1, K: 1 }) }));
    h.init(tube([1, 0, 1], [0.25, 0, 0.25]));
    h.run(0.1);
    for (let i = 0; i < h.ncell; i++) expect(h.U[5][i]).toBeCloseTo(1.5 * h.U[0][i], 12);
  });
});
