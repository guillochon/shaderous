import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos, quartic } from '../../src/eos/eos';

const grid: [number, number][] = [];
for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) grid.push([10 ** (i - 4), 10 ** (j - 4)]);
const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

describe('equations of state (float64 mirrors of the WGSL modules)', () => {
  it('ideal gas round-trips and has c^2 = gamma p / rho', () => {
    const e = makeEos('ideal', { ...defaultEosParams(), gamma: 1.4 });
    for (const [rho, p] of grid) {
      expect(rel(e.pressure(rho, e.eintFromP(rho, p)), p)).toBeLessThan(1e-14);
      expect(rel(e.soundSpeed(rho, p) ** 2, (1.4 * p) / rho)).toBeLessThan(1e-14);
    }
  });

  it('barotropic pressure depends on density only', () => {
    const iso = makeEos('barotropic', { ...defaultEosParams(), gamma: 1, K: 0.7 });
    const poly = makeEos('barotropic', { ...defaultEosParams(), gamma: 2, K: 0.3 });
    expect(iso.hasEnergy).toBe(false);
    for (const [rho] of grid) {
      expect(rel(iso.pressure(rho, 123), 0.7 * rho)).toBeLessThan(1e-14);
      expect(rel(iso.soundSpeed(rho, iso.pressure(rho, 0)) ** 2, 0.7)).toBeLessThan(1e-13);
      expect(rel(poly.pressure(rho, 0), 0.3 * rho * rho)).toBeLessThan(1e-13);
    }
  });

  it('quartic solver converges in its fixed iteration count', () => {
    for (const A of [0, 1e-6, 1, 1e4])
      for (const B of [1e-4, 1, 1e4])
        for (const C of [1e-6, 1, 1e6]) {
          const T = quartic(A, B, C);
          expect(rel(A * T ** 4 + B * T, C)).toBeLessThan(1e-9);
        }
  });

  it('gas + radiation round-trips and reduces to its limits', () => {
    const mixed = makeEos('gas_radiation', { ...defaultEosParams(), gamma: 5 / 3, R: 1, arad: 1 });
    const gas = makeEos('gas_radiation', { ...defaultEosParams(), gamma: 5 / 3, R: 1, arad: 0 });
    const rad = makeEos('gas_radiation', { ...defaultEosParams(), gamma: 5 / 3, R: 1e-14, arad: 1 });
    for (const [rho, p] of grid) {
      expect(rel(mixed.pressure(rho, mixed.eintFromP(rho, p)), p)).toBeLessThan(1e-8);
      // no radiation: ideal gas
      expect(rel(gas.eintFromP(rho, p), p / (5 / 3 - 1))).toBeLessThan(1e-12);
      expect(rel(gas.soundSpeed(rho, p) ** 2, ((5 / 3) * p) / rho)).toBeLessThan(1e-12);
      // radiation only: e = 3p, Gamma_1 = 4/3
      expect(rel(rad.eintFromP(rho, p), 3 * p)).toBeLessThan(1e-6);
      expect(rel(rad.soundSpeed(rho, p) ** 2, ((4 / 3) * p) / rho)).toBeLessThan(1e-6);
    }
  });

  it('gas + radiation sound speed lies between the two limits', () => {
    const e = makeEos('gas_radiation', { ...defaultEosParams(), gamma: 5 / 3, R: 1, arad: 1 });
    for (const [rho, p] of grid) {
      const g1 = (e.soundSpeed(rho, p) ** 2 * rho) / p;
      expect(g1).toBeGreaterThan(4 / 3 - 1e-9);
      expect(g1).toBeLessThan(5 / 3 + 1e-9);
    }
  });
});
