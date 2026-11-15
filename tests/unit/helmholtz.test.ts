import { beforeAll, describe, expect, it } from 'vitest';
import CASES from './helmholtz_cases.json';
import { helmIterations, helmSolve, helmThermo, loadHelmholtz } from '../../src/eos/helmholtz';
import { defaultEosParams, makeEos, mixtureParams, physicalEosParams, type EosParams } from '../../src/eos/eos';
import { COMPOSITIONS } from '../../src/units/composition';
import { WD_UNITS } from '../../src/problems/white_dwarf';
import { momentCoefficients, mixMoments } from '../../src/species/mixture';
import { SPECIES } from '../../src/units/composition';

// States of white dwarf matter (rho Ye >= 10 g/cm^3, 1e4 to 1e11 K, Abar 4 to 56) with their
// exact thermal energy, pressure and sound speed (tools/eos/gen_inverse.py, held out from
// everything the table was built from).
interface Case { rho: number; T: number; abar: number; e: number; p: number; cs2: number; c02: number }
const cases = CASES as Case[];
const rel = (a: number, b: number) => Math.abs(a / b - 1);
const q = (a: number[], f: number) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];

beforeAll(async () => {
  await loadHelmholtz();
});

describe('Helmholtz table', () => {
  it('gives the exact gas energy and pressure', () => {
    const e: number[] = [];
    const p: number[] = [];
    const c: number[] = [];
    for (const k of cases) {
      const t = helmThermo(k.rho, Math.log(k.T), 0.5, 1 / k.abar);
      e.push(rel(t.e, k.e));
      p.push(rel(t.pr * k.rho, k.p));
      c.push(rel(k.c02 + t.dprho + (t.dpr * t.dpr) / t.de, k.cs2));
    }
    expect(q(e, 0.99)).toBeLessThan(2e-4);
    expect(q(p, 0.99)).toBeLessThan(2e-4);
    expect(Math.max(...e, ...p)).toBeLessThan(2e-3);
    expect(q(c, 0.99)).toBeLessThan(2e-3);
  });

  it('inverts the energy and the pressure for the temperature', () => {
    for (const k of cases) {
      const ia = 1 / k.abar;
      expect(rel(Math.exp(helmSolve(k.rho, k.e, 0.5, ia, false)), k.T)).toBeLessThan(2e-3);
      expect(rel(Math.exp(helmSolve(k.rho, k.p / k.rho, 0.5, ia, true)), k.T)).toBeLessThan(2e-3);
    }
  });

  it('starts Newton close enough to converge in a few iterations', () => {
    const its = cases.map((k) => helmIterations(k.rho, k.e, 0.5, 1 / k.abar, false));
    const mean = its.reduce((s, x) => s + x, 0) / its.length;
    expect(mean).toBeLessThan(4);
    expect(Math.max(...its)).toBeLessThan(10);
  });
});

describe('Helmholtz equation of state in code units', () => {
  const params: EosParams = { ...defaultEosParams(), ...physicalEosParams(WD_UNITS, COMPOSITIONS.carbonOxygen) };
  const eos = makeEos('helmholtz', params);

  it('round-trips energy, pressure and temperature', () => {
    for (const rho of [1e-3, 1, 17]) {
      for (const TK of [3e7, 1e9, 5e9]) {
        const T = TK / params.helmT;
        const ei = eos.eintFromT(rho, T);
        expect(rel(eos.temperature(rho, ei), T)).toBeLessThan(1e-9);
        const p = eos.pressure(rho, ei);
        expect(rel(eos.eintFromP(rho, p), ei)).toBeLessThan(1e-9);
        expect(eos.soundSpeed(rho, p)).toBeGreaterThan(0);
      }
    }
  });

  it('is the zero-temperature degenerate gas when cold', () => {
    const deg = makeEos('degenerate', params);
    for (const rho of [1e-3, 1, 17]) {
      const p0 = deg.pressure(rho, deg.eintFromT(rho, 0));
      expect(rel(eos.pressure(rho, eos.eintFromT(rho, 0)), p0)).toBeLessThan(1e-12);
    }
  });

  it('reads shocked degenerate fuel hotter than the ideal-electron model', () => {
    // The same thermal energy: the degenerate electrons take less of it, so the gas is hotter.
    const deg = makeEos('degenerate', params);
    const rho = 5;   // 1e7 g/cm^3
    const ei = eos.eintFromT(rho, 1e9 / params.helmT);
    expect(deg.temperature(rho, ei) * params.helmT).toBeLessThan(0.6e9);
  });

  it('follows a cell composition: helium ash has more ions than C/O', () => {
    const coeffs = momentCoefficients([SPECIES.he4, SPECIES.c12, SPECIES.o16], COMPOSITIONS.carbonOxygen);
    const he = makeEos('helmholtz', mixtureParams('helmholtz', params, mixMoments(coeffs, [1, 0, 0])));
    expect(he.params.helmIons).toBeCloseTo(0.25, 6);
    expect(he.params.helmYe).toBeCloseTo(0.5, 6);
    // Same energy, more particles: cooler.
    const rho = 5;
    const ei = eos.eintFromT(rho, 3e9 / params.helmT);
    expect(he.temperature(rho, ei)).toBeLessThan(eos.temperature(rho, ei));
  });
});
