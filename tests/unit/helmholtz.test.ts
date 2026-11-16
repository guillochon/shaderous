import { beforeAll, describe, expect, it } from 'vitest';
import CASES from './helmholtz_cases.json';
import { HELM_FADE, HELM_GAMMA_T, helmCoulombCold, helmCoulombFns, helmCoulombFree, helmCoulombThermal, helmIterations, helmSolve, helmSolveShader, helmThermo, loadHelmholtz } from '../../src/eos/helmholtz';
import { HELM_META } from '../../src/eos/helmholtz_meta';
import { defaultEosParams, degFG, makeEos, mixtureParams, physicalEosParams, type EosParams } from '../../src/eos/eos';
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

  it('is the zero-temperature degenerate gas when cold, without the Coulomb term', () => {
    const deg = makeEos('degenerate', params);
    const bare = makeEos('helmholtz', { ...params, helmCoulomb: 0 });
    for (const rho of [1e-3, 1, 17]) {
      const p0 = deg.pressure(rho, deg.eintFromT(rho, 0));
      expect(rel(bare.pressure(rho, bare.eintFromT(rho, 0)), p0)).toBeLessThan(1e-12);
      // With it, the cold pressure is lower: by a percent or two where the electrons are
      // relativistic, by more where they are not.
      const drop = 1 - eos.pressure(rho, eos.eintFromT(rho, 0)) / p0;
      expect(drop).toBeGreaterThan(0.01);
      expect(drop).toBeLessThan(0.35);
    }
  });

  it('derives the ions per atomic mass unit as the shader does, from mu and Ye', () => {
    for (const comp of [COMPOSITIONS.carbonOxygen, COMPOSITIONS.solar]) {
      const q = { ...defaultEosParams(), ...physicalEosParams(WD_UNITS, comp) };
      expect(rel(1 / q.helmMu - q.helmYe, q.helmIons)).toBeLessThan(1e-12);
    }
    const d = defaultEosParams();
    expect(rel(1 / d.helmMu - d.helmYe, d.helmIons)).toBeLessThan(1e-12);
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

describe('Coulomb corrections', () => {
  // (Zbar, Abar): hydrogen, helium, carbon, C/O, silicon, nickel.
  const ions: [number, number][] = [[1, 1], [2, 4], [6, 12], [6.857, 13.714], [14, 28], [28, 56]];
  const states: [number, number][] = [];
  for (const lr of [0, 2, 3, 4, 6, 8, 10]) for (const lT of [4, 6, 7.5, 9, 10]) states.push([10 ** lr, 10 ** lT]);
  const { me, h, mu, mec2, k: kB } = HELM_META.consts;
  const c = Math.sqrt(mec2 / me);
  // The zero-temperature electrons, cgs: the pressure scale and the density at which p_F = m_e c, per Ye.
  const A = (Math.PI * me ** 4 * c ** 5) / (3 * h ** 3);
  const rhoD = ((mu * 8 * Math.PI) / 3) * ((me * c) / h) ** 3;

  it('has the pressure and energy of one free energy', () => {
    // e = F - dF/dln T, p / rho = dF/dln rho, by central differences of the free energy.
    const d = 1e-4;
    for (const [z, a] of ions) {
      const ye = z / a;
      const ia = 1 / a;
      for (const [rho, T] of states) {
        const lnT = Math.log(T);
        const F = (r: number, l: number) => helmCoulombFree(r, l, ye, ia, 1);
        const cold = helmCoulombCold(rho, ye, ia, 1);
        const th = helmCoulombThermal(rho, lnT, ye, ia, 1);
        const e = F(rho, lnT) - (F(rho, lnT + d) - F(rho, lnT - d)) / (2 * d);
        const pr = (F(rho * Math.exp(d), lnT) - F(rho * Math.exp(-d), lnT)) / (2 * d);
        const scale = Math.abs(cold.e) + Math.abs(th.e);
        expect(Math.abs(cold.e + th.e - e) / scale).toBeLessThan(1e-6);
        expect(Math.abs(cold.pr + th.pr - pr) / scale).toBeLessThan(1e-6);
        // The derivatives the solver and the sound speed use.
        const up = helmCoulombThermal(rho, lnT + d, ye, ia, 1);
        const dn = helmCoulombThermal(rho, lnT - d, ye, ia, 1);
        expect(rel((up.e - dn.e) / (2 * d), th.de)).toBeLessThan(1e-5);
        expect(rel((up.pr - dn.pr) / (2 * d), th.dpr)).toBeLessThan(1e-5);
        const [rp, rm] = [rho * Math.exp(d), rho * Math.exp(-d)];
        const dp = (rp * helmCoulombThermal(rp, lnT, ye, ia, 1).pr - rm * helmCoulombThermal(rm, lnT, ye, ia, 1).pr) / (rp - rm);
        expect(Math.abs(dp - th.dprho) / Math.abs(th.pr)).toBeLessThan(1e-5);
        const dc = (rp * helmCoulombCold(rp, ye, ia, 1).pr - rm * helmCoulombCold(rm, ye, ia, 1).pr) / (rp - rm);
        expect(Math.abs(dc - cold.dprho) / Math.abs(cold.pr)).toBeLessThan(1e-5);
      }
    }
  });

  it('is continuous where the two fits meet, and positive and rising with T in its thermal part', () => {
    const lo = helmCoulombFns(1 - 1e-9);
    const hi = helmCoulombFns(1 + 1e-9);
    expect(Math.abs(lo[0] - hi[0])).toBeLessThan(1e-8);
    expect(Math.abs(lo[1] - hi[1])).toBeLessThan(1e-4);
    expect(Math.abs(lo[2] - hi[2])).toBeLessThan(1e-3);
    for (let lg = -6; lg <= 4; lg += 0.05) {
      const [f, g, gp] = helmCoulombFns(10 ** lg);
      expect(f).toBeGreaterThan(0);
      expect(g).toBeGreaterThan(0);
      expect(gp).toBeGreaterThan(0);
      expect(g - gp).toBeGreaterThan(0); // the heat capacity
      expect(f - g).toBeGreaterThan(0); // d(p / rho)/dT
    }
  });

  it('is Debye-Hueckel when weakly coupled and the ion-sphere energy when strongly', () => {
    // Carbon, dense enough that the fade is gone.
    const [ye, ia, rho] = [0.5, 1 / 12, 1e9];
    const gT = HELM_GAMMA_T * 36 * Math.cbrt(rho * ia);
    const perIon = (T: number) => (helmCoulombCold(rho, ye, ia, 1).e + helmCoulombThermal(rho, Math.log(T), ye, ia, 1).e) / ((ia * kB * T) / mu);
    expect(rel(perIon(gT / 1e-3), -(Math.sqrt(3) / 2) * 1e-3 ** 1.5)).toBeLessThan(0.02);
    expect(rel(perIon(gT / 150), -0.9 * 150)).toBeLessThan(0.03);
    // Gamma of C/O at 1e7 g/cm^3 and 1e9 K is near one.
    const g = (HELM_GAMMA_T * 6.857 ** 2 * Math.cbrt(1e7 / 13.714)) / 1e9;
    expect(g).toBeGreaterThan(0.8);
    expect(g).toBeLessThan(1.2);
  });

  it('lowers the pressure of relativistic electrons by 0.0046 Z^(2/3)', () => {
    for (const [z, a] of ions) {
      const rho = 1e11;
      const pDeg = A * degFG(Math.cbrt((rho * z) / a / rhoD))[0];
      const ratio = (helmCoulombCold(rho, z / a, 1 / a, 1).pr * rho) / pDeg;
      expect(rel(ratio, -0.004553 * z ** (2 / 3))).toBeLessThan(0.02);
    }
  });

  it('leaves the cold pressure positive and rising with density, at every charge', () => {
    let worst = 0;
    for (const [z, a] of ions) {
      const ye = z / a;
      for (let lr = -6; lr <= 11; lr += 0.02) {
        const rho = 10 ** lr;
        const x = Math.cbrt((rho * ye) / rhoD);
        const pDeg = A * degFG(x)[0];
        const dDeg = (A * 8 * x ** 5) / (3 * rho * Math.sqrt(1 + x * x));
        const cold = helmCoulombCold(rho, ye, 1 / a, 1);
        expect(pDeg + cold.pr * rho).toBeGreaterThan(0.5 * pDeg);
        expect(dDeg + cold.dprho).toBeGreaterThan(0.4 * dDeg);
        worst = Math.max(worst, (-cold.pr * rho) / pDeg);
      }
    }
    // The fade caps the reduction near a third; for carbon it sets in near 1e3 g/cm^3.
    expect(worst).toBeLessThan(0.35);
    expect(worst).toBeGreaterThan(0.2);
    expect((HELM_FADE * 36) / 0.5).toBeGreaterThan(500);
    expect((HELM_FADE * 36) / 0.5).toBeLessThan(2000);
  });

  it('loses nothing by taking its last small step on trust', () => {
    // The shader stops once a step is under 3e-4 in ln T, without evaluating again. Newton's
    // method leaves the square of that times half the curvature: here, against the iteration
    // run to convergence, with and without the Coulomb term, by the energy and by the pressure.
    let worst = 0;
    let worstOut = 0;
    for (const k of cases) {
      const ia = 1 / k.abar;
      for (const cm of [0, 1]) {
        const t = helmThermo(k.rho, Math.log(k.T), 0.5, ia, cm);
        for (const byP of [false, true]) {
          const goal = byP ? t.pr : t.e;
          const lnT = helmSolveShader(k.rho, goal, 0.5, ia, byP, cm).lnT;
          worst = Math.max(worst, Math.abs(lnT - helmSolve(k.rho, goal, 0.5, ia, byP, cm)));
          // What the shader returns is the other quantity at that temperature, moved by one
          // more Newton step taken from the same evaluation (helm_fix).
          const s = helmThermo(k.rho, lnT, 0.5, ia, cm);
          const fix = (goal - (byP ? s.pr : s.e)) / (byP ? s.dpr : s.de);
          const out = byP ? s.e + s.de * fix : s.pr + s.dpr * fix;
          worstOut = Math.max(worstOut, rel(out, byP ? t.e : t.pr));
        }
      }
    }
    console.log(`last step on trust: worst error in ln T ${worst.toExponential(2)}, in what is returned ${worstOut.toExponential(2)}`);
    // Single precision holds ln T (about 25) to 1.5e-6, and a value to 6e-8.
    expect(worst).toBeLessThan(1e-6);
    expect(worstOut).toBeLessThan(1e-9);
  });

  it('still finds the temperature in a few iterations', () => {
    const its: number[] = [];
    for (const k of cases) {
      const ia = 1 / k.abar;
      const t = helmThermo(k.rho, Math.log(k.T), 0.5, ia, 1);
      expect(rel(Math.exp(helmSolve(k.rho, t.e, 0.5, ia, false, 1)), k.T)).toBeLessThan(1e-9);
      expect(rel(Math.exp(helmSolve(k.rho, t.pr, 0.5, ia, true, 1)), k.T)).toBeLessThan(1e-9);
      its.push(helmIterations(k.rho, t.e, 0.5, ia, false, 1), helmIterations(k.rho, t.pr, 0.5, ia, true, 1));
    }
    const mean = its.reduce((s, x) => s + x, 0) / its.length;
    console.log(`Newton with the Coulomb term: mean ${mean.toFixed(2)}, most ${Math.max(...its)}`);
    expect(mean).toBeLessThan(5);
    expect(Math.max(...its)).toBeLessThan(10);
  });

  it('makes cold dense matter cooler for the same energy, and changes hot matter little', () => {
    const params: EosParams = { ...defaultEosParams(), ...physicalEosParams(WD_UNITS, COMPOSITIONS.carbonOxygen) };
    const on = makeEos('helmholtz', params);
    const off = makeEos('helmholtz', { ...params, helmCoulomb: 0 });
    const rho = 5; // 1e7 g/cm^3
    // The same thermal energy above each one's cold curve.
    const thermal = (e: typeof on, T: number) => e.eintFromT(rho, T / params.helmT) - e.eintFromT(rho, 0);
    expect(on.temperature(rho, on.eintFromT(rho, 0) + thermal(off, 3e7)) * params.helmT).toBeLessThan(2.9e7);
    expect(rel(on.temperature(rho, on.eintFromT(rho, 0) + thermal(off, 5e9)) * params.helmT, 5e9)).toBeLessThan(0.02);
  });
});
