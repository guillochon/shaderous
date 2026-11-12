import { describe, expect, it } from 'vitest';
import { defaultEosParams, degFG, ionizedFraction, makeEos, physicalEosParams, type EosKind, type EosParams } from '../../src/eos/eos';
import { COMPOSITIONS, composition, meanMolecularWeight } from '../../src/units/composition';
import { SI } from '../../src/units/constants';
import { flowUnits, kelvinPerCodeT } from '../../src/units/units';

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

// Code units that are cgs: centimetres, g/cm^3, cm/s.
const CGS = flowUnits('cgs', 1e-2, 1e3, 1e-2);

describe('degenerate electron gas', () => {
  it('series and closed forms of f and g agree where both are accurate', () => {
    // The closed form is accurate in float64 from x ~ 0.3 up; the series is used below 0.5.
    for (const x of [0.3, 0.35, 0.4, 0.45, 0.4999]) {
      const s = x * x;
      const r = Math.sqrt(1 + s);
      const f = x * (2 * s - 3) * r + 3 * Math.asinh(x);
      const g = 8 * x ** 3 * (r - 1) - f;
      const [fs, gs] = degFG(x);
      expect(rel(fs, f)).toBeLessThan(1e-7);
      expect(rel(gs, g)).toBeLessThan(1e-7);
    }
  });

  it('has the textbook constants and limits for C/O (Ye = 1/2), in cgs', () => {
    const p = { ...defaultEosParams(), ...physicalEosParams(CGS, COMPOSITIONS.carbonOxygen) } as EosParams;
    expect(rel(p.degRho, 2 * 9.7388e5)).toBeLessThan(1e-3);
    expect(rel(p.degP, 6.0022e22)).toBeLessThan(1e-3);
    const cold = makeEos('degenerate', { ...p, arad: 0 });
    // Non-relativistic: P = 1.0036e13 (rho / mu_e)^(5/3); ultra-relativistic: 1.2435e15 (rho / mu_e)^(4/3).
    expect(rel(cold.pressure(1e2, 0), 1.0036e13 * (1e2 / 2) ** (5 / 3))).toBeLessThan(2e-3);
    expect(rel(cold.pressure(1e13, 0), 1.2435e15 * (1e13 / 2) ** (4 / 3))).toBeLessThan(2e-3);
    // E / P is 3/2 in the first limit and 3 in the second, approached as x^2 (here x = 0.04)
    // and as 1/x (here x = 170).
    expect(rel(cold.eintFromP(1e2, 0) / cold.pressure(1e2, 0), 1.5)).toBeLessThan(1e-3);
    expect(rel(cold.eintFromP(1e13, 0) / cold.pressure(1e13, 0), 3)).toBeLessThan(1e-2);
  });

  it('adds the thermal gas and radiation on top, and round-trips', () => {
    const e = makeEos('degenerate', { ...defaultEosParams(), gamma: 5 / 3, R: 1, arad: 1e-3, degRho: 1, degP: 1 });
    for (const rho of [1e-3, 0.1, 1, 10, 1e3])
      for (const T of [1e-2, 1, 1e2]) {
        const eint = e.eintFromT(rho, T);
        expect(rel(e.temperature(rho, eint), T)).toBeLessThan(1e-9);
        const p = e.pressure(rho, eint);
        expect(rel(e.eintFromP(rho, p), eint)).toBeLessThan(1e-9);
      }
  });
});

describe('ionizing hydrogen', () => {
  it('matches the Saha equation worked directly in cgs', () => {
    const comp = COMPOSITIONS.hydrogen;
    const p = { ...defaultEosParams(), ...physicalEosParams(CGS, comp) } as EosParams;
    const kpt = kelvinPerCodeT(CGS, meanMolecularWeight(comp));
    const kB = SI.kB * 1e7, me = SI.me * 1e3, h = SI.h * 1e7, mu = SI.mu * 1e3, chi = SI.chiH * 1e7;
    for (const rho of [1e-12, 1e-9, 1e-6])
      for (const TK of [5e3, 1e4, 2e4, 5e4]) {
        const nH = rho / mu;
        const S = ((2 * Math.PI * me * kB * TK) / h ** 2) ** 1.5 * Math.exp(-chi / (kB * TK)) / nH;
        const x = (-S + Math.sqrt(S * S + 4 * S)) / 2;
        expect(rel(ionizedFraction(p, rho, TK / kpt), x)).toBeLessThan(1e-9);
      }
  });

  it('reduces to ideal gas of the right mu when fully neutral or fully ionized', () => {
    const p = { ...defaultEosParams(), ...physicalEosParams(CGS, COMPOSITIONS.solar), gamma: 5 / 3 } as EosParams;
    const e = makeEos('ionizing', p);
    const rho = 1e-9;
    // Hot: everything ionized, as the composition (fully ionized solar) says: p = rho T.
    const hot = 1e3 * p.ionT;
    expect(rel(e.pressure(rho, e.eintFromT(rho, hot)), rho * hot)).toBeLessThan(1e-9);
    // Cold: hydrogen neutral, so fewer particles: p = rho T (ionH + ionOther).
    const cold = 1e-3 * p.ionT;
    expect(rel(e.pressure(rho, e.eintFromT(rho, cold)), rho * cold * (p.ionH + p.ionOther))).toBeLessThan(1e-9);
    // In the ionization zone Gamma_1 dips below 5/3; outside it, it is 5/3.
    const g1 = (T: number) => {
      const pr = e.pressure(rho, e.eintFromT(rho, T));
      return (e.soundSpeed(rho, pr) ** 2 * rho) / pr;
    };
    expect(rel(g1(hot), 5 / 3)).toBeLessThan(1e-6);
    expect(rel(g1(cold), 5 / 3)).toBeLessThan(1e-6);
    let lowest = Infinity;
    for (let T = 0.02 * p.ionT; T < 0.5 * p.ionT; T *= 1.05) lowest = Math.min(lowest, g1(T));
    expect(lowest).toBeLessThan(1.3);
    expect(lowest).toBeGreaterThan(1);
  });

  it('round-trips through e_int, p and T across the ionization zone', () => {
    const p = { ...defaultEosParams(), ...physicalEosParams(CGS, composition('h+he', { h1: 0.75, he4: 0.25 }, 'neutral')) } as EosParams;
    const e = makeEos('ionizing', p);
    for (const rho of [1e-12, 1e-8, 1e-4])
      for (let T = 0.01 * p.ionT; T < 10 * p.ionT; T *= 1.7) {
        const eint = e.eintFromT(rho, T);
        expect(rel(e.temperature(rho, eint), T)).toBeLessThan(1e-9);
        expect(rel(e.eintFromP(rho, e.pressure(rho, eint)), eint)).toBeLessThan(1e-9);
      }
  });
});

describe('every equation of state', () => {
  // The sound speed must be dp/drho along an adiabat, where d(e/rho) = p / rho^2 drho.
  const cases: [EosKind, Partial<EosParams>][] = [
    ['ideal', { gamma: 1.4 }],
    ['gas_radiation', { gamma: 5 / 3, R: 1, arad: 1 }],
    ['degenerate', { gamma: 5 / 3, R: 1, arad: 0.1, degRho: 1, degP: 1 }],
    ['ionizing', { gamma: 5 / 3, ...physicalEosParams(CGS, COMPOSITIONS.solar) }],
  ];
  for (const [kind, over] of cases) {
    it(`${kind}: sound speed is the adiabatic derivative`, () => {
      const e = makeEos(kind, { ...defaultEosParams(), ...over });
      const rhos = kind === 'ionizing' ? [1e-12, 1e-9, 1e-6] : [1e-2, 1, 1e2];
      const temps = kind === 'ionizing' ? [3e3, 8e3, 1.2e4, 2e4, 1e5].map((T) => T / kelvinPerCodeT(CGS, meanMolecularWeight(COMPOSITIONS.solar))) : [1e-2, 1, 1e2];
      for (const rho of rhos)
        for (const T of temps) {
          const u = e.eintFromT(rho, T) / rho;
          const p = e.pressure(rho, u * rho);
          const d = 1e-5 * rho;
          const at = (s: number) => {
            const r = rho + s * d;
            return e.pressure(r, (u + (s * d * p) / rho ** 2) * r);
          };
          const c2 = (at(1) - at(-1)) / (2 * d);
          expect(rel(e.soundSpeed(rho, p) ** 2, c2)).toBeLessThan(1e-5);
        }
    });
  }
});
