import { beforeAll, describe, expect, it } from 'vitest';
import CASES from './aneos_cases.json';
import PYROLITE from '../../src/eos/aneos/pyrolite.json';
import FE from '../../src/eos/aneos/fe85si15.json';
import WATER from '../../src/eos/aneos/water.json';
import FORSTERITE from '../../src/eos/aneos/forsterite.json';
import { ANEOS_MATERIALS, ANEOS_NEWTON, ANEOS_T_MIN, aneosEnergyFloor, aneosFromE, aneosFromP, aneosFromT, aneosSplit, loadAneos, type AneosMaterial, type AneosTable } from '../../src/eos/aneos';

// ANEOS itself (tools/eos/aneos_cases.py) at places off the tables' nodes: rho, T, p, e, phase.
const cases = CASES as unknown as Record<AneosMaterial, [number, number, number, number, number][]>;
const E_OFF: Record<AneosMaterial, number> = { pyrolite: PYROLITE.eOff, fe85si15: FE.eOff, water: WATER.eOff, forsterite: FORSTERITE.eOff };
const rel = (a: number, b: number) => (a === b ? 0 : Math.abs(a - b) / Math.abs(b));
const q = (a: number[], f: number) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
/** ANEOS's phase flag for the vapor dome, where the pressure spans tens of decades along T. */
const DOME = 2;

const tab = {} as Record<string, AneosTable>;
beforeAll(async () => {
  for (const m of ANEOS_MATERIALS) tab[m] = await loadAneos(m);
  tab.pair = await loadAneos('pyrolite+fe85si15');
}, 120000);

describe('ANEOS tables', () => {
  for (const m of ANEOS_MATERIALS) {
    it(`${m}: pressure and energy at a temperature are ANEOS's`, () => {
      const p: number[] = [];
      const e: number[] = [];
      for (const [rho, T, pr, en, kpa] of cases[m]) {
        if (kpa === DOME) continue;
        const s = aneosFromT(tab[m], rho, T);
        p.push(rel(s.p, pr));
        e.push(rel(s.e - E_OFF[m], en));
      }
      expect(p.length).toBeGreaterThan(100);
      expect(q(p, 0.5)).toBeLessThan(1e-5);
      expect(q(p, 0.9)).toBeLessThan(2e-3);
      expect(q(e, 0.5)).toBeLessThan(1e-5);
      expect(q(e, 0.9)).toBeLessThan(2e-3);
    });

    it(`${m}: the energy and the pressure are inverted for the temperature`, () => {
      const byE: number[] = [];
      const byP: number[] = [];
      const shader: number[] = [];
      for (const [rho, T, , , kpa] of cases[m]) {
        const s = aneosFromT(tab[m], rho, T);
        // A cold solid pressed to thousands of GPa holds less heat than ANEOS resolves beside its
        // cold energy: there the energy does not tell the temperature, and nothing asks it to.
        if (s.e - aneosFromT(tab[m], rho, 50).e < 1e-6 * s.e) continue;
        const a = aneosFromE(tab[m], rho, s.e);
        byE.push(rel(a.p, s.p), rel(a.T, T));
        // As many Newton steps as the shader takes.
        shader.push(rel(aneosFromE(tab[m], rho, s.e, 0, ANEOS_NEWTON).e, s.e));
        // By the pressure only where it tells the temperature: not in the dome's cold end, where
        // it is the same to rounding along the density and all but flat along T.
        if (kpa !== DOME && s.p > 1e6) byP.push(rel(aneosFromP(tab[m], rho, s.p).e, s.e));
      }
      expect(Math.max(...byE)).toBeLessThan(1e-6);
      expect(Math.max(...shader)).toBeLessThan(1e-5);
      expect(q(byP, 0.99)).toBeLessThan(1e-5);
    });

    it(`${m}: the sound speed is real and that of the pressure's own slopes`, () => {
      const err: number[] = [];
      for (const [rho, T, , , kpa] of cases[m]) {
        if (kpa === DOME) continue;
        const s = aneosFromT(tab[m], rho, T);
        expect(s.c2).toBeGreaterThan(0);
        // Along the adiabat de = p / rho^2 drho, by differences.
        const h = 1e-4;
        const up = aneosFromE(tab[m], rho * (1 + h), s.e + (s.p / rho) * h);
        const dn = aneosFromE(tab[m], rho * (1 - h), s.e - (s.p / rho) * h);
        const c2 = (up.p - dn.p) / (2 * h * rho);
        if (c2 > 1e8) err.push(rel(s.c2, c2));
      }
      // The sound speed is the thermodynamic one, which is the adiabat's where ANEOS is consistent
      // with a free energy; in the ionized fluid it is not, by a few percent.
      expect(q(err, 0.5)).toBeLessThan(2e-3);
      expect(q(err, 0.9)).toBeLessThan(4e-2);
    });
  }

  it('has a least energy: the lowest isotherm, and a state below it reads as that', () => {
    for (const m of ANEOS_MATERIALS) {
      for (const rho of [1e-6, 1e-2, 2, 5, 12]) {
        const floor = aneosEnergyFloor(tab[m], rho);
        expect(rel(floor, aneosFromT(tab[m], rho, ANEOS_T_MIN).e)).toBeLessThan(1e-12);
        // (Equal where a pressed solid's heat is below what ANEOS resolves.)
        expect(aneosFromT(tab[m], rho, 300).e).toBeGreaterThanOrEqual(floor);
        const below = aneosFromE(tab[m], rho, 0.5 * floor);
        expect(below.T).toBeCloseTo(ANEOS_T_MIN, 6);
        expect(below.p).toBe(aneosFromT(tab[m], rho, ANEOS_T_MIN).p);
      }
    }
  });

  it('gives a rock and an iron their densities at no pressure', () => {
    // 300 K, one bar: within a few percent of the decks' reference densities.
    const bar = 1e6;
    const find = (m: AneosMaterial, lo: number, hi: number) => {
      for (let k = 0; k < 60; k++) {
        const mid = 0.5 * (lo + hi);
        if (aneosFromT(tab[m], mid, 300).p > bar) hi = mid;
        else lo = mid;
      }
      return lo;
    };
    expect(find('pyrolite', 2, 5)).toBeGreaterThan(3.2);
    expect(find('pyrolite', 2, 5)).toBeLessThan(3.5);
    expect(find('fe85si15', 5, 10)).toBeGreaterThan(7.2);
    expect(find('fe85si15', 5, 10)).toBeLessThan(7.7);
    expect(find('forsterite', 2, 5)).toBeGreaterThan(3.1);
    expect(find('forsterite', 2, 5)).toBeLessThan(3.4);
  });
});

describe('ANEOS, two materials in a cell', () => {
  it("the ends of a pair's table are the materials themselves", () => {
    for (const [rho, T] of [[4, 3000], [9, 5000], [0.01, 8000]]) {
      expect(aneosFromT(tab.pair, rho, T, 0).p).toBe(aneosFromT(tab.pyrolite, rho, T).p);
      expect(aneosFromT(tab.pair, rho, T, 1).p).toBe(aneosFromT(tab.fe85si15, rho, T).p);
    }
  });

  it("a mixed cell's pressure is each material's at its own density", () => {
    const err: number[] = [];
    for (const X of [0.25, 0.5, 0.75]) {
      for (const rho of [3.5, 5, 7, 10, 14]) {
        for (const T of [500, 2000, 5000, 12000]) {
          const s = aneosSplit(tab.pyrolite, tab.fe85si15, rho, T, X);
          expect(rel(s.pA, s.pB)).toBeLessThan(1e-6);
          // Both materials condensed and pressed: the pressure is no accident of rounding.
          if (s.pA > 1e10) err.push(rel(aneosFromT(tab.pair, rho, T, X).p, s.pA));
        }
      }
    }
    expect(err.length).toBeGreaterThan(30);
    expect(q(err, 0.9)).toBeLessThan(5e-3);
    expect(Math.max(...err)).toBeLessThan(5e-2);
  });

  it('between its tables a mixture lies between them, and inverts', () => {
    for (const X of [0.1, 0.4, 0.6, 0.9]) {
      const rho = 6;
      const T = 4000;
      const s = aneosFromT(tab.pair, rho, T, X);
      const lo = aneosFromT(tab.pair, rho, T, Math.floor(X * 4) / 4);
      const hi = aneosFromT(tab.pair, rho, T, Math.ceil(X * 4) / 4);
      expect(s.p).toBeGreaterThan(Math.min(lo.p, hi.p));
      expect(s.p).toBeLessThan(Math.max(lo.p, hi.p));
      expect(rel(aneosFromE(tab.pair, rho, s.e, X).T, T)).toBeLessThan(1e-6);
    }
  });
});
