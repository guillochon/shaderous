import { beforeAll, describe, expect, it } from 'vitest';
import { HHE, h2Rotation, hheParticles, hheSecond, hheState } from '../../src/eos/hhe_model';
import { HHE_META } from '../../src/eos/hhe_meta';
import { hheSolve, hheSolveShader, hheSound2, hheTableFor, hheThermo, loadHhe } from '../../src/eos/hhe';
import { defaultEosParams, hheUnsupported, ionizedFraction, makeEos, physicalEosParams, type EosParams } from '../../src/eos/eos';
import { COMPOSITIONS, meanMolecularWeight } from '../../src/units/composition';
import { flowUnits, gravityUnits, kelvinPerCodeT, toCode } from '../../src/units/units';
import { isentropicCore } from '../../src/setup/parts/flows/first_core';

const rel = (a: number, b: number) => Math.abs(a / b - 1);
const KMU = HHE.k / HHE.mu;
const X = 0.7381;
const SOLAR = { X };
// Code units that are cgs: centimetres, g/cm^3, cm/s.
const CGS = flowUnits('cgs', 1e-2, 1e3, 1e-2);

// States between the table's nodes, the same every run.
function states(n: number, lr0: number, lr1: number, lt0 = 1, lt1 = 8): [number, number][] {
  let seed = 4242;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  return Array.from({ length: n }, () => [10 ** (lr0 + (lr1 - lr0) * rnd()), 10 ** (lt0 + (lt1 - lt0) * rnd())]);
}

beforeAll(async () => {
  await loadHhe('solar');
  await loadHhe('hydrogen');
});

describe('hydrogen and helium: the free energy', () => {
  it('is molecular gas when cold: p = rho k T (X / 2 + Y / 4) / m_u, gamma 5/3 at 30 K and 7/5 at 300 K', () => {
    const n = X / 2 + (1 - X) / 4;
    for (const rho of [1e-18, 1e-13, 1e-9]) {
      const cold = hheSecond(rho, 30, SOLAR);
      // (to the stand-in for pressure ionization, 2 rho / 0.33 g/cm^3 of it)
      expect(rel(cold.pr, n * KMU * 30)).toBeLessThan(1e-8);
      expect(rel(cold.e, 1.5 * n * KMU * 30)).toBeLessThan(1e-3);
      expect(cold.gamma1).toBeCloseTo(5 / 3, 2);
      const warm = hheSecond(rho, 300, SOLAR);
      expect(rel(warm.pr, n * KMU * 300)).toBeLessThan(1e-8);
      // The rotation of H2 is nearly all there (ortho to para fixed at 3:1), the vibration not.
      expect(Math.abs(warm.gamma1 - 1.4)).toBeLessThan(0.05);
    }
  });

  it('is a fully ionized ideal gas when hot: mu = 0.598 for solar, 1/2 for hydrogen', () => {
    for (const [rho, T] of [[1e-10, 1e6], [1e-3, 1e7], [1, 1e7], [150, 1.5e7]]) {
      const s = hheSecond(rho, T, SOLAR);
      expect(rel(1 / hheParticles(s), 1 / (2 * X + 0.75 * (1 - X)))).toBeLessThan(2e-3);
      expect(s.gamma1).toBeCloseTo(5 / 3, 2);
      expect(rel(1 / hheParticles(hheState(rho, T, { X: 1 })), 0.5)).toBeLessThan(2e-3);
    }
    expect(1 / (2 * X + 0.75 * (1 - X))).toBeCloseTo(0.598, 3);
  });

  it('has the energies of dissociation and ionization in e', () => {
    // Fully ionized: each H nucleus has cost D / 2 + chi_H, each He chi_1 + chi_2, over 3/2 k T a particle.
    const s = hheState(1e-8, 1e6, SOLAR);
    const bound = KMU * (X * (HHE.tDis / 2 + HHE.tH) + ((1 - X) / 4) * (HHE.tHe1 + HHE.tHe2));
    expect(rel(s.e, 1.5 * hheParticles(s) * KMU * 1e6 + bound)).toBeLessThan(1e-4);
    expect(bound).toBeGreaterThan(1.4e13);
  });

  it('gives hydrogen the Saha ionization of the ionizing equation of state', () => {
    const comp = COMPOSITIONS.hydrogen;
    const p = { ...defaultEosParams(), ...physicalEosParams(CGS, comp) } as EosParams;
    const kpt = kelvinPerCodeT(CGS, meanMolecularWeight(comp));
    // Hot enough that no H2 is left (which the other model does not have), thin enough that
    // pressure ionization is nothing.
    for (const rho of [1e-14, 1e-10, 1e-7])
      for (const T of [6e3, 1e4, 1.5e4, 3e4]) {
        const s = hheState(rho, T, { X: 1 });
        if (s.y[0] > 1e-6) continue;
        expect(rel(s.y[2] / (s.y[1] + s.y[2]), ionizedFraction(p, rho, T / kpt))).toBeLessThan(1e-5);
      }
  });

  it('obeys the first law: p and e are derivatives of F, and the Maxwell relation holds', () => {
    const d = 1e-4;
    for (const [rho, T] of states(60, -20, 0)) {
      const h = (r: number, t: number) => hheState(r, t, SOLAR);
      const s = h(rho, T);
      const rp = h(rho * Math.exp(d), T), rm = h(rho * Math.exp(-d), T);
      const tp = h(rho, T * Math.exp(d)), tm = h(rho, T * Math.exp(-d));
      expect(rel((rp.h - rm.h) / (2 * d), s.hn)).toBeLessThan(1e-5);
      expect(rel((tp.h - tm.h) / (2 * d), s.ht)).toBeLessThan(1e-5);
      // d/dln T of p equals d/dln rho of -e, both as second derivatives of h.
      const hnt = (tp.hn - tm.hn) / (2 * d);
      const htn = (rp.ht - rm.ht) / (2 * d);
      expect(Math.abs(hnt - htn)).toBeLessThan(1e-5 * (1 + Math.abs(s.ht)));
    }
  });

  it('softens to Gamma_1 near 1.1 where H2 dissociates and where hydrogen ionizes (Tomida et al. 2013, figure 1)', () => {
    const least = (rho: number, t0: number, t1: number) => {
      let m = [Infinity, 0];
      for (let lt = Math.log10(t0); lt <= Math.log10(t1); lt += 0.01) {
        const g = hheSecond(rho, 10 ** lt, SOLAR).gamma1;
        if (g < m[0]) m = [g, 10 ** lt];
      }
      return m;
    };
    // Dissociation at 1e-8 g/cm^3: near 2000 K, where the second collapse begins.
    const dis = least(1e-8, 500, 4000);
    expect(dis[0]).toBeGreaterThan(1.05);
    expect(dis[0]).toBeLessThan(1.15);
    expect(dis[1]).toBeGreaterThan(1500);
    expect(dis[1]).toBeLessThan(2600);
    // Below 4/3 there: the gas cannot hold itself up.
    expect(hheSecond(1e-8, 2000, SOLAR).gamma1).toBeLessThan(4 / 3);
    const ion = least(1e-6, 6000, 4e4);
    expect(ion[0]).toBeGreaterThan(1.1);
    expect(ion[0]).toBeLessThan(1.25);
    // And the two valleys are apart, with a ridge between them.
    expect(hheSecond(1e-8, 4500, SOLAR).gamma1).toBeGreaterThan(dis[0] + 0.1);
  });

  it('has ortho and para hydrogen at 3:1, or in equilibrium as an option', () => {
    // Hot, the two are the same up to the energy the ortho form is counted from (2 theta_rot each).
    const [f] = h2Rotation(3000);
    const [e] = h2Rotation(3000, true);
    expect(f - e).toBeCloseTo((0.75 * 2 * HHE.thetaRot) / 3000, 4);
    // Cold, equilibrium hydrogen is para: converting it takes heat near 50 K, which softens the gas.
    const fixed = hheSecond(1e-13, 50, SOLAR).gamma1;
    const eq = hheSecond(1e-13, 50, { X, orthoParaEquilibrium: true }).gamma1;
    expect(fixed).toBeGreaterThan(1.6);
    expect(eq).toBeLessThan(fixed - 0.1);
  });

  it('ionizes dense hot gas, as a stellar interior is (the stand-in for pressure ionization)', () => {
    // Along a Sun-like interior, T = 1.5e7 K (rho / 150 g/cm^3)^(1/3): bare Saha equations leave
    // a third of the hydrogen neutral at the centre.
    for (const rho of [0.1, 1, 10, 150]) {
      const T = 1.5e7 * Math.cbrt(rho / 150);
      const s = hheState(rho, T, SOLAR);
      expect(s.y[2] / X).toBeGreaterThan(0.98);
      // The pressure is the ideal gas's to a percent.
      expect(rel(s.hn, 2 * X + 0.75 * (1 - X))).toBeLessThan(0.01);
    }
    const bare = hheState(150, 1.5e7, { X, rhoPi: Infinity });
    expect(bare.y[2] / X).toBeLessThan(0.8);
    // And it leaves thin gas alone.
    const a = hheState(1e-4, 1.2e4, SOLAR), b = hheState(1e-4, 1.2e4, { X, rhoPi: Infinity });
    expect(rel(a.pr, b.pr)).toBeLessThan(1e-3);
  });
});

describe('hydrogen and helium: the table', () => {
  it('gives the model’s pressure, energy and sound speed between its nodes', () => {
    for (const [name, o] of [['solar', SOLAR], ['hydrogen', { X: 1 }]] as const) {
      const e: number[] = [], p: number[] = [], c: number[] = [];
      for (const [rho, T] of states(1500, -20, -1)) {
        const x = hheSecond(rho, T, o);
        const t = hheThermo(name, rho, Math.log(T));
        e.push(rel(t.e, x.e));
        p.push(rel(t.pr, x.pr));
        c.push(rel(t.dprho + (t.dpr * t.dpr) / t.de, x.cs2));
      }
      const q = (a: number[], f: number) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
      expect(Math.max(...p)).toBeLessThan(2e-4);
      expect(Math.max(...e)).toBeLessThan(1e-3);
      expect(q(e, 0.99)).toBeLessThan(2e-4);
      expect(q(c, 0.99)).toBeLessThan(4e-3);
      expect(Math.max(...c)).toBeLessThan(2e-2);
    }
  });

  it('holds to 1e3 g/cm^3 where the gas is hot', () => {
    for (const [rho, T] of states(300, -1, 3, 6.5, 8)) {
      const x = hheSecond(rho, T, SOLAR);
      const t = hheThermo('solar', rho, Math.log(T));
      expect(rel(t.pr, x.pr)).toBeLessThan(1e-3);
      expect(rel(t.e, x.e)).toBeLessThan(1e-3);
    }
  });

  it('inverts the energy and the pressure for the temperature, in a few steps', () => {
    const its: number[] = [];
    for (const [rho, T] of states(600, -20, -1)) {
      const t = hheThermo('solar', rho, Math.log(T));
      expect(rel(Math.exp(hheSolve('solar', rho, t.e, false)), T)).toBeLessThan(1e-9);
      expect(rel(Math.exp(hheSolve('solar', rho, t.pr, true)), T)).toBeLessThan(1e-9);
      for (const byP of [false, true]) {
        const s = hheSolveShader('solar', rho, byP ? t.pr : t.e, byP);
        its.push(s.its);
        // What the shader's few steps leave, before its last correction.
        expect(rel(Math.exp(s.lnT), T)).toBeLessThan(1e-5);
      }
    }
    expect(its.reduce((s, x) => s + x, 0) / its.length).toBeLessThan(3);
    expect(Math.max(...its)).toBeLessThan(8);
  });

  it('has the energy rise with temperature everywhere, and the pressure below 1 g/cm^3', () => {
    const L = HHE_META.table;
    for (let i = 0; i < L.nr; i += 5) {
      const rho = 10 ** (L.r0 + i * L.dr);
      for (let j = 0; j < L.nt - 1; j += 3) {
        const t = hheThermo('solar', rho, (L.t0 + (j + 0.5) * L.dt) * Math.LN10);
        expect(t.de).toBeGreaterThan(0);
        if (rho < 1) expect(t.dpr).toBeGreaterThan(0);
        if (rho < 0.1) expect(t.dprho).toBeGreaterThan(0);
      }
    }
  });

  it('adds radiation when asked, and extends past its ends in temperature', () => {
    const rho = 1e-6;
    const lnT = Math.log(3e5);
    const g = hheThermo('solar', rho, lnT);
    const r = hheThermo('solar', rho, lnT, 1);
    expect(rel(r.e - g.e, (HHE.arad * 3e5 ** 4) / rho)).toBeLessThan(1e-9);
    expect(rel(r.pr - g.pr, (HHE.arad * 3e5 ** 4) / (3 * rho))).toBeLessThan(1e-9);
    for (const rad of [0, 1])
      for (const T of [0.5, 3, 3e8, 2e9]) {
        const t = hheThermo('solar', rho, Math.log(T), rad);
        expect(rel(Math.exp(hheSolve('solar', rho, t.e, false, rad)), T)).toBeLessThan(1e-3);
        expect(rel(Math.exp(hheSolve('solar', rho, t.pr, true, rad)), T)).toBeLessThan(1e-3);
      }
    // Below 10 K the gas is cold H2 and He, e = 3/2 p.
    const c = hheThermo('solar', 1e-18, Math.log(3));
    expect(rel(c.e, 1.5 * c.pr)).toBeLessThan(1e-3);
  });
});

describe('hydrogen and helium in code units', () => {
  // A solar mass and an astronomical unit.
  const units = gravityUnits('au', 1.495978707e11, 1.98847e30);
  const params: EosParams = { ...defaultEosParams(), ...physicalEosParams(units, COMPOSITIONS.solar), hheRad: 0 };
  const eos = makeEos('hhe', params);

  it('round-trips energy, pressure and temperature', () => {
    for (const rhoCgs of [1e-16, 1e-11, 1e-7, 1e-3])
      for (const TK of [10, 300, 1800, 9000, 2e4, 1e6]) {
        const rho = rhoCgs / params.helmRho;
        const T = TK / params.helmT;
        const ei = eos.eintFromT(rho, T);
        expect(rel(eos.temperature(rho, ei), T)).toBeLessThan(1e-9);
        const p = eos.pressure(rho, ei);
        expect(rel(eos.eintFromP(rho, p), ei)).toBeLessThan(1e-9);
        const cs = eos.soundSpeed(rho, p);
        expect((cs * cs * rho) / p).toBeGreaterThan(1.05);
        expect((cs * cs * rho) / p).toBeLessThan(1.7);
      }
  });

  it('reads p = rho T for fully ionized gas, as the other equations of state do', () => {
    const rho = 1e-6 / params.helmRho;
    const T = 1e6 / params.helmT;
    expect(rel(eos.pressure(rho, eos.eintFromT(rho, T)), rho * T)).toBeLessThan(2e-3);
  });

  it('carries the radiation term unless radiation transport does', () => {
    const rho = 1e-8 / params.helmRho;
    const T = 1e6 / params.helmT;
    const withRad = makeEos('hhe', { ...params, hheRad: 1 });
    expect(withRad.eintFromT(rho, T)).toBeGreaterThan(2 * eos.eintFromT(rho, T));
    // syncRadiationEos sets arad to 0 while the transport carries the radiation.
    const carried = makeEos('hhe', { ...params, hheRad: 1, arad: 0 });
    expect(carried.eintFromT(rho, T)).toBe(eos.eintFromT(rho, T));
  });

  it('has tables for solar and hydrogen gas, and says so for any other', () => {
    expect(hheTableFor(physicalEosParams(undefined, COMPOSITIONS.solar).hheX!)).toBe('solar');
    expect(hheTableFor(physicalEosParams(undefined, COMPOSITIONS.solarNeutral).hheX!)).toBe('solar');
    expect(hheTableFor(physicalEosParams(undefined, COMPOSITIONS.hydrogen).hheX!)).toBe('hydrogen');
    const co = { ...defaultEosParams(), ...physicalEosParams(undefined, COMPOSITIONS.carbonOxygen) };
    expect(hheUnsupported('hhe', co, 'C/O')).toMatch(/solar \(X = 0.7381\) and hydrogen gas, not C\/O/);
    expect(hheUnsupported('ideal', co)).toBeNull();
    expect(() => makeEos('hhe', co)).toThrow(/solar/);
    expect(hheSound2(hheThermo('hydrogen', 1e-10, Math.log(1e4)))).toBeGreaterThan(0);
  });
});

describe('protostellar collapse', () => {
  it('follows the published track: from 10 K at 1e-13 g/cm^3 to 2000 K at 1e-8, then through dissociation', () => {
    // The adiabat of gas that turned opaque at 1e-13 g/cm^3 and 10 K, dln T/dln rho = Gamma_3 - 1
    // (Masunaga & Inutsuka 2000, figure 1; Tomida et al. 2013, figure 2).
    const slope = (lr: number, lt: number) => {
      const s = hheSecond(Math.exp(lr), Math.exp(lt), SOLAR);
      return s.dpr / s.de;
    };
    const N = 1500;
    const lr0 = Math.log(1e-13);
    const h = (Math.log(1e-2) - lr0) / N;
    const lt = [Math.log(10)];
    for (let i = 0; i < N; i++) {
      const x = lr0 + i * h, y = lt[i];
      const k1 = slope(x, y), k2 = slope(x + h / 2, y + (h * k1) / 2), k3 = slope(x + h / 2, y + (h * k2) / 2), k4 = slope(x + h, y + h * k3);
      lt.push(y + (h * (k1 + 2 * k2 + 2 * k3 + k4)) / 6);
    }
    const T = (rho: number) => Math.exp(lt[Math.round((Math.log(rho) - lr0) / h)]);
    // gamma = 5/3 while H2 does not turn, then 7/5.
    expect(T(1e-12)).toBeCloseTo(10 * 10 ** (2 / 3), 0);
    expect(T(1e-10)).toBeGreaterThan(400);
    expect(T(1e-10)).toBeLessThan(600);
    // The second collapse begins: 2000 K at 1e-8 g/cm^3.
    expect(T(1e-8)).toBeGreaterThan(1900);
    expect(T(1e-8)).toBeLessThan(2200);
    // Dissociation holds the temperature down over four decades of density.
    expect(T(1e-4) / T(1e-8)).toBeLessThan(3);
    // Then the gas is atomic and stiffens: the second core, above 1e4 K.
    expect(T(1e-2)).toBeGreaterThan(2e4);
    expect(hheSecond(1e-2, T(1e-2), SOLAR).gamma1).toBeGreaterThan(4 / 3);
  });

  it('builds first cores that grow to 0.04 solar masses and are unstable beyond', () => {
    // Code units of an astronomical unit and 0.01 solar masses, as the second-collapse preset.
    const units = gravityUnits('core', 1.495978707e11, 1.98847e28);
    const p: EosParams = { ...defaultEosParams(), ...physicalEosParams(units, COMPOSITIONS.solar), hheRad: 0 };
    const eos = makeEos('hhe', p);
    const code = (l: number) => toCode(units, 'density', 10 ** l * 1e3);
    const rhoA = code(-13);
    const uA = eos.eintFromT(rhoA, 10 / p.helmT) / rhoA;
    const core = (l: number) => {
      const c = isentropicCore(eos.pressure, 1, code(l), rhoA, uA, rhoA);
      return { mass: c.mass * 0.01, radius: c.r[c.r.length - 1], Tc: eos.temperature(code(l), eos.eintFromP(code(l), c.p[0])) * p.helmT, c };
    };
    const m = [-10, -9, -8.5, -8, -7.5].map(core);
    // Mass rises with the central density up to 3e-9 g/cm^3 (1800 K), and falls beyond: there
    // the center is dissociating, and the core is unstable.
    expect(m[1].mass).toBeGreaterThan(m[0].mass);
    expect(m[2].mass).toBeGreaterThan(m[1].mass);
    expect(m[2].mass).toBeGreaterThan(0.038);
    expect(m[2].mass).toBeLessThan(0.043);
    expect(m[2].Tc).toBeGreaterThan(1700);
    expect(m[2].Tc).toBeLessThan(1900);
    expect(m[3].mass).toBeLessThan(m[2].mass);
    expect(m[4].mass).toBeLessThan(m[3].mass);
    // A few astronomical units across (Masunaga & Inutsuka 2000: about 5 au and 0.05 solar masses).
    expect(m[2].radius).toBeGreaterThan(2);
    expect(m[2].radius).toBeLessThan(6);
    // In balance: dp/dr = -G m rho / r^2 between neighbouring points of the profile.
    const { r, rho, p: pr } = m[3].c;
    let mass = 0;
    for (let k = 1; k < r.length - 1; k++) {
      mass += (4 / 3) * Math.PI * (r[k] ** 3 - r[k - 1] ** 3) * Math.sqrt(rho[k] * rho[k - 1]);
      if (k % 50 || k < 100) continue;
      const grad = (pr[k + 1] - pr[k - 1]) / (r[k + 1] - r[k - 1]);
      expect(rel(grad, (-mass * rho[k]) / (r[k] * r[k]))).toBeLessThan(0.02);
    }
  });
});
