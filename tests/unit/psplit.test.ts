// The float64 prototype of the implicit-pressure step (src/reference/psplit.ts; plans/incompressible.md 4.2).
import { describe, expect, it } from 'vitest';
import { exactRiemann } from '../../src/reference/riemann_exact';
import { RefSplit, splitDefaults, type SplitConfig } from '../../src/reference/psplit';
import { defaultEosParams, makeEos, type Eos } from '../../src/eos/eos';
import { greshoProfile } from '../../src/setup/parts/flows/low_mach';

const ideal = (gamma: number) => makeEos('ideal', { ...defaultEosParams(), gamma });
const liquid = (c0: number) => makeEos('liquid', { ...defaultEosParams(), liqRho: 1, liqC: c0 });
const cfg = (over: Partial<SplitConfig>): SplitConfig => ({
  nx: 24, ny: 24, nz: 1, dx: 1 / 24, eos: ideal(1.4), bcLo: [0, 0, 0], bcHi: [0, 0, 0], ...splitDefaults, acousticCfl: Infinity, predGradP: 'solved', tol: 1e-6, ...over,
});
const sum = (a: Float64Array) => a.reduce((s, v) => s + v, 0);

/** A Gresho vortex in a liquid of sound speed 1 / mach, in exact balance (its density follows the enthalpy). */
const gresho = (mach: number) => (x: number, y: number) => {
  const r = Math.hypot(x, y);
  const { u, dp } = greshoProfile(r);
  const c = r > 0 ? [-y / r, x / r] : [0, 0];
  return { rho: Math.exp(dp * mach * mach), vx: u * c[0], vy: u * c[1], vz: 0, p: 0, s: 0 };
};
function runGresho(mach: number, over: Partial<SplitConfig> = {}) {
  const eos: Eos = liquid(1 / mach);
  const s = new RefSplit(cfg({ eos, ...over }));
  s.init(gresho(mach));
  const before = { ke: s.kinetic(), mass: sum(s.U[0]), px: sum(s.U[1]), py: sum(s.U[2]) };
  let scale = 0;
  for (let o = 0; o < s.ncell; o++) scale += Math.hypot(s.U[1][o], s.U[2][o]);
  s.run(1);
  let dev = 0;
  for (let o = 0; o < s.ncell; o++) dev = Math.max(dev, Math.abs(s.U[0][o] - 1));
  return { kept: s.kinetic() / before.ke, steps: s.steps, dev, mass: sum(s.U[0]) / before.mass - 1, mom: Math.max(Math.abs(sum(s.U[1]) - before.px), Math.abs(sum(s.U[2]) - before.py)) / scale };
}

describe('implicit-pressure prototype', () => {
  it('captures the Sod shock at the flow’s timestep, conserving mass and energy', () => {
    const n = 128;
    const exact = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
    const s = new RefSplit(cfg({ nx: n, ny: 1, dx: 1 / n, bcLo: [1, 0, 0], bcHi: [1, 0, 0], acousticCfl: 20, predGradP: 'none', tol: 1e-10 }));
    s.init((x) => (x < 0 ? { rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 1 } : { rho: 0.125, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }));
    const [m0, e0] = [sum(s.U[0]), sum(s.U[4])];
    s.run(0.2);
    expect(s.t).toBeCloseTo(0.2, 12);
    // Nothing has reached the open ends, so nothing has left.
    expect(Math.abs(sum(s.U[0]) / m0 - 1)).toBeLessThan(1e-12);
    expect(Math.abs(sum(s.U[4]) / e0 - 1)).toBeLessThan(1e-12);
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(s.U[0][i] - exact(((i + 0.5) / n - 0.5) / 0.2).rho) / n;
    expect(l1).toBeLessThan(0.012);
    // The shock: where the density falls through the middle of its jump, against 1.75216 t.
    let xs = NaN;
    for (let i = n - 2; i > 0; i--)
      if (s.U[0][i] > 0.195 && s.U[0][i + 1] <= 0.195) {
        xs = (i + 0.5 + (s.U[0][i] - 0.195) / (s.U[0][i] - s.U[0][i + 1])) / n - 0.5;
        break;
      }
    expect(Math.abs(xs - 0.2 * 1.75216)).toBeLessThan(1 / n);
    // Fewer steps than the sound speed would allow (151 for the explicit scheme at CFL 0.8).
    expect(s.steps).toBeLessThan(100);
  });

  it('holds a liquid Gresho vortex alike at Mach 0.1, 0.01 and 0.001, in the same number of steps', () => {
    const runs = [0.1, 0.01, 0.001].map((m) => ({ m, ...runGresho(m) }));
    for (const r of runs) {
      expect(r.kept, `Mach ${r.m}`).toBeGreaterThan(0.95);
      expect(r.kept, `Mach ${r.m}`).toBeLessThan(1);
      // Mach 0.01 and 0.001 are the same flow; at 0.1 the gas's own compressibility shows (0.007 here).
      expect(Math.abs(r.kept - runs[1].kept), `Mach ${r.m}`).toBeLessThan(r.m > 0.05 ? 0.015 : 1e-3);
      expect(Math.abs(r.steps - runs[0].steps), `Mach ${r.m}`).toBeLessThanOrEqual(0.05 * runs[0].steps);
      expect(r.dev, `Mach ${r.m}`).toBeLessThan(2 * r.m * r.m);
      // Mass and momentum are sums of face differences: round-off.
      expect(Math.abs(r.mass), `Mach ${r.m}`).toBeLessThan(1e-12);
      expect(r.mom, `Mach ${r.m}`).toBeLessThan(1e-12);
    }
  });

  it('is as good in single precision, and needs the mass flux corrected', () => {
    const double = runGresho(0.001);
    const single = runGresho(0.001, { single: true });
    expect(Math.abs(single.kept - double.kept)).toBeLessThan(1e-3);
    // The density left as advection made it (the scheme as published, for a gas with an
    // energy): explicit in the pressure of a liquid, and it runs away.
    const plain = runGresho(0.1, { baro: 'kwatra', maxIter: 200 });
    expect(plain.kept > 2 || !Number.isFinite(plain.kept)).toBe(true);
  });

  it('carries sound at its speed when resolved in time, and damps it when not', () => {
    // A plane wave in a liquid at rest, one wavelength across 64 cells: after one period it is back.
    const n = 64;
    const wave = (courant: number) => {
      const s = new RefSplit(cfg({ nx: n, ny: 1, dx: 1 / n, eos: liquid(1), cfl: 1, acousticCfl: courant, predGradP: 'none', tol: 1e-12 }));
      const a = 1e-4;
      s.init((x) => ({ rho: 1 + a * Math.sin(2 * Math.PI * x), vx: a * Math.sin(2 * Math.PI * x), vy: 0, vz: 0, p: 0, s: 0 }));
      let peak = 0;
      while (s.t < 1 - 1e-12) {
        s.step(1);
        for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(s.U[0][i] - 1));
      }
      // Phase and amplitude of the first Fourier mode of the density.
      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) / n - 0.5;
        re += (s.U[0][i] - 1) * Math.sin(2 * Math.PI * x);
        im += (s.U[0][i] - 1) * Math.cos(2 * Math.PI * x);
      }
      return { shift: Math.atan2(-im, re) / (2 * Math.PI), amp: (2 * Math.hypot(re, im)) / n / a, peak: peak / a, steps: s.steps };
    };
    const fine = wave(0.5);
    // A wavelength's travel off by under 1%: the speed to 1%.
    expect(Math.abs(fine.shift)).toBeLessThan(0.01);
    expect(fine.amp).toBeGreaterThan(0.7);
    expect(fine.amp).toBeLessThan(1);
    const coarse = wave(10);
    expect(coarse.steps).toBeLessThan(10);
    expect(coarse.amp).toBeLessThan(0.2);
    expect(coarse.peak).toBeLessThanOrEqual(1 + 1e-9);
  });
});

describe('implicit pressure, as a setting', () => {
  it('is taken for the liquid on a uniform grid and refused elsewhere, with the reason', async () => {
    const { configFor, hydroSlots, implicitUnsupported, packParams, refusal } = await import('../../src/sim');
    const { problemById } = await import('../../src/presets/index');
    const liquid = configFor(problemById('gresho'), 64, { ndim: 2, pressure: 'implicit' });
    expect(implicitUnsupported(liquid)).toBeNull();
    expect(refusal(liquid)).toBeNull();
    // Its flux is advection's alone, and its predictor takes the last solve's gradient.
    expect(hydroSlots(liquid)).toMatchObject({ riemann: 'hydro/riemann/advect.wgsl', psplit: 'const PSPLIT = true;\nconst PSPLIT_LAG = true;\n', psplit_grad: 'pressure/predict_on.wgsl' });
    expect(hydroSlots({ ...liquid, pressure: 'explicit' })).toMatchObject({ riemann: 'hydro/riemann/hllclm.wgsl', psplit: 'const PSPLIT = false;\nconst PSPLIT_LAG = false;\n' });
    // The sound speed's weight in the timestep: one over the acoustic steps a step may span.
    const weight = (c: typeof liquid) => new Float32Array(packParams(c, { dims: [64, 64, 1], dx: 1 / 64, rhoMean: 1, selfGravity: false, masses: 0 }))[69];
    expect(weight({ ...liquid, acousticCfl: 50 })).toBeCloseTo(0.02, 8);
    expect(weight({ ...liquid, pressure: 'explicit' })).toBe(0);
    expect(refusal(configFor(problemById('kelvin-helmholtz'), 64, { ndim: 2, pressure: 'implicit' }))).toMatch(/Implicit pressure: .*liquid/);
    expect(refusal({ ...liquid, amrLevels: 1 })).toBeNull();
    expect(refusal({ ...liquid, gext: [0, -1, 0] })).toBeNull();
    // A liquid's own refusals come first.
    expect(refusal({ ...liquid, gravity: 'isolated' })).toMatch(/Liquid: .*self-gravity/);
  });
});
