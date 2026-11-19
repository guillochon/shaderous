// The float64 prototype of the implicit-pressure step (src/reference/psplit.ts; plans/incompressible.md 4.2).
import { describe, expect, it } from 'vitest';
import { exactRiemann } from '../../src/reference/riemann_exact';
import { RefSplit, splitCentered, splitDefaults, type SplitConfig } from '../../src/reference/psplit';
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

describe('implicit-pressure prototype, a gas with an energy', () => {
  // A Gresho vortex in an ideal gas at Mach `mach`, 32 x 32 cells, to t = 1.
  const run = (mach: number, over: Partial<SplitConfig>) => {
    const s = new RefSplit(cfg({ nx: 32, ny: 32, dx: 1 / 32, tol: 1e-4, pa: 'upwind', ...over }));
    const p0 = 1 / (1.4 * mach * mach);
    s.init((x, y) => {
      const r = Math.hypot(x, y);
      const { u, dp } = greshoProfile(r);
      const c = r > 0 ? [-y / r, x / r] : [0, 0];
      return { rho: 1, vx: u * c[0], vy: u * c[1], vz: 0, p: p0 + dp, s: 0 };
    });
    const [e0, m0, en0] = [s.kinetic(), sum(s.U[0]), sum(s.U[4])];
    s.run(1, 3000);
    let dev = 0;
    for (let o = 0; o < s.ncell; o++) dev = Math.max(dev, Math.abs(s.U[0][o] - 1));
    return { kept: s.kinetic() / e0, steps: s.steps, done: s.t > 0.999, dev, mass: Math.abs(sum(s.U[0]) / m0 - 1), energy: Math.abs(sum(s.U[4]) / en0 - 1) };
  };

  it('holds the vortex alike at Mach 0.1, 0.01 and 0.001 once mass and energy ride on the corrected face velocities', () => {
    const runs = [0.1, 0.01, 0.001].map((m) => ({ m, ...run(m, { gas: 'carried' }) }));
    for (const r of runs) {
      expect(r.done, `Mach ${r.m}`).toBe(true);
      expect(r.kept, `Mach ${r.m}`).toBeGreaterThan(0.96);
      expect(r.kept, `Mach ${r.m}`).toBeLessThan(1);
      expect(Math.abs(r.kept - runs[1].kept), `Mach ${r.m}`).toBeLessThan(0.005);
      expect(Math.abs(r.steps - runs[0].steps), `Mach ${r.m}`).toBeLessThanOrEqual(0.15 * runs[0].steps);
      // The density should stay at 1: what moves it is the divergence the solve leaves.
      expect(r.dev, `Mach ${r.m}`).toBeLessThan(2e-3);
      expect(r.mass, `Mach ${r.m}`).toBeLessThan(1e-11);
      expect(r.energy, `Mach ${r.m}`).toBeLessThan(1e-11);
    }
  });

  it('drifts in density, and needs a ceiling on the step, as published', () => {
    // Under 20 acoustic steps a step it is stable, but the density is off by percents...
    const capped = run(0.01, { gas: 'kwatra', acousticCfl: 20 });
    expect(capped.done).toBe(true);
    expect(capped.dev).toBeGreaterThan(0.02);
    // ... and takes several times the steps, the ceiling binding.
    expect(capped.steps).toBeGreaterThan(4 * run(0.01, { gas: 'carried' }).steps);
  });

  it('is as good with the state in single precision at Mach 0.001', () => {
    const double = run(0.001, { gas: 'carried' });
    const single = run(0.001, { gas: 'carried', single: true });
    expect(Math.abs(single.kept - double.kept)).toBeLessThan(2e-3);
    expect(single.dev).toBeLessThan(2e-3);
  });

  it('holds the vortex alike at Mach 0.1, 0.01 and 0.001 time-centered too, and keeps more of it', () => {
    const be = run(0.01, { gas: 'carried' });
    const runs = [0.1, 0.01, 0.001].map((m) => ({ m, ...run(m, { gas: 'carried', ...splitCentered }) }));
    for (const r of runs) {
      expect(r.done, `Mach ${r.m}`).toBe(true);
      expect(r.kept, `Mach ${r.m}`).toBeGreaterThan(be.kept);
      expect(r.kept, `Mach ${r.m}`).toBeLessThan(1);
      expect(Math.abs(r.steps - be.steps), `Mach ${r.m}`).toBeLessThanOrEqual(0.15 * be.steps);
      expect(r.dev, `Mach ${r.m}`).toBeLessThan(2e-3);
      expect(r.mass, `Mach ${r.m}`).toBeLessThan(1e-11);
      expect(r.energy, `Mach ${r.m}`).toBeLessThan(1e-11);
    }
  });
});

// The time-centered step (pressure/theta.wgsl; splitCentered): theta = max(chi, 1/2 + K C) per
// cell, the gas carried on the mean of the old and corrected face velocities, and the predictor's
// pressure compressed where sound is resolved. One solve a step, as backward Euler.
describe('implicit-pressure prototype, time-centered', () => {
  const l1 = (a: Float64Array, b: Float64Array) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;

  it('carries resolved sound without backward Euler’s damping, and still damps sound it does not resolve', () => {
    const n = 64;
    const wave = (courant: number) => {
      const s = new RefSplit(cfg({ nx: n, ny: 1, dx: 1 / n, eos: liquid(1), cfl: 1, acousticCfl: courant, predGradP: 'none', tol: 1e-12, ...splitCentered }));
      const a = 1e-4;
      s.init((x) => ({ rho: 1 + a * Math.sin(2 * Math.PI * x), vx: a * Math.sin(2 * Math.PI * x), vy: 0, vz: 0, p: 0, s: 0 }));
      let peak = 0;
      while (s.t < 1 - 1e-12) {
        s.step(1);
        for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(s.U[0][i] - 1));
      }
      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) / n - 0.5;
        re += (s.U[0][i] - 1) * Math.sin(2 * Math.PI * x);
        im += (s.U[0][i] - 1) * Math.cos(2 * Math.PI * x);
      }
      return { amp: (2 * Math.hypot(re, im)) / n / a, peak: peak / a, steps: s.steps };
    };
    // theta = 0.55 at an acoustic Courant number of 0.5: 1.5% a period (backward Euler: 14%).
    const fine = wave(0.5);
    expect(fine.amp).toBeGreaterThan(0.98);
    expect(fine.amp).toBeLessThan(1);
    // theta = 1 at 10: damped as before, with no overshoot.
    const coarse = wave(10);
    expect(coarse.steps).toBeLessThan(10);
    expect(coarse.amp).toBeLessThan(0.2);
    expect(coarse.peak).toBeLessThanOrEqual(1 + 1e-9);
  });

  it('is second order in the step for a smooth wave in moving gas, where backward Euler is first', () => {
    // A simple wave of amplitude 0.05 on a flow at 0.5 (Mach 0.42), 128 cells, to t = 0.3: each
    // run against the same scheme at a fortieth of the default step.
    const n = 128;
    const run = (cfl: number, over: Partial<SplitConfig>) => {
      const s = new RefSplit(cfg({ nx: n, ny: 1, dx: 1 / n, cfl, acousticCfl: Infinity, pa: 'upwind', gas: 'carried', tol: 1e-12, ...over }));
      s.init((x) => {
        const w = 0.05 * Math.sin(2 * Math.PI * x);
        return { rho: 1 + w, vx: 0.5 + Math.sqrt(1.4) * w, vy: 0, vz: 0, p: 1 + 1.4 * w, s: 0 };
      });
      s.run(0.3);
      return [Float64Array.from(s.U[0]), Float64Array.from(s.U[1])];
    };
    const error = (over: Partial<SplitConfig>) => {
      const ref = run(0.01, over);
      return [0.4, 0.2, 0.1].map((cfl) => {
        const r = run(cfl, over);
        return l1(r[0], ref[0]) + l1(r[1], ref[1]);
      });
    };
    const be = error({});
    const tc = error(splitCentered);
    // Backward Euler: halving the step halves the error.
    expect(be[0] / be[1]).toBeLessThan(2.2);
    // Time-centered: it falls by four (measured: 3.9 and 3.6), and is a tenth of backward Euler's.
    expect(tc[0] / tc[1]).toBeGreaterThan(3.4);
    expect(tc[1] / tc[2]).toBeGreaterThan(3.2);
    expect(tc[0]).toBeLessThan(0.1 * be[0]);
  });

  it('takes the Sod tube’s shock by backward Euler and the rest centered: less error, no ringing, as many steps', () => {
    const n = 128;
    const exact = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
    const tube = (cfl: number, over: Partial<SplitConfig>) => {
      const s = new RefSplit(cfg({ nx: n, ny: 1, dx: 1 / n, bcLo: [1, 0, 0], bcHi: [1, 0, 0], cfl, acousticCfl: 20, pa: 'upwind', gas: 'carried', tol: 1e-10, ...over }));
      s.init((x) => (x < 0 ? { rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 1 } : { rho: 0.125, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }));
      const [m0, e0] = [sum(s.U[0]), sum(s.U[4])];
      s.run(0.2);
      let err = 0;
      let vmax = 0;
      for (let i = 0; i < n; i++) {
        err += Math.abs(s.U[0][i] - exact(((i + 0.5) / n - 0.5) / 0.2).rho) / n;
        vmax = Math.max(vmax, s.U[1][i] / s.U[0][i]);
      }
      return { err, vmax, steps: s.steps, mass: Math.abs(sum(s.U[0]) / m0 - 1), energy: Math.abs(sum(s.U[4]) / e0 - 1) };
    };
    for (const cfl of [0.8, 0.4]) {
      const be = tube(cfl, {});
      const tc = tube(cfl, splitCentered);
      // Measured: 0.0107 against 0.0167 at 0.8, 0.0081 against 0.0117 at 0.4.
      expect(tc.err, `CFL ${cfl}`).toBeLessThan(0.7 * be.err);
      // Behind the shock the gas moves at 0.927: theta near 1/2 there rang to 1.3.
      expect(tc.vmax, `CFL ${cfl}`).toBeLessThan(1.03);
      expect(tc.steps, `CFL ${cfl}`).toBeLessThanOrEqual(1.1 * be.steps);
      expect(tc.mass).toBeLessThan(1e-12);
      expect(tc.energy).toBeLessThan(1e-12);
    }
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
    expect(hydroSlots(liquid)).toMatchObject({ riemann: 'hydro/riemann/advect.wgsl', psplit: 'const PSPLIT = true;\nconst PSPLIT_LAG = true;\nconst PSPLIT_CENTER = true;\n', psplit_grad: 'pressure/predict_on.wgsl' });
    expect(hydroSlots({ ...liquid, pressure: 'explicit' })).toMatchObject({ riemann: 'hydro/riemann/hllclm.wgsl', psplit: 'const PSPLIT = false;\nconst PSPLIT_LAG = false;\nconst PSPLIT_CENTER = false;\n' });
    // The sound speed's weight in the timestep: one over the acoustic steps a step may span.
    const weight = (c: typeof liquid) => new Float32Array(packParams(c, { dims: [64, 64, 1], dx: 1 / 64, rhoMean: 1, selfGravity: false, masses: 0 }))[69];
    expect(weight({ ...liquid, acousticCfl: 50 })).toBeCloseTo(0.02, 8);
    expect(weight({ ...liquid, pressure: 'explicit' })).toBe(0);
    expect(refusal(configFor(problemById('kelvin-helmholtz'), 64, { ndim: 2, pressure: 'implicit' }))).toBeNull();
    expect(refusal(configFor(problemById('kelvin-helmholtz'), 64, { ndim: 2, pressure: 'implicit', eos: 'barotropic' }))).toMatch(/Implicit pressure: .*liquid and the ideal gas/);
    expect(hydroSlots(configFor(problemById('kelvin-helmholtz'), 64, { ndim: 2, pressure: 'implicit' })).psplit_store).toBe('pressure/store_on.wgsl');
    expect(refusal({ ...liquid, amrLevels: 1 })).toBeNull();
    expect(refusal({ ...liquid, gext: [0, -1, 0] })).toBeNull();
    // A liquid's own refusals come first.
    expect(refusal({ ...liquid, gravity: 'isolated' })).toMatch(/Liquid: .*self-gravity/);
  });

  it('notes its solves in the status line', async () => {
    const { pressureNote, configFor, refusal, implicitBlocksUnsupported } = await import('../../src/sim');
    const { problemById } = await import('../../src/presets/index');
    expect(pressureNote({ solves: 0, residual: 0, unconverged: 0 })).toBe('');
    expect(pressureNote({ solves: 40, residual: 2.34e-5, unconverged: 0 })).toBe('pressure solve 2.3e-5');
    expect(pressureNote({ solves: 40, residual: 3e-3, unconverged: 7 })).toBe('pressure solve 3.0e-3, 7 of 40 above tolerance');
    // On blocks per-block timesteps are refused, and nothing of the gas is.
    const gas = configFor(problemById('kelvin-helmholtz'), 64, { ndim: 2, pressure: 'implicit', amrLevels: 1 });
    expect(refusal(gas)).toBeNull();
    expect(implicitBlocksUnsupported(gas)).toBeNull();
    expect(implicitBlocksUnsupported({ ...gas, gext: [0, -1, 0] })).toBeNull();
    expect(implicitBlocksUnsupported({ ...gas, gravity: 'periodic' })).toBeNull();
    expect(implicitBlocksUnsupported({ ...gas, timeBins: 2 })).toMatch(/per-block timesteps/);
    // Point masses: with a gas, and without sinks (the default sink is the softening length).
    const { implicitUnsupported } = await import('../../src/sim');
    const withMass = (m: object, eos = gas.eos) => ({ ...gas, eos, problem: { ...gas.problem, masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], ...m }] } }) as typeof gas;
    expect(implicitUnsupported(withMass({ sinkRadius: 0 }))).toBeNull();
    expect(implicitUnsupported(withMass({}))).toMatch(/take gas in/);
    expect(implicitUnsupported(withMass({ sinkRadius: 0 }, 'liquid'))).toMatch(/for a liquid/);
  });
});
