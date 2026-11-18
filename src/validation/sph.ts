// Validation tests for spherical (r, theta) runs (plans/spherical-plan.md, coordinate plan E9).
// Registered through validation/frames.ts.
import { OUTFLOW, OUTFLOW_ONLY, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
import { SEDOV_XI0 } from '../presets/names';
import type { Sim } from '../sim';
import { sphGrid } from '../spherical';
import { withSim, type TestDef } from './tests';

/**
 * A spherical run from `rMin` to `rMax` (logarithmic in r with `log`), ideal gas, from
 * `init(r, theta)` with (v_r, v_theta, v_phi) as (vx, vy, vz).
 */
function shell(rMin: number, rMax: number, log: boolean, init: (r: number, th: number) => Prim, over: Partial<Problem> = {}): Problem {
  return {
    id: 'spherical', label: 'Spherical', blurb: '', eos: 'ideal', eosParams: { gamma: 1.4 },
    bcLo: [REFLECT, REFLECT, PERIODIC], bcHi: [REFLECT, REFLECT, PERIODIC], gravity: 'off', G: 0, box: rMax - rMin, tEnd: 0.1, supports2d: true,
    geometry: 'spherical', sphere: { rMin, log }, init: () => (r, th) => init(r, th), ...over,
  };
}

/** Totals over a spherical run's cells: mass, angular momentum about the axis, total energy. */
async function totals(sim: Sim) {
  const { u1, u2 } = await sim.readState();
  const g = sphGrid(sim.cfg);
  const n = g.n;
  let m = 0;
  let l = 0;
  let e = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const o = 4 * (j * n + i);
      const dv = g.volume(i, j);
      m += u1[o] * dv;
      l += g.r(i) * Math.sin(g.theta(j)) * u1[o + 3] * dv;
      e += u2[o] * dv;
    }
  return { m, l, e };
}

/** The shell average of the density at each radial index. */
function profile(sim: Sim, u1: Float32Array): number[] {
  const g = sphGrid(sim.cfg);
  const n = g.n;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    let v = 0;
    for (let j = 0; j < n; j++) {
      s += u1[4 * (j * n + i)] * g.volume(i, j);
      v += g.volume(i, j);
    }
    out.push(s / v);
  }
  return out;
}

/** Where the density along one ray (theta index j) falls to half way from its peak to `amb`, outside the peak, interpolated. */
function edge(sim: Sim, u1: Float32Array, j: number, amb: number): number {
  const g = sphGrid(sim.cfg);
  const n = g.n;
  const rho = (i: number) => u1[4 * (j * n + i)];
  let ip = 0;
  for (let i = 0; i < n; i++) if (rho(i) > rho(ip)) ip = i;
  const half = 0.5 * (rho(ip) + amb);
  for (let i = ip; i < n - 1; i++) {
    if (rho(i + 1) < half) {
      const f = (rho(i) - half) / (rho(i) - rho(i + 1));
      return g.r(i) + f * (g.r(i + 1) - g.r(i));
    }
  }
  return g.r(n - 1);
}

export const SPH_TESTS: TestDef[] = [
  {
    id: 'sph-rest',
    name: 'Spherical: uniform gas at rest stays at rest',
    criterion: 'Uniform gas in (r, theta), 64^2 between reflecting spheres at r = 0.1 and 1, both poles in the grid: after a sound crossing the largest Mach number is below 1e-5, on a uniform and on a logarithmic grid in r (the faces’ pressures and the geometric sources cancel to rounding)',
    async run(gpu) {
      const out: Record<string, number> = {};
      for (const log of [false, true]) {
        const p = shell(0.1, 1, log, () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }), { tEnd: 1 });
        await withSim(gpu, p, 64, { ndim: 2 }, async (sim) => {
          await sim.runToEnd();
          const d = await sim.diagnostics();
          out[log ? 'max_mach_log' : 'max_mach_uniform'] = d.maxMach;
          out[log ? 'mass_log' : 'mass_uniform'] = d.mass;
        });
      }
      out.mass_expected = (4 / 3) * Math.PI * (1 - 0.001);
      return { pass: out.max_mach_uniform < 1e-5 && out.max_mach_log < 1e-5 && Math.abs(out.mass_log / out.mass_expected - 1) < 1e-5, metrics: out };
    },
  },
  {
    id: 'sph-conserve',
    name: 'Spherical: a swirling blob thrown across the poles',
    criterion: 'A dense swirling blob moving across the grid between reflecting spheres at r = 0.1 and 1 (64^2, logarithmic in r): mass, total energy and angular momentum about the axis are kept to 1e-5 over a crossing',
    async run(gpu) {
      const p = shell(0.1, 1, true, (r, th) => {
        const R = r * Math.sin(th);
        const z = r * Math.cos(th);
        const d = Math.hypot(R - 0.3, z - 0.3);
        const inside = d < 0.2;
        // Moving along -z: v_r = -cos(theta), v_theta = sin(theta).
        const vz = -1;
        return { rho: inside ? 3 : 1, vx: vz * Math.cos(th), vy: -vz * Math.sin(th), vz: inside ? 2 * R * (1 - d / 0.2) : 0, p: 1, s: inside ? 1 : 0 };
      }, { tEnd: 0.6 });
      return withSim(gpu, p, 64, { ndim: 2 }, async (sim) => {
        const a = await totals(sim);
        await sim.runToEnd();
        const b = await totals(sim);
        const m = { mass: Math.abs(b.m / a.m - 1), energy: Math.abs(b.e / a.e - 1), angular_momentum: Math.abs(b.l / a.l - 1), angular_momentum_initial: a.l };
        return { pass: m.mass < 1e-5 && m.energy < 1e-5 && m.angular_momentum < 1e-5, metrics: m };
      });
    },
  },
  {
    id: 'sph-sedov',
    name: 'Spherical: a point blast at the origin',
    criterion: 'A Sedov blast (gamma 1.4, E = 1 in the cells within r = 0.03) in (r, theta) at 256^2 from a reflecting sphere at r = 0.005 out to 0.5, uniform in r: at t = 0.05 the shock radius (where the density falls half way from its peak to the ambient) along every ray is within 1% of the similarity solution, and its spread over theta is below 0.5%',
    async run(gpu) {
      const n = 256;
      const rMin = 0.005;
      const rMax = 0.5;
      const r0 = 0.03;
      const pr = shell(rMin, rMax, false, () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1e-5, s: 0 }));
      const g = sphGrid({ problem: pr, n, ndim: 2 });
      let vol = 0;
      for (let i = 0; g.r(i) < r0; i++) vol += g.shell(i);
      const pIn = 0.4 / vol;
      const p = shell(rMin, rMax, false, (r) => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: r < r0 ? pIn : 1e-5, s: 0 }), { bcHi: [OUTFLOW, REFLECT, PERIODIC], tEnd: 0.05 });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const { u1 } = await sim.readState();
        const want = SEDOV_XI0['1.4'] * Math.pow(0.05 * 0.05, 0.2);
        const radii = Array.from({ length: n }, (_, j) => edge(sim, u1, j, 1));
        const lo = Math.min(...radii);
        const hi = Math.max(...radii);
        const worst = Math.max(Math.abs(lo / want - 1), Math.abs(hi / want - 1));
        const mean = radii.reduce((a, b) => a + b, 0) / n;
        const m = { similarity: want, radius_min: lo, radius_max: hi, radius_pole: radii[0], radius_equator: radii[n / 2], worst_error: worst, spread: (hi - lo) / mean };
        return { pass: m.worst_error < 0.01 && m.spread < 0.005, metrics: m };
      });
    },
  },
  {
    id: 'sph-polytrope',
    name: 'Spherical: an n = 1 polytrope in its own monopole field',
    criterion: 'An n = 1 polytrope (gamma 2, radius 1, central density 1, G = 1) on a 128^2 logarithmic grid from r = 0.05 to 2, the mass inside r = 0.05 a point mass at the origin and the gas’s own gravity the monopole of its shells: over 10 dynamical times (t = 18) the density of the innermost shell stays within 1% of its start',
    async run(gpu) {
      const n = 128;
      const rMin = 0.05;
      const rMax = 2;
      const alpha = 1 / Math.PI;
      const K = 2 * Math.PI * alpha * alpha;
      const rhoAmb = 1e-6;
      const lane = (r: number) => (r < 1 ? Math.sin(Math.PI * r) / (Math.PI * r) : 0);
      // The mass inside r: 4 pi alpha^3 (sin x - x cos x), x = r / alpha.
      const mIn = (r: number) => 4 * Math.PI * alpha ** 3 * (Math.sin(r / alpha) - (r / alpha) * Math.cos(r / alpha));
      const p = shell(rMin, rMax, true, (r) => {
        const rho = Math.max(rhoAmb, lane(r));
        return { rho, vx: 0, vy: 0, vz: 0, p: K * rho * rho, s: r < 1 ? 1 : 0 };
      }, {
        eosParams: { gamma: 2 }, G: 1, gravity: 'isolated', bcHi: [OUTFLOW_ONLY, REFLECT, PERIODIC], tEnd: 18, rhoFloor: 1e-9, pFloor: 1e-14,
        masses: () => [{ gm: mIn(rMin), pos: [0, 0, 0], vel: [0, 0, 0] }],
      });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        const a = profile(sim, (await sim.readState()).u1);
        let worst = 0;
        let tWorst = 0;
        for (let k = 1; k <= 6; k++) {
          sim.cfg.tEnd = (18 * k) / 6;
          sim.writeParams();
          await sim.runToEnd();
          const b = profile(sim, (await sim.readState()).u1);
          const e = Math.abs(b[0] / a[0] - 1);
          if (e > worst) {
            worst = e;
            tWorst = sim.cfg.tEnd;
          }
        }
        const d = await sim.diagnostics();
        const m = { central_density_start: a[0], worst_change: worst, at_time: tWorst, mass: d.mass, mass_expected: mIn(1) - mIn(rMin), steps: d.steps };
        return { pass: m.worst_change < 0.01, metrics: m };
      });
    },
  },
  {
    id: 'sph-bondi',
    name: 'Spherical: Bondi accretion through the inner sphere',
    criterion: 'Isothermal gas (c = 1, density 1e-5 far away) around a point mass GM = 1 at the origin, on a 128^2 logarithmic grid from r = 0.1 (open for outflow only, well inside the sonic point) to 16: at t = 30 the accretion rate through the shells inside the Bondi radius is within 2% of Bondi’s, 4 pi lambda (GM)^2 rho / c^3 with lambda = e^1.5 / 4',
    async run(gpu) {
      const n = 128;
      const rho0 = 1e-5;
      const p = shell(0.1, 16, true, () => ({ rho: rho0, vx: 0, vy: 0, vz: 0, p: rho0, s: 0 }), {
        eos: 'barotropic', eosParams: { gamma: 1, K: 1 }, G: 1, bcLo: [OUTFLOW_ONLY, REFLECT, PERIODIC], bcHi: [OUTFLOW, REFLECT, PERIODIC], tEnd: 30,
        rhoFloor: 1e-12,
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0] }],
      });
      const bondi = 4 * Math.PI * (Math.exp(1.5) / 4) * rho0;
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const { u1 } = await sim.readState();
        const g = sphGrid(sim.cfg);
        // The inflow through each shell: -sum over theta of rho v_r r^2 2 pi dcos(theta).
        const rate = (i: number) => {
          let s = 0;
          for (let j = 0; j < n; j++) s -= u1[4 * (j * n + i) + 1] * g.r(i) ** 2 * 2 * Math.PI * (Math.cos(g.thetaFace(j)) - Math.cos(g.thetaFace(j + 1)));
          return s;
        };
        const rates: number[] = [];
        for (let i = 0; g.r(i) < 1; i++) rates.push(rate(i));
        const lo = Math.min(...rates);
        const hi = Math.max(...rates);
        const worst = Math.max(Math.abs(lo / bondi - 1), Math.abs(hi / bondi - 1));
        const d = await sim.diagnostics();
        const m = { bondi, rate_min: lo, rate_max: hi, worst_relative_difference: worst, steps: d.steps };
        return { pass: m.worst_relative_difference < 0.02, metrics: m };
      });
    },
  },
];
