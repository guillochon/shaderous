// Validation tests for spherical (r, theta) runs (plans/spherical-plan.md, coordinate plan E9).
// Registered through validation/frames.ts.
import { OUTFLOW, OUTFLOW_ONLY, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
import { SEDOV_XI0 } from '../presets/names';
import type { Sim, SimConfig } from '../sim';
import type { Gpu } from '../gpu/device';
import { sphGrid } from '../spherical';
import { withAmr, withSim, type TestDef } from './tests';
import type { AmrSim } from '../amr/amr_sim';
import { compileSetup } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';

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

/** sph-conserve's dense swirling blob, moving along -z: v_r = -cos(theta), v_theta = sin(theta). */
function blob(r: number, th: number): Prim {
  const R = r * Math.sin(th);
  const z = r * Math.cos(th);
  const d = Math.hypot(R - 0.3, z - 0.3);
  const inside = d < 0.2;
  const vz = -1;
  return { rho: inside ? 3 : 1, vx: vz * Math.cos(th), vy: -vz * Math.sin(th), vz: inside ? 2 * R * (1 - d / 0.2) : 0, p: 1, s: inside ? 1 : 0 };
}

/** The Sedov blast of sph-sedov: n^2 from a reflecting sphere at r = 0.005 to 0.5, uniform in r, E = 1 within r = 0.03 (on the finest grid of n). */
function sedov(n: number): Problem {
  const r0 = 0.03;
  const pr = shell(0.005, 0.5, false, () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1e-5, s: 0 }));
  const g = sphGrid({ problem: pr, n, ndim: 2 });
  let vol = 0;
  for (let i = 0; g.r(i) < r0; i++) vol += g.shell(i);
  const pIn = 0.4 / vol;
  return shell(0.005, 0.5, false, (r) => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: r < r0 ? pIn : 1e-5, s: 0 }), { bcHi: [OUTFLOW, REFLECT, PERIODIC], tEnd: 0.05 });
}

/** The n = 1 polytrope of sph-polytrope: radius 1, central density 1, G = 1. */
const LANE = (r: number) => (r < 1 ? Math.sin(Math.PI * r) / (Math.PI * r) : 0);
/** Its mass inside r: 4 pi alpha^3 (sin x - x cos x), x = r / alpha, alpha = 1 / pi. */
const polyMass = (r: number) => (4 / Math.PI ** 2) * (Math.sin(Math.PI * r) - Math.PI * r * Math.cos(Math.PI * r));

/**
 * sph-polytrope's star, on a logarithmic grid from r = 0.05 to 2: the mass inside r = 0.05 a
 * point mass at the origin, the gas's own gravity its monopole.
 */
function polytrope(): Problem {
  const rMin = 0.05;
  const K = 2 / Math.PI;
  return shell(rMin, 2, true, (r) => {
    const rho = Math.max(1e-6, LANE(r));
    return { rho, vx: 0, vy: 0, vz: 0, p: K * rho * rho, s: r < 1 ? 1 : 0 };
  }, {
    eosParams: { gamma: 2 }, G: 1, gravity: 'isolated', bcHi: [OUTFLOW_ONLY, REFLECT, PERIODIC], tEnd: 18, rhoFloor: 1e-9, pFloor: 1e-14,
    masses: () => [{ gm: polyMass(rMin), pos: [0, 0, 0], vel: [0, 0, 0] }],
  });
}

/**
 * The leaves of a spherical run on blocks, each cell with its level's grid (sphGrid at the
 * level's cells per side) and its indices on that level.
 */
async function leafCells(sim: AmrSim) {
  const c = await sim.readLeaves();
  const box = sim.cfg.problem.box;
  const rMin = sim.cfg.problem.sphere!.rMin;
  const grids = new Map<number, ReturnType<typeof sphGrid>>();
  const cells: { g: ReturnType<typeof sphGrid>; i: number; j: number; q: number }[] = [];
  for (let q = 0; q < c.count; q++) {
    const n = Math.round(box / c.dx[q]);
    let g = grids.get(n);
    if (!g) grids.set(n, (g = sphGrid({ problem: sim.cfg.problem, n, ndim: 2 })));
    cells.push({ g, i: Math.round((c.x[q] - rMin) / c.dx[q] - 0.5), j: Math.round(c.y[q] / c.dx[q] - 0.5), q });
  }
  return { c, cells };
}

/** Totals over a spherical run's leaves: mass, angular momentum about the axis, total energy. */
async function leafTotals(sim: AmrSim) {
  const { c, cells } = await leafCells(sim);
  let m = 0;
  let l = 0;
  let e = 0;
  for (const { g, i, j, q } of cells) {
    const dv = g.volume(i, j);
    m += c.u1[4 * q] * dv;
    l += g.r(i) * Math.sin(g.theta(j)) * c.u1[4 * q + 3] * dv;
    e += c.u2[4 * q] * dv;
  }
  return { m, l, e, levels: new Set(cells.map((x) => x.g.n)).size };
}

/** sph-polytrope's star on blocks, settled as there: how far its innermost cells' density moves over t = 18. */
async function polytropeOnBlocks(gpu: Gpu, n: number, over: Partial<SimConfig>) {
  return withAmr(gpu, polytrope(), n, { ndim: 2, ...over }, async (sim) => {
    // The volume-weighted density of the cells at r_min, at whatever level.
    const central = async () => {
      const { c, cells } = await leafCells(sim);
      let m = 0;
      let v = 0;
      for (const { g, i, j, q } of cells) {
        if (i !== 0) continue;
        m += c.u1[4 * q] * g.volume(i, j);
        v += g.volume(i, j);
      }
      return m / v;
    };
    for (let k = 0; k < 30; k++) {
      sim.cfg.tEnd = 0.2;
      sim.writeParams();
      await sim.runToEnd();
      const { u1, u2 } = await sim.readLeaves();
      for (let q = 0; q < u1.length; q += 4) {
        u1[q + 1] = u1[q + 2] = u1[q + 3] = 0;
        u2[q] = u2[q + 1];
      }
      sim.setLeaves(u1, u2);
    }
    const a = await central();
    let worst = 0;
    let tWorst = 0;
    for (let k = 1; k <= 18; k++) {
      sim.cfg.tEnd = k;
      sim.writeParams();
      await sim.runToEnd();
      const e = Math.abs((await central()) / a - 1);
      if (e > worst) {
        worst = e;
        tWorst = k;
      }
    }
    const d = await sim.diagnostics();
    const levels = new Set((await leafCells(sim)).cells.map((x) => x.g.n)).size;
    const work = over.timeBins ? await sim.binWork() : null;
    return {
      central_density_settled: a, worst_change: worst, at_time: tWorst, mass: d.mass, mass_expected: polyMass(1) - polyMass(0.05), levels, steps: d.steps,
      block_updates_over_single: work ? work.updates / work.single : 1,
    };
  });
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
      const p = shell(0.1, 1, true, blob, { tEnd: 0.6 });
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
      return withSim(gpu, sedov(n), n, { ndim: 2 }, async (sim) => {
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
    criterion: 'An n = 1 polytrope (gamma 2, radius 1, central density 1, G = 1) on a 128^2 logarithmic grid from r = 0.05 to 2, the mass inside r = 0.05 a point mass at the origin and the gas’s own gravity the monopole of its shells, settled (stopped every 0.2 for 6): over 10 dynamical times after (t = 18) the density of the innermost shell stays within 1% of the settled one',
    async run(gpu) {
      const n = 128;
      const p = polytrope();
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        // Settle it into the grid's own balance: run, and stop the gas, every 0.2 for 6 (about
        // three dynamical times), keeping its internal energy.
        for (let k = 0; k < 30; k++) {
          sim.cfg.tEnd = 0.2;
          sim.writeParams();
          await sim.runToEnd();
          const { u1, u2 } = await sim.readState();
          for (let q = 0; q < u1.length; q += 4) {
            u1[q + 1] = u1[q + 2] = u1[q + 3] = 0;
            u2[q] = u2[q + 1];
          }
          sim.setState(u1, u2, 0);
        }
        const a = profile(sim, (await sim.readState()).u1);
        let worst = 0;
        let tWorst = 0;
        for (let k = 1; k <= 18; k++) {
          sim.cfg.tEnd = k;
          sim.writeParams();
          await sim.runToEnd();
          const e = Math.abs(profile(sim, (await sim.readState()).u1)[0] / a[0] - 1);
          if (e > worst) {
            worst = e;
            tWorst = k;
          }
        }
        const d = await sim.diagnostics();
        const m = { central_density_analytic: LANE(sphGrid(sim.cfg).r(0)), central_density_settled: a[0], worst_change: worst, at_time: tWorst, mass: d.mass, mass_expected: polyMass(1) - polyMass(0.05), steps: d.steps };
        return { pass: m.worst_change < 0.01, metrics: m };
      });
    },
  },
  {
    id: 'sph-bondi',
    name: 'Spherical: Bondi accretion through the inner sphere',
    criterion: 'Isothermal gas (c = 1, density 1e-5 far away) around a point mass GM = 1 at the origin, on a 128^2 logarithmic grid from r = 0.1 (open for outflow only, well inside the sonic point) to 16, its outer two shells held at the isothermal atmosphere’s state: the accretion rate through every shell inside the Bondi radius, averaged over t = 40 to 80, is within 2% of Bondi’s, 4 pi lambda (GM)^2 rho / c^3 with lambda = e^1.5 / 4',
    async run(gpu) {
      const n = 128;
      const rho0 = 1e-5;
      const rMax = 16;
      // The outer two shells are held at the isothermal atmosphere's state at rest, rho0 e^(GM / r):
      // a reservoir that stands for the gas at infinity (an open face would let in whatever the
      // infall from the uniform start draws, and the density there grew without bound).
      const edgeR = sphGrid({ problem: { box: rMax - 0.1, geometry: 'spherical', sphere: { rMin: 0.1, log: true } }, n, ndim: 2 }).rFace(n - 2);
      const p = shell(0.1, rMax, true, (r) => ({ rho: r > edgeR ? rho0 * Math.exp(1 / r) : rho0, vx: 0, vy: 0, vz: 0, p: r > edgeR ? rho0 * Math.exp(1 / r) : rho0, s: 0, fixed: r > edgeR }), {
        eos: 'barotropic', eosParams: { gamma: 1, K: 1 }, G: 1, bcLo: [OUTFLOW_ONLY, REFLECT, PERIODIC], bcHi: [OUTFLOW, REFLECT, PERIODIC], tEnd: 80,
        rhoFloor: 1e-12,
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0] }],
      });
      const bondi = 4 * Math.PI * (Math.exp(1.5) / 4) * rho0;
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        const g = sphGrid(sim.cfg);
        // The inflow through shell i: -sum over theta of rho v_r r^2 2 pi dcos(theta).
        const rate = (u1: Float32Array, i: number) => {
          let s = 0;
          for (let j = 0; j < n; j++) s -= u1[4 * (j * n + i) + 1] * g.r(i) ** 2 * 2 * Math.PI * (Math.cos(g.thetaFace(j)) - Math.cos(g.thetaFace(j + 1)));
          return s;
        };
        // The rate through each shell, averaged over t = 40 to 80 (the flow inside the
        // reservoir still breathes by a couple of percent about its mean).
        const times = Array.from({ length: 6 }, (_, k) => 40 + 8 * k);
        const sum: number[] = [];
        let lo = Infinity;
        let hi = 0;
        for (const t of times) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd();
          const { u1 } = await sim.readState();
          for (let i = 0; g.r(i) < 1; i++) {
            const r = rate(u1, i);
            sum[i] = (sum[i] ?? 0) + r / times.length;
            lo = Math.min(lo, r);
            hi = Math.max(hi, r);
          }
        }
        const worst = Math.max(...sum.map((r) => Math.abs(r / bondi - 1)));
        const d = await sim.diagnostics();
        const m = { bondi, mean_rate_innermost: sum[0], worst_relative_difference: worst, instantaneous_min: lo / bondi, instantaneous_max: hi / bondi, steps: d.steps };
        return { pass: m.worst_relative_difference < 0.02, metrics: m };
      });
    },
  },
  {
    id: 'sph-blocks',
    name: 'Spherical: the blast and the swirling blob on blocks',
    criterion: 'On an adaptive mesh of two levels: sph-sedov’s blast (256^2 finest) has its shock radius within 2% of the similarity solution along every one of 64 rays, and sph-conserve’s swirling blob (128^2 finest, logarithmic in r) keeps mass, energy and angular momentum about the axis to 1e-5 through its re-meshing; both with a single timestep and with per-block timesteps (two bins)',
    async run(gpu) {
      const n = 256;
      const blastOn = (bins: boolean) => withAmr(gpu, sedov(n), n, { ndim: 2, amrLevels: 2, ...(bins ? { timeBins: 2, timeBinsAlways: true } : {}) }, async (sim) => {
        await sim.runToEnd();
        const { c, cells } = await leafCells(sim);
        // Along 64 rays (the centers of the coarsest level's theta cells, nudged off the finer
        // levels' faces), the leaves the ray crosses, outward: where the density falls half way
        // from its peak to the ambient, outside the peak.
        const rays = 64;
        const radii: number[] = [];
        let finest = 0;
        for (let k = 0; k < rays; k++) {
          const th = ((k + 0.5) / rays) * Math.PI + 1e-7;
          const ray = cells.filter(({ g, j }) => g.thetaFace(j) <= th && th < g.thetaFace(j + 1)).sort((a, b) => a.g.r(a.i) - b.g.r(b.i));
          const rho = ray.map(({ q }) => c.u1[4 * q]);
          const r = ray.map(({ g, i }) => g.r(i));
          let ip = 0;
          for (let i = 0; i < rho.length; i++) if (rho[i] > rho[ip]) ip = i;
          const half = 0.5 * (rho[ip] + 1);
          let at = r[r.length - 1];
          for (let i = ip; i < rho.length - 1; i++) {
            if (rho[i + 1] < half) {
              at = r[i] + ((rho[i] - half) / (rho[i] - rho[i + 1])) * (r[i + 1] - r[i]);
              finest = Math.max(finest, ray[i].g.n);
              break;
            }
          }
          radii.push(at);
        }
        return { radii, finest, levels: new Set(cells.map((x) => x.g.n)).size };
      });
      const blast = await blastOn(false);
      const blastBins = await blastOn(true);
      const want = SEDOV_XI0['1.4'] * Math.pow(0.05 * 0.05, 0.2);
      const lo = Math.min(...blast.radii);
      const hi = Math.max(...blast.radii);
      const binsWorst = Math.max(...blastBins.radii.map((r) => Math.abs(r / want - 1)));
      const p = shell(0.1, 1, true, blob, { tEnd: 0.6 });
      const conserve = (bins: boolean) =>
        withAmr(gpu, p, 128, { ndim: 2, amrLevels: 2, ...(bins ? { timeBins: 2, timeBinsAlways: true } : {}) }, async (sim) => {
          const a = await leafTotals(sim);
          await sim.runToEnd();
          const b = await leafTotals(sim);
          const work = bins ? await sim.binWork() : null;
          return {
            mass: Math.abs(b.m / a.m - 1), energy: Math.abs(b.e / a.e - 1), angular_momentum: Math.abs(b.l / a.l - 1), levels: b.levels, regrids: sim.regrids,
            ...(work ? { block_updates_over_single: work.updates / work.single } : {}),
          };
        });
      const cons = await conserve(false);
      const binned = await conserve(true);
      const m = {
        similarity: want, radius_min: lo, radius_max: hi, worst_error: Math.max(Math.abs(lo / want - 1), Math.abs(hi / want - 1)), spread: (hi - lo) / want,
        blast_levels: blast.levels, shock_cells_per_side: blast.finest, ...cons,
        bins_mass: binned.mass, bins_energy: binned.energy, bins_angular_momentum: binned.angular_momentum, bins_work: binned.block_updates_over_single ?? 1, bins_blast_worst_error: binsWorst,
      };
      const kept = (x: { mass: number; energy: number; angular_momentum: number; levels: number }) => x.mass < 1e-5 && x.energy < 1e-5 && x.angular_momentum < 1e-5 && x.levels > 1;
      return { pass: m.worst_error < 0.02 && binsWorst < 0.02 && kept(cons) && kept(binned), metrics: m };
    },
  },
  {
    id: 'sph-polytrope-blocks',
    name: 'Spherical: the polytrope on blocks, in the monopole of its leaves',
    criterion: 'sph-polytrope’s star on an adaptive mesh of two levels (128^2 finest, logarithmic in r), its monopole summed over the leaves into the finest shells: settled as there (stopped every 0.2 for 6), over 10 dynamical times after (t = 18) the volume-weighted density of the innermost cells stays within 1% of the settled one, with a single timestep and with per-block timesteps (two bins)',
    async run(gpu) {
      const one = await polytropeOnBlocks(gpu, 128, { amrLevels: 2 });
      const bins = await polytropeOnBlocks(gpu, 128, { amrLevels: 2, timeBins: 2, timeBinsAlways: true });
      const m = { ...one, bins_worst_change: bins.worst_change, bins_at_time: bins.at_time, bins_block_updates_over_single: bins.block_updates_over_single };
      return { pass: one.worst_change < 0.01 && bins.worst_change < 0.01 && one.levels > 1, metrics: m };
    },
  },
  {
    id: 'sph-swirl',
    name: 'Spherical: a fast swirl around the origin, reaching both poles',
    criterion: 'A dense cold blob (rho 28, sound speed 0.01 around it) swirling at v_phi = 0.65 right to the axis, on the uniform grid and on blocks: for 300 steps the values stay finite, no cell is floored, and the mass holds to 1e-5 (walls on every face). The timestep weighs the cells next to the axis by their outer faces and holds the centrifugal acceleration; without that the axis cells emptied in the first step',
    async run(gpu) {
      const doc = {
        format: 'shaderous-setup', version: 1, name: 'sph-swirl',
        domain: { ndim: 2, n: 128, box: 0.5, geometry: 'spherical', supports2d: true, rMin: 0.005, bc: { lo: [REFLECT, REFLECT, PERIODIC], hi: [REFLECT, REFLECT, PERIODIC] } },
        physics: { eos: 'ideal', eosParams: { gamma: 5 / 3 }, gravity: 'off', G: 0 },
        background: { rho: 1, p: 1e-4 },
        parts: [{ kind: 'region', shape: { type: 'sphere', center: [0, 0, 0], radius: 0.06 }, rho: 28, v: [0, 0, 0.65], s: 1 }],
        run: { tEnd: 10 },
      } as SetupSpec;
      const p = compileSetup(doc).problem;
      const m: Record<string, number> = {};
      const check = async (sim: Sim | AmrSim, tag: string) => {
        const d0 = await sim.diagnostics();
        for (let k = 0; k < 300; k++) sim.step();
        const d = await sim.diagnostics();
        m[`${tag}_mass_change`] = Math.abs(d.mass / d0.mass - 1);
        m[`${tag}_floored`] = d.floored;
        m[`${tag}_finite`] = Number.isFinite(d.eTot) && Number.isFinite(d.mass) ? 1 : 0;
        m[`${tag}_t`] = d.t;
      };
      await withSim(gpu, p, 128, { ndim: 2 }, (sim) => check(sim, 'uniform'));
      await withAmr(gpu, p, 128, { ndim: 2, amrLevels: 1 }, (sim) => check(sim, 'blocks'));
      const pass = ['uniform', 'blocks'].every((t) => m[`${t}_finite`] === 1 && m[`${t}_floored`] === 0 && m[`${t}_mass_change`] < 1e-5);
      return { pass, metrics: m };
    },
  },
];
