// Validation of radiation transport (radiation/radiation.ts; docs/numerics.md section 24).
// Registered in tests.ts after the others. On the uniform grid and on blocks (rad-blocks-*), in
// 2D but for the star, small enough to share the GPU.
import { all, OUTFLOW, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
import type { Sim, SimConfig } from '../sim';
import { withAmr, withSim, type TestDef } from './tests';
import { AmrSim } from '../amr/amr_sim';
import { capture, restartFrom } from '../io/capture';
import { configFor, Sim as SimClass } from '../sim';
import { marshakDiffusion, RefRad1D } from '../reference/radiation';
import { presetSpec } from '../presets/index';
import { compileSetup } from '../setup/compile';
import { cloneSetup } from '../setup/spec';
import { laneEmdenCached, polytropeK } from '../problems/lane_emden';
import type { StarPart } from '../setup/spec';

/**
 * An n = 3 polytrope (the polytrope preset's, radius 0.3, central density 1, G = 1) whose pressure
 * is a tenth radiation everywhere: Eddington's standard model, with the gas and radiation
 * equation of state's a = 3 (1 - beta) / (beta^4 K^3), beta = 0.9. With `transport`, the
 * radiation is carried by E (opacity kappa per unit mass, chat), else by the equation of state.
 */
function radStar(transport: boolean, kappa: number, chat: number, vx = 0) {
  const spec = cloneSetup(presetSpec('polytrope')!);
  const star = spec.parts![0] as StarPart;
  star.n = 3;
  if (vx) star.vel = [vx, 0, 0];
  const beta = 0.9;
  const K = polytropeK(laneEmdenCached(3), star.rhoc!, star.radius!, 1);
  const a = (3 * (1 - beta)) / (beta ** 4 * K ** 3);
  spec.physics = {
    ...spec.physics, eos: 'gas_radiation', eosParams: { gamma: 5 / 3, R: 1, arad: a },
    ...(transport ? { radiation: 'm1', opacity: 'constant', opacityParams: { absorption: kappa, scattering: 0 }, radChat: chat } : {}),
  };
  return compileSetup(spec).problem;
}

/** Gas at rest in a periodic unit box with radiation on: c = 100 and a = 1 unless changed. */
const radGas = (over: Partial<Problem>): Problem => ({
  id: 'rad-test', label: 'Radiation test', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3, arad: 1 },
  bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
  radiation: 'm1', radC: 100, opacity: 'constant', opacityParams: { absorption: 1, scattering: 0 },
  init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }),
  ...over,
});

const D2: Partial<SimConfig> = { ndim: 2 };

/** The state and radiation of every cell. */
async function cells(sim: Sim) {
  const { u1, u2 } = await sim.readState();
  const R = await sim.rad!.read();
  return { u1, u2, R, n: sim.nx * sim.ny * sim.nz };
}

/** Run to time t (the run's end time set to it). */
async function runTo(sim: Sim | AmrSim, t: number) {
  sim.cfg.tEnd = t;
  sim.writeParams();
  return sim.runToEnd();
}

/** The root of 1.5 T + a T^4 = e (ideal gas, gamma 5/3, rho 1): the equilibrium temperature. */
function equilibriumT(a: number, e: number): number {
  let lo = 0;
  let hi = Math.max(e, 1) * 10;
  for (let i = 0; i < 200; i++) {
    const m = 0.5 * (lo + hi);
    if (1.5 * m + a * m ** 4 > e) hi = m;
    else lo = m;
  }
  return lo;
}

/** Centroid along x and variance about it of E - floor, over cells of a 2D grid (x fastest). */
function moments(R: Float32Array, nx: number, ny: number, dx: number, floor: number, axis: 0 | 1 = 0) {
  let m0 = 0;
  let m1 = 0;
  let m2 = 0;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const w = R[4 * (j * nx + i)] - floor;
      const x = ((axis === 0 ? i : j) + 0.5) * dx;
      m0 += w;
      m1 += w * x;
      m2 += w * x * x;
    }
  const mean = m1 / m0;
  return { total: m0, mean, variance: m2 / m0 - mean * mean };
}

/** Linear interpolation of ys(xs) at x. */
function interp(xs: Float64Array, ys: Float64Array, x: number): number {
  for (let i = 1; i < xs.length; i++)
    if (xs[i] >= x) {
      const w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] * (1 - w) + ys[i] * w;
    }
  return ys[ys.length - 1];
}


/**
 * The jump of a shock made by gas of density rho1 and temperature T1 driven at u0 into a wall,
 * with radiation in equilibrium on both sides (P = rho T + a T^4 / 3, e = 1.5 rho T + a T^4 for
 * gamma 5/3): the gas behind it at rest, and the shock's speed D away from the wall. Rankine and
 * Hugoniot's conditions in the shock's frame, solved by Newton's method.
 */
function radiativeJump(rho1: number, T1: number, u0: number, a: number): { rho2: number; T2: number; D: number } {
  const P = (r: number, T: number) => r * T + (a * T ** 4) / 3;
  const e = (r: number, T: number) => 1.5 * r * T + a * T ** 4;
  const fluxes = (r: number, T: number, w: number) => [r * w, r * w * w + P(r, T), w * (0.5 * r * w * w + e(r, T) + P(r, T))];
  const residual = (x: number[]) => {
    const [rho2, T2, D] = x;
    const up = fluxes(rho1, T1, u0 + D);
    const down = fluxes(rho2, T2, D);
    return up.map((f, i) => (f - down[i]) / Math.max(Math.abs(f), 1));
  };
  // Start from the strong hydrodynamic shock.
  let x = [4 * rho1, (u0 * u0) / 3 + T1, u0 / 3];
  for (let it = 0; it < 100; it++) {
    const f = residual(x);
    const J = [0, 1, 2].map((j) => {
      const h = 1e-7 * Math.max(Math.abs(x[j]), 1e-3);
      const y = [...x];
      y[j] += h;
      const g = residual(y);
      return g.map((v, i) => (v - f[i]) / h);
    });
    // Solve J^T-column system: J[j][i] = d f_i / d x_j.
    const A = [0, 1, 2].map((i) => [J[0][i], J[1][i], J[2][i], -f[i]]);
    for (let k = 0; k < 3; k++) {
      let p = k;
      for (let i = k + 1; i < 3; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
      [A[k], A[p]] = [A[p], A[k]];
      for (let i = k + 1; i < 3; i++) {
        const w = A[i][k] / A[k][k];
        for (let j = k; j < 4; j++) A[i][j] -= w * A[k][j];
      }
    }
    const dx = [0, 0, 0];
    for (let i = 2; i >= 0; i--) dx[i] = (A[i][3] - A[i][2] * (i < 2 ? dx[2] : 0) - A[i][1] * (i < 1 ? dx[1] : 0)) / A[i][i];
    x = x.map((v, i) => Math.max(v + dx[i], 1e-6));
    if (Math.max(...f.map(Math.abs)) < 1e-13) break;
  }
  return { rho2: x[0], T2: x[1], D: x[2] };
}

/**
 * Moments of the radiation energy above `floor` over a run's cells, uniform grid or blocks: its
 * total (times the cells' areas), centroid and variance along x and y, positions from the box's
 * center.
 */
async function radMoments(sim: Sim | AmrSim, floor: number) {
  let m0 = 0;
  const m1 = [0, 0];
  const m2 = [0, 0];
  const add = (E: number, x: number, y: number, dx: number) => {
    const w = (E - floor) * dx * dx;
    m0 += w;
    m1[0] += w * x;
    m1[1] += w * y;
    m2[0] += w * x * x;
    m2[1] += w * y * y;
  };
  if (sim instanceof AmrSim) {
    const c = await sim.readLeaves();
    const R = await sim.readLeafRadiation();
    for (let q = 0; q < c.count; q++) add(R[4 * q], c.x[q], c.y[q], c.dx[q]);
  } else {
    const R = await sim.rad!.read();
    for (let j = 0; j < sim.ny; j++) for (let i = 0; i < sim.nx; i++) add(R[4 * (j * sim.nx + i)], (i + 0.5) * sim.dx - 0.5 * sim.nx * sim.dx, (j + 0.5) * sim.dx - 0.5 * sim.ny * sim.dx, sim.dx);
  }
  const mean = m1.map((v) => v / m0);
  return { total: m0, mean, variance: m2.map((v, a) => v / m0 - mean[a] ** 2) };
}

/**
 * Run a star (radStar) to each of `times` and record under `tag`: its central density against the
 * start, the drift of the total energy (gas and E / r), and at the end the mass-weighted rms and
 * largest speeds, about its own motion vx, inside 1% of the central density.
 */
async function starRun(sim: Sim | AmrSim, tag: string, vx: number, times: readonly number[], m: Record<string, number>) {
  const d0 = await sim.diagnostics();
  const rho0 = d0.ranges[0][1];
  for (const t of times) {
    await runTo(sim, t);
    const d = await sim.diagnostics();
    m[`${tag}_rhoc_t${t}`] = d.ranges[0][1] / rho0;
    const e0 = d0.eTot + (d0.radiation?.eRadReduced ?? 0);
    m[`${tag}_drift_t${t}`] = (d.eTot + (d.radiation?.eRadReduced ?? 0) - e0) / Math.abs(e0);
    if (d.radiation) {
      m[`${tag}_too_fast_t${t}`] = d.radiation.tooFast;
      m[`${tag}_lum_t${t}`] = d.radiation.luminosity;
    }
  }
  // (On blocks each cell counts by its volume.)
  const leaf = sim instanceof AmrSim ? await sim.readLeaves() : null;
  const u1 = leaf ? leaf.u1 : (await (sim as Sim).readState()).u1;
  let mv2 = 0;
  let mass = 0;
  let vmax = 0;
  for (let q = 0; q < u1.length / 4; q++) {
    const r = u1[4 * q];
    if (r < 0.01 * rho0) continue;
    const w = r * (leaf ? leaf.dx[q] ** 3 : 1);
    const v2 = ((u1[4 * q + 1] - vx * r) ** 2 + u1[4 * q + 2] ** 2 + u1[4 * q + 3] ** 2) / (r * r);
    mv2 += w * v2;
    mass += w;
    vmax = Math.max(vmax, Math.sqrt(v2));
  }
  m[`${tag}_vrms`] = Math.sqrt(mv2 / mass);
  m[`${tag}_vmax`] = vmax;
  m[`${tag}_steps`] = (await sim.diagnostics()).steps;
  // The star's central sound speed, sqrt(5/3 K rho_c^(1/3)) (radStar: radius 0.3, rho_c = 1, G = 1).
  m.central_sound_speed = Math.sqrt((5 / 3) * polytropeK(laneEmdenCached(3), 1, 0.3, 1));
}

/** How many leaves a block mesh has at each level, coarsest first. */
const leavesByLevel = (sim: AmrSim) => {
  const n: number[] = [];
  for (const b of sim.tree.leaves()) n[b.level] = (n[b.level] ?? 0) + 1;
  return Array.from(n, (v) => v ?? 0);
};

/** The radiation tests on blocks (docs/numerics.md section 24). */
const RAD_BLOCK_TESTS: TestDef[] = [
  {
    id: 'rad-blocks-diffusion',
    name: 'Radiation on blocks: a pulse diffuses across a refinement boundary as on the fine uniform grid, and no energy is lost (2D, 128 cells, one level)',
    criterion: 'A Gaussian of radiation centered on the face between coarse and fine blocks (the middle of the box refined, 20 and 10 optical depths a cell, scattering only, chat = c) spreads for 4 time units: the total energy (the radiation is 95% of it) is kept to 1e-4, as the uniform grid keeps it to 1e-5 (the rounding of 2048 sweeps in 32-bit floats), and the centroid (to 0.002 of the box) and the widths along x and y (to 2%) are those of the uniform grid at the fine resolution',
    async run(gpu) {
      const m: Record<string, number> = {};
      const n = 128;
      const p = radGas({
        radC: 1, radChat: 1, opacityParams: { absorption: 0, scattering: (10 * n) / 1e6 }, tEnd: 4,
        // Heavy gas, which the radiation does not move; twice as dense in the middle, which the mesh refines.
        init: () => (x, y): Prim => {
          const rho = Math.abs(x) < 0.2 && Math.abs(y) < 0.2 ? 2e6 : 1e6;
          return { rho, vx: 0, vy: 0, vz: 0, p: 1e-3, s: 0, erad: 0.01 + Math.exp(-(((x - 0.25) ** 2 + y ** 2) / (2 * 0.06 ** 2))), frad: [0, 0, 0] };
        },
      });
      const run = async (tag: string, sim: Sim | AmrSim) => {
        // (The radiation's push does a little work on the gas: the total is what is kept.)
        const total = (x: Awaited<ReturnType<Sim['diagnostics']>>) => x.eTot + x.radiation!.eRadReduced;
        const e0 = total(await sim.diagnostics());
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const r = await radMoments(sim, 0.01);
        m[`${tag}_energy_change`] = total(d) / e0 - 1;
        m[`${tag}_centroid_x`] = r.mean[0];
        m[`${tag}_width_x`] = Math.sqrt(r.variance[0]);
        m[`${tag}_width_y`] = Math.sqrt(r.variance[1]);
        m[`${tag}_steps`] = d.steps;
      };
      await withSim(gpu, p, n, D2, (sim) => run('uniform', sim));
      await withAmr(gpu, p, n, { ...D2, amrLevels: 1, refineHi: 2, refineDense: 1.5e6 }, async (sim) => {
        const by = leavesByLevel(sim);
        m.coarse_leaves = by[0];
        m.fine_leaves = by[1];
        await run('blocks', sim);
      });
      const pass =
        m.coarse_leaves > 0 && m.fine_leaves > 0 && Math.abs(m.blocks_energy_change) < 1e-4 && Math.abs(m.blocks_energy_change - m.uniform_energy_change) < 1e-5 && Math.abs(m.blocks_centroid_x - m.uniform_centroid_x) < 0.002 &&
        Math.abs(m.blocks_width_x / m.uniform_width_x - 1) < 0.02 && Math.abs(m.blocks_width_y / m.uniform_width_y - 1) < 0.02;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-blocks-advect',
    name: 'Radiation on blocks: trapped radiation moves with the gas while the mesh follows it (2D, 256 cells, one level)',
    criterion: 'The pulse of rad-advect on a mesh refined on its gradients, which re-meshes as it moves (at least twice, with coarse and fine blocks throughout): its centroid within 0.002 of the box of where the gas took it, its width kept to 5%, and E_gas + E / r to 1e-5',
    async run(gpu) {
      const m: Record<string, number> = {};
      const v = 1;
      const p = radGas({
        radC: 100, radChat: 1, opacityParams: { absorption: 1e6, scattering: 0 }, tEnd: 0.25,
        init: () => (x) => {
          const T = 1 + 0.3 * Math.exp(-((x / 0.08) ** 2));
          const rho = (2 - T ** 4 / 3) / T;
          return { rho, vx: v, vy: 0, vz: 0, p: rho * T, s: 0 };
        },
      });
      await withAmr(gpu, p, 256, { ...D2, amrLevels: 1, refineHi: 0.1, refineLo: 0.03 }, async (sim) => {
        const total = (d: Awaited<ReturnType<AmrSim['diagnostics']>>) => d.eTot + d.radiation!.eRadReduced;
        const d0 = await sim.diagnostics();
        const r0 = await radMoments(sim, 1);
        const by0 = leavesByLevel(sim);
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const r1 = await radMoments(sim, 1);
        const by = leavesByLevel(sim);
        m.moved = r1.mean[0] - r0.mean[0];
        m.expected = v * 0.25;
        m.width_ratio = Math.sqrt(r1.variance[0] / r0.variance[0]);
        m.energy_drift = total(d) / total(d0) - 1;
        m.regrids = sim.regrids;
        m.coarse_leaves = Math.min(by0[0], by[0]);
        m.fine_leaves = Math.min(by0[1], by[1]);
        m.steps = d.steps;
      });
      const pass = Math.abs(m.moved - m.expected) < 0.002 && Math.abs(m.width_ratio - 1) < 0.05 && Math.abs(m.energy_drift) < 1e-5 && m.regrids >= 2 && m.coarse_leaves > 0 && m.fine_leaves > 0;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-blocks-cooling-sphere',
    name: 'Radiation on blocks: the hot core behind its shell shines as on the fine uniform grid (2D, 128 cells, one level)',
    criterion: 'The sphere of rad-cooling-sphere with only the blocks that hold the shell refined: after 1.5 time units its luminosity through the box faces (which coarse blocks hold) is within 1% of the uniform fine grid’s, steady to 1% over 20 steps, and the same to 0.1% as the energy counted out of the box over them gives',
    async run(gpu) {
      const m: Record<string, number> = {};
      const n = 128;
      const p = radGas({
        bcLo: all(OUTFLOW), bcHi: all(OUTFLOW), radChat: 5, opacityParams: { absorption: 0, scattering: 5 * n * 1e-4 }, tEnd: 1.5,
        init: () => (x, y): Prim => {
          const r = Math.hypot(x, y);
          const rho = r < 0.2 ? 1e4 : 1e-4;
          return { rho, vx: 0, vy: 0, vz: 0, p: rho * 1e-6, s: 0, fixed: r < 0.1, erad: r < 0.1 ? 1 : 1e-8, frad: [0, 0, 0] };
        },
      });
      const run = async (tag: string, sim: Sim | AmrSim) => {
        await sim.runToEnd();
        sim.cfg.tEnd = 100;
        sim.writeParams();
        let sum = 0;
        let lo = Infinity;
        let hi = 0;
        const before = await sim.diagnostics();
        for (let k = 0; k < 20; k++) {
          sim.step(1);
          const l = (await sim.diagnostics()).radiation!.luminosity;
          sum += l / 20;
          lo = Math.min(lo, l);
          hi = Math.max(hi, l);
        }
        const after = await sim.diagnostics();
        m[`${tag}_counted_luminosity`] = (after.radiation!.escaped - before.radiation!.escaped) / (after.t - before.t);
        m[`${tag}_luminosity`] = sum;
        m[`${tag}_swing`] = (hi - lo) / (hi + lo);
      };
      await withSim(gpu, p, n, D2, (sim) => run('uniform', sim));
      await withAmr(gpu, p, n, { ...D2, amrLevels: 1, refineHi: 2, refineDense: 1 }, async (sim) => {
        const by = leavesByLevel(sim);
        m.coarse_leaves = by[0];
        m.fine_leaves = by[1];
        await run('blocks', sim);
      });
      m.ratio = m.blocks_luminosity / m.uniform_luminosity;
      return { pass: m.coarse_leaves > 0 && m.fine_leaves > 0 && Math.abs(m.ratio - 1) < 0.01 && m.blocks_swing < 0.01 && Math.abs(m.blocks_counted_luminosity / m.blocks_luminosity - 1) < 1e-3, metrics: m };
    },
  },
  {
    id: 'rad-blocks-star',
    name: 'Radiation on blocks: an n = 3 polytrope with a tenth of its pressure in radiation stays at rest (octant, 128 cells across at the finest of two levels)',
    criterion: 'The star of rad-star on a mesh of three levels. At rest for one dynamical time: its central density within 5% of the start, its mass-weighted rms speed inside 1% of the central density under 3% of the central sound speed and the largest under 10%, total energy (gas and E / r) kept to 2%. Moving at ten times its sound speed for half the box, the mesh following it (at least two re-meshes), against the same star with its radiation in the equation of state: central density within 2% of the baseline’s, rms speed about its own motion under 1.5 times the baseline’s, energy kept to 1%',
    async run(gpu) {
      const m: Record<string, number> = {};
      for (const [tag, transport, vx] of [['eos', false, 0], ['transport', true, 0], ['moving_eos', false, 6], ['moving_transport', true, 6]] as const) {
        await withAmr(gpu, radStar(transport, 1e6, 50, vx), 128, { amrLevels: 2, symmetry: [!vx, true, true], radSweeps: 16, tEnd: 0 }, async (sim) => {
          const by = leavesByLevel(sim);
          by.forEach((n, l) => (m[`${tag}_leaves_level${l}`] = n));
          await starRun(sim, tag, vx, vx ? [0.025, 0.05] : [0.25, 0.5, 1], m);
          m[`${tag}_regrids`] = sim.regrids;
          m[`${tag}_settle_ms`] = sim.settleMs;
          m[`${tag}_settle_rounds`] = sim.settleLog.length;
        });
      }
      const cs = m.central_sound_speed;
      const pass =
        Math.abs(m.transport_rhoc_t1 - 1) < 0.05 && m.transport_vrms < 0.03 * cs && m.transport_vmax < 0.1 * cs && Math.abs(m.transport_drift_t1) < 0.02 &&
        Math.abs(m['moving_transport_rhoc_t0.05'] - m['moving_eos_rhoc_t0.05']) < 0.02 && m.moving_transport_vrms < 1.5 * m.moving_eos_vrms &&
        Math.abs(m['moving_transport_drift_t0.05']) < 0.01 && m.moving_transport_regrids >= 2;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-blocks-restart',
    name: 'Radiation on blocks: a snapshot restarts the run exactly',
    criterion: 'The run of rad-blocks-advect part way, saved to a snapshot and restarted from it, has the same mesh and the same radiation in every cell bit for bit, and after ten more steps the two agree to 1e-5 in E',
    async run(gpu) {
      const m: Record<string, number> = {};
      const p = radGas({
        radC: 100, radChat: 1, opacityParams: { absorption: 1e6, scattering: 0 },
        init: () => (x) => {
          const T = 1 + 0.3 * Math.exp(-((x / 0.08) ** 2));
          const rho = (2 - T ** 4 / 3) / T;
          return { rho, vx: 1, vy: 0, vz: 0, p: rho * T, s: 0 };
        },
      });
      const over = { ...D2, amrLevels: 1, refineHi: 0.1, refineLo: 0.03 };
      // Each cell's E by its position: leaves may be stored in another order.
      const byCell = async (sim: AmrSim) => {
        const c = await sim.readLeaves();
        const R = await sim.readLeafRadiation();
        const at = new Map<string, number>();
        for (let q = 0; q < c.count; q++) at.set(`${c.x[q]},${c.y[q]}`, R[4 * q]);
        return at;
      };
      const compare = (a: Map<string, number>, b: Map<string, number>) => {
        let same = 0;
        let worst = 0;
        for (const [k, v] of a) {
          const w = b.get(k);
          if (w === v) same++;
          worst = Math.max(worst, w === undefined ? Infinity : Math.abs(w - v) / Math.abs(v));
        }
        return { share: same / a.size, worst };
      };
      await withAmr(gpu, p, 256, { ...over, tEnd: 0.05 }, async (sim) => {
        await sim.runToEnd();
        sim.flush();
        const snap = await capture(sim);
        const again = await AmrSim.create(gpu, { ...configFor(p, 256, { ...over, tEnd: 0.05 }), ...restartFrom(snap, p, true, 128) });
        try {
          const key = (s: AmrSim) => s.tree.leaves().map((l) => `${l.level}:${l.c.join(',')}`).sort().join(' ');
          m.same_mesh = key(sim) === key(again) ? 1 : 0;
          m.identical_share = compare(await byCell(sim), await byCell(again)).share;
          await sim.advance(10);
          await again.advance(10);
          m.same_mesh_later = key(sim) === key(again) ? 1 : 0;
          m.radiation_difference = compare(await byCell(sim), await byCell(again)).worst;
        } finally {
          again.destroy();
        }
      });
      return { pass: m.same_mesh === 1 && m.identical_share === 1 && m.radiation_difference < 1e-5, metrics: m };
    },
  },
];

export const RAD_TESTS: TestDef[] = [
  {
    id: 'rad-restart',
    name: 'Radiation: a snapshot restarts the run exactly',
    criterion: 'A run with radiation (the advected pulse, part way) saved to a snapshot and restarted from it holds the same radiation bit for bit, and after ten more steps the two runs agree to 1e-5 in E and in the gas’s energy',
    async run(gpu) {
      const m: Record<string, number> = {};
      const p = radGas({
        radC: 100, radChat: 1, opacityParams: { absorption: 1e6, scattering: 0 },
        init: () => (x) => {
          const T = 1 + 0.3 * Math.exp(-((x / 0.08) ** 2));
          const rho = (2 - T ** 4 / 3) / T;
          return { rho, vx: 1, vy: 0, vz: 0, p: rho * T, s: 0 };
        },
      });
      await withSim(gpu, p, 64, D2, async (sim) => {
        await runTo(sim, 0.05);
        const snap = await capture(sim);
        const R0 = await sim.rad!.read();
        const r = restartFrom(snap, sim.cfg.problem, false, sim.n);
        const again = await SimClass.create(gpu, configFor(r.problem, 64, { ...D2, restart: r.restart }));
        try {
          const R1 = await again.rad!.read();
          let same = 0;
          for (let i = 0; i < R0.length; i++) if (R0[i] === R1[i]) same++;
          m.identical_share = same / R0.length;
          await runTo(sim, 0.08);
          await runTo(again, 0.08);
          const a = await cells(sim);
          const b = await cells(again);
          let dE = 0;
          let dG = 0;
          for (let q = 0; q < a.n; q++) {
            dE = Math.max(dE, Math.abs(a.R[4 * q] - b.R[4 * q]) / Math.abs(a.R[4 * q]));
            dG = Math.max(dG, Math.abs(a.u2[4 * q] - b.u2[4 * q]) / Math.abs(a.u2[4 * q]));
          }
          m.radiation_difference = dE;
          m.gas_energy_difference = dG;
        } finally {
          again.destroy();
        }
      });
      return { pass: m.identical_share === 1 && m.radiation_difference < 1e-5 && m.gas_energy_difference < 1e-5, metrics: m };
    },
  },
  {
    id: 'rad-off',
    name: 'Radiation: off adds nothing to a run',
    criterion: 'With radiation off a run has no radiation passes, its parameter block has P.rad = 0, and its equation of state keeps its radiation constant; on, the constant moves to the radiation and the equation of state has none',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const gr = radGas({ eos: 'gas_radiation', eosParams: { gamma: 5 / 3, arad: 0.5 } });
      await withSim(gpu, { ...gr, radiation: 'off' }, 32, D2, async (sim) => {
        await runTo(sim, 0.01);
        m.off_passes = sim.rad ? 1 : 0;
        m.off_eos_arad = sim.cfg.eosParams.arad;
        pass &&= !sim.rad && sim.cfg.eosParams.arad === 0.5 && !(await sim.diagnostics()).radiation;
      });
      await withSim(gpu, gr, 32, D2, async (sim) => {
        m.on_eos_arad = sim.cfg.eosParams.arad;
        m.on_rad_a = sim.cfg.radA;
        pass &&= !!sim.rad && sim.cfg.eosParams.arad === 0 && sim.cfg.radA === 0.5;
        // The pressure is split between gas and radiation at the full equation of state's T.
        const c = await cells(sim);
        const T = (c.u2[1] * (2 / 3)) / c.u1[0];
        m.split_error = Math.abs(T * c.u1[0] + (0.5 * T ** 4) / 3 - 1);
        m.erad_error = Math.abs(c.R[0] - 0.5 * T ** 4);
        pass &&= m.split_error < 1e-5 && m.erad_error < 1e-5;
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-equilibrium',
    name: 'Radiation: gas and radiation relax to their equilibrium in a closed box',
    criterion: 'With r = 1 (chat = c) the temperature reaches the root of e(T) + a T^4 = e0 + E0 to 1e-4 and E_gas + E is kept to 1e-5; with one reduced speed chat = c/10 everywhere it misses by over 5% (the known error); with the per-cell rule in an opaque box it is right to 1e-4',
    async run(gpu) {
      const m: Record<string, number> = {};
      const a = 1;
      const exact = equilibriumT(a, 1);
      m.exact_T = exact;
      const go = async (tag: string, chat: number, local: boolean, kappa: number) => {
        const p = radGas({
          eosParams: { gamma: 5 / 3, arad: a }, radChat: chat, opacityParams: { absorption: kappa, scattering: 0 },
          init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 2 / 3, s: 0, erad: 0, frad: [0, 0, 0] }),
        });
        await withSim(gpu, p, 16, { ...D2, radLocalC: local }, async (sim) => {
          const d0 = await sim.diagnostics();
          const e0 = d0.eTot + d0.radiation!.eRadReduced;
          await runTo(sim, 0.2);
          const d1 = await sim.diagnostics();
          const c = await cells(sim);
          m[`${tag}_T`] = (c.u2[1] * (2 / 3)) / c.u1[0];
          m[`${tag}_drift`] = Math.abs(d1.eTot + d1.radiation!.eRadReduced - e0) / e0;
          m[`${tag}_steps`] = d1.steps;
        });
      };
      await go('true_c', 100, true, 10);
      await go('uniform', 10, false, 10);
      await go('local', 10, true, 300);
      const err = (t: string) => Math.abs(m[`${t}_T`] - exact) / exact;
      m.true_c_error = err('true_c');
      m.uniform_error = err('uniform');
      m.local_error = err('local');
      const pass = m.true_c_error < 1e-4 && m.true_c_drift < 1e-5 && m.uniform_error > 0.05 && m.local_error < 1e-4 && m.local_drift < 1e-5;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-f32',
    name: 'Radiation: equilibrium at tau_c = 1e10 holds in float32',
    criterion: 'Gas and radiation started in equilibrium (E = a T^4, T varying by 10% across the box, and uniform) at a cell optical depth of 1e10 keep E / a T^4 within 1e-5 of 1 over 10^4 sweeps; their total energy is kept to 1e-6 at one temperature and to 1e-4 where the gas moves (1e-8 a step, under one float32 rounding)',
    async run(gpu) {
      const m: Record<string, number> = {};
      const n = 16;
      const kappa = 1e10 * n;
      const p = radGas({
        opacityParams: { absorption: kappa, scattering: 0 },
        init: () => (x) => {
          const T = 1 + 0.1 * Math.sin(2 * Math.PI * x);
          // Pressure balance with the radiation: rho T + a T^4 / 3 = 2.
          const rho = (2 - T ** 4 / 3) / T;
          return { rho, vx: 0, vy: 0, vz: 0, p: rho * T, s: 0, erad: T ** 4 };
        },
      });
      await withSim(gpu, p, n, { ...D2, radSweeps: 16 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const e0 = d0.eTot + d0.radiation!.eRadReduced;
        // 10^4 sweeps: 625 steps of 16, each at the sweeps' longest, 16 (0.5 dx / 2 c).
        await runTo(sim, 625 * 16 * (0.5 / n / 200) * 1.0001);
        const d1 = await sim.diagnostics();
        const c = await cells(sim);
        let worst = 0;
        for (let q = 0; q < c.n; q++) {
          const T = (c.u2[4 * q + 1] * (2 / 3)) / c.u1[4 * q];
          worst = Math.max(worst, Math.abs(c.R[4 * q] / T ** 4 - 1));
        }
        m.sweeps = d1.steps * 16;
        m.max_departure = worst;
        m.energy_drift = Math.abs(d1.eTot + d1.radiation!.eRadReduced - e0) / e0;
      });
      // The same at one temperature, where nothing moves.
      const still = radGas({ opacityParams: { absorption: kappa, scattering: 0 }, init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0, erad: 1 }) });
      await withSim(gpu, still, n, { ...D2, radSweeps: 16 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const e0 = d0.eTot + d0.radiation!.eRadReduced;
        await runTo(sim, 625 * 16 * (0.5 / n / 200) * 1.0001);
        const d1 = await sim.diagnostics();
        m.uniform_energy_drift = Math.abs(d1.eTot + d1.radiation!.eRadReduced - e0) / e0;
      });
      // In moving gas the total is kept to the rounding of adding each step's exchange to the
      // gas's energy: about 1e-8 of it a step, under one float32 rounding (it did not grow with
      // the sweeps: 1e-5 to 4e-5 at 1, 4 and 16 a step). Nothing moving, nothing is lost at all.
      const pass = m.max_departure < 1e-5 && m.energy_drift < 1e-4 && m.uniform_energy_drift < 1e-6 && m.sweeps >= 1e4;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-streaming',
    name: 'Radiation: a pulse streams through vacuum at chat',
    criterion: 'A pulse with F = c E in transparent gas moves at chat (to 2% of the distance), keeps its energy to 1e-5 in a periodic box and its reduced flux above 0.95 at its peak; with chat = c it moves at c',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const [tag, chat] of [['reduced', 10], ['true_c', 100]] as const) {
        const floor = 1e-8;
        const p = radGas({
          radChat: chat, opacityParams: { absorption: 0, scattering: 0 },
          init: () => (x) => {
            const e = Math.exp(-(((x + 0.3) / 0.05) ** 2));
            return { rho: 1, vx: 0, vy: 0, vz: 0, p: 1e-4, s: 0, erad: floor + e, frad: [100 * e, 0, 0] };
          },
        });
        await withSim(gpu, p, 128, D2, async (sim) => {
          const r0 = moments((await cells(sim)).R, sim.nx, sim.ny, sim.dx, floor);
          const t = 0.4 / chat;
          await runTo(sim, t);
          const c = await cells(sim);
          const r1 = moments(c.R, sim.nx, sim.ny, sim.dx, floor);
          m[`${tag}_distance`] = (r1.mean - r0.mean) / (chat * t);
          m[`${tag}_energy`] = Math.abs(r1.total - r0.total) / r0.total;
          let best = 0;
          for (let i = 0; i < sim.nx; i++) if (c.R[4 * i] > c.R[4 * best]) best = i;
          m[`${tag}_peak_f`] = c.R[4 * best + 1] / (100 * c.R[4 * best]);
          pass &&= Math.abs(m[`${tag}_distance`] - 1) < 0.02 && m[`${tag}_energy`] < 1e-5 && m[`${tag}_peak_f`] > 0.95;
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-diffusion',
    name: 'Radiation: a pulse diffuses at c / (3 kappa rho) from tau_c = 1 to 1e4',
    criterion: 'The variance of a Gaussian pulse in scattering gas grows as 2 D t, D = c / (3 kappa rho), to 10%, at cell optical depths 1, 100 and 1e4, with the per-cell rule (chat chosen for about 1000 sweeps, r = 1 in the opaque cases)',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const n = 64;
      const dx = 1 / n;
      for (const tau of [1, 100, 1e4]) {
        // Dense enough that the radiation's force does not move the gas over the run (at 1e6 it
        // carried the pulse apart at tau 1e4, over its 1e4 time units).
        const rho = 1e12;
        const kappa = tau / (rho * dx);
        const c = 1;
        const D = c / (3 * kappa * rho);
        const s0 = 4 * dx;
        const t = (1.5 * s0 * s0) / D;
        const chat = Math.min(c, (0.5 * dx * 1000) / (2 * t));
        const floor = 1e-6;
        const p = radGas({
          radC: c, radChat: chat, opacityParams: { absorption: 0, scattering: kappa },
          init: () => (x) => ({ rho, vx: 0, vy: 0, vz: 0, p: 1e-12, s: 0, erad: floor + Math.exp(-(x * x) / (2 * s0 * s0)) }),
        });
        await withSim(gpu, p, n, { ...D2, radSweeps: 16 }, async (sim) => {
          const v0 = moments((await cells(sim)).R, sim.nx, sim.ny, dx, floor).variance;
          await runTo(sim, t);
          const r = moments((await cells(sim)).R, sim.nx, sim.ny, dx, floor);
          m[`tau${tau}_ratio`] = (r.variance - v0) / (2 * D * t);
          m[`tau${tau}_chat`] = chat;
          m[`tau${tau}_steps`] = (await sim.diagnostics()).steps;
          pass &&= Math.abs(m[`tau${tau}_ratio`] - 1) < 0.1;
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-shadow',
    name: 'Radiation: an opaque clump casts a shadow (2D)',
    criterion: 'A beam entering through fixed cells at the left edge crosses transparent gas; behind an opaque scattering disc (optical depth about 1000) the radiation energy is under 10% of that beside it, where the beam keeps E > 0.85 and f > 0.9 (the clump scatters some light back)',
    async run(gpu) {
      const m: Record<string, number> = {};
      // In pressure balance, and scattering: an absorbing clump heats at its surface and blows off
      // gas into the thin surroundings faster than light, which the scheme (moments in the gas's
      // frame) is not for; the diagnostics count such cells.
      const p = radGas({
        bcLo: all(OUTFLOW), bcHi: all(OUTFLOW), radChat: 10, opacity: 'constant', opacityParams: { absorption: 0, scattering: 1 },
        init: () => (x, y): Prim => {
          const clump = Math.hypot(x, y) < 0.1;
          const beam = x < -0.48;
          const rho = clump ? 1e4 : 1e-6;
          return { rho, vx: 0, vy: 0, vz: 0, p: 1e-9, s: 0, fixed: beam, erad: beam ? 1 : 1e-10, frad: beam ? [100, 0, 0] : [0, 0, 0] };
        },
      });
      await withSim(gpu, p, 128, D2, async (sim) => {
        await runTo(sim, 0.15);
        const c = await cells(sim);
        const at = (x: number, y: number) => 4 * (Math.floor((y + 0.5) * sim.ny) * sim.nx + Math.floor((x + 0.5) * sim.nx));
        m.behind = c.R[at(0.3, 0)];
        m.beside = c.R[at(0.3, 0.3)];
        m.beside_f = c.R[at(0.3, 0.3) + 1] / (100 * m.beside);
        m.ratio = m.behind / m.beside;
        m.too_fast = (await sim.diagnostics()).radiation!.tooFast;
      });
      return { pass: m.ratio < 0.1 && m.beside > 0.85 && m.beside_f > 0.9, metrics: m };
    },
  },
  {
    id: 'rad-advect',
    name: 'Radiation: trapped radiation moves with the gas (beta = 1e-2, beta tau = 1e4, chat = c / 100)',
    criterion: 'A pulse of radiation trapped in gas moving at 1% of c, the box of optical depth 1e6, keeps with the gas (its centroid within 1% of the box of where the gas took it) and its width to 10%, at chat = c / 100 (r = 1 in the gas by the per-cell rule)',
    async run(gpu) {
      const m: Record<string, number> = {};
      const c = 100;
      const v = 1;
      const kappa = 1e6;
      const p = radGas({
        radC: c, radChat: 1, opacityParams: { absorption: kappa, scattering: 0 },
        init: () => (x) => {
          const T = 1 + 0.3 * Math.exp(-((x / 0.08) ** 2));
          // Total pressure balance: rho T + a T^4 / 3 = 2.
          const rho = (2 - T ** 4 / 3) / T;
          return { rho, vx: v, vy: 0, vz: 0, p: rho * T, s: 0 };
        },
      });
      await withSim(gpu, p, 128, D2, async (sim) => {
        const r0 = moments((await cells(sim)).R, sim.nx, sim.ny, sim.dx, 1);
        const t = 0.25;
        await runTo(sim, t);
        const r1 = moments((await cells(sim)).R, sim.nx, sim.ny, sim.dx, 1);
        m.moved = r1.mean - r0.mean;
        m.expected = v * t;
        m.width_ratio = Math.sqrt(r1.variance / r0.variance);
        m.steps = (await sim.diagnostics()).steps;
      });
      return { pass: Math.abs(m.moved - m.expected) < 0.01 && Math.abs(m.width_ratio - 1) < 0.1, metrics: m };
    },
  },
  {
    id: 'rad-marshak',
    name: 'Radiation: Su & Olson non-equilibrium Marshak wave (2D, 128 cells across 16 mean free paths)',
    criterion: 'Radiation falls in through a held column (Marshak boundary) on a cold material of heat capacity 4 a T^3. The GPU matches the float64 reference of the same scheme to 2e-3 in E and a T^4 at tau = 1, 3.16 and 10; and both match the float64 non-equilibrium diffusion solution with the Marshak condition to 0.03 at tau = 3.16 (x <= 2) and 0.015 at tau = 10 (x <= 5), in units of the incident energy',
    async run(gpu) {
      const m: Record<string, number> = {};
      const n = 128;
      const L = 16;
      const dx = L / n;
      const R = 1e-12;
      const rho = 1e6;
      const eint = (r: number, T: number) => 1.5 * R * r * T + T ** 4;
      const temperature = (r: number, e: number) => {
        let lo = 0;
        let hi = Math.max(1, e) * 10;
        for (let i = 0; i < 100; i++) {
          const mid = 0.5 * (lo + hi);
          if (eint(r, mid) > e) hi = mid;
          else lo = mid;
        }
        return lo;
      };
      // Gas and radiation equation of state with its a T^4 kept: the material's own energy.
      const p = radGas({
        eos: 'gas_radiation', eosParams: { gamma: 5 / 3, R, arad: 1 }, radKeepEos: true, radC: 1, radChat: 1,
        box: L, bcLo: [OUTFLOW, PERIODIC, PERIODIC], bcHi: [OUTFLOW, PERIODIC, PERIODIC],
        opacityParams: { absorption: 1 / rho, scattering: 0 },
        init: () => (x): Prim => {
          const held = x < -L / 2 + dx;
          const T = held ? 1 : 1e-3;
          return { rho, vx: 0, vy: 0, vz: 0, p: R * rho * T + T ** 4 / 3, s: 0, fixed: held, erad: held ? 0.5 : 1e-12, frad: held ? [0.25, 0, 0] : [0, 0, 0] };
        },
      });
      const ref = new RefRad1D({ n, dx, c: 1, chat: 1, a: 1, gamma: 5 / 3, bc: 'outflow', localC: true, dims: 2, holdFirst: true, eos: { eintFromT: eint, temperature }, opacity: () => ({ planck: 1 / rho, rosseland: 1 / rho }) });
      ref.rho.fill(rho);
      for (let i = 0; i < n; i++) {
        ref.eint[i] = eint(rho, i === 0 ? 1 : 1e-3);
        ref.E[i] = i === 0 ? 0.5 : 1e-12;
      }
      ref.F[0] = 0.25;
      let pass = true;
      await withSim(gpu, p, n, D2, async (sim) => {
        const sweeps = sim.cfg.radSweeps;
        let t = 0;
        for (const T of [1, 3.1623, 10]) {
          await runTo(sim, T);
          // The reference takes the GPU's steps: whole steps of sweeps at the sweep's longest, then
          // one shorter step to land on T.
          const step = sweeps * ref.sweepDt;
          while (T - t > step * (1 + 1e-6)) {
            for (let q = 0; q < sweeps; q++) ref.sweep(ref.sweepDt);
            t += step;
          }
          for (let q = 0; q < sweeps; q++) ref.sweep((T - t) / sweeps);
          t = T;
          const c = await cells(sim);
          const diff = marshakDiffusion({ c: 1, k: 1, u0: 1, length: 30, cells: 1500, t: T, steps: 2000, marshak: true });
          let dRef = 0;
          let dDiff = 0;
          const xmax = T > 5 ? 5 : 2;
          for (let i = 1; i < n; i++) {
            const u = c.R[4 * i];
            const v = temperature(rho, c.u2[4 * i + 1]) ** 4;
            dRef = Math.max(dRef, Math.abs(u - ref.E[i]), Math.abs(v - ref.temperature(i) ** 4));
            const x = (i - 0.5) * dx;
            // (Within half a mean free path of the boundary M1 and diffusion part ways, as they should.)
            if (x >= 0.5 && x <= xmax) dDiff = Math.max(dDiff, Math.abs(u - interp(diff.x, diff.u, x)), Math.abs(v - interp(diff.x, diff.v, x)));
          }
          const tag = `tau${T > 5 ? 10 : T > 2 ? 3 : 1}`;
          m[`${tag}_vs_reference`] = dRef;
          pass &&= dRef < 2e-3;
          if (T > 2) {
            m[`${tag}_vs_diffusion`] = dDiff;
            pass &&= dDiff < 0.03;
          }
          m[`${tag}_u_at_1`] = c.R[4 * Math.round(1 / dx + 0.5)];
        }
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-star',
    name: 'Radiation: an n = 3 polytrope with a tenth of its pressure in radiation stays at rest (64^3, octant)',
    criterion: 'Against the same star with its radiation in the equation of state: at rest for one dynamical time, its central density within 10% of the start and its mass-weighted rms and largest speeds inside 1% of the central density under twice the baseline’s, total energy (gas and E / r) kept to 3%; moving at ten times its sound speed for half the box, its central density within 2% of the baseline’s, its rms speed about its own motion under 1.5 times the baseline’s, energy kept to 1%',
    async run(gpu) {
      const m: Record<string, number> = {};
      for (const [tag, transport, vx] of [['eos', false, 0], ['transport', true, 0], ['moving_eos', false, 6], ['moving_transport', true, 6]] as const) {
        // Moving: along x at about ten times the central sound speed, for half the box.
        const times = vx ? [0.025, 0.05] : [0.25, 0.5, 1];
        await withSim(gpu, radStar(transport, 1e6, 50, vx), 64, { symmetry: [!vx, true, true], radSweeps: 16, tEnd: 0 }, async (sim) => {
          await starRun(sim, tag, vx, times, m);
          m[`${tag}_settle_ms`] = sim.settleMs;
        });
      }
      const pass =
        Math.abs(m.transport_rhoc_t1 - 1) < 0.1 && m.transport_vrms < 2 * m.eos_vrms && m.transport_vmax < 2 * m.eos_vmax && Math.abs(m.transport_drift_t1) < 0.03 &&
        Math.abs(m['moving_transport_rhoc_t0.05'] - m['moving_eos_rhoc_t0.05']) < 0.02 && m.moving_transport_vrms < 1.5 * m.moving_eos_vrms && Math.abs(m['moving_transport_drift_t0.05']) < 0.01;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-shock',
    name: 'Radiation: radiative shocks against a wall, subcritical and supercritical (2D, 256 cells)',
    criterion: 'Gas at T = 1 driven into a wall at 1.5 (subcritical) and 5 (supercritical), against the jump with gas and radiation in equilibrium on both sides (Rankine-Hugoniot with radiation pressure and energy). Subcritical, chat = c/5: density and temperature behind to 1%, the shock where its speed puts it to 1%, a temperature spike at the front, the gas ahead heated by under 5%. Supercritical at the true c: the same to 1%, and the gas two cells ahead heated to over 3 times its temperature by the precursor. Supercritical at chat = c/5 and c/2.5: within 5% and 2%, the error shrinking as chat grows (a moving structure in cells whose speed of light is reduced: docs/numerics.md section 24)',
    async run(gpu) {
      const m: Record<string, number> = {};
      const n = 256;
      const dx = 1 / n;
      const a = 1.2e-4;
      for (const [tag, u0, chat, reach] of [['sub', 1.5, 200, 0.4], ['super', 5, 200, 0.4], ['super_400', 5, 400, 0.4], ['super_c', 5, 1000, 0.4]] as const) {
        const jump = radiativeJump(1, 1, u0, a);
        m[`${tag}_rho2_exact`] = jump.rho2;
        m[`${tag}_T2_exact`] = jump.T2;
        m[`${tag}_D_exact`] = jump.D;
        const p = radGas({
          eosParams: { gamma: 5 / 3, R: 1, arad: a }, radC: 1000, radChat: chat, opacityParams: { absorption: 240, scattering: 0 },
          bcLo: [REFLECT, PERIODIC, PERIODIC], bcHi: [OUTFLOW, PERIODIC, PERIODIC],
          init: () => () => ({ rho: 1, vx: -u0, vy: 0, vz: 0, p: 1, s: 0 }),
        });
        const t = reach / jump.D;
        await withSim(gpu, p, n, D2, async (sim) => {
          await runTo(sim, t);
          const c = await cells(sim);
          const row = (i: number) => 4 * (Math.floor(n / 2) * n + i);
          const rho = (i: number) => c.u1[row(i)];
          const T = (i: number) => (c.u2[row(i) + 1] * (2 / 3)) / rho(i);
          // The shock: where the density first falls through the mean of the two sides.
          let s = 0;
          while (s < n - 1 && rho(s) > 0.5 * (1 + jump.rho2)) s++;
          m[`${tag}_shock_position`] = (s * dx) / (jump.D * t);
          let rs = 0;
          let ts = 0;
          let k = 0;
          // The gas behind the shock, away from the wall (shocked first) and the front.
          for (let i = Math.round((reach > 0.5 ? 0.6 : 0.3) * s); i < Math.round(0.85 * s); i++, k++) {
            rs += rho(i);
            ts += T(i);
          }
          m[`${tag}_rho2`] = rs / k / jump.rho2;
          m[`${tag}_T2`] = ts / k / jump.T2;
          // The gas two cells ahead of the front, over the upstream temperature: the precursor.
          m[`${tag}_preheat`] = T(Math.min(n - 1, s + 2));
          let tmax = 0;
          for (let i = 0; i < n; i++) tmax = Math.max(tmax, T(i));
          m[`${tag}_Tmax`] = tmax / jump.T2;
          m[`${tag}_steps`] = (await sim.diagnostics()).steps;
          m[`${tag}_too_fast`] = (await sim.diagnostics()).radiation!.tooFast;
        });
      }
      const err = (t: string) => Math.max(Math.abs(m[`${t}_rho2`] - 1), Math.abs(m[`${t}_T2`] - 1), Math.abs(m[`${t}_shock_position`] - 1));
      m.sub_error = err('sub');
      m.super_c_error = err('super_c');
      m.super_error = err('super');
      m.super_400_error = err('super_400');
      const pass =
        m.sub_error < 0.01 && m.sub_Tmax > 1 && m.sub_preheat < 1.05 &&
        m.super_c_error < 0.01 && m.super_c_preheat > 3 &&
        m.super_error < 0.05 && m.super_400_error < 0.02 && m.super_400_error < m.super_error;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rad-cooling-sphere',
    name: 'Radiation: a hot core shining through an opaque shell gives the same luminosity at chat and 2 chat',
    criterion: 'A core held at E = 1 inside a scattering shell (cell optical depth 5, so its speed of light is reduced by the per-cell rule) in transparent gas reaches a steady luminosity through the box faces (over 30 steps it stays within 2% of its mean, and the energy counted out of the box step by step over them gives the same luminosity to 0.1%) that differs by under 5% between chat = c/20 and c/10; the shell is reduced (r < 1) in both',
    async run(gpu) {
      const m: Record<string, number> = {};
      const lum: number[] = [];
      let steady = true;
      let counted = true;
      const n = 64;
      for (const chat of [5, 10]) {
        const p = radGas({
          bcLo: all(OUTFLOW), bcHi: all(OUTFLOW), radChat: chat, opacityParams: { absorption: 0, scattering: 5 * n * 1e-4 },
          init: () => (x, y): Prim => {
            const r = Math.hypot(x, y);
            const core = r < 0.1;
            const shell = r < 0.2;
            // Dense and cold: the radiation's force does not move it over the run.
            const rho = shell ? 1e4 : 1e-4;
            return { rho, vx: 0, vy: 0, vz: 0, p: rho * 1e-6, s: 0, fixed: core, erad: core ? 1 : 1e-8, frad: [0, 0, 0] };
          },
        });
        // The shell's diffusion time is 0.1, and its reduced cells (r about 0.3 at chat = c/20)
        // settle r times slower: thirty of them.
        await withSim(gpu, p, n, D2, async (sim) => {
          // Steady over 30 steps too: with a steeper slope on E the shell's surface sent its
          // light out in pulses, 12% up and down at the faces (radiation/common.wgsl, rad_slope).
          await runTo(sim, 3);
          sim.cfg.tEnd = 100;
          sim.writeParams();
          let sum = 0;
          let lo = Infinity;
          let hi = 0;
          const before = await sim.diagnostics();
          for (let k = 0; k < 30; k++) {
            sim.step(1);
            const l = (await sim.diagnostics()).radiation!.luminosity;
            sum += l;
            lo = Math.min(lo, l);
            hi = Math.max(hi, l);
          }
          // The light curve's luminosity: the energy the GPU counted out of the box over the time.
          const after = await sim.diagnostics();
          m[`chat${chat}_counted_luminosity`] = (after.radiation!.escaped - before.radiation!.escaped) / (after.t - before.t);
          counted &&= Math.abs(m[`chat${chat}_counted_luminosity`] / (sum / 30) - 1) < 1e-3;
          m[`chat${chat}_swing`] = (hi - lo) / (hi + lo);
          steady &&= (hi - lo) / (hi + lo) < 0.02;
          const d = await sim.diagnostics();
          lum.push(sum / 30);
          m[`chat${chat}_luminosity`] = sum / 30;
          m[`chat${chat}_reduced_share`] = d.radiation!.reducedShare;
          m[`chat${chat}_steps`] = d.steps;
        });
      }
      m.ratio = lum[1] / lum[0];
      return { pass: steady && counted && Math.abs(m.ratio - 1) < 0.05 && lum[0] > 0 && m.chat5_reduced_share > 0 && m.chat10_reduced_share > 0, metrics: m };
    },
  },
  ...RAD_BLOCK_TESTS,
];

