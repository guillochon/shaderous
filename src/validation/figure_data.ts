// Data for the paper's figures that the scripting API (docs/api.md) cannot reach: the validation
// suite's own setups, run as its tests run them, and readbacks of the simulator's internals.
// paper/figures/collect.mjs imports this module into the headless page and writes what each
// function returns as JSON; each figure script names the functions it uses.
import { initGpu, type Gpu } from '../gpu/device';
import { configFor, Sim, type LimiterKind } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { column, withAmr, withSim } from './tests';
import { exactRiemann } from '../reference/riemann_exact';
import { sodX } from '../problems/shocktube';
import { advection } from '../problems/advection';
import { sedov } from '../problems/sedov';
import { sphere, spherePotential } from '../problems/sphere';
import { whiteDwarfStar } from '../problems/white_dwarf_star';
import type { Problem } from '../problems/types';

let gpuPromise: Promise<Gpu> | null = null;
const gpu = () => (gpuPromise ??= initGpu());

/** Run a simulation to time t (it may already be there). */
async function runTo(sim: Sim | AmrSim, t: number) {
  sim.cfg.tEnd = t;
  sim.writeParams();
  await sim.runToEnd();
}

// --- hydrodynamics ---------------------------------------------------------------------------

const SOD = { L: { rho: 1, u: 0, p: 1 }, R: { rho: 0.125, u: 0, p: 0.1 }, gamma: 1.4, t: 0.2 };

/**
 * The Sod tube along x on an n x n grid at t = 0.2: the middle row's density, velocity and
 * pressure, the exact solution at the same points, and the mean density error.
 */
export async function sod(n: number, limiter?: LimiterKind) {
  return withSim(await gpu(), sodX, n, { ndim: 2, tEnd: SOD.t, ...(limiter ? { limiter } : {}) }, async (sim) => {
    await sim.runToEnd();
    const { u1, u2 } = await sim.readState();
    const exact = exactRiemann(SOD.L, SOD.R, SOD.gamma);
    const j = n >> 1;
    const x: number[] = [];
    const rho: number[] = [];
    const v: number[] = [];
    const p: number[] = [];
    const ex = { rho: [] as number[], v: [] as number[], p: [] as number[] };
    let l1 = 0;
    for (let i = 0; i < n; i++) {
      const q = 4 * (j * n + i);
      const r = u1[q];
      const vx = u1[q + 1] / r;
      x.push((i + 0.5) / n);
      rho.push(r);
      v.push(vx);
      p.push((SOD.gamma - 1) * u2[q + 1]);
      const e = exact(((i + 0.5) / n - 0.5) / SOD.t);
      ex.rho.push(e.rho);
      ex.v.push(e.u);
      ex.p.push(e.p);
      l1 += Math.abs(r - e.rho);
    }
    return { n, t: SOD.t, x, rho, v, p, exact: ex, l1: l1 / n };
  });
}

/** Mean density error of the Sod tube (t = 0.2) on n x n grids, for each n. */
export async function sodConvergence(ns: number[]) {
  const out: { n: number; l1: number }[] = [];
  for (const n of ns) out.push({ n, l1: (await sod(n)).l1 });
  return out;
}

/**
 * Smooth advection: a Gaussian carried once around the periodic box diagonally (t = 2) on
 * n x n grids, as the advection-accuracy test does; the mean density error for each n.
 */
export async function advectionConvergence(ns: number[], limiter?: LimiterKind) {
  const out: { n: number; l1: number }[] = [];
  for (const n of ns) {
    const l1 = await withSim(await gpu(), advection, n, { ndim: 2, tEnd: 2, ...(limiter ? { limiter } : {}) }, async (sim) => {
      const before = (await sim.readState()).u1.slice();
      await sim.runToEnd(64);
      const after = (await sim.readState()).u1;
      let s = 0;
      for (let i = 0; i < n * n; i++) s += Math.abs(after[4 * i] - before[4 * i]);
      return s / (n * n);
    });
    out.push({ n, l1 });
  }
  return out;
}

/**
 * A 3D Sedov blast on the adaptive mesh (finest n cells per side, `levels` levels) at its end
 * time: the distance from the center and the density of the leaf cells, at most `keep` of them
 * (every k-th, in the order the mesh holds them), and how many there are.
 */
export async function sedovCells(n: number, levels: number, keep = 200000) {
  return withAmr(await gpu(), sedov, n, { amrLevels: levels }, async (sim) => {
    await sim.runToEnd();
    const c = await sim.readLeaves();
    const step = Math.max(1, Math.ceil(c.count / keep));
    const r: number[] = [];
    const rho: number[] = [];
    // Five figures: more than a plot shows, and the file stays small.
    const f5 = (x: number) => +x.toPrecision(5);
    for (let q = 0; q < c.count; q += step) {
      r.push(f5(Math.hypot(c.x[q], c.y[q], c.z[q])));
      rho.push(f5(c.u1[4 * q]));
    }
    return { n, levels, t: sim.cfg.tEnd, gamma: sim.cfg.eosParams.gamma, cells: c.count, r, rho };
  });
}

// --- hydrostatic balance ---------------------------------------------------------------------

/**
 * The well-balanced test's isothermal column (density falling by e every 4 cells, 64 cells
 * tall, 2D) over one sound crossing: the largest Mach number at `samples` times, with and
 * without the well-balanced reconstruction.
 */
export async function wellBalanced(samples = 40) {
  const series = async (wellBalanced: boolean) =>
    withSim(await gpu(), column(4), 64, { ndim: 2, tEnd: 0, wellBalanced }, async (sim) => {
      const t: number[] = [];
      const mach: number[] = [];
      for (let i = 1; i <= samples; i++) {
        await runTo(sim, i / samples);
        const d = await sim.diagnostics();
        t.push(d.t);
        mach.push(d.maxMach);
      }
      return { t, mach };
    });
  return { balanced: await series(true), plain: await series(false) };
}

/** Density and distance from the box center of every cell (every leaf cell on an adaptive mesh). */
async function cellsOf(sim: Sim | AmrSim): Promise<{ rho: Float32Array; r: Float32Array }> {
  if (sim instanceof AmrSim) {
    const c = await sim.readLeaves();
    const r = new Float32Array(c.count);
    const rho = new Float32Array(c.count);
    for (let q = 0; q < c.count; q++) {
      r[q] = Math.hypot(c.x[q], c.y[q], c.z[q]);
      rho[q] = c.u1[4 * q];
    }
    return { rho, r };
  }
  const { u1 } = await sim.readState();
  const n = sim.n;
  const h = sim.cfg.problem.box / n;
  const r = new Float32Array(n ** 3);
  const rho = new Float32Array(n ** 3);
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const q = (k * n + j) * n + i;
        r[q] = Math.hypot(i + 0.5 - n / 2, j + 0.5 - n / 2, k + 0.5 - n / 2) * h;
        rho[q] = u1[4 * q];
      }
  return { rho, r };
}

/**
 * A white dwarf of `mass` solar masses over `dyn` dynamical times, on a grid of n cells at the
 * finest level with `levels` levels of refinement (0: uniform): its central density against time,
 * settled into the grid's own balance (the preset) and mapped from its profile alone (the same
 * preset without settling). Time is in units of the dynamical time sqrt(R^3 / G M), R being the
 * largest distance from the center of a cell denser than 1e-4 of the center.
 */
export async function whiteDwarf(mass = 1, dyn = 6, n = 64, levels = 0, samples = 48) {
  const mapped: Problem = { ...whiteDwarfStar, star: undefined, settle: undefined };
  const over = { tEnd: 0, amrLevels: levels, params: { ...configFor(whiteDwarfStar, n).params, m: mass } };
  const series = async (problem: Problem) => {
    const go = async (sim: Sim | AmrSim) => {
      await sim.settled;
      const c = await cellsOf(sim);
      let rhoc = 0;
      for (const x of c.rho) rhoc = Math.max(rhoc, x);
      let R = 0;
      c.rho.forEach((x, q) => {
        if (x > 1e-4 * rhoc) R = Math.max(R, c.r[q]);
      });
      const tdyn = Math.sqrt(R ** 3 / (sim.cfg.G * mass));
      const t: number[] = [0];
      const rho: number[] = [(await sim.diagnostics()).ranges[0][1]];
      for (let i = 1; i <= samples; i++) {
        await runTo(sim, (i / samples) * dyn * tdyn);
        const d = await sim.diagnostics();
        t.push(d.t / tdyn);
        rho.push(d.ranges[0][1]);
      }
      return { t, rho, tdyn, radius: R };
    };
    return levels > 0 ? withAmr(await gpu(), problem, n, over, go) : withSim(await gpu(), problem, n, over, go);
  };
  return { mass, n, levels, settled: await series(whiteDwarfStar), mapped: await series(mapped) };
}

// --- throughput ------------------------------------------------------------------------------

/** Encode and submit `steps` steps, in batches, and wait for the GPU to finish them; seconds taken. */
async function timeSteps(sim: Sim | AmrSim, steps: number): Promise<number> {
  const d = sim.device;
  await d.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  let done = 0;
  while (done < steps) {
    const enc = d.createCommandEncoder();
    const k = sim.encodeSteps(enc, Math.min(16, steps - done));
    d.queue.submit([enc.finish()]);
    sim.submitted();
    done += k;
    if (k === 0) await (sim.stepsResume() ?? d.queue.onSubmittedWorkDone());
  }
  await d.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / 1000;
}

interface ThroughputGrid {
  label: string;
  problem: string;
  ndim: 2 | 3;
  /** [finest cells per side, levels of refinement] */
  runs: [number, number][];
}

/**
 * Cell updates per second on each grid: after 20 steps of warm-up, enough steps to take about two
 * seconds are timed on the GPU. A uniform grid updates every cell each step, an adaptive mesh
 * every leaf cell (per-block timesteps are off), counted from the mesh at the start and the end
 * of the timed steps.
 */
export async function throughput(grids: ThroughputGrid[]) {
  const g = await gpu();
  const { problemById } = await import('../problems');
  const out = [];
  for (const grid of grids) {
    const problem = problemById(grid.problem);
    const runs = [];
    for (const [n, levels] of grid.runs) {
      const over = { ndim: grid.ndim, tEnd: Infinity, amrLevels: levels, timeBins: 0 };
      const measure = async (sim: Sim | AmrSim) => {
        const cells = () => (sim instanceof AmrSim ? sim.tree.stats().cells : n ** grid.ndim);
        await timeSteps(sim, 20);
        const probe = await timeSteps(sim, 10);
        const steps = Math.max(20, Math.min(2000, Math.round((2 * 10) / Math.max(probe, 1e-4))));
        const c0 = cells();
        const seconds = await timeSteps(sim, steps);
        const mean = (c0 + cells()) / 2;
        return { n, levels, cells: Math.round(mean), steps, ms_per_step: (1000 * seconds) / steps, updates_per_second: (mean * steps) / seconds };
      };
      runs.push(levels > 0 ? await withAmr(g, problem, n, over, measure) : await withSim(g, problem, n, over, measure));
    }
    out.push({ label: grid.label, problem: grid.problem, ndim: grid.ndim, runs });
  }
  const adapter = `${g.info.vendor} ${g.info.architecture}`.trim() || 'WebGPU adapter';
  return { adapter: `${adapter}, ${navigator.userAgent.match(/Chrome\/[\d.]+/)?.[0] ?? 'browser'}`, grids: out };
}

// --- gravity ---------------------------------------------------------------------------------

/**
 * The potential of the uniform sphere with isolated boundaries against the analytic one, along
 * the x axis through the center: on a uniform 64^3 grid, and on adaptive meshes with 2 and 3
 * levels (finest 128 and 256 cells), from the leaf cells that touch the axis.
 */
export async function spherePotentialError() {
  const g = await gpu();
  const uniform = await withSim(g, sphere, 64, {}, async (sim) => {
    sim.solvePotential(10);
    const phi = await sim.readPotential();
    const n = 64;
    const x: number[] = [];
    const err: number[] = [];
    // The cells on either side of the axis: j and k at n/2 - 1 and n/2 straddle it.
    const j = n / 2;
    const k = n / 2;
    for (let i = 0; i < n; i++) {
      const cx = (i + 0.5) / n - 0.5;
      const r = Math.hypot(cx, 0.5 / n, 0.5 / n);
      x.push(cx);
      err.push(phi[(k * n + j) * n + i] / spherePotential(r) - 1);
    }
    return { label: 'uniform, 64^3', x, err };
  });
  const adaptive = [];
  for (const [n, levels] of [[128, 2], [256, 3]] as const) {
    adaptive.push(
      await withAmr(g, sphere, n, { amrLevels: levels, tEnd: Infinity }, async (sim) => {
        const c = await sim.readLeaves();
        const phi = await sim.readLeafPotential();
        const pts: [number, number][] = [];
        for (let q = 0; q < c.count; q++) {
          // Cells whose y and z extents both contain the axis.
          if (c.y[q] > 0 && c.y[q] < 0.51 * c.dx[q] && c.z[q] > 0 && c.z[q] < 0.51 * c.dx[q]) {
            pts.push([c.x[q], phi[q] / spherePotential(Math.hypot(c.x[q], c.y[q], c.z[q])) - 1]);
          }
        }
        pts.sort((a, b) => a[0] - b[0]);
        return { label: `${levels} levels, ${n}^3 finest`, x: pts.map((p) => p[0]), err: pts.map((p) => p[1]) };
      }),
    );
  }
  return [uniform, ...adaptive];
}
