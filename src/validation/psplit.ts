// Validation tests of implicit pressure (plans/incompressible.md, stage 2; src/pressure):
// the multigrid solver on its own, and the liquid's implicit step on the uniform grid.
// Registered in tests.ts after the others.
import type { Gpu } from '../gpu/device';
import { configFor, Sim, type SimConfig } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { defaultEosParams } from '../eos/eos';
import { presetSpec, problemById } from '../presets/index';
import { compileSetup } from '../setup/compile';
import { cloneSetup } from '../setup/spec';
import { mixMoments, problemCoefficients } from '../species/mixture';

/** How uniform the pressure must stay in psplit-species-eos (the explicit scheme keeps it to 0.017). */
const PTOL = 0.03;
import { all, PERIODIC, type Problem } from '../problems/types';
import { PF_CLOSED, PF_OPEN, PF_PERIODIC, PressureSolver } from '../pressure/multigrid';
import { cavityError, GHIA_RE100, greshoGas } from './liquid';
import { column } from './tests';
import { exactRiemann } from '../reference/riemann_exact';
import type { TestDef } from './tests';

async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
  const sim = await Sim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

async function withAmr<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: AmrSim) => Promise<T>) {
  const sim = await AmrSim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

/** A run on blocks from start to end: what the tests below read of it. */
async function measureBlocks(sim: AmrSim) {
  const levels = () => new Set(sim.tree.leaves().map((b) => b.level)).size;
  const d0 = await sim.diagnostics();
  const before = levels();
  const cells = await sim.readLeaves();
  const dim = sim.cfg.ndim;
  let scale = 0;
  for (let q = 0; q < cells.count; q++) scale += Math.hypot(cells.u1[q * 4 + 1], cells.u1[q * 4 + 2], cells.u1[q * 4 + 3]) * cells.dx[q] ** dim;
  // The residual part way through: the run's last steps, of no length, solve nothing.
  sim.step(10);
  await sim.diagnostics();
  const residual = (await sim.prs!.stats()).residual;
  const steps = await sim.runToEnd(64, 1e6);
  const d1 = await sim.diagnostics();
  const solve = await sim.prs!.stats();
  const leaves = await sim.readLeaves();
  let vz = 0;
  for (let q = 0; q < leaves.count; q++) vz = Math.max(vz, Math.abs(leaves.u1[q * 4 + 3] / leaves.u1[q * 4]));
  return {
    kept: d1.eKin / d0.eKin, steps, residual, vz, levels: [before, levels()],
    unconverged: solve.unconverged / Math.max(solve.solves, 1),
    dev: Math.max(Math.abs(d1.ranges[0][0] - 1), Math.abs(d1.ranges[0][1] - 1)),
    mass: Math.abs(d1.mass / d0.mass - 1),
    etot: Math.abs(d1.eTot / d0.eTot - 1),
    mom: Math.max(...[0, 1, 2].map((a) => Math.abs(d1.momentum[a] - d0.momentum[a]))) / scale,
  };
}

const IMPLICIT: Partial<SimConfig> = { pressure: 'implicit', acousticCfl: 1e6 };
/** The liquid's sound speed set to `c0` on a preset's configuration. */
const withC0 = (c0: number): Partial<SimConfig> => ({ eosParams: { ...defaultEosParams(), liqRho: 1, liqC: c0 } });

/**
 * The operator of pressure/stencil.wgsl in float64: (A p)_c for coefficients `coef` (four per
 * cell: the upper faces' betas and alpha) on a grid of `dims` cells of size h, the box's faces
 * of kinds `lo` and `hi`.
 */
function applyOperator(p: Float64Array, coef: Float32Array, dims: readonly number[], h: number, lo: readonly number[], hi: readonly number[]): Float64Array {
  const [nx, ny, nz] = dims;
  const at = (i: number, j: number, k: number) => (k * ny + j) * nx + i;
  const out = new Float64Array(p.length);
  const axes = nz === 1 ? 2 : 3;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const c = [i, j, k];
        const o = at(i, j, k);
        let sum = 0;
        let diag = coef[4 * o + 3] * h * h;
        for (let a = 0; a < axes; a++) {
          const bu = coef[4 * o + a];
          const up = [...c];
          up[a]++;
          if (up[a] < dims[a]) {
            sum += bu * p[at(up[0], up[1], up[2])];
            diag += bu;
          } else if (hi[a] === PF_PERIODIC) {
            up[a] = 0;
            sum += bu * p[at(up[0], up[1], up[2])];
            diag += bu;
          } else if (hi[a] === PF_OPEN) diag += 2 * bu;
          const dn = [...c];
          dn[a]--;
          if (dn[a] < 0 && lo[a] === PF_PERIODIC) dn[a] = dims[a] - 1;
          if (dn[a] >= 0) {
            const q = at(dn[0], dn[1], dn[2]);
            sum += coef[4 * q + a] * p[q];
            diag += coef[4 * q + a];
          } else if (lo[a] === PF_OPEN) diag += 2 * (bu === 0 ? 1 : bu);
        }
        out[o] = (diag * p[o] - sum) / (h * h);
      }
  return out;
}

/** A Gresho vortex's start, run and end, as the tests below read them. */
async function measure(sim: Sim) {
  const d0 = await sim.diagnostics();
  const { u1 } = await sim.readState();
  let scale = 0;
  for (let o = 0; o < u1.length; o += 4) scale += Math.hypot(u1[o + 1], u1[o + 2], u1[o + 3]);
  scale *= sim.cfg.ndim === 2 ? (sim.cfg.problem.box / sim.n) ** 2 : (sim.cfg.problem.box / sim.n) ** 3;
  const steps = await sim.runToEnd(64, 1e7);
  const d1 = await sim.diagnostics();
  const solve = (await sim.prs?.stats()) ?? { solves: 0, residual: 0, unconverged: 0 };
  return {
    unconverged: solve.unconverged / Math.max(solve.solves, 1),
    kept: d1.eKin / d0.eKin, steps,
    dev: Math.max(Math.abs(d1.ranges[0][0] - 1), Math.abs(d1.ranges[0][1] - 1)),
    mass: Math.abs(d1.mass / d0.mass - 1),
    mom: Math.max(...[0, 1, 2].map((a) => Math.abs(d1.momentum[a] - d0.momentum[a]))) / scale,
  };
}

export const PSPLIT_TESTS: TestDef[] = [
  {
    id: 'psplit-solver',
    name: 'Pressure multigrid: a manufactured solution with walls, an obstacle and variable coefficients (2D 128^2, 3D 32^3)',
    criterion:
      'For a smooth pressure whose right-hand side is formed in float64 by the same operator, with coefficients varying by a factor of 3, an obstacle (faces of beta = 0 around cells that keep their value) and each kind of box face, eight V-cycles from zero bring the pressure within 1e-4 of it (relative to its largest value) and the residual below 1e-4 of the right-hand side; the last cycle still cuts the residual by 3 or more unless it is at round-off. Periodic, closed and open boxes, in 2D and 3D',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const [tag, dims, lo, hi] of [
        ['2d_periodic', [128, 128, 1], [PF_PERIODIC, PF_PERIODIC, PF_PERIODIC], [PF_PERIODIC, PF_PERIODIC, PF_PERIODIC]],
        ['2d_closed', [128, 128, 1], [PF_CLOSED, PF_CLOSED, PF_PERIODIC], [PF_CLOSED, PF_CLOSED, PF_PERIODIC]],
        ['2d_open', [128, 128, 1], [PF_OPEN, PF_CLOSED, PF_PERIODIC], [PF_OPEN, PF_CLOSED, PF_PERIODIC]],
        ['3d_closed', [32, 32, 32], [PF_CLOSED, PF_CLOSED, PF_CLOSED], [PF_CLOSED, PF_CLOSED, PF_CLOSED]],
        ['3d_mixed', [32, 32, 32], [PF_PERIODIC, PF_OPEN, PF_CLOSED], [PF_PERIODIC, PF_CLOSED, PF_CLOSED]],
      ] as const) {
        const [nx, ny, nz] = dims;
        const h = 1 / nx;
        const N = nx * ny * nz;
        const at = (i: number, j: number, k: number) => (k * ny + j) * nx + i;
        const axes = nz === 1 ? 2 : 3;
        // An obstacle: a block of cells a fifth of the box across, off center.
        const solid = (i: number, j: number, k: number) => Math.abs(i / nx - 0.3) < 0.1 && Math.abs(j / ny - 0.6) < 0.1 && (nz === 1 || Math.abs(k / nz - 0.5) < 0.2);
        const coef = new Float32Array(4 * N);
        const exact = new Float64Array(N);
        for (let k = 0; k < nz; k++)
          for (let j = 0; j < ny; j++)
            for (let i = 0; i < nx; i++) {
              const o = at(i, j, k);
              const c = [i, j, k];
              if (solid(i, j, k)) continue;
              const [x, y, z] = [(i + 0.5) / nx, (j + 0.5) / ny, (k + 0.5) / nz];
              exact[o] = Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y) + 0.5 * Math.cos(4 * Math.PI * (x + y)) * (nz === 1 ? 1 : Math.cos(2 * Math.PI * z)) + 0.3;
              coef[4 * o + 3] = 5 * (1 + 0.5 * Math.sin(2 * Math.PI * x));
              for (let a = 0; a < axes; a++) {
                const up = [...c];
                up[a]++;
                const edge = up[a] >= dims[a];
                if (edge && hi[a] === PF_PERIODIC) up[a] = 0;
                const closed = edge && hi[a] === PF_CLOSED;
                const wall = !closed && !(edge && hi[a] === PF_OPEN) && solid(up[0], up[1], up[2]);
                coef[4 * o + a] = closed || wall ? 0 : 1 + 0.5 * Math.sin(2 * Math.PI * (x + 2 * y) + a);
              }
            }
        // The lower faces of the cells above an obstacle are the obstacle's own upper faces: zero already.
        const b = applyOperator(exact, coef, dims, h, lo, hi);
        for (let o = 0; o < N; o++) if (coef[4 * o + 3] === 0) b[o] = 0;
        const solver = await PressureSolver.create(gpu.device, dims, h, lo, hi);
        try {
          solver.write(Float32Array.from(b), coef);
          const residual = async () => {
            const enc = gpu.device.createCommandEncoder();
            const p = enc.beginComputePass();
            solver.encodeResidual(p);
            p.end();
            gpu.device.queue.submit([enc.finish()]);
            return (await solver.read('res')).reduce((s, v) => Math.max(s, Math.abs(v)), 0);
          };
          const bmax = b.reduce((s, v) => Math.max(s, Math.abs(v)), 0);
          const history: number[] = [];
          for (let cycle = 0; cycle < 8; cycle++) {
            const enc = gpu.device.createCommandEncoder();
            const p = enc.beginComputePass();
            if (cycle === 0) solver.encodeCoarsen(p);
            solver.encodeVCycle(p);
            p.end();
            gpu.device.queue.submit([enc.finish()]);
            history.push((await residual()) / bmax);
          }
          const got = await solver.read('p');
          let err = 0;
          let pmax = 0;
          for (let o = 0; o < N; o++) {
            if (coef[4 * o + 3] === 0) continue;
            err = Math.max(err, Math.abs(got[o] - exact[o]));
            pmax = Math.max(pmax, Math.abs(exact[o]));
          }
          m[`${tag}_error`] = err / pmax;
          m[`${tag}_residual`] = history[7];
          m[`${tag}_first_cycle_factor`] = 1 / history[0];
          m[`${tag}_last_cycle_factor`] = history[6] / history[7];
          pass &&= err / pmax < 1e-4 && history[7] < 1e-4 && (history[6] / history[7] >= 3 || history[7] < 1e-5);
        } finally {
          solver.destroy();
        }
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-gresho',
    name: 'Implicit pressure: liquid Gresho vortex at Mach 0.1, 0.01 and 0.001 (2D 64^2)',
    criterion:
      'To t = 2 the three runs take the same number of steps (to 2%), under a tenth of the explicit scheme\'s at Mach 0.1; each keeps at least the kinetic energy the explicit scheme keeps at Mach 0.1 and gains none, and the same as the others to 0.002; the density stays within 2 M^2 of the reference; mass is conserved to 2e-6 and each momentum component to 1e-6 of the sum of |momentum|; fewer than 2% of the steps\' solves (3 V-cycles each) end above the tolerance of 1e-4',
    async run(gpu) {
      const gresho = problemById('gresho');
      const m: Record<string, number> = {};
      let pass = true;
      const explicit = await withSim(gpu, gresho, 64, { ndim: 2, tEnd: 2 }, measure);
      m.explicit_kept = explicit.kept;
      m.explicit_steps = explicit.steps;
      const runs = [];
      for (const mach of [0.1, 0.01, 0.001]) {
        const r = await withSim(gpu, gresho, 64, { ndim: 2, tEnd: 2, ...IMPLICIT, ...withC0(1 / mach) }, measure);
        runs.push(r);
        m[`mach_${mach}_kept`] = r.kept;
        m[`mach_${mach}_steps`] = r.steps;
        m[`mach_${mach}_density_deviation_over_mach2`] = r.dev / (mach * mach);
        m[`mach_${mach}_mass_drift`] = r.mass;
        m[`mach_${mach}_momentum_drift`] = r.mom;
        m[`mach_${mach}_unconverged_fraction`] = r.unconverged;
        pass &&= r.kept < 1 && r.kept >= explicit.kept && Math.abs(r.kept - runs[0].kept) < 0.002 && Math.abs(r.steps - runs[0].steps) <= 0.02 * runs[0].steps;
        pass &&= r.dev < 2 * mach * mach && r.mass < 2e-6 && r.mom < 1e-6 && r.unconverged < 0.02;
      }
      pass &&= runs[0].steps < 0.1 * explicit.steps;
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-2d-3d',
    name: 'Implicit pressure: a 2D flow and the same flow as a column in 3D (Gresho, 32^2 and 32^3)',
    criterion: 'The column keeps the 2D run\'s kinetic energy to 1e-4 by t = 0.5, in the same number of steps, with no motion along z above 1e-6 of the flow speed, and conserves mass to 2e-6',
    async run(gpu) {
      const gresho = problemById('gresho');
      const flat = await withSim(gpu, gresho, 32, { ndim: 2, tEnd: 0.5, ...IMPLICIT }, measure);
      return withSim(gpu, gresho, 32, { ndim: 3, tEnd: 0.5, ...IMPLICIT }, async (sim) => {
        const r = await measure(sim);
        const { u1 } = await sim.readState();
        let vz = 0;
        for (let o = 0; o < u1.length; o += 4) vz = Math.max(vz, Math.abs(u1[o + 3] / u1[o]));
        const m = { kept_2d: flat.kept, kept_3d: r.kept, steps_2d: flat.steps, steps_3d: r.steps, max_vz: vz, mass_drift_3d: r.mass };
        return { pass: Math.abs(r.kept - flat.kept) < 1e-4 && r.steps === flat.steps && vz < 1e-6 && r.mass < 2e-6, metrics: m };
      });
    },
  },
  {
    id: 'psplit-taylor-green',
    name: 'Implicit pressure: viscous Taylor-Green vortex in a liquid (2D, 64 cells per wavelength)',
    criterion: 'The decay rate of the kinetic energy over 0 < t < 1 is within 1% of 4 nu k^2, as the explicit scheme\'s is, in at most a fifth of its steps',
    async run(gpu) {
      const tg = problemById('taylor-green');
      const rate = async (sim: Sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64);
        const d1 = await sim.diagnostics();
        const k = 2 * Math.PI;
        return { ratio: -Math.log(d1.eKin / d0.eKin) / (4 * sim.cfg.viscosity * k * k * sim.cfg.tEnd), steps };
      };
      const explicit = await withSim(gpu, tg, 64, { ndim: 2, tEnd: 1 }, rate);
      const implicit = await withSim(gpu, tg, 64, { ndim: 2, tEnd: 1, ...IMPLICIT }, rate);
      const m = { decay_rate_over_exact: implicit.ratio, steps: implicit.steps, explicit_decay_rate_over_exact: explicit.ratio, explicit_steps: explicit.steps };
      return { pass: Math.abs(implicit.ratio - 1) < 0.01 && implicit.steps <= 0.2 * explicit.steps, metrics: m };
    },
  },
  {
    id: 'psplit-cavity',
    name: 'Implicit pressure: lid-driven cavity at Reynolds number 100 (2D 128^2, liquid)',
    criterion:
      'By t = 12 the vertical-centerline u is within 0.015 of the lid speed of Ghia, Ghia & Shin (1982) at every tabulated height, as with explicit pressure, and mass is conserved to 2e-6 (no-slip walls and a moving lid; the step is viscosity\'s here, a third of the explicit scheme\'s count)',
    async run(gpu) {
      const n = 128;
      return withSim(gpu, problemById('lid-driven-cavity-liquid'), n, { ndim: 2, tEnd: 12, viscosity: 0.01, ...IMPLICIT }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd(64, 1e7);
        const d = await sim.diagnostics();
        const c = cavityError((await sim.readState()).u1, n, GHIA_RE100);
        const m = { cavity_max_error: c.error, cavity_min_u: c.minU, steps: d.steps, mass_drift: Math.abs(d.mass / d0.mass - 1) };
        return { pass: c.error < 0.015 && m.mass_drift < 2e-6, metrics: m };
      });
    },
  },
  {
    id: 'psplit-obstacle',
    name: 'Implicit pressure: flow past a cylinder, with an inlet and open faces (2D 256^2, liquid, Reynolds number 100)',
    criterion:
      'The cylinder wake preset to t = 20: the run stays finite with the density within 5% of the reference and nothing floored, and the wake sheds, the transverse velocity two diameters behind the cylinder swinging past 0.2 of the stream on both sides with a period within 10% of the explicit scheme\'s 1.43 (a Strouhal number of 0.174)',
    async run(gpu) {
      return withSim(gpu, problemById('cylinder-wake'), 256, { ndim: 2, tEnd: 20, ...IMPLICIT }, async (sim) => {
        const n = 256;
        // The cell two diameters behind the cylinder's center, on the axis.
        const probe = 4 * ((n / 2) * n + Math.round(((-0.5 + 2) / 4) * n));
        const ts: number[] = [];
        const vy: number[] = [];
        for (let t = 0.1; t < 20.05; t += 0.1) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd(64, 1e7);
          const { u1 } = await sim.readState();
          ts.push(t);
          vy.push(u1[probe + 2] / u1[probe]);
        }
        const d = await sim.diagnostics();
        const cross: number[] = [];
        for (let i = 1; i < vy.length; i++) if (ts[i] > 12 && vy[i - 1] < 0 && vy[i] >= 0) cross.push(ts[i - 1] + ((0 - vy[i - 1]) / (vy[i] - vy[i - 1])) * (ts[i] - ts[i - 1]));
        const period = cross.length > 1 ? (cross[cross.length - 1] - cross[0]) / (cross.length - 1) : NaN;
        const late = vy.slice(120);
        const m = {
          period, strouhal: 0.25 / period, vy_min: Math.min(...late), vy_max: Math.max(...late), steps: d.steps, floored: d.floored,
          density_deviation: Math.max(Math.abs(d.ranges[0][0] - 1), Math.abs(d.ranges[0][1] - 1)),
        };
        return { pass: Math.abs(period / 1.43 - 1) < 0.1 && m.vy_min < -0.2 && m.vy_max > 0.2 && m.floored === 0 && m.density_deviation < 0.05, metrics: m };
      });
    },
  },
  {
    id: 'psplit-balance',
    name: 'Implicit pressure: a liquid column in a uniform field, between walls (2D 64^2)',
    criterion:
      'A column whose pressure differences obey the trapezoid rule p[j+1] - p[j] = -g dx (rho[j] + rho[j+1]) / 2 (g = 1, sound speed 4, so the density falls by 6% up the box) stays at rest: after t = 2, 60 sound crossings in a handful of steps, its largest Mach number is below 1e-5. A column that starts at uniform density instead settles into balance: by t = 20 its largest Mach number is below 1e-3 and the density difference between its bottom and top rows is within 2% of g box / c^2. Mass is conserved to 2e-6 in both',
    async run(gpu) {
      const n = 64;
      const c0 = 4;
      const column = (balanced: boolean): Problem => ({
        id: 'liquid-column', label: 'Liquid column', blurb: '', eos: 'liquid', eosParams: { liqRho: 1, liqC: c0 },
        bcLo: [0, 2, 0], bcHi: [0, 2, 0], gravity: 'off', G: 1, box: 1, tEnd: 2, supports2d: true,
        gext: () => [0, -1, 0],
        init: (cells) => {
          const a = 1 / cells / (2 * c0 * c0); // g dx / 2 c^2
          const q = (1 - a) / (1 + a);
          return (_x, y) => ({ rho: balanced ? Math.pow(q, Math.round((y + 0.5) * cells - 0.5)) : 1, vx: 0, vy: 0, vz: 0, p: 0, s: 0 });
        },
      });
      const run = (balanced: boolean, tEnd: number) =>
        withSim(gpu, column(balanced), n, { ndim: 2, tEnd, pressure: 'implicit', acousticCfl: 50 }, async (sim) => {
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd(64, 1e6);
          const d = await sim.diagnostics();
          const { u1 } = await sim.readState();
          return { mach: d.maxMach, steps, mass: Math.abs(d.mass / d0.mass - 1), drop: u1[0] - u1[4 * n * (n - 1)] };
        });
      const still = await run(true, 2);
      const settled = await run(false, 20);
      const want = (1 - 1 / n) / (c0 * c0);
      const m = {
        balanced_max_mach: still.mach, balanced_steps: still.steps, balanced_mass_drift: still.mass,
        settled_max_mach: settled.mach, settled_steps: settled.steps, settled_mass_drift: settled.mass, settled_density_drop_over_expected: settled.drop / want,
      };
      return { pass: still.mach < 1e-5 && settled.mach < 1e-3 && Math.abs(m.settled_density_drop_over_expected - 1) < 0.02 && still.mass < 2e-6 && settled.mass < 2e-6, metrics: m };
    },
  },
  {
    id: 'psplit-acoustic',
    name: 'Implicit pressure: a sound wave resolved in time, and one that is not (2D 64^2, liquid at rest)',
    criterion:
      'A plane wave of one wavelength across the box: at an acoustic Courant number of 0.5 it is back where it started after one period to 0.5% of a wavelength, with more than 97% of its amplitude and no more than all of it (the time-centered step, theta = 0.55 there: backward Euler kept 86%); at a Courant number of 10 (theta = 1) it is damped to under 20% of its amplitude and its density never exceeds the starting amplitude',
    async run(gpu) {
      const n = 64;
      const a = 1e-3;
      const still: Problem = {
        id: 'liquid-wave', label: 'Sound in a liquid', blurb: '', eos: 'liquid', eosParams: { liqRho: 1, liqC: 1 },
        bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
        init: () => (x) => ({ rho: 1 + a * Math.sin(2 * Math.PI * x), vx: a * Math.sin(2 * Math.PI * x), vy: 0, vz: 0, p: 0, s: 0 }),
      };
      // The step is cfl dx / (|v| + 2 c / acousticCfl): a Courant number of cfl acousticCfl / 2.
      const wave = (courant: number) =>
        withSim(gpu, still, n, { ndim: 2, tEnd: 1, pressure: 'implicit', cfl: 0.5, acousticCfl: 4 * courant, pressureCycles: 6 }, async (sim) => {
          let peak = 0;
          let steps = 0;
          for (;;) {
            sim.step(1);
            await sim.device.queue.onSubmittedWorkDone();
            const { u1 } = await sim.readState();
            for (let o = 0; o < u1.length; o += 4) peak = Math.max(peak, Math.abs(u1[o] - 1));
            const d = await sim.diagnostics();
            steps = d.steps;
            if (d.t >= 1 - 1e-6 || steps > 400) break;
          }
          const { u1 } = await sim.readState();
          let re = 0;
          let im = 0;
          for (let i = 0; i < n; i++) {
            const x = (i + 0.5) / n - 0.5;
            re += (u1[4 * i] - 1) * Math.sin(2 * Math.PI * x);
            im += (u1[4 * i] - 1) * Math.cos(2 * Math.PI * x);
          }
          return { shift: Math.atan2(-im, re) / (2 * Math.PI), amp: (2 * Math.hypot(re, im)) / n / a, peak: peak / a, steps };
        });
      const fine = await wave(0.5);
      const coarse = await wave(10);
      const m = { resolved_shift: fine.shift, resolved_amplitude: fine.amp, resolved_steps: fine.steps, coarse_amplitude: coarse.amp, coarse_peak: coarse.peak, coarse_steps: coarse.steps };
      return { pass: Math.abs(fine.shift) < 0.005 && fine.amp > 0.97 && fine.amp < 1.001 && coarse.amp < 0.2 && coarse.peak < 1.001, metrics: m };
    },
  },
  {
    id: 'psplit-amr-unrefined',
    name: 'Implicit pressure on blocks without refinement against the uniform grid (Gresho, 2D 64^2)',
    criterion: 'The block grid (16 root blocks, its pressure solved through the root-level multigrid) keeps the uniform grid\'s kinetic energy to 1e-3 by t = 1, in the same number of steps to one, conserving mass to 2e-6',
    async run(gpu) {
      const gresho = problemById('gresho');
      const uni = await withSim(gpu, gresho, 64, { ndim: 2, tEnd: 1, ...IMPLICIT }, measure);
      const blk = await withAmr(gpu, gresho, 64, { ndim: 2, tEnd: 1, amrLevels: 0, ...IMPLICIT }, measureBlocks);
      const m = { kept_uniform: uni.kept, kept_blocks: blk.kept, steps_uniform: uni.steps, steps_blocks: blk.steps, blocks_mass_drift: blk.mass, blocks_residual: blk.residual };
      return { pass: Math.abs(blk.kept - uni.kept) < 1e-3 && Math.abs(blk.steps - uni.steps) <= 1 && blk.mass < 2e-6, metrics: m };
    },
  },
  {
    id: 'psplit-amr-gresho',
    name: 'Implicit pressure on the adaptive mesh: a Gresho vortex across refinement boundaries at Mach 0.1, 0.01 and 0.001 (2D, 512 finest, 3 levels)',
    criterion:
      'The mesh of liquid-amr (three levels, their boundaries through the vortex, no block merged). To t = 1 the three runs take the same number of steps (to 2%), under a tenth of the explicit scheme\'s 12900; each keeps more than 99.8% of its kinetic energy and gains none, the same as the others to 0.002; the density stays within 2 M^2 of the reference; mass is conserved to 2e-6 and each momentum component to 1e-6 of the sum of |momentum|, across the levels; the composite residual after a step\'s three cycles is under 3e-3, and fewer than 2% of the steps end above that. (At this resolution the residual is single-precision round-off, about 3e-4 of the right-hand side at Mach 0.1: it is the same after 2, 4 and 8 cycles)',
    async run(gpu) {
      const gresho = problemById('gresho');
      const m: Record<string, number> = {};
      let pass = true;
      const runs = [];
      for (const mach of [0.1, 0.01, 0.001]) {
        const r = await withAmr(gpu, gresho, 512, { ndim: 2, tEnd: 1, amrLevels: 2, refineHi: 0.6, refineLo: 0, pressureTol: 3e-3, ...IMPLICIT, ...withC0(1 / mach) }, measureBlocks);
        runs.push(r);
        m[`mach_${mach}_kept`] = r.kept;
        m[`mach_${mach}_steps`] = r.steps;
        m[`mach_${mach}_density_deviation_over_mach2`] = r.dev / (mach * mach);
        m[`mach_${mach}_mass_drift`] = r.mass;
        m[`mach_${mach}_momentum_drift`] = r.mom;
        m[`mach_${mach}_residual`] = r.residual;
        m[`mach_${mach}_unconverged_fraction`] = r.unconverged;
        pass &&= r.levels[0] === 3 && r.levels[1] === 3 && r.kept < 1 && r.kept > 0.998 && Math.abs(r.kept - runs[0].kept) < 0.002 && Math.abs(r.steps - runs[0].steps) <= 0.02 * runs[0].steps;
        pass &&= r.steps < 1290 && r.dev < 2 * mach * mach && r.mass < 2e-6 && r.mom < 1e-6 && r.residual < 3e-3 && r.unconverged < 0.02;
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-regrid',
    name: 'Implicit pressure on the adaptive mesh while it regrids (Gresho, 2D, 512 finest, 3 levels at the start)',
    criterion:
      'With the merging threshold just under the refining one (0.5 against 0.6) the mesh merges fine blocks as the run goes (three levels at the start, fewer at t = 1). No pressure pulse comes of it: the density stays within 2 M^2 of the reference at the end, mass is conserved to 2e-6, and the vortex keeps more than 99% of its kinetic energy and gains none',
    async run(gpu) {
      const r = await withAmr(gpu, problemById('gresho'), 512, { ndim: 2, tEnd: 1, amrLevels: 2, refineHi: 0.6, refineLo: 0.5, ...IMPLICIT }, measureBlocks);
      const m = { levels_at_start: r.levels[0], levels_at_end: r.levels[1], kept: r.kept, steps: r.steps, density_deviation_over_mach2: r.dev / 0.01, mass_drift: r.mass, unconverged_fraction: r.unconverged };
      return { pass: r.levels[0] === 3 && r.levels[1] < 3 && r.kept > 0.99 && r.kept < 1 && r.dev < 0.02 && r.mass < 2e-6, metrics: m };
    },
  },
  {
    id: 'psplit-amr-3d',
    name: 'Implicit pressure on blocks in 3D (a Gresho column, 64^3 finest, one level of refinement allowed) and with viscosity (Taylor-Green, 2D blocks)',
    criterion:
      'The column to t = 0.25: mass conserved to 2e-6, no motion along z above 1e-5 of the flow speed, more than 98% of the kinetic energy kept and none gained, fewer than 2% of the steps unconverged. The viscous Taylor-Green vortex on a block grid (128^2, one level allowed) decays at a rate within 1% of 4 nu k^2 over 0 < t < 0.5',
    async run(gpu) {
      const col = await withAmr(gpu, problemById('gresho'), 64, { ndim: 3, tEnd: 0.25, amrLevels: 1, ...IMPLICIT }, measureBlocks);
      const tg = problemById('taylor-green');
      const k = 2 * Math.PI;
      const decay = await withAmr(gpu, tg, 128, { ndim: 2, tEnd: 0.5, amrLevels: 1, ...IMPLICIT }, measureBlocks);
      const ratio = -Math.log(decay.kept) / (4 * (tg.viscosity ?? 0) * k * k * 0.5);
      const m = { column_kept: col.kept, column_steps: col.steps, column_mass_drift: col.mass, column_max_vz: col.vz, column_unconverged_fraction: col.unconverged, taylor_green_decay_rate_over_exact: ratio, taylor_green_steps: decay.steps };
      return { pass: col.mass < 2e-6 && col.vz < 1e-5 && col.kept > 0.98 && col.kept < 1 && col.unconverged < 0.02 && Math.abs(ratio - 1) < 0.01, metrics: m };
    },
  },
  {
    id: 'psplit-amr-shear',
    name: 'Adaptive mesh in a liquid: the velocity indicator refines a sharp double shear layer (2D, 512 finest, 2 levels allowed, implicit pressure)',
    criterion:
      'Layers a fifth of a root cell thick, in a liquid whose density and pressure are uniform: the velocity indicator alone refines the mesh to three levels along the layers, at the start and as they move to t = 0.3; through the regrids mass is conserved to 2e-6, the density stays within 5% of the reference, and fewer than 2% of the steps\' solves end above the tolerance',
    async run(gpu) {
      const cfg = configFor(problemById('shear-layers'), 512, { ndim: 2, tEnd: 0.3, amrLevels: 2, ...IMPLICIT });
      cfg.params.sharp = 400;
      const sim = await AmrSim.create(gpu, cfg);
      try {
        const r = await measureBlocks(sim);
        const m = { levels_at_start: r.levels[0], levels_at_end: r.levels[1], leaves_at_end: sim.tree.leaves().length, steps: r.steps, mass_drift: r.mass, density_deviation: r.dev, unconverged_fraction: r.unconverged };
        return { pass: r.levels[0] === 3 && r.levels[1] === 3 && r.mass < 2e-6 && r.dev < 0.05 && r.unconverged < 0.02, metrics: m };
      } finally {
        sim.destroy();
      }
    },
  },
  {
    id: 'psplit-gas-gresho',
    name: 'Implicit pressure for an ideal gas: Gresho vortex at Mach 0.1, 0.01 and 0.001 (2D 64^2)',
    criterion:
      'To t = 2, at Mach 0.1 and 0.01: the same number of steps (to 2%), under a tenth of the explicit low-Mach HLLC\'s at Mach 0.1; at least its kinetic energy kept, less 0.002, and none gained; the density, which should not change, within 2e-3 of 1; mass conserved to 2e-6 and total energy to 2e-6. At Mach 0.001, where the pressure is the difference of an energy of 1e6 and a kinetic energy of order 1 in single precision: the same steps to 5%, more than 95% of the kinetic energy kept, the density within 5%',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const energy = async (sim: Sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64, 1e7);
        const d1 = await sim.diagnostics();
        return { kept: d1.eKin / d0.eKin, steps, dev: Math.max(Math.abs(d1.ranges[0][0] - 1), Math.abs(d1.ranges[0][1] - 1)), mass: Math.abs(d1.mass / d0.mass - 1), etot: Math.abs(d1.eTot / d0.eTot - 1) };
      };
      const explicit = await withSim(gpu, greshoGas(0.1), 64, { ndim: 2, tEnd: 2, riemann: 'hllclm' }, energy);
      m.explicit_kept = explicit.kept;
      m.explicit_steps = explicit.steps;
      const runs = [];
      for (const mach of [0.1, 0.01, 0.001]) {
        const r = await withSim(gpu, greshoGas(mach), 64, { ndim: 2, tEnd: 2, ...IMPLICIT }, energy);
        runs.push(r);
        m[`mach_${mach}_kept`] = r.kept;
        m[`mach_${mach}_steps`] = r.steps;
        m[`mach_${mach}_density_deviation`] = r.dev;
        m[`mach_${mach}_mass_drift`] = r.mass;
        m[`mach_${mach}_energy_drift`] = r.etot;
        if (mach > 0.005) pass &&= r.kept < 1 && r.kept > explicit.kept - 0.002 && Math.abs(r.steps - runs[0].steps) <= 0.02 * runs[0].steps && r.dev < 2e-3 && r.mass < 2e-6 && r.etot < 2e-6;
        else pass &&= r.kept < 1 && r.kept > 0.95 && Math.abs(r.steps - runs[0].steps) <= 0.05 * runs[0].steps && r.dev < 0.05 && r.mass < 2e-6;
      }
      pass &&= runs[0].steps < 0.1 * explicit.steps;
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-gas',
    name: 'Implicit pressure for an ideal gas on the adaptive mesh: a Gresho vortex across refinement boundaries at Mach 0.1 and 0.01 (2D, 512 finest, 3 levels)',
    criterion:
      'Three levels with their boundaries through the vortex, no block merged. To t = 1 the two runs take the same number of steps (to 2%); each keeps more than 99.5% of its kinetic energy and gains none, the same as the other to 0.002; the density, which should not change, stays within 5e-3 of 1 (the start, whose pressure is the setup\'s and not the scheme\'s, sheds a sound wave at the refinement boundaries); mass and total energy are conserved to 2e-6 across the levels; the composite residual after a step\'s three cycles is under 1e-4 early in the run, and fewer than 2% of the steps end above 1e-3',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const runs = [];
      for (const mach of [0.1, 0.01]) {
        const r = await withAmr(gpu, greshoGas(mach), 512, { ndim: 2, tEnd: 1, amrLevels: 2, refineHi: 0.6, refineLo: 0, pressureTol: 1e-3, ...IMPLICIT }, measureBlocks);
        runs.push(r);
        m[`mach_${mach}_levels`] = r.levels[0];
        m[`mach_${mach}_kept`] = r.kept;
        m[`mach_${mach}_steps`] = r.steps;
        m[`mach_${mach}_density_deviation`] = r.dev;
        m[`mach_${mach}_mass_drift`] = r.mass;
        m[`mach_${mach}_energy_drift`] = r.etot;
        m[`mach_${mach}_residual`] = r.residual;
        m[`mach_${mach}_unconverged_fraction`] = r.unconverged;
        pass &&= r.levels[0] === 3 && r.levels[1] === 3 && r.kept < 1 && r.kept > 0.995 && Math.abs(r.kept - runs[0].kept) < 0.002 && Math.abs(r.steps - runs[0].steps) <= 0.02 * runs[0].steps;
        pass &&= r.dev < 5e-3 && r.mass < 2e-6 && r.etot < 2e-6 && r.residual < 1e-4 && r.unconverged < 0.02;
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-sod',
    name: 'Implicit pressure for an ideal gas: Sod shock tube (128 cells along x)',
    criterion:
      'At t = 0.2 the shock is within one cell of its exact position, mass and total energy are conserved to 1e-6, and the run takes under 0.4 of the explicit scheme\'s steps (a third, before the flow\'s and the pressure\'s limits on the step were combined: psplit-moving-jump); the L1 density error against the exact solution is under 0.012 at the default Courant number and under 0.009 at half of it (the explicit scheme: 0.004; backward Euler for the pressure, before the step was time-centered: 0.016 and 0.011)',
    async run(gpu) {
      const n = 128;
      const exact = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
      const tube = (over: Partial<SimConfig>) =>
        withSim(gpu, problemById('sod-x'), n, { ndim: 2, ...over }, async (sim) => {
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd(64, 1e6);
          const d = await sim.diagnostics();
          const { u1 } = await sim.readState();
          const row = (i: number) => u1[4 * ((n / 2) * n + i)];
          let l1 = 0;
          for (let i = 0; i < n; i++) l1 += Math.abs(row(i) - exact(((i + 0.5) / n - 0.5) / d.t).rho) / n;
          // The shock: where the density falls through the middle of its jump.
          let shock = NaN;
          for (let i = n - 2; i > 0; i--)
            if (row(i) > 0.195 && row(i + 1) <= 0.195) {
              shock = (i + 0.5 + (row(i) - 0.195) / (row(i) - row(i + 1))) / n - 0.5;
              break;
            }
          return { l1, steps, shock: Math.abs(shock - 1.75216 * d.t), mass: Math.abs(d.mass / d0.mass - 1), etot: Math.abs(d.eTot / d0.eTot - 1) };
        });
      const explicit = await tube({});
      const implicit = await tube({ pressure: 'implicit' });
      const half = await tube({ pressure: 'implicit', cfl: 0.4 });
      const m = {
        l1: implicit.l1, l1_half_courant: half.l1, l1_explicit: explicit.l1, steps: implicit.steps, steps_half_courant: half.steps, steps_explicit: explicit.steps,
        shock_error_cells: implicit.shock * n, shock_error_cells_half_courant: half.shock * n, mass_drift: implicit.mass, energy_drift: implicit.etot,
      };
      const pass = implicit.shock < 1 / n && half.shock < 1 / n && implicit.mass < 1e-6 && implicit.etot < 1e-6 && half.etot < 1e-6 && implicit.steps < 0.4 * explicit.steps && implicit.l1 < 0.012 && half.l1 < 0.009;
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-sod',
    name: 'Implicit pressure for an ideal gas on the adaptive mesh: Sod shock tube (256 cells along x at the finest, 2 levels)',
    criterion:
      'With the default refinement and timestep settings (open faces along x, the mesh following the waves): at t = 0.2 mass and total energy are conserved to 1e-6, the run takes within 10% of the uniform grid\'s steps at 256 cells (the solved pressure\'s acceleration limits the first steps on blocks as there), the volume-weighted L1 density error against the exact solution is under 0.008 (0.0093 before the step was time-centered), and no step\'s solve ends above the tolerance',
    async run(gpu) {
      const exact = exactRiemann({ rho: 1, u: 0, p: 1 }, { rho: 0.125, u: 0, p: 0.1 }, 1.4);
      const sod = problemById('sod-x');
      const uniform = await withSim(gpu, sod, 256, { ndim: 2, pressure: 'implicit' }, (sim) => sim.runToEnd(64, 1e6));
      return withAmr(gpu, sod, 256, { ndim: 2, amrLevels: 1, pressure: 'implicit' }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64, 1e6);
        const d = await sim.diagnostics();
        const solve = await sim.prs!.stats();
        const cells = await sim.readLeaves();
        let l1 = 0;
        let area = 0;
        for (let q = 0; q < cells.count; q++) {
          const a = cells.dx[q] ** 2;
          l1 += a * Math.abs(cells.u1[q * 4] - exact(cells.x[q] / d.t).rho);
          area += a;
        }
        const m = {
          l1: l1 / area, steps, steps_uniform: uniform, mass_drift: Math.abs(d.mass / d0.mass - 1), energy_drift: Math.abs(d.eTot / d0.eTot - 1),
          unconverged: solve.unconverged, leaf_levels: new Set(sim.tree.leaves().map((b) => b.level)).size,
        };
        const pass = m.l1 < 0.008 && Math.abs(steps - uniform) <= 0.1 * uniform && m.mass_drift < 1e-6 && m.energy_drift < 1e-6 && m.unconverged === 0 && m.leaf_levels === 2;
        return { pass, metrics: m };
      });
    },
  },
  {
    id: 'psplit-moving-jump',
    name: 'Implicit pressure for an ideal gas: a hot spot in dense gas that is already moving (2D 128^2, uniform grid and blocks)',
    criterion:
      'Gas of density 30 and pressure 1 crossing a periodic box at 0.42, with a spot of pressure 10 in it, at the default settings (the flow alone allows a step of 0.7 cells, and the spot\'s pressure another: the step has to answer to both together). To t = 0.1, on the uniform grid and on a two-level mesh: no cell is floored, the density stays within a factor of 6 of 30 (the blast thins its middle to 9), the first step moves the gas under 0.6 of a cell, and mass is conserved to 1e-6',
    async run(gpu) {
      const n = 128;
      const v = 0.42;
      const spot: Problem = {
        id: 'moving-jump', label: '', blurb: '', eos: 'ideal', eosParams: { gamma: 1.4 }, bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 0.1, supports2d: true,
        init: () => (x, y) => ({ rho: 30, vx: 0, vy: v, vz: 0, p: Math.hypot(x, y) < 0.04 ? 10 : 1, s: 0 }),
      };
      const m: Record<string, number> = {};
      let pass = true;
      const read = async (tag: string, sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        sim.step(1);
        const first = (await sim.diagnostics()).t;
        const steps = 1 + (await sim.runToEnd(64, 1e6));
        const d = await sim.diagnostics();
        Object.assign(m, {
          [`${tag}_first_step_cells`]: first * v * n, [`${tag}_steps`]: steps, [`${tag}_rho_min`]: d.ranges[0][0], [`${tag}_rho_max`]: d.ranges[0][1],
          [`${tag}_floored`]: d.floored, [`${tag}_mass_drift`]: Math.abs(d.mass / d0.mass - 1),
        });
        pass &&= d.floored === 0 && d.ranges[0][0] > 5 && d.ranges[0][1] < 180 && first * v * n < 0.6 && Math.abs(d.mass / d0.mass - 1) < 1e-6;
      };
      await withSim(gpu, spot, n, { ndim: 2, pressure: 'implicit' }, (sim) => read('uniform', sim));
      await withAmr(gpu, spot, n, { ndim: 2, amrLevels: 1, pressure: 'implicit' }, (sim) => read('blocks', sim));
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-balance',
    name: 'Implicit pressure on the adaptive mesh: a liquid column in a uniform field, with a refinement boundary across it (2D, 128 finest, 2 levels)',
    criterion:
      'The column of psplit-balance (g = 1, sound speed 4), started in the uniform grid\'s discrete balance at each block\'s own cell size, with the lower half of the box refined (density above 1.03): the mesh has two levels throughout; after t = 2 the largest Mach number is below 1e-3 (the balance across the refinement boundary holds to the order of the density\'s change over a cell, not to round-off), and mass is conserved to 2e-6',
    async run(gpu) {
      const c0 = 4;
      const column: Problem = {
        id: 'liquid-column', label: 'Liquid column', blurb: '', eos: 'liquid', eosParams: { liqRho: 1, liqC: c0 },
        bcLo: [0, 2, 0], bcHi: [0, 2, 0], gravity: 'off', G: 1, box: 1, tEnd: 2, supports2d: true,
        gext: () => [0, -1, 0],
        // Hydrostatic for p = c^2 (rho - 1): rho = exp(-g (y - y0) / c^2), 1 at the top.
        init: () => (_x, y) => ({ rho: Math.exp((0.5 - y) / (c0 * c0)), vx: 0, vy: 0, vz: 0, p: 0, s: 0 }),
      };
      return withAmr(gpu, column, 128, { ndim: 2, tEnd: 2, amrLevels: 1, refineDense: 1.03, refineLo: 0, pressure: 'implicit', acousticCfl: 50 }, async (sim) => {
        const levels = () => new Set(sim.tree.leaves().map((b) => b.level)).size;
        const before = levels();
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64, 1e6);
        const d = await sim.diagnostics();
        const m = { levels_at_start: before, levels_at_end: levels(), max_mach: d.maxMach, steps, mass_drift: Math.abs(d.mass / d0.mass - 1) };
        return { pass: before === 2 && m.levels_at_end === 2 && d.maxMach < 1e-3 && m.mass_drift < 2e-6, metrics: m };
      });
    },
  },
  {
    id: 'psplit-amr-field',
    name: 'Implicit pressure on the adaptive mesh: columns at rest in a uniform field with refinement boundaries along the field (2D, 128 finest, 2 levels)',
    criterion:
      'A species marker refines a patch or a strip of the box, so that refinement boundaries run along the field as well as across it. To t = 2 the largest Mach number stays below: 1e-4 for the liquid column of psplit-amr-balance with a patch in its middle (it was 1e-2 before the fine cells\' ghosts took the field into account); for an isothermal ideal gas whose density falls by e over the box, 1e-4 with the lower part refined (a boundary across the field), 5e-4 with a strip from wall to wall, and 2e-3 with a patch, whose corners admit no state of rest (the explicit scheme on the same meshes: 6e-4, 8e-3 and 2e-3). The gas\'s total energy changes by less than 1e-6',
    async run(gpu) {
      const patch = (x: number, y: number) => Math.abs(x) < 0.1 && Math.abs(y) < 0.1;
      const strip = (x: number) => Math.abs(x) < 0.1;
      const base = { label: '', blurb: '', bcLo: [0, 2, 0], bcHi: [0, 2, 0], gravity: 'off', G: 1, box: 1, tEnd: 2, supports2d: true, gext: () => [0, -1, 0] } as const;
      const mark = (on: boolean) => (on ? [1, 0] : [0, 1]);
      const liquid = (inside: (x: number, y: number) => boolean): Problem => ({
        ...base, id: 'liquid-column', eos: 'liquid', eosParams: { liqRho: 1, liqC: 4 }, species: ['a', 'b'],
        init: () => (x: number, y: number) => ({ rho: Math.exp((0.5 - y) / 16), vx: 0, vy: 0, vz: 0, p: 0, s: 0, X: mark(inside(x, y)) }),
      } as unknown as Problem);
      const gas = (inside: ((x: number, y: number) => boolean) | null): Problem => ({
        ...base, id: 'gas-column', eos: 'ideal', eosParams: { gamma: 1.4 }, species: inside ? ['a', 'b'] : undefined,
        init: () => (x: number, y: number) => ({ rho: Math.exp(-(y + 0.5)), vx: 0, vy: 0, vz: 0, p: Math.exp(-(y + 0.5)), s: 0, X: inside ? mark(inside(x, y)) : undefined }),
      } as unknown as Problem);
      const m: Record<string, number> = {};
      let pass = true;
      const cases: [string, Problem, Partial<SimConfig>, number][] = [
        ['liquid_patch', liquid(patch), {}, 1e-4],
        ['gas_across', gas(null), { refineDense: 0.62, refineHi: 1e9 }, 1e-4],
        ['gas_strip', gas(strip), {}, 5e-4],
        ['gas_patch', gas(patch), {}, 2e-3],
      ];
      for (const [tag, problem, over, limit] of cases) {
        await withAmr(gpu, problem, 128, { ndim: 2, tEnd: 2, amrLevels: 1, refineLo: 0, pressure: 'implicit', acousticCfl: 50, ...over }, async (sim) => {
          const levels = new Set(sim.tree.leaves().map((b) => b.level)).size;
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64, 1e6);
          const d = await sim.diagnostics();
          m[`${tag}_max_mach`] = d.maxMach;
          m[`${tag}_levels`] = levels;
          pass &&= levels === 2 && d.maxMach < limit && Math.abs(d.mass / d0.mass - 1) < 2e-6;
          if (problem.eos === 'ideal') {
            m[`${tag}_energy_drift`] = Math.abs(d.eTot / d0.eTot - 1);
            pass &&= m[`${tag}_energy_drift`] < 1e-6;
          }
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-species',
    name: 'Implicit pressure with species carried by the gas (2D 64^2): a liquid vortex, and an ideal gas carrying a density bump across a periodic box',
    criterion:
      'Three species (a disc, a uniform 0.2, the rest) are moved after the pressure correction, with the mass fluxes of the whole step (pressure/mf_on.wgsl), and scaled to its density. In both runs, at the end: the mass fractions of every cell sum to 1 within 1e-5 and lie in [-1e-4, 1 + 1e-4]; the uniform species is still 0.2 to 1e-4 everywhere; and each species\' total mass is kept to 1e-6 (moved with the advective mass fluxes alone, the scaling was by the density\'s correction: 2e-3 in the liquid). A species\' sharp edge can overshoot by a percent of its jump. The liquid run again on a block grid, where the correction leaves the whole step\'s mass fluxes in each block\'s records (amr/pressure/mf_on.wgsl): the same sums, bounds, uniform species and species\' masses (with the advective fluxes alone: 5e-4 of their mass)',
    async run(gpu) {
      const n = 64;
      const X = (r: number) => (r < 0.2 ? [0.7, 0.2, 0.1] : [0.1, 0.2, 0.7]);
      const base = problemById('gresho');
      const liquid: Problem = { ...base, id: 'liquid-species', setup: undefined, species: ['a', 'b', 'c'], init: (cells, eos, ndim, params) => { const f = base.init(cells, eos, ndim, params); return (x, y, z) => ({ ...f(x, y, z), X: X(Math.hypot(x, y)) }); } };
      const gas: Problem = {
        id: 'gas-species', label: '', blurb: '', eos: 'ideal', eosParams: { gamma: 1.4 }, bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
        species: ['a', 'b', 'c'],
        init: () => (x, y) => ({ rho: 1 + 0.5 * Math.exp(-(x * x + y * y) / 0.02), vx: 1, vy: 0.5, vz: 0, p: 20, s: 0, X: X(Math.hypot(x, y)) }),
      };
      const m: Record<string, number> = {};
      let pass = true;
      for (const [tag, problem] of [['liquid', liquid], ['gas', gas]] as const) {
        await withSim(gpu, problem, n, { ndim: 2, tEnd: 1, riemann: 'hllclm', ...IMPLICIT }, async (sim) => {
          const mass = (Xs: Float32Array[], u1: Float32Array) => Xs.map((x) => x.reduce((t, v, q) => t + v * u1[4 * q], 0));
          const m0 = mass(await sim.readSpecies(), (await sim.readState()).u1);
          const steps = await sim.runToEnd(64, 1e6);
          const Xs = await sim.readSpecies();
          const m1 = mass(Xs, (await sim.readState()).u1);
          let sum = 0;
          let lo = Infinity;
          let hi = -Infinity;
          let flat = 0;
          for (let q = 0; q < n * n; q++) {
            sum = Math.max(sum, Math.abs(Xs[0][q] + Xs[1][q] + Xs[2][q] - 1));
            for (const x of Xs) {
              lo = Math.min(lo, x[q]);
              hi = Math.max(hi, x[q]);
            }
            flat = Math.max(flat, Math.abs(Xs[1][q] - 0.2));
          }
          const drift = Math.max(...m1.map((v, i) => Math.abs(v / m0[i] - 1)));
          Object.assign(m, { [`${tag}_sum_error`]: sum, [`${tag}_min`]: lo, [`${tag}_max`]: hi, [`${tag}_uniform_species_error`]: flat, [`${tag}_species_mass_drift`]: drift, [`${tag}_steps`]: steps });
          pass &&= sum < 1e-5 && lo > -1e-4 && hi < 1 + 1e-4 && flat < 1e-4 && drift < 1e-6;
        });
      }
      // And on blocks (the liquid, 128 finest, one level of refinement allowed).
      await withAmr(gpu, liquid, 128, { ndim: 2, tEnd: 1, amrLevels: 1, ...IMPLICIT }, async (sim) => {
        // Each species' mass: X rho dx^2 over the leaves (readLeaves and readLeafSpecies list them alike).
        const mass = async () => {
          const c = await sim.readLeaves();
          const Xs = await sim.readLeafSpecies();
          return Xs.map((x) => x.reduce((t, v, q) => t + v * c.u1[4 * q] * c.dx[q] * c.dx[q], 0));
        };
        const m0 = await mass();
        const steps = await sim.runToEnd(64, 1e6);
        const m1 = await mass();
        const X = await sim.readLeafSpecies();
        let sum = 0;
        let lo = Infinity;
        let hi = -Infinity;
        let flat = 0;
        for (let q = 0; q < X[0].length; q++) {
          sum = Math.max(sum, Math.abs(X[0][q] + X[1][q] + X[2][q] - 1));
          for (const x of X) {
            lo = Math.min(lo, x[q]);
            hi = Math.max(hi, x[q]);
          }
          flat = Math.max(flat, Math.abs(X[1][q] - 0.2));
        }
        const drift = Math.max(...m1.map((v, i) => Math.abs(v / m0[i] - 1)));
        Object.assign(m, { blocks_sum_error: sum, blocks_min: lo, blocks_max: hi, blocks_uniform_species_error: flat, blocks_species_mass_drift: drift, blocks_steps: steps });
        pass &&= sum < 1e-5 && lo > -1e-4 && hi < 1 + 1e-4 && flat < 1e-4 && drift < 1e-6;
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-species-eos',
    name: 'Implicit pressure with species that set the equation of state: a carbon blob carried through helium (2D, uniform 64^2 and blocks 128 finest, 1 level)',
    criterion:
      'A blob of carbon (gamma 1.3) in helium (gamma 5/3), at one pressure, carried across a periodic box at Mach 0.05: the pressure, worked out from each cell\'s internal energy and composition, stays uniform to 0.03 (the explicit scheme: 0.017; a solve blind to the composition would read the blob at 2.2 times the rest), the mass fractions sum to 1 within 1e-5 and lie in [-1e-4, 1 + 1e-4], and each species keeps its mass to 1e-6; on the uniform grid and on blocks. (Refused until the pressure passes read each cell\'s composition.)',
    async run(gpu) {
      const gas: Problem = {
        id: 'carbon-blob', label: '', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 }, bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 4, supports2d: true,
        species: ['he4', 'c12'], speciesGamma: { c12: 1.3 },
        init: () => (x, y) => {
          const f = 0.5 * (1 - Math.tanh((Math.hypot(x, y) - 0.15) / 0.02));
          return { rho: 1 + 2 * f, vx: 0.065, vy: 0.03, vz: 0, p: 1, s: 0, X: [1 - f, f] };
        },
      };
      const coeffs = problemCoefficients(gas, 5 / 3)!;
      // p = (gamma - 1) rho e of the mixture, gamma - 1 = x / w (species/mixture.ts).
      const pressures = (eint: number[], X: Float32Array[]) => eint.map((e, q) => {
        const k = mixMoments(coeffs, X.map((x) => x[q]));
        return (e * k[0]) / k[3];
      });
      const m: Record<string, number> = {};
      let pass = true;
      const judge = (tag: string, p: number[], X: Float32Array[], m0: number[], m1: number[]) => {
        let dev = 0;
        let sum = 0;
        let lo = Infinity;
        let hi = -Infinity;
        for (let q = 0; q < p.length; q++) {
          dev = Math.max(dev, Math.abs(p[q] - 1));
          sum = Math.max(sum, Math.abs(X[0][q] + X[1][q] - 1));
          for (const x of X) {
            lo = Math.min(lo, x[q]);
            hi = Math.max(hi, x[q]);
          }
        }
        const drift = Math.max(...m1.map((v, i) => Math.abs(v / m0[i] - 1)));
        Object.assign(m, { [`${tag}_pressure_deviation`]: dev, [`${tag}_sum_error`]: sum, [`${tag}_min`]: lo, [`${tag}_max`]: hi, [`${tag}_species_mass_drift`]: drift });
        pass &&= dev < PTOL && sum < 1e-5 && lo > -1e-4 && hi < 1 + 1e-4 && drift < 1e-6;
      };
      await withSim(gpu, gas, 64, { ndim: 2, ...IMPLICIT }, async (sim) => {
        const mass = (X: Float32Array[], u1: Float32Array) => X.map((x) => x.reduce((t, v, q) => t + v * u1[4 * q], 0));
        const s0 = await sim.readState();
        const m0 = mass(await sim.readSpecies(), s0.u1);
        m.uniform_steps = await sim.runToEnd(64, 1e6);
        const s1 = await sim.readState();
        const X = await sim.readSpecies();
        const eint = Array.from({ length: X[0].length }, (_, q) => s1.u2[4 * q + 1]);
        judge('uniform', pressures(eint, X), X, m0, mass(X, s1.u1));
      });
      // The explicit scheme on the same problem, for scale (not judged).
      await withSim(gpu, gas, 64, { ndim: 2 }, async (sim) => {
        await sim.runToEnd(64, 1e6);
        const s1 = await sim.readState();
        const X = await sim.readSpecies();
        const p = pressures(Array.from({ length: X[0].length }, (_, q) => s1.u2[4 * q + 1]), X);
        m.explicit_pressure_deviation = Math.max(...p.map((x) => Math.abs(x - 1)));
      });
      await withAmr(gpu, gas, 128, { ndim: 2, amrLevels: 1, ...IMPLICIT }, async (sim) => {
        const read = async () => {
          const c = await sim.readLeaves();
          const X = await sim.readLeafSpecies();
          return { c, X, mass: X.map((x) => x.reduce((t, v, q) => t + v * c.u1[4 * q] * c.dx[q] * c.dx[q], 0)) };
        };
        const r0 = await read();
        m.blocks_steps = await sim.runToEnd(64, 1e6);
        const r1 = await read();
        const eint = Array.from({ length: r1.X[0].length }, (_, q) => r1.c.u2[4 * q + 1]);
        judge('blocks', pressures(eint, r1.X), r1.X, r0.mass, r1.mass);
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-gas-balance',
    name: 'Implicit pressure for an ideal gas in a uniform field: the stratified columns of the well-balanced test (2D 64^2)',
    criterion:
      'Isothermal columns at rest between walls, in the discrete balance p[j+1] - p[j] = -g dx (rho[j] + rho[j+1]) / 2, the density falling by e every 16 cells (a factor of 55 in all) and every 4 (9e6): after one sound crossing the largest Mach number is below 1e-3 in both, as the explicit scheme\'s well-balanced reconstruction holds them, and total energy has changed by less than 1e-5',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const cells of [16, 4]) {
        await withSim(gpu, column(cells), 64, { ndim: 2, tEnd: 1, pressure: 'implicit', acousticCfl: 50 }, async (sim) => {
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd(64, 1e6);
          const d = await sim.diagnostics();
          m[`e_fold_${cells}_cells_max_mach`] = d.maxMach;
          m[`e_fold_${cells}_cells_steps`] = steps;
          m[`e_fold_${cells}_cells_energy_drift`] = Math.abs(d.eTot / d0.eTot - 1);
          pass &&= d.maxMach < 1e-3 && Math.abs(d.eTot / d0.eTot - 1) < 1e-5;
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-gas-polytrope',
    name: 'Implicit pressure with self-gravity: a polytropic star at rest (3D 64^3)',
    criterion:
      'The polytrope preset, settled as usual, over two dynamical times with implicit pressure: the central density stays within 2% of its start (it does with explicit pressure to 3%), mass within 0.5% and total energy within 2%, in under a fifth of the explicit scheme\'s steps. (Gravity kicks the face velocities by the potential\'s difference across each face, and the potential is solved after the pressure correction.)',
    async run(gpu) {
      const star = problemById('polytrope');
      const T = 2 * (star.tDyn ?? 1);
      const hold = (over: Partial<SimConfig>) =>
        withSim(gpu, star, 64, { ndim: 3, tEnd: T, ...over }, async (sim) => {
          await sim.settled;
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd(16, 1e5);
          const d = await sim.diagnostics();
          return { steps, rhoc: d.ranges[0][1] / d0.ranges[0][1], mass: Math.abs(d.mass / d0.mass - 1), etot: Math.abs(d.eTot / d0.eTot - 1) };
        });
      const explicit = await hold({});
      const implicit = await hold({ pressure: 'implicit' });
      const m = {
        central_density: implicit.rhoc, central_density_explicit: explicit.rhoc, steps: implicit.steps, steps_explicit: explicit.steps,
        mass_drift: implicit.mass, energy_drift: implicit.etot, energy_drift_explicit: explicit.etot,
      };
      return { pass: Math.abs(implicit.rhoc - 1) < 0.02 && implicit.mass < 5e-3 && implicit.etot < 0.02 && implicit.steps < 0.2 * explicit.steps, metrics: m };
    },
  },
  {
    id: 'psplit-gas-star-vacuum',
    name: 'Implicit pressure with self-gravity: a star at rest in near-vacuum (3D 64^3)',
    criterion:
      'The polytrope preset with the gas around it at 1e-7 of its central density, settled as usual, over one dynamical time with implicit pressure: no cell is ever hotter than 1.5 times the hottest at the start (the center), none reaches the floors, and the central density stays within 5% of its start. (Gravity\'s work on a face is shared between its two cells by their densities. Shared evenly, the first cell of near-vacuum beyond the surface took half the work done on the star\'s gas and was 3e5 times hotter after one step, and the run was not finite after ten: a random setup found it.)',
    async run(gpu) {
      const spec = cloneSetup(presetSpec('polytrope')!);
      spec.background = { rho: 1e-7, p: 1e-9 };
      spec.physics = { ...spec.physics, floors: { rho: 1e-11, p: 1e-15 } };
      const star = compileSetup(spec).problem;
      const m: Record<string, number> = {};
      let pass = true;
      await withSim(gpu, star, 64, { ndim: 3, tEnd: 0, pressure: 'implicit' }, async (sim) => {
        await sim.settled;
        const d0 = await sim.diagnostics();
        let hottest = 0;
        let floored = 0;
        let finite = true;
        for (const t of [0.05, 0.1, 0.2, 0.4, 0.7, 1]) {
          sim.cfg.tEnd = t * (star.tDyn ?? 1);
          sim.writeParams();
          await sim.runToEnd(16, 1e4);
          const d = await sim.diagnostics();
          finite &&= [d.mass, d.eTot, d.maxMach, d.ranges[1][1]].every(Number.isFinite);
          hottest = Math.max(hottest, d.ranges[1][1] / d0.ranges[1][1]);
          floored = Math.max(floored, d.floored);
          m.central_density = d.ranges[0][1] / d0.ranges[0][1];
          m.steps = d.steps;
        }
        m.hottest_over_central = hottest;
        m.floored_cells = floored;
        m.density_contrast = d0.ranges[0][1] / d0.ranges[0][0];
        pass = finite && hottest < 1.5 && floored === 0 && Math.abs(m.central_density - 1) < 0.05;
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-gas-star-moving',
    name: 'Implicit pressure with self-gravity: a star crossing the grid (3D 64^3, and blocks of two levels, 128^3 finest)',
    criterion:
      'The polytrope preset, settled as usual, set moving at 0.1 (0.3 of its central sound speed: 6.4 cells in the dynamical time run, 0.8 of a cell a step) with implicit pressure. On the uniform grid its central density stays within 2% of its start, its momentum within 1% and its total energy within 1%; on blocks of two levels (the star at the finer, 128 cells across) the central density within 2% and the momentum within 4% (it gains 1.6% there, which the multi-level gravity solve is suspected of: the explicit scheme gains a tenth of that). (Two things a random setup found. The uniform grid\'s update added gravity\'s work on the moving gas as the explicit scheme does, and the pressure correction added it again: the heat stayed where it was while the mass moved on, and the star\'s central density tripled in a dynamical time. And the faces were kicked with the potential of the step before, extrapolated half a step, while the gas they act on is a whole step on: the star lost 10% of its momentum. The potential is now solved for the advected density, before the pressure passes.)',
    async run(gpu) {
      const spec = cloneSetup(presetSpec('polytrope')!);
      (spec.parts![0] as { vel?: number[] }).vel = [0.1, 0, 0];
      const star = compileSetup(spec).problem;
      const T = star.tDyn ?? 1;
      const m: Record<string, number> = {};
      const go = async (tag: string, sim: Sim | AmrSim) => {
        await sim.settled;
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(16, 1e4);
        const d = await sim.diagnostics();
        m[`${tag}_central_density`] = d.ranges[0][1] / d0.ranges[0][1];
        m[`${tag}_momentum`] = d.momentum[0] / d0.momentum[0];
        m[`${tag}_energy_drift`] = Math.abs(d.eTot / d0.eTot - 1);
        m[`${tag}_steps`] = steps;
      };
      await withSim(gpu, star, 64, { ndim: 3, tEnd: T, pressure: 'implicit' }, (sim) => go('uniform', sim));
      const blocks = await AmrSim.create(gpu, configFor(star, 128, { ndim: 3, tEnd: T, amrLevels: 1, timeBins: 0, pressure: 'implicit' }));
      try {
        await go('blocks', blocks);
        m.blocks_leaf_levels = new Set(blocks.tree.leaves().map((b) => b.level)).size;
      } finally {
        blocks.destroy();
      }
      const pass =
        Math.abs(m.uniform_central_density - 1) < 0.02 && Math.abs(m.uniform_momentum - 1) < 0.01 && m.uniform_energy_drift < 0.01 &&
        Math.abs(m.blocks_central_density - 1) < 0.02 && Math.abs(m.blocks_momentum - 1) < 0.04 && m.blocks_leaf_levels === 2;
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-polytrope',
    name: 'Implicit pressure with self-gravity on the adaptive mesh: a polytropic star at rest (3D, 128^3 finest, 2 levels)',
    criterion:
      'The polytrope preset on a mesh with the star at the finer level and the box around it at the coarser, settled as usual, over two dynamical times: the central density stays within 2% of its start, mass within 0.5% and total energy within 2%, in under a fifth of the explicit scheme\'s steps, with two levels throughout. (The potential\'s difference across each face kicks the face velocities, fine faces\' own at a refinement boundary, and the potential is solved after the pressure correction.)',
    async run(gpu) {
      const star = problemById('polytrope');
      const T = 2 * (star.tDyn ?? 1);
      const hold = (over: Partial<SimConfig>) =>
        withAmr(gpu, star, 128, { ndim: 3, tEnd: T, amrLevels: 1, timeBins: 0, ...over }, async (sim) => {
          await sim.settled;
          const levels = () => new Set(sim.tree.leaves().map((b) => b.level)).size;
          const before = levels();
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd(16, 1e5);
          const d = await sim.diagnostics();
          return { steps, levels: Math.min(before, levels()), rhoc: d.ranges[0][1] / d0.ranges[0][1], mass: Math.abs(d.mass / d0.mass - 1), etot: Math.abs(d.eTot / d0.eTot - 1) };
        });
      const explicit = await hold({});
      const implicit = await hold({ pressure: 'implicit' });
      const m = {
        central_density: implicit.rhoc, central_density_explicit: explicit.rhoc, steps: implicit.steps, steps_explicit: explicit.steps,
        mass_drift: implicit.mass, energy_drift: implicit.etot, energy_drift_explicit: explicit.etot, leaf_levels: implicit.levels,
      };
      return { pass: implicit.levels === 2 && Math.abs(implicit.rhoc - 1) < 0.02 && implicit.mass < 5e-3 && implicit.etot < 0.02 && implicit.steps < 0.2 * explicit.steps, metrics: m };
    },
  },
  {
    id: 'psplit-masses',
    name: 'Implicit pressure with point masses (ideal gas, 2D 128^2, uniform grid and blocks): an atmosphere at rest around one, and one moving through the gas',
    criterion:
      'An isothermal atmosphere in balance with a softened point mass (G M = 0.2, softening 0.1, central density e^2 times the outer), between walls, over one sound crossing: on the uniform grid the largest Mach number stays under 2e-3 (the explicit scheme\'s, whose kick is the centered difference of the potential its balanced reconstruction takes, under 1e-4), in under a fifth of its steps, total energy to 1e-5 and the mass where it is to 1e-6; on a two-level mesh with the center refined, under 1e-2 (the refined patch\'s corners) and energy to 2e-5. Two masses moving at Mach 0.25 and 0.08 through uniform gas in a periodic box, to t = 1: the gas\'s mass conserved to 1e-6 and the momentum of gas and masses together to 1e-4 of what they exchanged, at the default Courant number (the masses are given the sum of the very kicks the gas was given), on the uniform grid and on a mesh refined around the wakes; and the first mass\'s velocity is within 4% of the explicit run\'s at the default Courant number and within 1% at an eighth of it, on the uniform grid and on the mesh (backward Euler for the pressure: 8% and 1%, and 1.3% on the mesh)',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const base = { label: '', blurb: '', eos: 'ideal', eosParams: { gamma: 1.4 }, gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true } as const;
      const gm = 0.2;
      const soft = 0.1;
      const atmosphere = {
        ...base, id: 'atmosphere', bcLo: [2, 2, 2], bcHi: [2, 2, 2],
        masses: () => [{ gm, pos: [0, 0, 0], vel: [0, 0, 0], soft, sinkRadius: 0 }],
        init: () => (x: number, y: number) => {
          const rho = Math.exp(gm / Math.sqrt(x * x + y * y + soft * soft));
          return { rho, vx: 0, vy: 0, vz: 0, p: rho, s: 0 };
        },
      } as unknown as Problem;
      const rest = async (sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64, 1e6);
        const d = await sim.diagnostics();
        return { steps, mach: d.maxMach, etot: Math.abs(d.eTot / d0.eTot - 1), moved: Math.hypot(...d.masses[0].pos) };
      };
      const blocks: Partial<SimConfig> = { amrLevels: 1, refineDense: 3, refineLo: 0, timeBins: 0 };
      const re = await withSim(gpu, atmosphere, 128, { ndim: 2 }, rest);
      const ri = await withSim(gpu, atmosphere, 128, { ndim: 2, pressure: 'implicit' }, rest);
      const rb = await withAmr(gpu, atmosphere, 128, { ndim: 2, pressure: 'implicit', ...blocks }, async (sim) => ({ ...(await rest(sim)), levels: new Set(sim.tree.leaves().map((b) => b.level)).size }));
      Object.assign(m, {
        rest_max_mach: ri.mach, rest_max_mach_explicit: re.mach, rest_steps: ri.steps, rest_steps_explicit: re.steps, rest_energy_drift: ri.etot, rest_mass_moved: ri.moved,
        rest_blocks_max_mach: rb.mach, rest_blocks_energy_drift: rb.etot, rest_blocks_levels: rb.levels,
      });
      pass &&= ri.mach < 2e-3 && re.mach < 1e-4 && ri.steps < 0.2 * re.steps && ri.etot < 1e-5 && ri.moved < 1e-6;
      pass &&= rb.levels === 2 && rb.mach < 1e-2 && rb.etot < 2e-5 && rb.moved < 1e-6;

      const gmm = 0.05;
      const mover = {
        ...base, id: 'mover', bcLo: all(PERIODIC), bcHi: all(PERIODIC),
        masses: () => [
          { gm: gmm, pos: [-0.2, 0, 0], vel: [0.3, 0.1, 0], soft: 0.08, sinkRadius: 0 },
          { gm: 0.03, pos: [0.2, 0.2, 0], vel: [-0.1, 0, 0], soft: 0.08, sinkRadius: 0 },
        ],
        init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }),
      } as unknown as Problem;
      const moving = async (sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd(64, 1e6);
        const d = await sim.diagnostics();
        const total = (q: typeof d) => [0, 1].map((a) => q.momentum[a] + gmm * q.masses[0].vel[a] + 0.03 * q.masses[1].vel[a]);
        const [t0, t1] = [total(d0), total(d)];
        const given = Math.hypot(d.momentum[0] - d0.momentum[0], d.momentum[1] - d0.momentum[1]);
        return { vel: d.masses[0].vel, lost: Math.hypot(t1[0] - t0[0], t1[1] - t0[1]) / given, mass: Math.abs(d.mass / d0.mass - 1) };
      };
      const me = await withSim(gpu, mover, 128, { ndim: 2 }, moving);
      const off = (r: { vel: number[] }) => Math.hypot(r.vel[0] - me.vel[0], r.vel[1] - me.vel[1]) / Math.hypot(me.vel[0], me.vel[1]);
      const mi = await withSim(gpu, mover, 128, { ndim: 2, pressure: 'implicit' }, moving);
      const mb = await withAmr(gpu, mover, 128, { ndim: 2, pressure: 'implicit', amrLevels: 1, refineHi: 0.3, timeBins: 0 }, moving);
      const mq = await withSim(gpu, mover, 128, { ndim: 2, pressure: 'implicit', cfl: 0.1 }, moving);
      const bq = await withAmr(gpu, mover, 128, { ndim: 2, pressure: 'implicit', cfl: 0.1, amrLevels: 1, refineHi: 0.3, timeBins: 0 }, moving);
      Object.assign(m, {
        moving_momentum_lost: mi.lost, moving_gas_mass_drift: mi.mass, moving_momentum_lost_explicit: me.lost,
        moving_blocks_momentum_lost: mb.lost, moving_blocks_gas_mass_drift: mb.mass,
        moving_velocity_difference: off(mi), moving_velocity_difference_quarter_courant: off(mq), moving_blocks_velocity_difference_quarter_courant: off(bq),
      });
      pass &&= mi.lost < 1e-4 && mi.mass < 1e-6 && mb.lost < 1e-4 && mb.mass < 1e-6 && mq.lost < 1e-4 && bq.lost < 1e-4 && off(mi) < 0.04 && off(mq) < 0.01 && off(bq) < 0.01;
      return { pass, metrics: m };
    },
  },
  {
    id: 'psplit-amr-wake',
    name: 'Implicit pressure on the adaptive mesh with an obstacle and an inlet: the cylinder wake (2D, 512 finest, 2 levels, liquid, Reynolds number 100)',
    criterion:
      'The cylinder wake preset on blocks (the cylinder\'s surface is always at the finest level, 32 cells across it) to t = 16: the mesh has more than one level, the run stays finite with the density within 5% of the reference, and the wake sheds, the transverse velocity two diameters behind the cylinder swinging past 0.2 of the stream on both sides with a period within 10% of the uniform grid\'s 1.43',
    async run(gpu) {
      return withAmr(gpu, problemById('cylinder-wake'), 512, { ndim: 2, tEnd: 16, amrLevels: 2, ...IMPLICIT }, async (sim) => {
        const ts: number[] = [];
        const vy: number[] = [];
        for (let t = 0.1; t < 16.05; t += 0.1) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd(64, 1e7);
          const cell = await sim.probe([(-0.5 + 2) / 4, 0.5, 0.5]);
          ts.push(t);
          vy.push(cell ? cell.u1[2] / cell.u1[0] : NaN);
        }
        const d = await sim.diagnostics();
        const cross: number[] = [];
        for (let i = 1; i < vy.length; i++) if (ts[i] > 9 && vy[i - 1] < 0 && vy[i] >= 0) cross.push(ts[i - 1] + ((0 - vy[i - 1]) / (vy[i] - vy[i - 1])) * (ts[i] - ts[i - 1]));
        const period = cross.length > 1 ? (cross[cross.length - 1] - cross[0]) / (cross.length - 1) : NaN;
        const late = vy.slice(90);
        const solve = await sim.prs!.stats();
        const m = {
          period, strouhal: 0.25 / period, vy_min: Math.min(...late), vy_max: Math.max(...late), steps: d.steps, floored: d.floored,
          levels: new Set(sim.tree.leaves().map((b) => b.level)).size, leaves: sim.tree.leaves().length,
          density_deviation: Math.max(Math.abs(d.ranges[0][0] - 1), Math.abs(d.ranges[0][1] - 1)), unconverged_fraction: solve.unconverged / Math.max(solve.solves, 1),
        };
        return { pass: Math.abs(period / 1.43 - 1) < 0.1 && m.vy_min < -0.2 && m.vy_max > 0.2 && m.levels > 1 && m.density_deviation < 0.05, metrics: m };
      });
    },
  },
];
