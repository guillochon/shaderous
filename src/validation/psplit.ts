// Validation tests of implicit pressure (plans/incompressible.md, stage 2; src/pressure):
// the multigrid solver on its own, and the liquid's implicit step on the uniform grid.
// Registered in tests.ts after the others.
import type { Gpu } from '../gpu/device';
import { configFor, Sim, type SimConfig } from '../sim';
import { defaultEosParams } from '../eos/eos';
import { problemById } from '../presets/index';
import { all, PERIODIC, type Problem } from '../problems/types';
import { PF_CLOSED, PF_OPEN, PF_PERIODIC, PressureSolver } from '../pressure/multigrid';
import { cavityError, GHIA_RE100 } from './liquid';
import type { TestDef } from './tests';

async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
  const sim = await Sim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
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
  return {
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
      'To t = 2 the three runs take the same number of steps (to 2%), under a tenth of the explicit scheme\'s at Mach 0.1; each keeps at least the kinetic energy the explicit scheme keeps at Mach 0.1 and gains none, and the same as the others to 0.002; the density stays within 2 M^2 of the reference; mass is conserved to 2e-6 and each momentum component to 1e-6 of the sum of |momentum|',
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
        pass &&= r.kept < 1 && r.kept >= explicit.kept && Math.abs(r.kept - runs[0].kept) < 0.002 && Math.abs(r.steps - runs[0].steps) <= 0.02 * runs[0].steps;
        pass &&= r.dev < 2 * mach * mach && r.mass < 2e-6 && r.mom < 1e-6;
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
    id: 'psplit-acoustic',
    name: 'Implicit pressure: a sound wave resolved in time, and one that is not (2D 64^2, liquid at rest)',
    criterion:
      'A plane wave of one wavelength across the box: at an acoustic Courant number of 0.5 it is back where it started after one period to 1% of a wavelength, with more than 70% of its amplitude; at a Courant number of 10 it is damped to under 20% of its amplitude and its density never exceeds the starting amplitude',
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
      return { pass: Math.abs(fine.shift) < 0.01 && fine.amp > 0.7 && fine.amp < 1.001 && coarse.amp < 0.2 && coarse.peak < 1.001, metrics: m };
    },
  },
];
