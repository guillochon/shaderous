// Validation tests against closed-form solutions from ExactPack (Los Alamos's suite of exact
// solutions to the equations of hydrodynamics): the Riemann problems of its table that Sod and
// the strong shock are not, the Noh implosion in its three symmetries, and the uniform collapse.
// Each runs its preset. Registered through validation/frames.ts.
import { exactRiemann } from '../reference/riemann_exact';
import { nohExact, nohShockRadius, uniformCollapseExact, type RadialState } from '../reference/converging';
import { RIEMANN_TUBES } from '../presets/names';
import { problemById } from '../presets/index';
import { sphGrid } from '../spherical';
import type { Sim } from '../sim';
import { withSim, type TestDef } from './tests';

/** A profile of one run: distance from the center (or position along a tube), density, velocity along it, pressure. */
interface Profile {
  r: number[];
  rho: number[];
  u: number[];
  p: number[];
}

/**
 * The run's profile away from what the gas converges on, from a 2D run: along the middle row
 * (Cartesian, planar: x from the box's center), along the middle row from the axis (R, z), or
 * shell-averaged by volume (r, theta).
 */
async function profileOf(sim: Sim): Promise<Profile> {
  const { u1, u2 } = await sim.readState();
  const c = sim.cfg;
  const n = c.n;
  const gm1 = c.eosParams.gamma - 1;
  const out: Profile = { r: [], rho: [], u: [], p: [] };
  if (c.problem.geometry === 'spherical') {
    const g = sphGrid(c);
    for (let i = 0; i < n; i++) {
      let m = 0;
      let mr = 0;
      let e = 0;
      let v = 0;
      for (let j = 0; j < n; j++) {
        const o = 4 * (j * n + i);
        const dv = g.volume(i, j);
        m += u1[o] * dv;
        mr += u1[o + 1] * dv;
        e += u2[o + 1] * dv;
        v += dv;
      }
      out.r.push(g.r(i));
      out.rho.push(m / v);
      out.u.push(mr / m);
      out.p.push((gm1 * e) / v);
    }
    return out;
  }
  const dx = c.problem.box / n;
  const j = n >> 1;
  const fromAxis = c.problem.geometry === 'cylindrical';
  for (let i = 0; i < n; i++) {
    const o = 4 * (j * n + i);
    out.r.push(fromAxis ? (i + 0.5) * dx : (i + 0.5) * dx - 0.5 * c.problem.box);
    out.rho.push(u1[o]);
    out.u.push(u1[o + 1] / u1[o]);
    out.p.push(gm1 * u2[o + 1]);
  }
  return out;
}

/** Sum |run - exact| over sum |exact|, for one quantity, over the samples `keep` admits. */
function l1(prof: Profile, key: 'rho' | 'u' | 'p', exact: (r: number) => RadialState, keep: (r: number) => boolean): number {
  let d = 0;
  let s = 0;
  prof.r.forEach((r, i) => {
    if (!keep(r)) return;
    const e = exact(r)[key];
    d += Math.abs(prof[key][i] - e);
    s += Math.abs(e);
  });
  return d / s;
}

/** Mean of one quantity where `keep` admits. */
function mean(prof: Profile, key: 'rho' | 'u' | 'p', keep: (r: number) => boolean): number {
  let s = 0;
  let k = 0;
  prof.r.forEach((r, i) => {
    if (keep(r)) (s += prof[key][i]), k++;
  });
  return s / k;
}

/** The outermost distance at which the density is still at `level`, interpolated: a shock moving out, with less behind it outside. */
function outerCrossing(prof: Profile, level: number): number {
  for (let i = prof.r.length - 1; i > 0; i--) {
    if (prof.rho[i - 1] >= level && prof.rho[i] < level) {
      const f = (prof.rho[i - 1] - level) / (prof.rho[i - 1] - prof.rho[i]);
      return prof.r[i - 1] + f * (prof.r[i] - prof.r[i - 1]);
    }
  }
  return NaN;
}

const N = 256;

/** The Noh preset of symmetry k at N^2 in 2D: its post-shock density, pre-shock profile and shock radius against the exact solution. */
async function nohRun(gpu: Parameters<TestDef['run']>[0], id: string, k: 1 | 2 | 3) {
  const pr = problemById(id);
  return withSim(gpu, pr, N, { ndim: 2 }, async (sim) => {
    const steps = await sim.runToEnd();
    const t = sim.cfg.tEnd;
    const gamma = sim.cfg.eosParams.gamma;
    const q = { rho0: 1, u0: 1, p0: 1e-6 };
    const prof = await profileOf(sim);
    if (k === 1) {
      // Fold the two halves of the planar run onto |x|.
      prof.u = prof.u.map((u, i) => u * Math.sign(prof.r[i]));
      prof.r = prof.r.map(Math.abs);
    }
    const exact = (r: number) => nohExact(k, gamma, q, r, t);
    const rs = nohShockRadius(gamma, 1, t);
    const dx = 1 / N;
    const post = exact(0).rho;
    const preAtShock = exact(rs * 1.000001).rho;
    // Planar: sorted by |x| for the crossing, from the right half.
    const half: Profile = k === 1 ? { r: prof.r.slice(N / 2), rho: prof.rho.slice(N / 2), u: prof.u.slice(N / 2), p: prof.p.slice(N / 2) } : prof;
    const shock = outerCrossing(half, 0.5 * (post + preAtShock));
    // Gas that started beyond 1 - t came in through the outer face: measured only inside it.
    const reach = 1 - t - 2 * dx;
    return {
      post_shock_density: mean(prof, 'rho', (r) => r > 0.3 * rs && r < rs - 4 * dx),
      post_expected: post,
      post_shock_pressure: mean(prof, 'p', (r) => r > 0.3 * rs && r < rs - 4 * dx),
      post_pressure_expected: exact(0).p,
      pre_shock_density_l1: l1(prof, 'rho', exact, (r) => r > rs + 4 * dx && r < reach),
      pre_shock_velocity_l1: l1(prof, 'u', exact, (r) => r > rs + 4 * dx && r < reach),
      shock_radius: shock,
      shock_expected: rs,
      shock_error_cells: (shock - rs) / dx,
      center_density: prof.rho[k === 1 ? N / 2 : 0],
      steps,
    };
  });
}

export const EXACT_TESTS: TestDef[] = [
  {
    id: 'exact-riemann',
    name: 'ExactPack Riemann problems: Einfeldt 1-2-3, stationary contact, slow shock, shock-contact-shock, LeBlanc',
    criterion:
      'Each preset at 256^2 (2D) against the exact solution, along the middle row: relative L1 error (sum |run - exact| over sum |exact|) of density and of pressure under 0.025 (Einfeldt 1-2-3, its near-vacuum: measured 0.0095 and 0.015), 0.05 (stationary contact, whose shell is 6 times denser than the gas around it and a few cells wide: 0.038 and 0.009), 0.01 (slow shock: 0.0039 and 0.0046; shock-contact-shock: 0.0027 and 0.0026) and 0.02 (LeBlanc: 0.0085 and 0.0078); no floored cells',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      const limit: Record<string, number> = { 'riemann-einfeldt': 0.025, 'riemann-stationary-contact': 0.05, 'riemann-slow-shock': 0.01, 'riemann-shock-contact-shock': 0.01, 'riemann-leblanc': 0.02 };
      let pass = true;
      for (const tube of RIEMANN_TUBES) {
        const tag = tube.id.replace('riemann-', '').replace(/-/g, '_');
        const exact = exactRiemann(tube.left, tube.right, tube.eosParams.gamma!);
        const r = await withSim(gpu, tube, N, { ndim: 2 }, async (sim) => {
          const steps = await sim.runToEnd();
          const prof = await profileOf(sim);
          const d = await sim.diagnostics();
          const at = (x: number) => exact((x - tube.x0) / sim.cfg.tEnd);
          const all = () => true;
          return { rho: l1(prof, 'rho', at, all), p: l1(prof, 'p', at, all), u: l1(prof, 'u', at, all), floored: d.floored, steps };
        });
        metrics[`${tag}_rho_l1`] = r.rho;
        metrics[`${tag}_p_l1`] = r.p;
        metrics[`${tag}_u_l1`] = r.u;
        metrics[`${tag}_floored`] = r.floored;
        metrics[`${tag}_steps`] = r.steps;
        pass &&= r.rho < limit[tube.id] && r.p < limit[tube.id] && r.floored === 0;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'exact-noh',
    name: 'ExactPack Noh implosion: planar, cylindrical (R, z) and spherical (r, theta)',
    criterion:
      'Each preset (gamma = 5/3, unit inflow, t = 0.6) at 256^2: the post-shock density, averaged from 0.3 times the shock radius (outside the wall heating every code has at the center) to 4 cells inside it, within 1%, 3% and 6% of the exact 4, 16 and 64 (planar, cylindrical, spherical: measured 0.01%, 1.5%, 4.2%), and its pressure the same (0.01%, 1.2%, 3.5%); the shock within 1.5 cells of r = 0.2 (0.07, 0.6, 0.8); the pre-shock density, rising as (1 + t / r)^(k - 1), within 1e-3 in L1 where no gas from the outer face has arrived (2e-5)',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [id, k, tag, tol] of [['noh-planar', 1, 'planar', 0.01], ['noh-cylindrical', 2, 'cylindrical', 0.03], ['noh-spherical', 3, 'spherical', 0.06]] as const) {
        const m = await nohRun(gpu, id, k);
        for (const [key, v] of Object.entries(m)) metrics[`${tag}_${key}`] = v;
        pass &&=
          Math.abs(m.post_shock_density / m.post_expected - 1) < tol &&
          Math.abs(m.post_shock_pressure / m.post_pressure_expected - 1) < tol &&
          Math.abs(m.shock_error_cells) < 1.5 &&
          m.pre_shock_density_l1 < 1e-3;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'exact-noh-cartesian',
    name: 'ExactPack Noh implosion: cylindrical, on a 2D Cartesian grid',
    criterion:
      'The noh-cartesian preset (a box of side 2) at 512^2: gas falling toward the center from every direction in the plane (Liska and Wendroff 2003). The post-shock density, averaged over the ring 0.06 < r < 0.17, within 2% of 16 (measured 15.89); the shock radius along the axes and along the diagonal each within 1 cell of 0.2 (0.33 and 0.06 cells)',
    async run(gpu) {
      const n = 512;
      return withSim(gpu, problemById('noh-cartesian'), n, { ndim: 2 }, async (sim) => {
        const steps = await sim.runToEnd();
        const { u1 } = await sim.readState();
        const dx = sim.cfg.problem.box / n;
        const rho = (i: number, j: number) => u1[4 * (j * n + i)];
        let s = 0;
        let k = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const r = Math.hypot((i + 0.5 - n / 2) * dx, (j + 0.5 - n / 2) * dx);
            if (r > 0.06 && r < 0.17) (s += rho(i, j)), k++;
          }
        // Along a ray from the center: cells (c + a s, c + b s) for the axis (1, 0) and the diagonal (1, 1).
        const ray = (a: number, b: number): Profile => {
          const p: Profile = { r: [], rho: [], u: [], p: [] };
          // out to r = 0.35: beyond it, gas that came in through the faces
          for (let q = 0; (q + 0.5) * dx * Math.hypot(a, b) < 0.35; q++) {
            p.r.push((q + 0.5) * dx * Math.hypot(a, b));
            p.rho.push(rho(n / 2 + a * q, n / 2 + b * q));
          }
          return p;
        };
        const level = 0.5 * (16 + 4);
        const axis = [ray(1, 0), ray(0, 1)].map((p) => outerCrossing(p, level));
        const diag = [ray(1, 1)].map((p) => outerCrossing(p, level));
        const m = {
          post_shock_density: s / k, post_expected: 16,
          shock_along_axes: (axis[0] + axis[1]) / 2, shock_along_diagonal: diag[0], shock_expected: 0.2,
          axis_error_cells: ((axis[0] + axis[1]) / 2 - 0.2) / dx, diagonal_error_cells: (diag[0] - 0.2) / dx,
          center_density: rho(n / 2, n / 2), steps,
        };
        return { pass: Math.abs(m.post_shock_density / 16 - 1) < 0.02 && Math.abs(m.axis_error_cells) < 1 && Math.abs(m.diagonal_error_cells) < 1, metrics: m };
      });
    },
  },
  {
    id: 'exact-uniform-collapse',
    name: 'ExactPack uniform collapse (noh2): planar, cylindrical (R, z) and spherical (r, theta)',
    criterion:
      'Each preset (gamma = 5/3, u = -r, e = 1, t = 0.5, when the density is 2, 4 and 8 times its start) at 256^2: density, pressure and velocity within 0.5% of the exact solution in L1 (measured at most 2e-5 planar, 6e-5 cylindrical, 1e-3 spherical, its velocity), where the rarefaction from the open outer face has not arrived (it moves inward at the sound speed: by t = 0.5, 0.82, 0.93 and 1.05 in r / (1 - t), less 0.15 for its smeared front)',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [id, k, tag] of [['uniform-collapse-planar', 1, 'planar'], ['uniform-collapse-cylindrical', 2, 'cylindrical'], ['uniform-collapse-spherical', 3, 'spherical']] as const) {
        const m = await withSim(gpu, problemById(id), N, { ndim: 2 }, async (sim) => {
          const steps = await sim.runToEnd();
          const t = sim.cfg.tEnd;
          const gamma = sim.cfg.eosParams.gamma;
          const prof = await profileOf(sim);
          const exact = (r: number) => uniformCollapseExact(k, gamma, { rho0: 1, e0: 1, tau: 1 }, r, t);
          // How far in, in r / (1 - t), the open face's rarefaction has come: at the sound speed,
          // c0 (1 - t)^-a with a = (gamma - 1) k / 2, which is c0 / (1 - t)^(a + 1) in r / (1 - t).
          const a = ((gamma - 1) * k) / 2;
          const reach = (Math.sqrt(gamma * (gamma - 1)) / a) * ((1 - t) ** -a - 1);
          const outer = k === 1 ? sim.cfg.problem.box / 2 : prof.r[prof.r.length - 1];
          const valid = (outer - reach - 0.15) * (1 - t);
          // Planar: velocity is -x on both sides, which the exact solution's u(r) has as a signed r.
          const keep = (r: number) => Math.abs(r) < valid;
          return {
            rho_l1: l1(prof, 'rho', (r) => exact(Math.abs(r)), keep),
            p_l1: l1(prof, 'p', (r) => exact(Math.abs(r)), keep),
            u_l1: l1(prof, 'u', exact, keep),
            density: mean(prof, 'rho', keep), density_expected: exact(0).rho, compared_within: valid, steps,
          };
        });
        for (const [key, v] of Object.entries(m)) metrics[`${tag}_${key}`] = v;
        pass &&= m.rho_l1 < 0.005 && m.p_l1 < 0.005 && m.u_l1 < 0.005;
      }
      return { pass, metrics };
    },
  },
];
