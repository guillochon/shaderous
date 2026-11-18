// Validation of radiation transport (radiation/radiation.ts; plans/radiation-m1.md section 9).
// Registered in tests.ts after the others. All on the uniform grid in 2D, small enough to share
// the GPU.
import { all, OUTFLOW, PERIODIC, type Prim, type Problem } from '../problems/types';
import type { Sim, SimConfig } from '../sim';
import { withSim, type TestDef } from './tests';
import { capture, restartFrom } from '../io/capture';
import { configFor, Sim as SimClass } from '../sim';

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
async function runTo(sim: Sim, t: number) {
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
    id: 'rad-cooling-sphere',
    name: 'Radiation: a hot core shining through an opaque shell gives the same luminosity at chat and 2 chat',
    criterion: 'A core held at E = 1 inside a scattering shell (cell optical depth 5, so its speed of light is reduced by the per-cell rule) in transparent gas reaches a steady luminosity through the box faces that differs by under 5% between chat = c/20 and c/10; the shell is reduced (r < 1) in both',
    async run(gpu) {
      const m: Record<string, number> = {};
      const lum: number[] = [];
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
        // The shell's diffusion time is 0.1; ten of them.
        await withSim(gpu, p, n, D2, async (sim) => {
          await runTo(sim, 1);
          const d = await sim.diagnostics();
          lum.push(d.radiation!.luminosity);
          m[`chat${chat}_luminosity`] = d.radiation!.luminosity;
          m[`chat${chat}_reduced_share`] = d.radiation!.reducedShare;
          m[`chat${chat}_steps`] = d.steps;
        });
      }
      m.ratio = lum[1] / lum[0];
      return { pass: Math.abs(m.ratio - 1) < 0.05 && lum[0] > 0 && m.chat5_reduced_share > 0 && m.chat10_reduced_share > 0, metrics: m };
    },
  },
];

