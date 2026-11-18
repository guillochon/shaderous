// Validation tests for the shearing box (plans/coordinate-systems-plan.md, Part D). Registered
// through validation/frames.ts.
import { all, PERIODIC, type Prim, type Problem } from '../problems/types';
import type { Sim } from '../sim';
import { withSim, type TestDef } from './tests';

const OMEGA = 1;
const Q = 1.5;

/** A 2D shearing sheet (Keplerian, Omega = 1, box 1) whose state is the background flow v_y = -q Omega x plus `extra`. */
function sheet(tEnd: number, extra: (x: number, y: number) => Partial<Prim> = () => ({})): Problem {
  return {
    id: 'shearing-sheet', label: 'Shearing sheet', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
    bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 0, box: 1, tEnd, supports2d: true,
    shear: { omega: OMEGA, q: Q },
    init: () => (x, y) => ({ rho: 1, vx: 0, vy: -Q * OMEGA * x, vz: 0, p: 1, s: 0, ...extra(x, y) }),
  };
}

/** Largest departure of the velocity from the background flow, over the grid. */
async function departure(sim: Sim): Promise<{ dv: number; drho: number }> {
  const { u1 } = await sim.readState();
  const n = sim.cfg.n;
  let dv = 0;
  let drho = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const o = 4 * (j * n + i);
      const x = (i + 0.5) / n - 0.5;
      dv = Math.max(dv, Math.abs(u1[o + 1] / u1[o]), Math.abs(u1[o + 2] / u1[o] + Q * OMEGA * x));
      drho = Math.max(drho, Math.abs(u1[o] - 1));
    }
  return { dv, drho };
}

export const SHEAR_TESTS: TestDef[] = [
  {
    id: 'shear-steady',
    name: 'Shearing box: the background flow is steady',
    criterion: 'Uniform gas in a Keplerian shearing sheet (2D, 64^2), moving with the background flow v_y = -q Omega x: over 10 orbits (16,700 steps) the velocity stays within 2e-3 of the shear speed at the box edge and the density within 1e-3 of 1, and the mass is kept to 1e-6. (What departs is float32 rounding of the Coriolis and tidal increments to a velocity of order the shear speed, the same every step in a steady state, so it adds up: about 1e-4 of the shear speed an orbit.)',
    async run(gpu) {
      return withSim(gpu, sheet(20 * Math.PI), 64, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d = await departure(sim);
        const d1 = await sim.diagnostics();
        const edge = Q * OMEGA * 0.5;
        return {
          pass: d.dv / edge < 2e-3 && d.drho < 1e-3 && Math.abs(d1.mass / d0.mass - 1) < 1e-6,
          metrics: { velocity_departure: d.dv / edge, density_departure: d.drho, mass_change: d1.mass / d0.mass - 1, steps: d1.steps },
        };
      });
    },
  },
  {
    id: 'shear-mass',
    name: 'Shearing box: a dense blob across the shearing boundary',
    criterion: 'A blob of twice the density straddling the x faces of a Keplerian sheet (2D, 64^2), carried along y by the shear for two orbits: its mass (the passive scalar) and the total kept to 1e-6, x momentum to 1e-6 of the gas’s mass times the shear speed, and the density within [0.9, 2.1] (its edges smeared, nothing made at the boundary)',
    async run(gpu) {
      const blob = (x: number, y: number) => {
        const dx = Math.abs(x) - 0.5;
        return Math.hypot(dx, y) < 0.15 ? { rho: 2, s: 1 } : {};
      };
      return withSim(gpu, sheet(4 * Math.PI, blob), 64, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const { u2: s0 } = await sim.readState();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const { u1, u2 } = await sim.readState();
        let m0 = 0;
        let m1 = 0;
        for (let q = 0; q < s0.length; q += 4) {
          m0 += s0[q + 2];
          m1 += u2[q + 2];
        }
        let lo = Infinity;
        let hi = -Infinity;
        for (let q = 0; q < u1.length; q += 4) {
          lo = Math.min(lo, u1[q]);
          hi = Math.max(hi, u1[q]);
        }
        const scale = d0.mass * Q * OMEGA * 0.5;
        const m = {
          mass_change: Math.abs(d1.mass / d0.mass - 1),
          blob_change: Math.abs(m1 / m0 - 1),
          x_momentum: Math.abs(d1.momentum[0] - d0.momentum[0]) / scale,
          density_min: lo, density_max: hi,
        };
        return { pass: m.mass_change < 1e-6 && m.blob_change < 1e-6 && m.x_momentum < 1e-6 && lo > 0.9 && hi < 2.1, metrics: m };
      });
    },
  },
  {
    id: 'shear-epicycle',
    name: 'Shearing box: epicycles',
    criterion: 'A uniform kick of 0.01 along x on the background flow of a Keplerian sheet (2D, 32^2) oscillates at the epicyclic frequency kappa = sqrt(2 (2 - q)) Omega: after one epicycle the kick is back to within 1e-3 of itself in phase and amplitude, and the epicyclic energy (dv_x^2 + 2 dv_y^2 / (2 - q)) / 2 keeps to 1e-4 throughout',
    async run(gpu) {
      const kappa = Math.sqrt(2 * (2 - Q)) * OMEGA;
      const period = (2 * Math.PI) / kappa;
      const kick = 0.01;
      return withSim(gpu, sheet(0, (x) => ({ vx: kick, vy: -Q * OMEGA * x })), 32, { ndim: 2 }, async (sim) => {
        const energy = (d: { momentum: number[]; mass: number }) => {
          const vx = d.momentum[0] / d.mass;
          const vy = d.momentum[1] / d.mass;
          return 0.5 * (vx * vx + (2 / (2 - Q)) * vy * vy);
        };
        const d0 = await sim.diagnostics();
        const e0 = energy(d0);
        let worst = 0;
        let last = d0;
        for (let i = 1; i <= 8; i++) {
          sim.cfg.tEnd = (period * i) / 8;
          sim.writeParams();
          await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(energy(last) / e0 - 1));
          // Where it should be: dv_x = kick cos(kappa t), dv_y = -kick (2 - q) Omega / kappa sin(kappa t).
        }
        const vx = last.momentum[0] / last.mass;
        const vy = last.momentum[1] / last.mass;
        const off = Math.hypot(vx - kick, vy) / kick;
        return { pass: off < 1e-3 && worst < 1e-4, metrics: { return_offset: off, energy_drift: worst, vx, vy, kappa } };
      });
    },
  },
];
