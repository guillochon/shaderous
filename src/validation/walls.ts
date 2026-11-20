// Validation of the forces on bodies (plans/wind-tunnel.md 4.2 and 11.2; hydro/forces.ts).
import type { Gpu } from '../gpu/device';
import { configFor, Sim, type Diagnostics, type SimConfig } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { compileSetup } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';
import type { TestDef } from './tests';

/**
 * A periodic box of liquid or gas at rest, a cylinder off center, and a slab of it moving
 * toward the cylinder at an angle: nothing enters or leaves the box, so the only thing that
 * changes the gas's momentum is the body.
 */
function balanceSpec(gas: boolean): SetupSpec {
  return {
    format: 'shaderous-setup', version: 1, id: 'force-balance', name: 'Force balance',
    domain: { ndim: 2, n: 128, amr: 0, box: 1, supports2d: true, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } },
    physics: gas
      ? { eos: 'ideal', eosParams: { gamma: 1.4 }, gravity: 'off', G: 0, viscosity: 0.002 }
      : { eos: 'liquid', eosParams: { liqRho: 1, liqC: 5 }, gravity: 'off', G: 0, riemann: 'hllc', viscosity: 0.002 },
    background: { rho: 1, p: gas ? 1 : 0, v: [0, 0, 0] },
    parts: [
      { kind: 'obstacle', id: 'cylinder', shape: { type: 'sphere', center: [0.1, 0.05, 0], radius: 0.08 } },
      { kind: 'region', id: 'slab', shape: { type: 'box', min: [-0.4, -0.15, null], max: [0, 0.2, null] }, v: [gas ? 0.3 : 1, 0.2, 0] },
    ],
  } as SetupSpec;
}

interface Balance {
  /** the change of the gas's momentum along x and y over the steps */
  dp: [number, number];
  /** the impulse the wall faces gave the body over them, sum of F dt */
  impulse: [number, number];
  steps: number;
}

/** Step one at a time, summing the force on the body times each step's dt. */
async function balance(sim: Sim | AmrSim, steps: number): Promise<Balance> {
  const step = async (): Promise<Diagnostics> => {
    const before = (await sim.diagnostics()).steps;
    for (let tries = 0; tries < 1000; tries++) {
      if (sim instanceof Sim) sim.step(1);
      else sim.step(1);
      const d = await sim.diagnostics();
      if (d.steps > before) return d;
    }
    throw new Error('the run took no step');
  };
  const d0 = await sim.diagnostics();
  const impulse: [number, number] = [0, 0];
  let d = d0;
  for (let i = 0; i < steps; i++) {
    d = await step();
    const f = d.bodies![0].force;
    impulse[0] += f[0] * d.dt;
    impulse[1] += f[1] * d.dt;
  }
  return { dp: [d.momentum[0] - d0.momentum[0], d.momentum[1] - d0.momentum[1]], impulse, steps: d.steps };
}

export const WALL_TESTS: TestDef[] = [
  {
    id: 'force-balance',
    name: 'Forces on bodies: the wall faces against the momentum balance of the whole box (2D 128, liquid and ideal gas, explicit and implicit pressure, uniform grid and 2 levels of blocks)',
    criterion:
      'A periodic box with a cylinder and a slab of fluid moving toward it, viscosity on: over 30 steps the gas\'s momentum changes by minus the impulse the force diagnostic gives the body (the sum of force times dt), to 1e-4 of that impulse, along x and y, in each of the eight runs (liquid, ideal gas; explicit, implicit; uniform, blocks). Measured 2026-11-20: 3e-7 to 5e-6, round-off',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const gas of [false, true]) {
        const problem = compileSetup(balanceSpec(gas)).problem;
        for (const implicit of [false, true]) {
          for (const amr of [false, true]) {
            const over: Partial<SimConfig> = { ndim: 2, tEnd: 1e9, ...(implicit ? { pressure: 'implicit' } : {}), ...(amr ? { amrLevels: 2 } : {}) };
            const cfg = configFor(problem, 128, over);
            const sim = amr ? await AmrSim.create(gpu, cfg) : await Sim.create(gpu, cfg);
            try {
              const b = await balance(sim, 30);
              const key = `${gas ? 'gas' : 'liquid'}_${implicit ? 'implicit' : 'explicit'}_${amr ? 'blocks' : 'uniform'}`;
              const size = Math.hypot(...b.impulse);
              const off = Math.hypot(b.dp[0] + b.impulse[0], b.dp[1] + b.impulse[1]) / size;
              m[`${key}_impulse_x`] = b.impulse[0];
              m[`${key}_impulse_y`] = b.impulse[1];
              m[`${key}_mismatch`] = off;
              pass &&= off < 1e-4 && size > 0;
            } finally {
              sim.destroy();
            }
          }
        }
      }
      return { pass, metrics: m };
    },
  },
];
