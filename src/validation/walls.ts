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

/** A 3D tunnel (box 4, stream along x at Mach 0.1, Reynolds number 100 on a body of size 0.5) holding one obstacle shape. */
function tunnelSpec(shape: Record<string, unknown>, over: { n?: number; amr?: number; nu?: number; box?: number } = {}): SetupSpec {
  return {
    format: 'shaderous-setup', version: 1, id: 'tunnel-test', name: 'Tunnel test',
    domain: { ndim: 3, n: over.n ?? 64, amr: over.amr ?? 0, box: over.box ?? 4 },
    physics: { viscosity: over.nu ?? 0.005 },
    parts: [{ kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 }, { kind: 'obstacle', id: 'body', shape }],
  } as SetupSpec;
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
  {
    id: 'mesh-vs-shape',
    name: 'A sphere from a mesh against the built-in sphere (3D 64 and 128, tunnel, Reynolds number 100)',
    criterion:
      'A sphere of diameter 0.5 in a tunnel, once as the built-in shape and once as an icosphere of 5120 triangles (geometry/builtin.ts) through the mesh path (distance field, placement): the solid cells number the same to 3%, and after t = 2 the drag differs by under 3%, at 8 and at 16 cells across. Measured 2026-11-20: the same cells, and the same drag to the bit',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const n of [64, 128]) {
        const solids: number[] = [];
        const drag: number[] = [];
        for (const shape of [{ type: 'sphere', center: [-1, 0, 0], radius: 0.25 }, { type: 'mesh', file: 'builtin:sphere', fit: 0.125, center: [-1, 0, 0] }]) {
          const problem = compileSetup(tunnelSpec(shape, { n })).problem;
          const sim = await Sim.create(gpu, configFor(problem, n, { ndim: 3, tEnd: 2, ...compileSetup(tunnelSpec(shape, { n })).overrides }));
          try {
            const { u2 } = await sim.readState();
            let k = 0;
            for (let i = 3; i < u2.length; i += 4) if (u2[i] > 0.5) k++;
            solids.push(k);
            await sim.runToEnd(64, 1e6);
            drag.push((await sim.diagnostics()).bodies![0].force[0]);
          } finally {
            sim.destroy();
          }
        }
        m[`n${n}_solid_shape`] = solids[0];
        m[`n${n}_solid_mesh`] = solids[1];
        m[`n${n}_drag_shape`] = drag[0];
        m[`n${n}_drag_mesh`] = drag[1];
        const ds = Math.abs(solids[1] / solids[0] - 1);
        const dd = Math.abs(drag[1] / drag[0] - 1);
        pass &&= ds < 0.03 && dd < 0.03;
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'rotor-frame',
    name: 'A paddle in its own frame: the explicit and the implicit scheme give the same torque (3D 64, liquid, tunnel with a rotor frame)',
    criterion:
      'Two flat blades (boxes 1.5 long, 0.3 wide, 0.1 thick, pitched at +-45 degrees about their spans) about the stream’s axis, in the frame turning at Omega = 2 with the stream at U = 1 (Mach 0.1, nu = 0.01): the torque about the axis, averaged over t = 1.5 to 3, from the explicit scheme and from implicit pressure agree to 5%, and is not zero. Measured 2026-11-20: -3.99 and -4.06 (1.7%)',
    async run(gpu: Gpu) {
      const spec: SetupSpec = {
        format: 'shaderous-setup', version: 1, id: 'rotor-test', name: 'Rotor test',
        domain: { ndim: 3, n: 64, amr: 0, box: 4 },
        physics: { viscosity: 0.01 },
        parts: [
          { kind: 'tunnel', speed: 1, axis: 'x', sponge: 0.15, mach: 0.1, rotor: 2 },
          { kind: 'obstacle', id: 'blade 1', shape: { type: 'box', min: [-0.05, 0.1, -0.15], max: [0.05, 1.6, 0.15], rotate: [0, 45, 0] } },
          { kind: 'obstacle', id: 'blade 2', shape: { type: 'box', min: [-0.05, -1.6, -0.15], max: [0.05, -0.1, 0.15], rotate: [0, -45, 0] } },
        ],
      } as SetupSpec;
      const compiled = compileSetup(spec);
      const torque = async (pressure: 'explicit' | 'implicit') => {
        const sim = await Sim.create(gpu, configFor(compiled.problem, 64, { ndim: 3, tEnd: 1.5, ...compiled.overrides, pressure }));
        try {
          await sim.runToEnd(64, 1e7);
          let sum = 0;
          let k = 0;
          for (let t = 1.6; t <= 3.001; t += 0.1) {
            sim.cfg.tEnd = t;
            sim.writeParams();
            await sim.runToEnd(64, 1e7);
            const d = await sim.diagnostics();
            // About the x axis through the box's center: each blade's torque about its own center, plus its center's lever.
            for (const b of d.bodies!) {
              const c = compiled.problem.bodies!.find((x) => x.id === b.id)!.center;
              sum += b.torque[0] + (c[1] * b.force[2] - c[2] * b.force[1]);
            }
            k++;
          }
          return sum / k;
        } finally {
          sim.destroy();
        }
      };
      const ex = await torque('explicit');
      const im = await torque('implicit');
      const m = { torque_explicit: ex, torque_implicit: im, difference: Math.abs(im / ex - 1) };
      return { pass: m.difference < 0.05 && Math.abs(ex) > 1e-3, metrics: m };
    },
  },
];
