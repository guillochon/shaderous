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
    id: 'force-balance-moving',
    name: 'Forces on a moving body: its wall faces and the cells it crosses against the momentum balance of the box (2D 256^2, liquid, explicit and implicit pressure)',
    criterion:
      'A cylinder 0.25 across sliding at -1 through a periodic box of liquid at rest (Mach 0.1, nu = 0.0025), over 200 steps: the change of the liquid\u2019s momentum is minus the impulse the force diagnostic gives the body (its wall faces\u2019 fluxes and the momentum the exchange hands over where its cells change kind), to 1e-3 of that impulse, with either scheme',
    async run(gpu: Gpu) {
      const spec = {
        format: 'shaderous-setup', version: 1, id: 'moving-balance', name: 'Moving balance',
        domain: { ndim: 2, n: 256, amr: 0, box: 4, supports2d: true, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } },
        physics: { eos: 'liquid', eosParams: { liqRho: 1, liqC: 10 }, gravity: 'off', riemann: 'hllc', viscosity: 0.0025 },
        background: { rho: 1, p: 0, v: [0, 0, 0] },
        parts: [{ kind: 'obstacle', id: 'cylinder', shape: { type: 'sphere', center: [1, 0.01, 0], radius: 0.125 }, motion: { slide: { velocity: [-1, 0, 0] } } }],
      } as unknown as SetupSpec;
      const c = compileSetup(spec);
      const m: Record<string, number> = {};
      let pass = true;
      for (const pressure of ['explicit', 'implicit'] as const) {
        const sim = await Sim.create(gpu, configFor(c.problem, 256, { ndim: 2, tEnd: 1e9, ...c.overrides, pressure }));
        try {
          const b = await balance(sim, 200);
          const size = Math.hypot(...b.impulse);
          const off = Math.hypot(b.dp[0] + b.impulse[0], b.dp[1] + b.impulse[1]) / size;
          m[`${pressure}_impulse_x`] = b.impulse[0];
          m[`${pressure}_momentum_change_x`] = b.dp[0];
          m[`${pressure}_mismatch`] = off;
          pass &&= off < 1e-3;
        } finally {
          sim.destroy();
        }
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'cylinder-forces',
    name: 'A cylinder in a tunnel at Reynolds number 100: drag, lift and shedding (2D, 16 and 32 cells across, liquid, implicit pressure)',
    criterion:
      'A cylinder of diameter 0.25 in a tunnel 32 diameters long and wide (slip sides: 3% blockage), Mach 0.1, nu = 0.0025, to t = 60, statistics from t = 35. The staircase wall converges at first order, so the drag is extrapolated from 16 and 32 cells across, 2 Cd(32) - Cd(16): within 1.30 to 1.45 (published, unbounded: 1.32 to 1.35, Park et al. 1998, Posdziech & Grundmann 2007, Liu et al. 1998; the blockage adds a few percent). At 32 cells across the Strouhal number is within 0.160 to 0.178 (published 0.164 to 0.165) and the lift amplitude within 0.25 to 0.50 (published about 0.33; measured 0.43). Measured 2026-11-20: Cd 1.548 and 1.463 (1.378 extrapolated), St 0.168 and 0.171, lift amplitude 0.48 and 0.43',
    async run(gpu: Gpu) {
      const spec = (n: number): SetupSpec => ({
        format: 'shaderous-setup', version: 1, id: 'tunnel-cylinder', name: 'Cylinder in a tunnel',
        domain: { ndim: 2, n, amr: 0, box: 8, supports2d: true },
        physics: { viscosity: 0.0025 },
        parts: [
          { kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 },
          { kind: 'obstacle', id: 'cylinder', shape: { type: 'sphere', center: [-2, 0.01, 0], radius: 0.125 } },
        ],
      }) as SetupSpec;
      const measure = async (n: number) => {
        const c = compileSetup(spec(n));
        const sim = await Sim.create(gpu, configFor(c.problem, n, { ndim: 2, tEnd: 35, ...c.overrides }));
        try {
          await sim.runToEnd(64, 1e7);
          const t: number[] = [];
          const cd: number[] = [];
          const cl: number[] = [];
          for (let k = 1; k <= 500; k++) {
            sim.cfg.tEnd = 35 + 0.05 * k;
            sim.writeParams();
            await sim.runToEnd(64, 1e7);
            const d = await sim.diagnostics();
            t.push(d.t);
            cd.push(d.bodies![0].coefficients!.drag);
            cl.push(d.bodies![0].coefficients!.lift);
          }
          const mean = cl.reduce((a, b) => a + b, 0) / cl.length;
          const ups: number[] = [];
          for (let i = 1; i < t.length; i++) if (cl[i - 1] < mean && cl[i] >= mean) ups.push(t[i - 1] + ((mean - cl[i - 1]) / (cl[i] - cl[i - 1])) * (t[i] - t[i - 1]));
          const period = (ups[ups.length - 1] - ups[0]) / (ups.length - 1);
          return { cd: cd.reduce((a, b) => a + b, 0) / cd.length, amp: (Math.max(...cl) - Math.min(...cl)) / 2, st: 0.25 / period };
        } finally {
          sim.destroy();
        }
      };
      const a = await measure(512);
      const b = await measure(1024);
      const cdInf = 2 * b.cd - a.cd;
      const m = { cd_16: a.cd, cd_32: b.cd, cd_extrapolated: cdInf, st_16: a.st, st_32: b.st, lift_amplitude_16: a.amp, lift_amplitude_32: b.amp };
      return { pass: cdInf > 1.3 && cdInf < 1.45 && b.st > 0.16 && b.st < 0.178 && b.amp > 0.25 && b.amp < 0.5, metrics: m };
    },
  },
  {
    id: 'sphere-drag',
    name: 'A sphere in a tunnel at Reynolds number 100: its drag (3D, 16 cells across, 2 levels of blocks, liquid, implicit pressure)',
    criterion:
      'A sphere of diameter 0.25 in a tunnel 16 diameters across (slip sides: 1.2% blockage), Mach 0.1, nu = 0.0025, 256 finest cells with 2 levels: the mean drag coefficient over t = 10 to 12 is within 0.95 to 1.35 (published 1.08 to 1.09: Johnson & Patel 1999, Fornberg 1988; a staircase at 16 cells across reads high, as the cylinder’s does). Measured 2026-11-20: 1.253 (the runner’s run to t = 20: 1.158 over t = 15 to 20)',
    async run(gpu: Gpu) {
      const spec = {
        format: 'shaderous-setup', version: 1, id: 'tunnel-sphere', name: 'Sphere in a tunnel',
        domain: { ndim: 3, n: 256, amr: 2, box: 4 },
        physics: { viscosity: 0.0025 },
        parts: [
          { kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 },
          { kind: 'obstacle', id: 'sphere', shape: { type: 'sphere', center: [-1, 0, 0], radius: 0.125 } },
        ],
      } as SetupSpec;
      const c = compileSetup(spec);
      const sim = await AmrSim.create(gpu, configFor(c.problem, 256, { ndim: 3, amrLevels: 2, tEnd: 10, ...c.overrides }));
      try {
        await sim.runToEnd(16, 1e7);
        const cd: number[] = [];
        for (let k = 1; k <= 20; k++) {
          sim.cfg.tEnd = 10 + 0.1 * k;
          sim.writeParams();
          await sim.runToEnd(16, 1e7);
          cd.push((await sim.diagnostics()).bodies![0].coefficients!.drag);
        }
        const mean = cd.reduce((a, b) => a + b, 0) / cd.length;
        const m = { cd: mean, cd_min: Math.min(...cd), cd_max: Math.max(...cd), leaves: sim.tree.leaves().length };
        return { pass: mean > 0.95 && mean < 1.35, metrics: m };
      } finally {
        sim.destroy();
      }
    },
  },
  {
    id: 'piston',
    name: 'A moving wall: a piston in a closed box of ideal gas (2D 64^2, explicit)',
    criterion:
      'A slab across the box slides at 0.1 (Mach 0.085) for t = 2, compressing the gas ahead of it from 0.85 to 0.65 of the box and expanding the gas behind it: the gas’s mass is conserved to 1e-6, the mean pressure ahead of it is within 5% of the adiabat (0.85 / 0.65)^1.4, and the energy the gas gained equals the work the piston did on it (minus the sum over steps of the force on the body times its velocity times dt) to 5%. Measured 2026-11-20: mass to the bit, pressure 1.443 against 1.456, energy 0.18745 against work 0.18735',
    async run(gpu: Gpu) {
      const spec = {
        format: 'shaderous-setup', version: 1, id: 'piston', name: 'Piston',
        domain: { ndim: 2, n: 64, amr: 0, box: 1, supports2d: true, bc: { lo: [2, 2, 0], hi: [2, 2, 0] } },
        physics: { eos: 'ideal', eosParams: { gamma: 1.4 }, gravity: 'off' },
        background: { rho: 1, p: 1, v: [0, 0, 0] },
        parts: [{ kind: 'obstacle', id: 'piston', shape: { type: 'box', min: [-0.45, null, null], max: [-0.35, null, null] }, motion: { slide: { velocity: [0.1, 0, 0] } } }],
      } as unknown as SetupSpec;
      const c = compileSetup(spec);
      const sim = await Sim.create(gpu, configFor(c.problem, 64, { ndim: 2, tEnd: 2, ...c.overrides }));
      try {
        const d0 = await sim.diagnostics();
        let work = 0;
        for (let k = 0; k < 100000; k++) {
          sim.step(1);
          const d = await sim.diagnostics();
          work -= d.bodies![0].force[0] * 0.1 * d.dt;
          if (d.t >= 2 - 1e-6) break;
        }
        const d = await sim.diagnostics();
        const { u1, u2 } = await sim.readState();
        // The gas ahead of the piston (its front at x = -0.35 + 0.2 = -0.15), a cell clear of it.
        const n = 64;
        let ps = 0;
        let cells = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const x = (i + 0.5) / n - 0.5;
            const o = 4 * (j * n + i);
            if (x > -0.15 + 2 / n && u2[o + 3] === 0) {
              ps += 0.4 * u2[o + 1];
              cells++;
            }
          }
        void u1;
        const adiabat = (0.85 / 0.65) ** 1.4;
        const m = {
          mass_drift: Math.abs(d.mass / d0.mass - 1), pressure_ahead: ps / cells, adiabat, energy_gained: d.eTot - d0.eTot, work, steps: d.steps,
        };
        const pass = m.mass_drift < 1e-6 && Math.abs(m.pressure_ahead / adiabat - 1) < 0.05 && Math.abs(m.energy_gained / work - 1) < 0.05;
        return { pass, metrics: m };
      } finally {
        sim.destroy();
      }
    },
  },
  {
    id: 'galilean-drag',
    name: 'A body moving through still liquid against liquid streaming past it: the same drag (2D 512^2, cylinder, Reynolds number 100, explicit and implicit pressure)',
    criterion:
      'A cylinder of diameter 0.25 (16 cells across) in a periodic box of liquid 8 across (Mach 0.1, nu = 0.0025): at rest with the liquid at U = 1, and sliding at -1 through the liquid at rest. Over t = 1 to 3 the mean drag agrees to 5% with the explicit scheme; the moving body’s drag is noisier (each cell it crosses a small piston stroke): the rms of the difference of the two histories is reported. With implicit pressure (gas fractions near the body: plans/wind-tunnel.md 8.1’s cheap form) the same is measured and reported, not judged. Fails, 2026-11-20 (plans/wind-tunnel.md 6.7): explicit C_d 1.317 at rest, 2.459 moving; implicit 1.348 and 3.565: the staircase’s jumps, which the full 8.1 is to remove',
    async run(gpu: Gpu) {
      const spec = (moving: boolean): SetupSpec => ({
        format: 'shaderous-setup', version: 1, id: 'galilean', name: 'Galilean drag',
        domain: { ndim: 2, n: 512, amr: 0, box: 8, supports2d: true, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } },
        physics: { eos: 'liquid', eosParams: { liqRho: 1, liqC: 10 }, gravity: 'off', riemann: 'hllc', viscosity: 0.0025 },
        background: { rho: 1, p: 0, v: [moving ? 0 : 1, 0, 0] },
        parts: [{ kind: 'obstacle', id: 'cylinder', shape: { type: 'sphere', center: [moving ? 2 : 0, 0.01, 0], radius: 0.125 }, ...(moving ? { motion: { slide: { velocity: [-1, 0, 0] } } } : {}) }],
      }) as unknown as SetupSpec;
      const history = async (moving: boolean, pressure: 'explicit' | 'implicit') => {
        const c = compileSetup(spec(moving));
        const sim = await Sim.create(gpu, configFor(c.problem, 512, { ndim: 2, tEnd: 1, ...c.overrides, pressure }));
        try {
          await sim.runToEnd(64, 1e7);
          const f: number[] = [];
          for (let k = 1; k <= 200; k++) {
            sim.cfg.tEnd = 1 + 0.01 * k;
            sim.writeParams();
            await sim.runToEnd(64, 1e7);
            f.push((await sim.diagnostics()).bodies![0].force[0]);
          }
          return f;
        } finally {
          sim.destroy();
        }
      };
      const m: Record<string, number> = {};
      let pass = true;
      for (const pressure of ['explicit', 'implicit'] as const) {
        const a = await history(false, pressure);
        const b = await history(true, pressure);
        const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;
        const diff = Math.abs(mean(b) / mean(a) - 1);
        const rms = Math.sqrt(mean(a.map((v, i) => (b[i] - v) ** 2))) / Math.abs(mean(a));
        m[`${pressure}_cd_rest`] = mean(a) / 0.125;
        m[`${pressure}_cd_moving`] = mean(b) / 0.125;
        m[`${pressure}_mean_difference`] = diff;
        m[`${pressure}_rms_difference_over_mean`] = rms;
        if (pressure === 'explicit') pass &&= diff < 0.05;
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
