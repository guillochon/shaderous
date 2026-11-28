// Validation of partly filled cells on the GPU (plans/wind-tunnel.md 8.1; hydro/cut.ts), against
// the float64 prototype (src/reference/cutcell.ts) where they share a problem.
import type { Gpu } from '../gpu/device';
import { configFor, Sim, type SimConfig } from '../sim';
import { compileSetup } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';
import type { TestDef } from './tests';
import { preset } from '../presets/index';

/** A periodic box of liquid with a cylinder (a sphere in 3D), and a slab of it moving toward the body. */
function boxSpec(o: { n: number; box: number; radius: number; center: [number, number]; v: number; c: number; nu: number; slab?: boolean; ndim?: 1 | 2 | 3 }): SetupSpec {
  return {
    format: 'shaderous-setup', version: 1, id: 'cut-test', name: 'Cut cells',
    domain: { ndim: o.ndim ?? 2, n: o.n, amr: 0, box: o.box, supports2d: true, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } },
    physics: { eos: 'liquid', eosParams: { liqRho: 1, liqC: o.c }, gravity: 'off', G: 0, riemann: 'hllc', viscosity: o.nu },
    background: { rho: 1, p: 0, v: [o.v, 0, 0] },
    parts: [
      { kind: 'obstacle', id: 'body', shape: { type: 'sphere', center: [...o.center, 0], radius: o.radius } },
      ...(o.slab ? [{ kind: 'region', id: 'slab', shape: { type: 'box', min: [-0.4, -0.15, null], max: [0, 0.2, null] }, v: [1, 0.2, 0.1] }] : []),
    ],
  } as unknown as SetupSpec;
}

async function makeSim(gpu: Gpu, spec: SetupSpec, n: number, over: Partial<SimConfig>): Promise<Sim> {
  const c = compileSetup(spec);
  return Sim.create(gpu, configFor(c.problem, n, { ndim: spec.domain?.ndim ?? 2, tEnd: 1e9, ...c.overrides, pressure: 'implicit', ...over }));
}

/** A cube texture of rgba32float read back slab by slab (a whole 256^3 one is past one buffer's size). */
async function readSlabs(device: GPUDevice, tex: GPUTexture, n: number): Promise<Float32Array> {
  const out = new Float32Array(n * n * n * 4);
  const slab = 32;
  const row = n * 16;
  for (let z0 = 0; z0 < n; z0 += slab) {
    const buf = device.createBuffer({ size: row * n * slab, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex, origin: [0, 0, z0] }, { buffer: buf, bytesPerRow: row, rowsPerImage: n }, [n, n, slab]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    out.set(new Float32Array(buf.getMappedRange()), z0 * n * n * 4);
    buf.unmap();
    buf.destroy();
  }
  return out;
}

/**
 * The body's mean force over t0 to t1, stepping freely from t0 (where the run is stopped): each
 * step's force weighed by its dt. A stop lands on its time with a short step, whose force (the
 * pressure that enforces the step's volume over a short dt) is not the flow's: read at stops,
 * a rotor's torque came out up to four times too large (plans/wind-tunnel.md 8.1.4).
 */
async function meanForce(sim: Sim, t1: number): Promise<{ force: number[]; pressure: number[] }> {
  sim.cfg.tEnd = 1e9;
  sim.writeParams();
  const f = [0, 0, 0], p = [0, 0, 0];
  let w = 0;
  for (;;) {
    sim.step(1);
    const d = await sim.diagnostics();
    const b = d.bodies![0];
    for (let a = 0; a < 3; a++) {
      f[a] += b.force[a] * d.dt;
      p[a] += b.pressure[a] * d.dt;
    }
    w += d.dt;
    if (d.t >= t1) break;
  }
  return { force: f.map((x) => x / w), pressure: p.map((x) => x / w) };
}

/** Take exactly one step (the run may hold a step at a stop). */
async function oneStep(sim: Sim) {
  const before = (await sim.diagnostics()).steps;
  for (let k = 0; k < 100; k++) {
    sim.step(1);
    const d = await sim.diagnostics();
    if (d.steps > before) return d;
  }
  throw new Error('the run took no step');
}

export const CUT_TESTS: TestDef[] = [
  {
    id: 'cut-rest',
    name: 'Partly filled cells: liquid at rest around a cylinder (2D 128) and a sphere (3D 64) stays at rest (implicit pressure)',
    criterion: 'A periodic box of liquid at rest (c = 5, nu = 0.002) with a body cut into the cells: after 50 steps no cell moves faster than 1e-5, the mass changes by under 1e-6 of itself, and the force on the body is under 1e-6, in 2D and 3D',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const ndim of [2, 3] as const) {
        const n = ndim === 2 ? 128 : 64;
        const sim = await makeSim(gpu, boxSpec({ n, box: 1, radius: 0.13, center: [0.1, 0.05], v: 0, c: 5, nu: 0.002, ndim }), n, { cutCells: true });
        try {
          const d0 = await sim.diagnostics();
          let d = d0;
          for (let k = 0; k < 50; k++) d = await oneStep(sim);
          const { u1, u2 } = await sim.readState();
          let v = 0;
          for (let q = 0; q < u1.length; q += 4) if (u2[q + 3] < 0.5 && u1[q] > 0) v = Math.max(v, Math.hypot(u1[q + 1], u1[q + 2], u1[q + 3]) / u1[q]);
          const f = Math.hypot(...d.bodies![0].force);
          m[`d${ndim}_max_speed`] = v;
          m[`d${ndim}_mass_change`] = Math.abs(d.mass / d0.mass - 1);
          m[`d${ndim}_force`] = f;
          pass &&= v < 1e-5 && m[`d${ndim}_mass_change`] < 1e-6 && f < 1e-6;
        } finally {
          sim.destroy();
        }
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'cut-balance',
    name: 'Partly filled cells: the force on a cylinder (2D 128) and a sphere (3D 64) against the momentum balance of the whole box (liquid, implicit pressure)',
    criterion: 'As force-balance, with the body cut into the cells: over 30 steps the liquid’s momentum changes by minus the impulse the force diagnostic gives the body, to 1e-4 of that impulse; the mass is kept to 1e-6; in 2D and 3D',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      for (const ndim of [2, 3] as const) {
        const n = ndim === 2 ? 128 : 64;
        const sim = await makeSim(gpu, boxSpec({ n, box: 1, radius: 0.13, center: [0.1, 0.05], v: 0, c: 5, nu: 0.002, slab: true, ndim }), n, { cutCells: true });
        try {
          const d0 = await sim.diagnostics();
          const imp = [0, 0, 0];
          let d = d0;
          for (let k = 0; k < 30; k++) {
            d = await oneStep(sim);
            for (let a = 0; a < 3; a++) imp[a] += d.bodies![0].force[a] * d.dt;
          }
          const size = Math.hypot(...imp);
          const off = Math.hypot(...[0, 1, 2].map((a) => d.momentum[a] - d0.momentum[a] + imp[a])) / size;
          m[`d${ndim}_impulse_x`] = imp[0];
          m[`d${ndim}_mismatch`] = off;
          m[`d${ndim}_mass_change`] = Math.abs(d.mass / d0.mass - 1);
          pass &&= off < 1e-4 && size > 0 && m[`d${ndim}_mass_change`] < 1e-6;
        } finally {
          sim.destroy();
        }
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'cut-galilean-rest',
    name: 'Partly filled cells: a cylinder at rest in a stream against the float64 prototype (2D 256, 16 cells across, Re 100)',
    criterion: 'The prototype’s Galilean problem with the body at rest (periodic box 4, cylinder 0.25 across, U = 1, c = 10, nu = 0.0025): the mean drag coefficient over t = 1 to 2.2 on the GPU, with cut cells and with the staircase, against the prototype’s 1.220 (pressure 0.927, viscous 0.294: tools/cutcell/gal_split.ts). The GPU’s advection is not the prototype’s (MUSCL-Hancock with an HLLC flux against upwinded MUSCL), so they agree to the schemes’ difference, not to single precision: passes within 5% (measured 1.204, the staircase 1.300; the force averaged over free steps, meanForce)',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const cut of [true, false]) {
        const sim = await makeSim(gpu, boxSpec({ n: 256, box: 4, radius: 0.125, center: [0, 0.01], v: 1, c: 10, nu: 0.0025 }), 256, { cutCells: cut, tEnd: 1 });
        try {
          await sim.runToEnd(64, 1e7);
          const r = await meanForce(sim, 2.2);
          const key = cut ? 'cut' : 'staircase';
          m[`cd_${key}`] = r.force[0] / 0.125;
          m[`cd_${key}_pressure`] = r.pressure[0] / 0.125;
        } finally {
          sim.destroy();
        }
      }
      m.cd_prototype = 1.22;
      return { pass: Math.abs(m.cd_cut / 1.22 - 1) < 0.05, metrics: m };
    },
  },
  {
    id: 'cut-airfoil',
    name: 'Partly filled cells: a NACA 0012 at 5 degrees, Re 1000, against the float64 prototype and the staircase (2D, 16 to 64 cells along the chord)',
    criterion: 'The prototype’s airfoil (periodic box 8 chords across, U = 1, c = 10, nu = 0.001, from rest in the stream): the lift and drag coefficients averaged over t = 5 to 6 at 16, 32 and 64 cells along the chord (the pressure multigrid takes powers of two), with cut cells and with the staircase. The prototype (plans/wind-tunnel.md 8.1.3, probe shear) gives C_l 0.26, 0.32, 0.25 and C_d 0.085, 0.115, 0.134, converging to about 0.24 and 0.133 (112 cells). Reported; passes if the cut cells at 64 cells are within 0.05 of C_l 0.24 and 0.02 of C_d 0.133',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const cells of [16, 32, 64]) {
        const n = 8 * cells;
        const spec = {
          format: 'shaderous-setup', version: 1, id: 'cut-airfoil', name: 'Airfoil',
          domain: { ndim: 2, n, amr: 0, box: 8, supports2d: true, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } },
          physics: { eos: 'liquid', eosParams: { liqRho: 1, liqC: 10 }, gravity: 'off', G: 0, riemann: 'hllc', viscosity: 0.001 },
          background: { rho: 1, p: 0, v: [1, 0, 0] },
          parts: [{ kind: 'obstacle', id: 'wing', shape: { type: 'mesh', file: 'builtin:naca0012', scale: 1, center: [0.0037, 0.0051, 0], anchor: 'origin', up: 'y', rotate: [0, 0, -5] } }],
        } as unknown as SetupSpec;
        for (const cut of [true, false]) {
          const sim = await makeSim(gpu, spec, n, { cutCells: cut, tEnd: 5 });
          try {
            await sim.runToEnd(64, 1e7);
            const r = await meanForce(sim, 6);
            const key = `${cells}_${cut ? 'cut' : 'staircase'}`;
            m[`cl_${key}`] = r.force[1] / 0.5;
            m[`cd_${key}`] = r.force[0] / 0.5;
          } finally {
            sim.destroy();
          }
        }
      }
      return { pass: Math.abs(m.cl_64_cut - 0.24) < 0.05 && Math.abs(m.cd_64_cut - 0.133) < 0.02, metrics: m };
    },
  },
  {
    id: 'cut-cylinder-re40',
    name: 'Partly filled cells: the steady cylinder at Re 40 in a tunnel, cut cells against the staircase (2D, 8 to 32 cells across)',
    criterion: 'A cylinder of diameter 1 in a 2D tunnel 16 diameters long and wide (slip sides, Mach 0.1, nu = 1/40), to t = 30: the drag coefficient over t = 25 to 30 at 8, 16 and 32 cells across, with cut cells and with the staircase (published, unbounded: 1.50 to 1.57; the sides add some). Reported; passes if the cut cells at 16 across are within 3% of their value at 32',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const across of [8, 16, 32]) {
        const n = 16 * across;
        const spec = {
          format: 'shaderous-setup', version: 1, id: 'cut-cyl40', name: 'Cylinder at Re 40',
          domain: { ndim: 2, n, amr: 0, box: 16, supports2d: true },
          physics: { viscosity: 1 / 40 },
          parts: [
            { kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 },
            { kind: 'obstacle', id: 'cylinder', shape: { type: 'sphere', center: [-3.0037, 0.0051, 0], radius: 0.5 } },
          ],
        } as unknown as SetupSpec;
        for (const cut of [true, false]) {
          const sim = await makeSim(gpu, spec, n, { cutCells: cut, tEnd: 25 });
          try {
            await sim.runToEnd(64, 1e7);
            const r = await meanForce(sim, 30);
            m[`cd_${across}_${cut ? 'cut' : 'staircase'}`] = r.force[0] / 0.5;
          } finally {
            sim.destroy();
          }
        }
      }
      return { pass: Math.abs(m.cd_16_cut / m.cd_32_cut - 1) < 0.03, metrics: m };
    },
  },
  {
    id: 'cut-sphere-re100',
    name: 'Partly filled cells: a sphere at Re 100 in a 3D tunnel, cut cells against the staircase (uniform 128^3 and 256^3: 8 and 16 cells across)',
    criterion: 'A sphere of diameter 0.25 in a tunnel 16 diameters across (slip sides, Mach 0.1, nu = 0.0025), as sphere-drag but on the uniform grid: the drag coefficient over t = 6 to 8 at 8 and 16 cells across, with cut cells and with the staircase (published 1.08 to 1.09: Johnson & Patel 1999, Fornberg 1988; the staircase at 16 across on blocks read 1.25). Reported; passes if the cut cells at 16 across are within 0.95 to 1.25',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const n of [128, 256]) {
        const spec = {
          format: 'shaderous-setup', version: 1, id: 'cut-sphere', name: 'Sphere at Re 100',
          domain: { ndim: 3, n, amr: 0, box: 4 },
          physics: { viscosity: 0.0025 },
          parts: [
            { kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 },
            { kind: 'obstacle', id: 'sphere', shape: { type: 'sphere', center: [-1.0037, 0.0051, 0.0023], radius: 0.125 } },
          ],
        } as unknown as SetupSpec;
        for (const cut of [true, false]) {
          const sim = await makeSim(gpu, spec, n, { cutCells: cut, tEnd: 6 });
          try {
            await sim.runToEnd(64, 1e7);
            const r = await meanForce(sim, 8);
            m[`cd_${n / 16}_across_${cut ? 'cut' : 'staircase'}`] = r.force[0] / (0.5 * Math.PI * 0.125 * 0.125);
          } finally {
            sim.destroy();
          }
        }
      }
      return { pass: m.cd_16_across_cut > 0.95 && m.cd_16_across_cut < 1.25, metrics: m };
    },
  },
  {
    id: 'cut-turbine',
    name: 'Partly filled cells: the fast rotor (builtin:turbine-fast) in its frame, C_p against tip-speed ratio, grid and timestep, cut cells against the staircase (uniform 128^3 and 256^3)',
    criterion: 'The wind-turbine preset’s tunnel and frame with the fast rotor: the torque about the axis averaged over t = 0.8 to 1.6 (steady from t = 0.4), stepping freely (a stop lands with a short step whose force is not the flow’s), at Courant numbers 0.8, 0.4, 0.2 and extrapolated to no timestep (2 T(0.2) - T(0.4)), at a tip-speed ratio of 4 on 128^3 and 256^3 with cut cells and the staircase; then tip-speed ratios 2, 3, 5, 6 with cut cells on 256^3 at 0.2. Reported (plans/wind-tunnel.md 8.1.4)',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      const R = 0.8;
      const run = async (n: number, omega: number, cut: boolean, cfl: number) => {
        const spec = JSON.parse(JSON.stringify(preset('wind-turbine').spec));
        spec.parts[1].shape.file = 'builtin:turbine-fast';
        spec.parts[0].rotor = omega;
        spec.physics = { ...(spec.physics ?? {}), cutCells: cut };
        const c = compileSetup(spec);
        const sim = await Sim.create(gpu, configFor(c.problem, n, { ndim: 3, tEnd: 1e9, ...c.overrides, amrLevels: 0, cutCells: cut, cfl }));
        try {
          let tq = 0, fx = 0, k = 0;
          const t0 = performance.now();
          for (;;) {
            sim.step(50);
            const d = await sim.diagnostics();
            if (d.t > 0.8) {
              tq += d.bodies![0].torque[0];
              fx += d.bodies![0].force[0];
              k++;
            }
            if (d.t > 1.6) {
              m[`ms_per_step_${n}_${cut ? 'cut' : 'stair'}_tsr${(omega * R).toFixed(0)}_cfl${cfl}`] = (performance.now() - t0) / d.steps;
              break;
            }
          }
          const torque = tq / k;
          // C_p = Omega T / (rho U^3 pi R^2 / 2), U = 1, rho = 1.
          return { torque, cp: (omega * torque) / (0.5 * Math.PI * R * R), thrust: fx / k };
        } finally {
          sim.destroy();
        }
      };
      for (const n of [128, 256])
        for (const cut of [true, false]) {
          const r: Record<number, { cp: number }> = {};
          for (const cfl of [0.8, 0.4, 0.2]) {
            if (!cut && n === 256 && cfl === 0.2) continue;
            const x = await run(n, 5, cut, cfl);
            r[cfl] = x;
            m[`cp_${n}_${cut ? 'cut' : 'stair'}_tsr4_cfl${cfl}`] = x.cp;
            m[`thrust_${n}_${cut ? 'cut' : 'stair'}_tsr4_cfl${cfl}`] = x.thrust;
          }
          if (r[0.2]) m[`cp_${n}_${cut ? 'cut' : 'stair'}_tsr4_extrapolated`] = 2 * r[0.2].cp - r[0.4].cp;
        }
      for (const tsr of [2, 3, 5, 6]) {
        const x = await run(256, tsr / R, true, 0.2);
        m[`cp_256_cut_tsr${tsr}_cfl0.2`] = x.cp;
      }
      return { pass: true, metrics: m };
    },
  },
  {
    id: 'cut-turbine-span',
    name: 'Partly filled cells: the fast rotor (builtin:turbine-fast, tip-speed ratio 4) in its frame, its torque and thrust by span, pressure against shear (uniform 256^3)',
    criterion: 'The wind-turbine preset’s tunnel and frame with the fast rotor at Omega = 5, cut cells, Courant number 0.2, stepping freely to t = 1.2: the torque about the axis and the thrust from each cell’s wall forces, by radius in tenths of the rotor’s, the pressure’s and the shear’s apart, with C_p. Reported (plans/wind-tunnel.md 8.1.4)',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      const spec = JSON.parse(JSON.stringify(preset('wind-turbine').spec));
      spec.parts[1].shape.file = 'builtin:turbine-fast';
      spec.parts[0].rotor = 5;
      spec.physics = { ...(spec.physics ?? {}), cutCells: true };
      const c = compileSetup(spec);
      const n = 256;
      // Stepping freely (a stop's short landing step has a force of its own), at a Courant number
      // of 0.2 (on 256^3 its torque is about 11% short of the extrapolation to no timestep).
      // One instant's forces, not a mean: their sum is the instant's torque.
      const sim = await Sim.create(gpu, configFor(c.problem, n, { ndim: 3, tEnd: 1e9, ...c.overrides, amrLevels: 0, cutCells: true, cfl: 0.2 }));
      try {
        for (;;) {
          sim.step(50);
          if ((await sim.diagnostics()).t > 1.2) break;
        }
        const d = await sim.diagnostics();
        m.t = d.t;
        m.cp = d.bodies![0].coefficients?.power ?? NaN;
        m.torque = d.bodies![0].torque[0];
        const walls = (sim as unknown as { walls: { wp: GPUTexture; wv: GPUTexture } }).walls;
        const wp = await readSlabs(gpu.device, walls.wp, n);
        const wv = await readSlabs(gpu.device, walls.wv, n);
        const box = 3, dx = box / n, R = 0.8, cx = -0.4;
        const bins = 10;
        const tp = new Array(bins).fill(0), tv = new Array(bins).fill(0), fp = new Array(bins).fill(0), fv = new Array(bins).fill(0);
        for (let k = 0; k < n; k++)
          for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) {
              const q = (k * n + j) * n + i;
              if (!(wp[4 * q + 3] > 0.5 || wv[4 * q + 3] > 0.5)) continue;
              const x = -box / 2 + (i + 0.5) * dx, y = -box / 2 + (j + 0.5) * dx, z = -box / 2 + (k + 0.5) * dx;
              if (Math.abs(x - cx) > 0.5) continue;
              const r = Math.hypot(y, z);
              const b = Math.min(bins - 1, Math.floor((r / R) * bins));
              tp[b] += y * wp[4 * q + 2] - z * wp[4 * q + 1];
              tv[b] += y * wv[4 * q + 2] - z * wv[4 * q + 1];
              fp[b] += wp[4 * q];
              fv[b] += wv[4 * q];
            }
        for (let b = 0; b < bins; b++) {
          const tag = `r${(b / bins).toFixed(1)}_${((b + 1) / bins).toFixed(1)}`;
          m[`${tag}_torque_pressure`] = tp[b];
          m[`${tag}_torque_shear`] = tv[b];
          m[`${tag}_thrust_pressure`] = fp[b];
          m[`${tag}_thrust_shear`] = fv[b];
        }
        m.torque_pressure = tp.reduce((a, v) => a + v, 0);
        m.torque_shear = tv.reduce((a, v) => a + v, 0);
      } finally {
        sim.destroy();
      }
      return { pass: true, metrics: m };
    },
  },
  {
    id: 'cut-turbine-sense',
    name: 'Partly filled cells: the fast rotor (builtin:turbine-fast) turned the way it was built for and the other way (uniform 128^3)',
    criterion: 'The wind-turbine preset’s tunnel and frame with the fast rotor, cut cells, Courant number 0.4, stepping freely: the torque about the axis averaged over t = 0.8 to 1.6 with the frame turning at Omega = +5 (the blades’ leading edges ahead) and -5. Built the right way, the rotor brakes less (or drives) at +5: the torque at +5 must be above the torque at -5 times the sign of the turning (C_p(+5) > C_p(-5)). Reported (plans/wind-tunnel.md 8.1.5)',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      const R = 0.8;
      for (const omega of [5, -5]) {
        const spec = JSON.parse(JSON.stringify(preset('wind-turbine').spec));
        spec.parts[1].shape.file = 'builtin:turbine-fast';
        spec.parts[0].rotor = omega;
        spec.physics = { ...(spec.physics ?? {}), cutCells: true };
        const c = compileSetup(spec);
        const sim = await Sim.create(gpu, configFor(c.problem, 128, { ndim: 3, tEnd: 1e9, ...c.overrides, amrLevels: 0, cutCells: true, cfl: 0.4 }));
        try {
          let tq = 0, fx = 0, k = 0;
          for (;;) {
            sim.step(50);
            const d = await sim.diagnostics();
            if (d.t > 0.8) {
              tq += d.bodies![0].torque[0];
              fx += d.bodies![0].force[0];
              k++;
            }
            if (d.t > 1.6) break;
          }
          const tag = omega > 0 ? 'plus' : 'minus';
          m[`torque_${tag}`] = tq / k;
          m[`thrust_${tag}`] = fx / k;
          m[`cp_${tag}`] = (omega * tq) / k / (0.5 * Math.PI * R * R);
        } finally {
          sim.destroy();
        }
      }
      return { pass: m.cp_plus > m.cp_minus, metrics: m };
    },
  },
  {
    id: 'cut-sections-check',
    name: 'Partly filled cells: the fast rotor’s tip section at 9 degrees, Re 5000, on the GPU against the float64 prototype (2D, 16 and 32 cells on the chord)',
    criterion: 'builtin:section-tip (a NACA 0020 floored at 20% of the chord: a rectangle) at 9 degrees in a 2D tunnel 8 chords across (slip sides, Mach 0.1, nu = 2e-4), Courant 0.4: C_l and C_d over t = 8 to 16, at 16 and 32 cells on the chord, with cut cells. The prototype (tools/cutcell/section.ts, periodic with a sponge, plans/wind-tunnel.md 8.1.5) gave C_l 0.50 and 0.38, C_d 0.26 and 0.30 at 10 and 20 cells. Reported; passes if C_d at 32 cells is within 0.2 to 0.4',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const cells of [16, 32]) {
        const r = await sectionRun(gpu, 'section-tip', cells, { tEnd: 16 });
        for (const [k, v] of Object.entries(r)) m[`${k}_${cells}`] = v;
      }
      return { pass: m.cd_32 > 0.2 && m.cd_32 < 0.4, metrics: m };
    },
  },
  {
    id: 'cut-sections',
    name: 'Partly filled cells: the fast rotor’s blade sections and two airfoils at 9 degrees, Re 5000, converged (2D, 32 to 256 cells on the chord)',
    criterion: 'Five sections (builtin:section-tip, section-075, section-round, naca0020, naca0012) at 9 degrees in a 2D tunnel 8 chords across (slip sides, Mach 0.1, nu = 2e-4), Courant 0.4: C_l, C_d, the shear’s part of C_d and the rms of C_l over t = 8 to 24, at 32, 64, 128 and 256 cells on the chord with cut cells, and at 32 and 128 with the staircase (plans/wind-tunnel.md 8.1.6). Reported; passes if every run gives a finite positive drag',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const file of SECTIONS)
        for (const [cells, cut] of [[32, true], [64, true], [128, true], [256, true], [32, false], [128, false]] as const) {
          const r = await sectionRun(gpu, file, cells, { cutCells: cut });
          for (const [k, v] of Object.entries(r)) m[`${file}_${cells}${cut ? '' : '_staircase'}_${k}`] = v;
        }
      const cds = Object.entries(m).filter(([k]) => k.endsWith('_cd')).map(([, v]) => v);
      return { pass: cds.every((v) => Number.isFinite(v) && v > 0), metrics: m };
    },
  },
  {
    id: 'cut-sections-box',
    name: 'Partly filled cells: a blade section and the NACA 0012 at 9 degrees, Re 5000, in tunnels 8 and 16 chords across (2D, 64 cells on the chord)',
    criterion: 'builtin:section-tip and naca0012 as cut-sections at 64 cells on the chord, in a tunnel 8 chords across (n = 512) and 16 (n = 1024): what the tunnel’s sides add to C_l and C_d (plans/wind-tunnel.md 8.1.6). Reported',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const file of ['section-tip', 'naca0012'])
        for (const box of [8, 16]) {
          const r = await sectionRun(gpu, file, 64, { box });
          for (const [k, v] of Object.entries(r)) m[`${file}_box${box}_${k}`] = v;
        }
      return { pass: true, metrics: m };
    },
  },
  {
    id: 'cut-sections-dt',
    name: 'Partly filled cells: a blade section at 9 degrees, Re 5000, at Courant numbers 0.8, 0.4 and 0.2 (2D, 128 cells on the chord)',
    criterion: 'builtin:section-tip and naca0012 as cut-sections at 128 cells on the chord, at Courant numbers 0.8, 0.4 and 0.2: how much of C_l and C_d is the timestep’s (the fast rotor’s torque moved 11% between 0.4 and 0.2, plans/wind-tunnel.md 8.1.4). Reported',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const file of ['section-tip', 'naca0012'])
        for (const cfl of [0.8, 0.4, 0.2]) {
          const r = await sectionRun(gpu, file, 128, { cfl });
          for (const [k, v] of Object.entries(r)) m[`${file}_cfl${cfl}_${k}`] = v;
        }
      return { pass: true, metrics: m };
    },
  },
];

/** The blade sections of plans/wind-tunnel.md 8.1.6, as built-in wings. */
export const SECTIONS = ['section-tip', 'section-075', 'section-round', 'naca0020', 'naca0012'];

/**
 * A blade section (a built-in wing, chord 1) at `alpha` degrees in a 2D tunnel `box` chords
 * across, slip sides, Mach 0.1, U = 1; its quarter chord 1.5 chords upstream of the middle (as the
 * prototype's, tools/cutcell/section.ts). The setup document, for the command-line runner too.
 */
export function sectionSpec(file: string, cells: number, o: { box?: number; alpha?: number; re?: number } = {}): SetupSpec {
  const box = o.box ?? 8;
  return {
    format: 'shaderous-setup', version: 1, id: `section-${file}`, name: `Blade section ${file}`,
    domain: { ndim: 2, n: box * cells, amr: 0, box, supports2d: true },
    physics: { viscosity: 1 / (o.re ?? 5000), cutCells: true },
    parts: [
      { kind: 'tunnel', speed: 1, axis: 'x', sides: 'slip', sponge: 0.15, mach: 0.1 },
      { kind: 'obstacle', id: 'section', shape: { type: 'mesh', file: `builtin:${file}`, scale: 1, center: [-1.4963, 0.0051, 0], anchor: 'origin', up: 'y', rotate: [0, 0, -(o.alpha ?? 9)] } },
    ],
  } as unknown as SetupSpec;
}

/**
 * Run a section to `t0` and then step freely to `tEnd`: C_l and C_d (over the chord, rho U^2 / 2
 * = 1/2), the shear's part of C_d, and the rms of C_l about its mean, each step weighed by its dt.
 */
async function sectionRun(gpu: Gpu, file: string, cells: number, o: { box?: number; cutCells?: boolean; cfl?: number; t0?: number; tEnd?: number } = {}): Promise<Record<string, number>> {
  const spec = sectionSpec(file, cells, { box: o.box });
  const n = (o.box ?? 8) * cells;
  const cut = o.cutCells ?? true;
  const sim = await makeSim(gpu, spec, n, { cutCells: cut, cfl: o.cfl ?? 0.4, tEnd: o.t0 ?? 8 });
  try {
    const t0 = performance.now();
    await sim.runToEnd(64, 1e8);
    sim.cfg.tEnd = 1e9;
    sim.writeParams();
    let w = 0, l = 0, l2 = 0, dr = 0, dp = 0, steps = 0;
    for (;;) {
      sim.step(1);
      const d = await sim.diagnostics();
      const b = d.bodies![0];
      w += d.dt;
      l += b.force[1] * d.dt;
      l2 += b.force[1] * b.force[1] * d.dt;
      dr += b.force[0] * d.dt;
      dp += b.pressure[0] * d.dt;
      steps = d.steps;
      if (d.t >= (o.tEnd ?? 24)) break;
    }
    const cl = l / w / 0.5, cd = dr / w / 0.5;
    return { cl, cd, cd_shear: (dr - dp) / w / 0.5, cl_rms: Math.sqrt(Math.max(0, l2 / w - (l / w) ** 2)) / 0.5, ms_per_step: (performance.now() - t0) / steps, steps };
  } finally {
    sim.destroy();
  }
}
