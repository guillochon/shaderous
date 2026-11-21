// Validation of partly filled cells on the GPU (plans/wind-tunnel.md 8.1; hydro/cut.ts), against
// the float64 prototype (src/reference/cutcell.ts) where they share a problem.
import type { Gpu } from '../gpu/device';
import { configFor, Sim, type SimConfig } from '../sim';
import { compileSetup } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';
import type { TestDef } from './tests';

/** A periodic box of liquid with a cylinder (a sphere in 3D), and a slab of it moving toward the body. */
function boxSpec(o: { n: number; box: number; radius: number; center: [number, number]; v: number; c: number; nu: number; slab?: boolean; ndim?: 2 | 3 }): SetupSpec {
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
    criterion: 'The prototype’s Galilean problem with the body at rest (periodic box 4, cylinder 0.25 across, U = 1, c = 10, nu = 0.0025): the mean drag coefficient over t = 1 to 2.2 on the GPU, with cut cells and with the staircase, against the prototype’s 1.220 (pressure 0.927, viscous 0.294: tools/cutcell/gal_split.ts). The GPU’s advection is not the prototype’s (MUSCL-Hancock with an HLLC flux against upwinded MUSCL), so they agree to the schemes’ difference, not to single precision: passes within 10%',
    async run(gpu: Gpu) {
      const m: Record<string, number> = {};
      for (const cut of [true, false]) {
        const sim = await makeSim(gpu, boxSpec({ n: 256, box: 4, radius: 0.125, center: [0, 0.01], v: 1, c: 10, nu: 0.0025 }), 256, { cutCells: cut, tEnd: 1 });
        try {
          await sim.runToEnd(64, 1e7);
          const f: number[] = [];
          let fp = 0;
          for (let k = 1; k <= 60; k++) {
            sim.cfg.tEnd = 1 + 0.02 * k;
            sim.writeParams();
            await sim.runToEnd(64, 1e7);
            const b = (await sim.diagnostics()).bodies![0];
            f.push(b.force[0]);
            fp += b.pressure[0];
          }
          const key = cut ? 'cut' : 'staircase';
          m[`cd_${key}`] = f.reduce((a, b) => a + b, 0) / f.length / 0.125;
          m[`cd_${key}_pressure`] = fp / f.length / 0.125;
        } finally {
          sim.destroy();
        }
      }
      m.cd_prototype = 1.22;
      return { pass: Math.abs(m.cd_cut / 1.22 - 1) < 0.1, metrics: m };
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
            let fx = 0, fy = 0, k = 0;
            for (let q = 1; q <= 50; q++) {
              sim.cfg.tEnd = 5 + 0.02 * q;
              sim.writeParams();
              await sim.runToEnd(64, 1e7);
              const b = (await sim.diagnostics()).bodies![0];
              fx += b.force[0];
              fy += b.force[1];
              k++;
            }
            const key = `${cells}_${cut ? 'cut' : 'staircase'}`;
            m[`cl_${key}`] = fy / k / 0.5;
            m[`cd_${key}`] = fx / k / 0.5;
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
            let fx = 0, k = 0;
            for (let q = 1; q <= 50; q++) {
              sim.cfg.tEnd = 25 + 0.1 * q;
              sim.writeParams();
              await sim.runToEnd(64, 1e7);
              fx += (await sim.diagnostics()).bodies![0].force[0];
              k++;
            }
            m[`cd_${across}_${cut ? 'cut' : 'staircase'}`] = fx / k / 0.5;
          } finally {
            sim.destroy();
          }
        }
      }
      return { pass: Math.abs(m.cd_16_cut / m.cd_32_cut - 1) < 0.03, metrics: m };
    },
  },
];
