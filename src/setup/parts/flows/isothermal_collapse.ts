// Isothermal, self-gravitating periodic box seeded with a few long-wavelength velocity modes.
// About six Jeans lengths fit across the box, so the gas gathers into filaments and clumps.
//
// Isothermal collapse never stops on its own, so a Jeans pressure floor holds the cores at a
// fixed number of cells. That number matters for how long the run can be trusted: the cores
// then live for hundreds of their own dynamical times at that resolution, and truncation error
// in the gravity coupling slowly feeds them energy (they spin up and puff out). At 4 cells per
// Jeans length, the minimum that prevents artificial fragmentation, the cores lose most of
// their binding energy within about two box free-fall times of forming; at 8 the drift is
// several times slower. The preset uses 8 and ends at t = 2.
import type { FlowDef } from '../flow';

const isothermalCollapse: FlowDef<{ rho: number; p: number; vrms: number; modes: number; seed: number }> = {
  name: 'isothermal-collapse',
  label: 'Turbulent velocity field',
  tip: 'Uniform gas in a unit periodic box, stirred by a few long-wavelength velocity modes with random directions and phases. With the preset\u2019s isothermal gas (sound speed 0.1) the default speed is Mach 5.',
  wholeBox: true,
  fields: {
    rho: { label: 'Density', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas, the same everywhere' },
    p: { label: 'Pressure', default: 0.01, min: 1e-12, max: 1e6, factor: 1.25, tip: 'Pressure of the gas. A barotropic gas takes its pressure from its density instead' },
    vrms: { label: 'Rms speed', default: 0.5, min: 0, max: 100, factor: 1.25, tip: 'Root-mean-square speed of the stirring' },
    modes: { label: 'Velocity modes', default: 12, min: 1, max: 64, step: 1, tip: 'How many plane waves the velocity field is the sum of, each with wavenumbers up to 2 along every axis' },
    seed: { label: 'Random seed', default: 12345, min: 1, max: 4294967295, step: 1, tip: 'Which velocity field: each seed gives another' },
  },
  init: (_ctx, q) => {
    // Deterministic pseudo-random phases.
    let seed = Math.round(q.seed) >>> 0;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const modes: { k: number[]; a: number[]; ph: number }[] = [];
    for (let i = 0; i < Math.round(q.modes); i++) {
      const k: number[] = [0, 0, 0].map(() => Math.floor(rnd() * 5) - 2);
      if (Math.hypot(...k) === 0) k[0] = 1;
      const kk = Math.hypot(...k);
      modes.push({ k, a: [0, 0, 0].map(() => (rnd() - 0.5) / kk ** 2), ph: rnd() * 2 * Math.PI });
    }
    const velocity = (x: number, y: number, z: number) => {
      const v = [0, 0, 0];
      for (const m of modes) {
        const s = Math.sin(2 * Math.PI * (m.k[0] * x + m.k[1] * y + m.k[2] * z) + m.ph);
        for (let d = 0; d < 3; d++) v[d] += m.a[d] * s;
      }
      return v;
    };
    // Normalize to the rms speed asked for (Mach 5 for the preset's c_s = 0.1).
    let sum = 0;
    for (let i = 0; i < 512; i++) {
      const v = velocity(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
      sum += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
    const scale = q.vrms / Math.sqrt(sum / 512);
    return (x, y, z) => {
      const v = velocity(x, y, z);
      return { rho: q.rho, vx: scale * v[0], vy: scale * v[1], vz: scale * v[2], p: q.p, s: 0.5 + 0.5 * Math.sin(2 * Math.PI * x) };
    };
  },
};

export const flows = [isothermalCollapse];
