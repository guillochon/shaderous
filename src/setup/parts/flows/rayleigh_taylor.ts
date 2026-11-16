// Rayleigh-Taylor instability: heavy gas (rho = 2) resting on light gas (rho = 1) in a uniform
// gravitational field, in hydrostatic balance, with a single-mode velocity perturbation that
// vanishes at the walls. "Up" is y in 2D and z in 3D (the axis the volume view draws upward).
// The linear growth rate is sqrt(A g k) with Atwood number A = 1/3: about 1.45 for the
// k = 4 pi mode seeded here, so bubbles and spikes are well developed by t = 4.
import type { FlowDef } from '../flow';

export const RT = { g: 0.5, p0: 2.5, amp: 0.05 };

const rayleighTaylor: FlowDef<Record<string, never>> = {
  name: 'rayleigh-taylor',
  label: 'Rayleigh-Taylor layers',
  tip: 'Gas of density 2 over gas of density 1 in a unit box, in balance with a downward field of strength 0.5 (set in the physics), with one seeded mode. Up is y in 2D and z in 3D.',
  wholeBox: true,
  fields: {},
  init: (ctx) => {
    const ndim = ctx.ndim;
    return (x, y, z) => {
      const h = ndim === 2 ? y : z; // height
      const rho = h > 0 ? 2 : 1;
      // Hydrostatic: dp/dh = -rho g, continuous across the interface.
      const p = RT.p0 - RT.g * rho * h;
      const across = ndim === 2 ? 1 + Math.cos(4 * Math.PI * x) : 0.5 * (1 + Math.cos(4 * Math.PI * x)) * (1 + Math.cos(4 * Math.PI * y));
      const v = -0.25 * RT.amp * across * (1 + Math.cos(2 * Math.PI * h));
      return { rho, vx: 0, vy: ndim === 2 ? v : 0, vz: ndim === 2 ? 0 : v, p, s: h > 0 ? 1 : 0 };
    };
  },
};

export const flows = [rayleighTaylor];
