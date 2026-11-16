// Rayleigh-Taylor instability: heavy gas (rho = 2) resting on light gas (rho = 1) in a uniform
// gravitational field, in hydrostatic balance, with a single-mode velocity perturbation that
// vanishes at the walls. "Up" is y in 2D and z in 3D (the axis the volume view draws upward).
// The linear growth rate is sqrt(A g k) with Atwood number A = 1/3: about 1.45 for the
// k = 4 pi mode seeded here, so bubbles and spikes are well developed by t = 4.
import type { FlowDef } from '../flow';

export const RT = { g: 0.5, p0: 2.5, amp: 0.05 };

const rayleighTaylor: FlowDef<{ rhoTop: number; rhoBottom: number; p0: number; amp: number; modes: number }> = {
  name: 'rayleigh-taylor',
  label: 'Rayleigh-Taylor layers',
  tip: 'Heavy gas over light gas in a unit box, in hydrostatic balance with the uniform downward acceleration of the setup (Gravity: Uniform acceleration), with one seeded mode that vanishes at the walls. Up is y in 2D and z in 3D.',
  wholeBox: true,
  fields: {
    rhoTop: { label: 'Density above', default: 2, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas above the interface' },
    rhoBottom: { label: 'Density below', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas below the interface. Lighter than the gas above, the layers are unstable' },
    p0: { label: 'Pressure at the interface', default: RT.p0, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure at the interface: high enough that it stays positive at the top of the box' },
    amp: { label: 'Kick amplitude', default: RT.amp, min: 0, max: 1, step: 0.01, tip: 'Peak vertical velocity of the seeded mode' },
    modes: { label: 'Wavelengths across', default: 2, min: 1, max: 32, step: 1, tip: 'Wavelengths of the seeded mode across the box (along x, and y in 3D)' },
  },
  init: (ctx, q) => {
    const ndim = ctx.ndim;
    // The setup's own downward acceleration, which the layers are in balance with.
    const gext = ctx.spec.physics?.gext;
    const g = gext === undefined ? RT.g : Array.isArray(gext) ? -gext[ndim === 2 ? 1 : 2] : gext.down;
    const k = 2 * Math.round(q.modes);
    return (x, y, z) => {
      const h = ndim === 2 ? y : z; // height
      const rho = h > 0 ? q.rhoTop : q.rhoBottom;
      // Hydrostatic: dp/dh = -rho g, continuous across the interface.
      const p = q.p0 - g * rho * h;
      const across = ndim === 2 ? 1 + Math.cos(k * Math.PI * x) : 0.5 * (1 + Math.cos(k * Math.PI * x)) * (1 + Math.cos(k * Math.PI * y));
      const v = -0.25 * q.amp * across * (1 + Math.cos(2 * Math.PI * h));
      return { rho, vx: 0, vy: ndim === 2 ? v : 0, vz: ndim === 2 ? 0 : v, p, s: h > 0 ? 1 : 0 };
    };
  },
};

export const flows = [rayleighTaylor];
