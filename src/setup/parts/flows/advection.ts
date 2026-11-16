// Gaussian density blob carried by a uniform flow in pressure equilibrium. vx = 1 in a unit
// box, so t = 10 is ten box crossings along x.
//
// The background density is 1.5 rather than 1 on purpose. At exactly 1.0 the float32 spacing
// changes by a factor of two, so rounding in the blob's faint tail is biased and total mass
// creeps by about 2e-9 per step; away from a power of two the drift is unbiased round-off.
import type { FlowDef } from '../flow';

const advection: FlowDef<Record<string, never>> = {
  name: 'advection',
  label: 'Advected Gaussian blob',
  tip: 'A Gaussian density blob (width 0.1, height 1 over a density of 1.5) at the center, carried at velocity (1, 0.5, 0.25) at a uniform pressure of 1. The passive scalar follows the blob.',
  wholeBox: true,
  fields: {},
  init: () => (x, y, z) => {
    const g = Math.exp(-(x * x + y * y + z * z) / (2 * 0.1 * 0.1));
    return { rho: 1.5 + g, vx: 1, vy: 0.5, vz: 0.25, p: 1, s: g };
  },
};

export const flows = [advection];
