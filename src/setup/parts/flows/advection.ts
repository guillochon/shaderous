// Gaussian density blob carried by a uniform flow in pressure equilibrium. vx = 1 in a unit
// box, so t = 10 is ten box crossings along x.
//
// The background density is 1.5 rather than 1 on purpose. At exactly 1.0 the float32 spacing
// changes by a factor of two, so rounding in the blob's faint tail is biased and total mass
// creeps by about 2e-9 per step; away from a power of two the drift is unbiased round-off.
import type { FlowDef } from '../flow';

const advection: FlowDef<{ rho0: number; amp: number; width: number; vx: number; vy: number; vz: number; p: number }> = {
  name: 'advection',
  label: 'Advected Gaussian blob',
  tip: 'A Gaussian density blob at the center, carried by a uniform flow at a uniform pressure. The passive scalar follows the blob. On a periodic box it tests conservation.',
  wholeBox: true,
  fields: {
    rho0: { label: 'Density around the blob', default: 1.5, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas the blob sits in. Not a power of two by default: at exactly 1 the rounding of float32 in the blob\u2019s faint tail is biased, and the total mass creeps' },
    amp: { label: 'Blob height', default: 1, min: 0, max: 1e6, factor: 1.25, tip: 'Density of the blob\u2019s peak above the gas around it' },
    width: { label: 'Blob width', default: 0.1, min: 0.01, max: 1, step: 0.01, tip: 'Standard deviation of the Gaussian, in code lengths' },
    vx: { label: 'Velocity x', default: 1, min: -100, max: 100, step: 0.25, tip: 'Velocity of the flow along x' },
    vy: { label: 'Velocity y', default: 0.5, min: -100, max: 100, step: 0.25, tip: 'Velocity of the flow along y' },
    vz: { label: 'Velocity z', default: 0.25, min: -100, max: 100, step: 0.25, tip: 'Velocity of the flow along z (3D)' },
    p: { label: 'Pressure', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure, the same everywhere' },
  },
  init: (_ctx, q) => (x, y, z) => {
    const g = Math.exp(-(x * x + y * y + z * z) / (2 * q.width * q.width));
    return { rho: q.rho0 + q.amp * g, vx: q.vx, vy: q.vy, vz: q.vz, p: q.p, s: g };
  },
};

export const flows = [advection];
