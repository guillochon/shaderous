import { all, PERIODIC, type Problem } from './types';

// Gaussian density blob carried by a uniform flow in pressure equilibrium. vx = 1 in a unit
// box, so t = 10 is ten box crossings along x.
//
// The background density is 1.5 rather than 1 on purpose. At exactly 1.0 the float32 spacing
// changes by a factor of two, so rounding in the blob's faint tail is biased and total mass
// creeps by about 2e-9 per step; away from a power of two the drift is unbiased round-off.
export const advection: Problem = {
  id: 'advection',
  label: 'Uniform advection',
  blurb: 'Density blob advected diagonally through a periodic box; tests conservation.',
  eos: 'ideal',
  eosParams: { gamma: 1.4 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: 10,
  checkpointDt: 0.5,
  supports2d: true,
  view: { field: 0, log: false, mode: 'slice' },
  init: () => (x, y, z) => {
    const g = Math.exp(-(x * x + y * y + z * z) / (2 * 0.1 * 0.1));
    return { rho: 1.5 + g, vx: 1, vy: 0.5, vz: 0.25, p: 1, s: g };
  },
};
