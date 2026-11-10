import { all, PERIODIC, type Problem } from './types';

// One Jeans-unstable mode along x in a periodic box. With rho0 = 1, c_s = 1 and k = 2 pi,
// G is chosen so that 4 pi G rho0 = 2 k^2 c_s^2, giving a growth rate of k c_s.
export const JEANS = {
  amp: 1e-3,
  k: 2 * Math.PI,
  G: (2 * (2 * Math.PI) ** 2) / (4 * Math.PI),
  growth: 2 * Math.PI,
};

export const jeans: Problem = {
  id: 'jeans',
  label: 'Jeans instability',
  blurb: 'Linear growth of a self-gravitating density wave; delta grows as cosh(k c_s t).',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'periodic',
  G: JEANS.G,
  box: 1,
  tEnd: 0.5,
  view: { field: 0, log: false, mode: 'slice' },
  init: (_n, e) => (x) => {
    const g = e.params.gamma;
    const d = JEANS.amp * Math.cos(JEANS.k * x);
    // Adiabatic perturbation about rho0 = 1, p0 = 1 / gamma (c_s = 1).
    return { rho: 1 + d, vx: 0, vy: 0, vz: 0, p: (1 + g * d) / g, s: 0.5 + 0.5 * Math.cos(JEANS.k * x) };
  },
};
