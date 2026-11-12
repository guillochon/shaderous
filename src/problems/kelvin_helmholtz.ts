import { all, PERIODIC, type Problem } from './types';

export const kelvinHelmholtz: Problem = {
  id: 'kelvin-helmholtz',
  label: 'Kelvin-Helmholtz',
  blurb: 'Dense slab sliding through lighter gas; the passive scalar tags the slab.',
  eos: 'ideal',
  eosParams: { gamma: 1.4 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: Infinity,
  checkpointDt: 0.1,
  supports2d: true,
  view: { field: 0, log: false, mode: 'slice' },
  init: () => (x, y, z) => {
    const inner = Math.abs(y) < 0.25;
    const sig = 0.035;
    const env = Math.exp(-((y - 0.25) ** 2) / (2 * sig * sig)) + Math.exp(-((y + 0.25) ** 2) / (2 * sig * sig));
    // Single mode in x, weakly modulated in z so the flow is not exactly two-dimensional.
    const vy = 0.1 * Math.sin(4 * Math.PI * x) * (1 + 0.2 * Math.cos(2 * Math.PI * z)) * env;
    return { rho: inner ? 2 : 1, vx: inner ? 0.5 : -0.5, vy, vz: 0, p: 2.5, s: inner ? 1 : 0 };
  },
};
