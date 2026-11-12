import { all, OUTFLOW, type Problem } from './types';

// Evrard (1988) collapse: cold gas sphere with rho = M / (2 pi R^2 r), specific internal
// energy 0.05 G M / R, G = M = R = 1. It collapses, bounces through a strong accretion
// shock and settles toward virial equilibrium.
export const evrard: Problem = {
  id: 'evrard',
  label: 'Evrard collapse',
  blurb: 'Adiabatic collapse of a cold 1/r gas sphere: self-gravity plus strong shocks.',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'isolated',
  G: 1,
  box: 4,
  tEnd: Infinity,
  checkpointDt: 0.1,
  rhoFloor: 1e-7,
  pFloor: 1e-12,
  view: { field: 0, log: true, mode: 'volume' },
  init: (n) => {
    const dx = 4 / n;
    const u = 0.05;
    const rhoAmb = 1e-4;
    return (x, y, z) => {
      const r = Math.max(Math.hypot(x, y, z), 0.5 * dx);
      const rho = r < 1 ? 1 / (2 * Math.PI * r) : rhoAmb;
      return { rho, vx: 0, vy: 0, vz: 0, p: (2 / 3) * rho * u, s: r < 1 ? 1 : 0 };
    };
  },
};
