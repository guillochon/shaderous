import { all, OUTFLOW, type Problem } from './types';

// n = 1 polytrope: rho = rho_c sin(xi) / xi with xi = pi r / R, p = K rho^2, K = 2 G R^2 / pi.
// Evolved with an ideal gas of gamma = 2, so the energy equation and dual-energy switch are
// exercised. The star sits in a light, hot ambient medium.
export const POLY = { R: 0.3, rhoc: 1, G: 1, rhoAmb: 1e-4, rhoCut: 1e-2 };
export const POLY_K = (2 * POLY.G * POLY.R ** 2) / Math.PI;

export function polytropeDensity(r: number): number {
  if (r >= POLY.R) return 0;
  const xi = (Math.PI * r) / POLY.R;
  return xi < 1e-6 ? POLY.rhoc : (POLY.rhoc * Math.sin(xi)) / xi;
}

export const polytrope: Problem = {
  id: 'polytrope',
  label: 'Polytrope (hydrostatic star)',
  blurb: 'Self-gravitating n = 1 polytrope in equilibrium. One dynamical time is t = 1.',
  eos: 'ideal',
  eosParams: { gamma: 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'isolated',
  G: POLY.G,
  box: 1,
  tEnd: 10,
  rhoFloor: 1e-6,
  pFloor: 1e-10,
  view: { field: 0, log: true, mode: 'volume' },
  star: { n: 1, K: POLY_K, rhoCut: POLY.rhoCut, rhoAmb: POLY.rhoAmb },
  init: () => (x, y, z) => {
    const rho = polytropeDensity(Math.hypot(x, y, z));
    const p = POLY_K * Math.max(rho, POLY.rhoCut) ** 2;
    return { rho: Math.max(rho, POLY.rhoAmb), vx: 0, vy: 0, vz: 0, p, s: rho > POLY.rhoCut ? 1 : 0 };
  },
};
