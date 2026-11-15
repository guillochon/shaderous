import { all, OUTFLOW, type Problem, type ProblemParam } from './types';
import { dipoleField, randomField, type StarProfile } from '../mhd/star_fields';

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
  tEnd: Infinity,
  checkpointDt: 0.25,
  tDyn: 1,
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

/** The polytrope's pressure profile, for the fields below: confined to where p > p_c / 100. */
const POLY_STAR: StarProfile = {
  pressure: (r) => POLY_K * polytropeDensity(r) ** 2,
  pc: POLY_K * POLY.rhoc ** 2,
  pcut: 0.01 * POLY_K * POLY.rhoc ** 2,
  radius: POLY.R,
};

const betaParam: ProblemParam = {
  id: 'beta', label: 'Plasma beta at the center', value: 1000, min: 10, max: 1e7, factor: 10,
  title: 'Ratio of the gas pressure to the magnetic pressure B^2 / 2 at the star\'s center: the larger, the weaker the field',
};

/** The polytrope with a dipole field inside it, along z (mhd/star_fields.ts). */
export const polytropeDipole: Problem = {
  ...polytrope,
  id: 'polytrope-dipole',
  label: 'Polytrope with a dipole field',
  blurb: 'The n = 1 polytrope threaded by a dipole field confined to the star (beta 1000 at its center): its field lines close inside it, and at this beta it stays in balance.',
  params: [betaParam],
  mhd: { potential: (p) => dipoleField(POLY_STAR, p.beta) },
};

/** The polytrope with a random (Kolmogorov) field inside it (mhd/star_fields.ts). */
export const polytropeRandom: Problem = {
  ...polytrope,
  id: 'polytrope-random',
  label: 'Polytrope with a random field',
  blurb: 'The n = 1 polytrope threaded by a tangled field with a Kolmogorov spectrum, confined to the star (beta 1000 at its center on average).',
  params: [betaParam, { id: 'seed', label: 'Random seed', value: 1, min: 1, max: 9999, step: 1, title: 'Which random field: each seed gives another' }],
  mhd: { potential: (p) => randomField(POLY_STAR, p.beta, Math.round(p.seed)) },
};
