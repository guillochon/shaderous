import { all, OUTFLOW, type Problem, type ProblemParam } from '../../src/problems/types';
import { dipoleField, randomField, type StarProfile } from '../../src/mhd/star_fields';
import { laneEmdenCached, polytropeK, polytropeMass } from '../../src/problems/lane_emden';
import { settlingOf } from '../../src/problems/settle';
import { ASTRO } from '../../src/units/constants';
import { gravityUnits, type UnitSystem } from '../../src/units/units';

// A polytrope of index n at rest: p = K rho^(1 + 1/n). The default is n = 1, whose profile is
// rho = rho_c sin(xi) / xi with xi = pi r / R and K = 2 G R^2 / pi, evolved with an ideal gas of
// gamma = 2, so the energy equation and dual-energy switch are exercised. The star sits in a
// light, hot ambient medium.
//
// In code units the star has central density 1 and G = 1, and its radius is a share of the box
// (the `size` parameter, POLY.R by default), as the star tool's Size is. Its mass and radius in
// solar units only say what one code unit is (polytropeUnits), so with an ideal gas they change
// nothing but the readouts. An equation of state with physical constants in it
// (radiation, degenerate electrons, the tabulated Helmholtz one) takes them from those units:
// the star then keeps the polytrope's pressure at every density, and the equation of state
// finds the temperature that gives it.
export const POLY = { R: 0.3, rhoc: 1, G: 1, rhoAmb: 1e-4, rhoCut: 1e-2 };
export const POLY_K = (2 * POLY.G * POLY.R ** 2) / Math.PI;

/** Density of the n = 1 star at radius r. */
export function polytropeDensity(r: number): number {
  if (r >= POLY.R) return 0;
  const xi = (Math.PI * r) / POLY.R;
  return xi < 1e-6 ? POLY.rhoc : (POLY.rhoc * Math.sin(xi)) / xi;
}

/** The star of index n in code units. */
export interface PolyStar {
  n: number;
  K: number;
  mass: number;
  /** density below which a cell is ambient gas */
  rhoCut: number;
  density(r: number): number;
  pressure(rho: number): number;
}

const stars = new Map<string, PolyStar>();

/**
 * The star of index n (default 1). The ambient density is the same for every n; the star is
 * cut off where its pressure over that density equals the center's p / rho, so the ambient
 * gas, which carries the pressure at the cut, is as hot as the center and no hotter (for
 * n = 1, at POLY.rhoCut).
 */
export function polyStar(n = 1, R = POLY.R): PolyStar {
  const key = `${n}:${R}`;
  let s = stars.get(key);
  if (s) return s;
  const g = 1 + 1 / n;
  if (n === 1 && R === POLY.R) {
    s = { n, K: POLY_K, mass: (4 / Math.PI) * POLY.rhoc * POLY.R ** 3, rhoCut: POLY.rhoCut, density: polytropeDensity, pressure: (rho) => POLY_K * rho * rho };
  } else {
    const le = laneEmdenCached(n);
    const K = polytropeK(le, POLY.rhoc, R, POLY.G);
    s = {
      n,
      K,
      mass: polytropeMass(le, POLY.rhoc, R),
      rhoCut: POLY.rhoc * Math.pow(POLY.rhoAmb / POLY.rhoc, 1 / g),
      density: (r) => POLY.rhoc * Math.pow(le.theta(r / R), n),
      pressure: (rho) => K * Math.pow(rho, g),
    };
  }
  stars.set(key, s);
  return s;
}

/** The star's radius in code units: the `size` parameter, a share of the box (of side 1). */
const radiusOf = (p: Record<string, number> | undefined): number => p?.size ?? POLY.R;
/** The star these parameters ask for. */
const starOf = (p: Record<string, number> | undefined): PolyStar => polyStar(p?.n, radiusOf(p));

/**
 * What one code unit is for a star of `m` solar masses and `r` solar radii with index `n`:
 * the star's radius is `size` and its mass starOf(p).mass in code units, and G = 1.
 */
export function polytropeUnits(p: Record<string, number>): UnitSystem {
  return gravityUnits('star', ((p.r ?? 1) * ASTRO.Rsun) / radiusOf(p), ((p.m ?? 1) * ASTRO.Msun) / starOf(p).mass);
}

const SCALE_NOTE = 'With an ideal gas it sets only the physical units the readouts are shown in; an equation of state with physical constants (radiation, degenerate electrons, the tabulated one) takes them from it. Changing it starts the run again';
const starParams: ProblemParam[] = [
  { id: 'm', label: 'Mass (M☉)', title: `Mass of the star, in solar masses. ${SCALE_NOTE}`, value: 1, min: 1e-3, max: 1e3, factor: 2, scale: true },
  { id: 'r', label: 'Radius (R☉)', title: `Radius of the star, in solar radii (a white dwarf is about 0.01). ${SCALE_NOTE}`, value: 1, min: 1e-5, max: 1e4, factor: 2, scale: true },
  { id: 'size', label: 'Size (share of the box)', title: 'Radius of the star as a share of the box, as the star tool’s Size: how much room is left around it. The star is the same in physical units, on a finer or coarser grid. Changing it starts the run again', value: POLY.R, min: 0.05, max: 0.45, step: 0.05 },
  {
    id: 'n', label: 'Polytropic index n', value: 1, min: 0.5, max: 3.5, step: 0.5,
    title: 'The star\'s structure, p = K ρ^(1 + 1/n): the larger n, the more centrally concentrated (3/2 for a convective star, 3 for the Sun\'s interior). The star is stable against convection only if the gas\'s adiabatic index is at least 1 + 1/n: n = 1 needs the ideal gas\'s γ = 2, and a real gas (γ of 5/3 or less) needs n of 3/2 or more. Changing it starts the run again',
  },
];

export const polytrope: Problem = {
  id: 'polytrope',
  label: 'Polytrope (hydrostatic star)',
  blurb: 'Self-gravitating polytrope in equilibrium (n = 1 with a γ = 2 gas by default). One dynamical time is t = 1. Its mass and radius set the physical units, for an equation of state that needs them.',
  eos: 'ideal',
  eosParams: { gamma: 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'isolated',
  G: POLY.G,
  units: polytropeUnits,
  box: 1,
  tEnd: Infinity,
  checkpointDt: 0.25,
  tDyn: 1,
  rhoFloor: 1e-6,
  pFloor: 1e-10,
  refineRho: 10 * POLY.rhoAmb,
  view: { field: 0, log: true, mode: 'volume' },
  params: starParams,
  settle: (p) => {
    const st = starOf(p);
    return settlingOf({ n: st.n, K: st.K, rhoCut: st.rhoCut, rhoAmb: POLY.rhoAmb }, POLY.G);
  },
  init: (_n, _eos, _ndim, params) => {
    const st = starOf(params);
    return (x, y, z) => {
      const rho = st.density(Math.hypot(x, y, z));
      const p = st.pressure(Math.max(rho, st.rhoCut));
      return { rho: Math.max(rho, POLY.rhoAmb), vx: 0, vy: 0, vz: 0, p, s: rho > st.rhoCut ? 1 : 0 };
    };
  },
};

/** The star's pressure profile, for the fields below: confined to where p > p_c / 100. */
function starProfile(p: Record<string, number>): StarProfile {
  const st = starOf(p);
  const pc = st.pressure(POLY.rhoc);
  return { pressure: (r) => st.pressure(st.density(r)), pc, pcut: 0.01 * pc, radius: radiusOf(p) };
}

const betaParam: ProblemParam = {
  id: 'beta', label: 'Plasma beta at the center', value: 1000, min: 10, max: 1e7, factor: 10,
  title: 'Ratio of the gas pressure to the magnetic pressure B^2 / 2 at the star\'s center: the larger, the weaker the field',
};

/** The polytrope with a dipole field inside it, along z (mhd/star_fields.ts). */
export const polytropeDipole: Problem = {
  ...polytrope,
  id: 'polytrope-dipole',
  label: 'Polytrope with a dipole field',
  blurb: 'The polytrope (n = 1 by default) threaded by a dipole field confined to the star (beta 1000 at its center): its field lines close inside it, and at this beta it stays in balance.',
  params: [...starParams, betaParam],
  mhd: { potential: (p) => dipoleField(starProfile(p), p.beta) },
};

/** The polytrope with a random (Kolmogorov) field inside it (mhd/star_fields.ts). */
export const polytropeRandom: Problem = {
  ...polytrope,
  id: 'polytrope-random',
  label: 'Polytrope with a random field',
  blurb: 'The polytrope (n = 1 by default) threaded by a tangled field with a Kolmogorov spectrum, confined to the star (beta 1000 at its center on average).',
  params: [...starParams, betaParam, { id: 'seed', label: 'Random seed', value: 1, min: 1, max: 9999, step: 1, title: 'Which random field: each seed gives another' }],
  mhd: { potential: (p) => randomField(starProfile(p), p.beta, Math.round(p.seed)) },
};
