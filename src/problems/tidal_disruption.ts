import { all, OUTFLOW_ONLY, type Problem } from './types';
import { laneEmden, polytropeScales } from './lane_emden';

// Tidal disruption: a polytropic star passed by a point mass on a parabolic orbit.
//
// Units: G = M_star = R_star = 1, so the star's dynamical time is 1. The star is an n = 3/2
// polytrope of gamma = 5/3 gas. The run is in the star's rest frame: the frame accelerates
// with the gas's centre of mass, so the star stays at the box centre while the point mass
// swings past. The orbit lies in the x-y plane with pericentre on the +x axis.
//
//   mass ratio q = M_perturber / M_star        tidal radius r_t = R_star q^(1/3)
//   impact parameter beta = r_t / r_p          beta = 1 puts pericentre at the tidal radius
//
// The boundaries let gas leave but not enter. Once debris starts to leave, the frame follows
// the gas that is still in the box.
export const TDE = { n: 1.5, M: 1, R: 1, G: 1, start: 3, rhoAmb: 1e-6, rhoCut: 1e-4, box: 8 };

const star = laneEmden(TDE.n);
export const TDE_STAR = { ...polytropeScales(star, TDE.M, TDE.R, TDE.G), theta: (s: number) => star.theta(s) };

/** Relative parabolic orbit about the star at the origin: state at separation `r` before pericentre. */
export function parabolicOrbit(q: number, beta: number, r: number) {
  const rt = TDE.R * Math.cbrt(q);
  const rp = rt / beta;
  const mu = TDE.G * TDE.M * (1 + q);
  const f = -Math.acos(Math.min(1, Math.max(-1, (2 * rp) / r - 1))); // true anomaly, negative on the way in
  const speed = Math.sqrt(mu / (2 * rp));
  // Time until pericentre, from Barker's equation.
  const D = Math.tan(-f / 2);
  const tPeri = Math.sqrt((2 * rp ** 3) / mu) * (D + D ** 3 / 3);
  return {
    rt,
    rp,
    tPeri,
    pos: [r * Math.cos(f), r * Math.sin(f), 0],
    vel: [-speed * Math.sin(f), speed * (1 + Math.cos(f)), 0],
  };
}

const orbitFor = (p: Record<string, number>) => {
  const rt = TDE.R * Math.cbrt(p.q);
  // Start a few tidal radii out, and never inside pericentre.
  return parabolicOrbit(p.q, p.beta, Math.max(TDE.start * rt, (1.5 * rt) / p.beta));
};

export const tidalDisruption: Problem = {
  id: 'tidal-disruption',
  label: 'Tidal disruption',
  blurb: 'A star passed by a point mass on a parabolic orbit, in the star\u2019s rest frame.',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW_ONLY),
  bcHi: all(OUTFLOW_ONLY),
  gravity: 'isolated',
  G: TDE.G,
  box: TDE.box,
  tEnd: 14,
  rhoFloor: 1e-10,
  pFloor: 1e-14,
  // Ambient gas and the outermost wisps of debris: no hotter than the star's interior, and no
  // faster than a few times the orbital speed at pericentre.
  fluff: { rho: 1e-5 * TDE_STAR.rhoc, u: 1.5, v: 20 },
  // Look down on the orbital plane.
  view: { field: 0, log: true, mode: 'volume', pitch: 1.15 },
  params: [
    { id: 'q', label: 'Perturber : star mass', value: 100, min: 1, max: 1e6 },
    { id: 'beta', label: 'Impact parameter \u03b2', value: 1, min: 0.2, max: 8, step: 0.1 },
  ],
  gasFrame: true,
  star: { n: TDE.n, K: TDE_STAR.K, rhoCut: TDE.rhoCut * TDE_STAR.rhoc, rhoAmb: TDE.rhoAmb * TDE_STAR.rhoc },
  masses: (p, G) => {
    const o = orbitFor(p);
    return [{ gm: G * TDE.M * p.q, pos: o.pos, vel: o.vel }];
  },
  init: () => {
    const { rhoc, K } = TDE_STAR;
    const g = 1 + 1 / TDE.n;
    return (x, y, z) => {
      const rho = rhoc * Math.pow(TDE_STAR.theta(Math.hypot(x, y, z) / TDE.R), TDE.n);
      // Light, warm ambient gas in rough pressure balance with the star's outermost layer.
      const p = K * Math.pow(Math.max(rho, TDE.rhoCut * rhoc), g);
      return { rho: Math.max(rho, TDE.rhoAmb * rhoc), vx: 0, vy: 0, vz: 0, p, s: rho > TDE.rhoCut * rhoc ? 1 : 0 };
    };
  },
};

export const tdeOrbit = orbitFor;
