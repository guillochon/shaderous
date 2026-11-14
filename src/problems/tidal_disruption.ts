import { all, OUTFLOW_ONLY, type Problem } from './types';
import { laneEmden, polytropeScales } from './lane_emden';
import { ASTRO } from '../units/constants';
import { COMPOSITIONS } from '../units/composition';
import { codeG, gravityUnits } from '../units/units';

// Tidal disruption: a polytropic star passed by a point mass on a parabolic orbit.
//
// Units: G = M_star = R_star = 1, so the star's dynamical time is 1. The star has the Sun's
// mass and radius: one code unit is a solar radius, a solar mass, and
// sqrt(R_sun^3 / G M_sun) = 1593 s. It is a polytrope of index n, structural gamma = 1 + 1/n:
// n = 3/2 and gamma = 5/3 (convective, like a low-mass star), or n = 3 and gamma = 4/3
// (centrally concentrated, like the Sun's interior). Its gas is ideal with Gamma = 5/3 in
// either case, as in Guillochon & Ramirez-Ruiz (2013): a star's structure is softer than its
// gas only because radiation carries its heat outward, and once disrupted the gas responds
// with its own Gamma. (A gamma = 4/3 gas would also leave the star marginally stable.) The
// run is in the star's rest frame: the frame
// accelerates with the gas's center of mass, so the star stays at the box center while the
// point mass swings past. The orbit lies in the x-y plane with pericenter on the +x axis.
//
//   mass ratio q = M_perturber / M_star        tidal radius r_t = R_star q^(1/3)
//   impact parameter beta = r_t / r_p          beta = 1 puts pericenter at the tidal radius
//
// How deep the star must pass to be torn apart depends on how concentrated it is: Guillochon &
// Ramirez-Ruiz (2013) find full disruption from beta of about 0.9 for gamma = 5/3 and 1.85 for
// gamma = 4/3. The variants sit on either side: partial at 0.6 and 1.0, full at 1.5 and 2.5.
// All have a million-solar-mass black hole. The gamma = 4/3 star's dense core needs a finer
// grid, so its variants open on one: at 128 cells across with two levels it settles to about
// twice the Lane-Emden central density, is bound more tightly, and survives even beta = 4. On
// its own grid (finest cell 1/32 of its radius, central density 15% high) its peak density
// recollapses after beta = 1.8 and keeps falling after beta = 2 (by t = 12), so it is destroyed
// from between the two, as Guillochon & Ramirez-Ruiz find; the full variant opens at 2.5, well
// past that. (The census's self-bound core, an energy test, still holds a few tenths of a
// solar mass at t = 12 even at beta = 2.5, and keeps shrinking: some gas always moves with
// the core, as they note. Peak density is the surer test.)
//
// The boundaries let gas leave but not enter. Once debris starts to leave, the frame follows
// the gas that is still in the box. Every variant has a box 16 stellar radii across, so the debris
// and what is left of the star stay on the grid for longer; TDE.box, 8, is the box the validation
// tests were calibrated in.
export const TDE = { n: 1.5, M: 1, R: 1, G: 1, start: 3, rhoAmb: 1e-6, rhoCut: 1e-4, box: 8 };

export const TDE_UNITS = gravityUnits('solar', TDE.R * ASTRO.Rsun, TDE.M * ASTRO.Msun);

/** A star of polytropic index n with the code's mass and radius: its scales and profile. */
function tdeStar(n: number) {
  const le = laneEmden(n);
  return { ...polytropeScales(le, TDE.M, TDE.R, TDE.G), theta: (s: number) => le.theta(s) };
}
/** The gamma = 5/3 (n = 3/2) star. */
export const TDE_STAR = tdeStar(TDE.n);

/** Relative parabolic orbit about the star at the origin: state at separation `r` before pericenter. */
export function parabolicOrbit(q: number, beta: number, r: number) {
  const rt = TDE.R * Math.cbrt(q);
  const rp = rt / beta;
  const mu = TDE.G * TDE.M * (1 + q);
  const f = -Math.acos(Math.min(1, Math.max(-1, (2 * rp) / r - 1))); // true anomaly, negative on the way in
  const speed = Math.sqrt(mu / (2 * rp));
  // Time until pericenter, from Barker's equation.
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
  // Start a few tidal radii out, and never inside pericenter.
  return parabolicOrbit(p.q, p.beta, Math.max(TDE.start * rt, (1.5 * rt) / p.beta));
};

/** The gas's adiabatic index, whatever the star's polytropic index. */
const TDE_GAMMA = 5 / 3;

/** Specific internal energy at the center of a star of index n: p / ((Gamma - 1) rho), p = K rho^(1 + 1/n). */
const centralEint = (n: number, st: { K: number; rhoc: number }) => (st.K * st.rhoc ** (1 / n)) / (TDE_GAMMA - 1);

interface TdeVariant {
  id: string;
  blurb: string;
  /** polytropic index of the star; its gas has gamma = 1 + 1/n */
  n: number;
  /** default impact parameter */
  beta: number;
  /** finest cells per side and levels of refinement it opens on */
  grid: { n: number; amr: number };
  /** side of the box, in stellar radii (default TDE.box) */
  box?: number;
  /**
   * Where the star is cut off and the density of the ambient gas, as fractions of the central
   * density (defaults TDE.rhoCut and TDE.rhoAmb). The ambient gas carries the pressure at the
   * cut, so its specific internal energy relative to the center's is about
   * cut^(1 + 1/n) / ambient: the defaults put it at a fifth for n = 3/2, but at nearly five
   * times for n = 3, whose star needs lower values.
   */
  rhoCut?: number;
  rhoAmb?: number;
}

/** A tidal disruption of a star of index n, opening with impact parameter beta. */
function tde(v: TdeVariant): Problem {
  const n = v.n;
  const st = n === TDE.n ? TDE_STAR : tdeStar(n);
  const cut = (v.rhoCut ?? TDE.rhoCut) * st.rhoc;
  const amb = (v.rhoAmb ?? TDE.rhoAmb) * st.rhoc;
  return {
    id: v.id,
    label: 'Tidal disruption',
    blurb: v.blurb,
    eos: 'ideal',
    eosParams: { gamma: TDE_GAMMA },
    bcLo: all(OUTFLOW_ONLY),
    bcHi: all(OUTFLOW_ONLY),
    gravity: 'isolated',
    G: codeG(TDE_UNITS),
    units: TDE_UNITS,
    composition: COMPOSITIONS.solar,
    box: v.box ?? TDE.box,
    grid: { ndim: 3, ...v.grid },
    tEnd: Infinity,
    checkpointDt: 0.5,
    rhoFloor: 1e-10,
    pFloor: 1e-14,
    // Ambient gas and the outermost wisps of debris: no hotter than the star's interior (1.5 for
    // the n = 3/2 star, in proportion to the central internal energy for another), and no
    // faster than a few times the orbital speed at pericenter.
    fluff: { rho: 1e-5 * st.rhoc, u: (1.5 * centralEint(n, st)) / centralEint(TDE.n, TDE_STAR), v: 20 },
    // Look down on the orbital plane.
    view: { field: 0, log: true, mode: 'volume', pitch: 1.15 },
    params: [
      { id: 'q', label: 'Perturber : star mass', title: 'Mass of the black hole over the star’s. Sets the orbit: changing it starts the run again', value: 1e6, min: 1, max: 1e6, factor: 2 },
      { id: 'beta', label: 'Impact parameter β', title: 'Tidal radius over pericenter distance: above 1 the star passes inside the tidal radius. Changing it starts the run again', value: v.beta, min: 0.2, max: 8, step: 0.1 },
    ],
    frame: 'gas',
    star: { n, K: st.K, rhoCut: cut, rhoAmb: amb },
    masses: (p, G) => {
      const o = orbitFor(p);
      return [{ gm: G * TDE.M * p.q, pos: o.pos, vel: o.vel }];
    },
    // Debris energies about the black hole, in units of the spread a disruption gives them,
    // G M_bh R_star / r_t^2, with the time of pericenter for their return rate. Only the star's
    // gas (the passive scalar) is binned, not the ambient medium.
    debris: (p, G) => {
      const o = orbitFor(p);
      return { mass: 0, de: (G * TDE.M * p.q * TDE.R) / (o.rt * o.rt), tPeri: o.tPeri, refMass: TDE.M, scalar: true };
    },
    init: () => {
      const { rhoc, K } = st;
      const g = 1 + 1 / n;
      return (x, y, z) => {
        const rho = rhoc * Math.pow(st.theta(Math.hypot(x, y, z) / TDE.R), n);
        // Light, warm ambient gas in rough pressure balance with the star's outermost layer.
        const p = K * Math.pow(Math.max(rho, cut), g);
        return { rho: Math.max(rho, amb), vx: 0, vy: 0, vz: 0, p, s: rho > cut ? 1 : 0 };
      };
    },
  };
}

// The gamma = 5/3 star: finest cells of 1/16 of its radius, 256 across the box with three levels.
const N15_SETUP = { grid: { n: 256, amr: 3 }, box: 16 };

/** The default: a gamma = 5/3 star that loses its outer layers and survives. */
export const tidalDisruption = tde({
  id: 'tidal-disruption',
  blurb: 'A Sun-like γ = 5/3 star grazes a million-solar-mass black hole (β = 0.6): it loses its outer layers and survives. In the star’s rest frame.',
  n: 1.5,
  beta: 0.6,
  ...N15_SETUP,
});
export const tidalDisruptionFull = tde({
  id: 'tidal-disruption-full',
  blurb: 'A Sun-like γ = 5/3 star passes deep inside the tidal radius of a million-solar-mass black hole (β = 1.5) and is torn apart. In the star’s rest frame.',
  n: 1.5,
  beta: 1.5,
  ...N15_SETUP,
});
// The gamma = 4/3 star: finest cells of 1/32 of its radius, 512 across the box with four levels;
// an ambient medium as dense as the gamma = 5/3 star's in absolute terms, and a cut low enough to
// keep it cooler than the star's center.
const N3_SETUP = { grid: { n: 512, amr: 4 }, box: 16, rhoCut: 2e-6, rhoAmb: 1e-7 };

export const tidalDisruptionN3 = tde({
  id: 'tidal-disruption-n3',
  blurb: 'A centrally concentrated γ = 4/3 star grazes a million-solar-mass black hole (β = 1): it loses its outer layers and survives. In the star’s rest frame.',
  n: 3,
  beta: 1,
  ...N3_SETUP,
});
export const tidalDisruptionN3Full = tde({
  id: 'tidal-disruption-n3-full',
  blurb: 'A centrally concentrated γ = 4/3 star passes deep inside the tidal radius of a million-solar-mass black hole (β = 2.5) and is torn apart. In the star’s rest frame.',
  n: 3,
  beta: 2.5,
  ...N3_SETUP,
});

export const tdeOrbit = orbitFor;
