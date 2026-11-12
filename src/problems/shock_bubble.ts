import { MASS_FRACTION_FIELD, OUTFLOW, REFLECT, type Problem } from './types';
import { composition } from '../units/composition';
import { momentCoefficients, mixMoments, speciesNuclides } from '../species/mixture';

// A planar shock in air running into a bubble of helium: the shock-tube experiment of Haas &
// Sturtevant (1987), as Quirk & Karni (1996) set it up for computation. The bubble is helium
// with 28% air by mass, as theirs was: at the air's pressure and temperature that gives density
// 0.182 and gamma 1.648 from the mixing rule alone (species/mixture.ts), the values they used.
//
// Air and helium are species that set the equation of state, each with its own gamma (1.4 and
// 5/3). Code units: the air ahead of the shock has density 1 and pressure 1 (sound speed
// sqrt(1.4)); the box is the shock tube's cross-section, walls above and below.

const GAMMA_AIR = 1.4;
const SPECIES_NAMES = ['he4', 'air'] as const;
/** Every gas here is neutral; the reference is air, so code T = p / rho in the air. */
const AIR = composition('air', { air: 1 }, 'neutral');
const COEFFS = momentCoefficients(speciesNuclides(SPECIES_NAMES)!, AIR, GAMMA_AIR, { air: GAMMA_AIR, he4: 5 / 3 });

/** The gas behind a shock of Mach number m running into gas at rest with density 1 and pressure 1. */
export function postShock(m: number, g = GAMMA_AIR): { rho: number; p: number; u: number } {
  const m2 = m * m;
  const rho = ((g + 1) * m2) / ((g - 1) * m2 + 2);
  const p = 1 + ((2 * g) / (g + 1)) * (m2 - 1);
  return { rho, p, u: m * Math.sqrt(g) * (1 - 1 / rho) };
}

/** Density of the bubble gas at the air's pressure and temperature, helium mass fraction xHe. */
export function bubbleDensity(xHe: number): number {
  return mixMoments(COEFFS, [0, 1])[0] / mixMoments(COEFFS, [xHe, 1 - xHe])[0];
}

export const shockBubble: Problem = {
  id: 'shock-bubble',
  label: 'Shock hitting a helium bubble',
  blurb:
    'A Mach 1.22 shock in air runs into a bubble of helium (with 28% air by mass), after Haas & Sturtevant (1987) and Quirk & Karni (1996). Each gas has its own gamma. The light bubble is flattened, turned into a vortex pair (a ring in 3D), and pierced by a jet of air along the axis.',
  eos: 'ideal',
  eosParams: { gamma: GAMMA_AIR },
  composition: AIR,
  species: SPECIES_NAMES,
  speciesGamma: { air: GAMMA_AIR, he4: 5 / 3 },
  // Along x the tube is open; across it, walls.
  bcLo: [OUTFLOW, REFLECT, REFLECT],
  bcHi: [OUTFLOW, REFLECT, REFLECT],
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: 1.2,
  checkpointDt: 0.1,
  supports2d: true,
  view: { field: MASS_FRACTION_FIELD, log: false, mode: 'slice', axis: 2 },
  params: [
    { id: 'mach', label: 'Shock Mach number', title: 'Mach number of the shock in the air. Changing it starts the run again', value: 1.22, min: 1.01, max: 5, step: 0.01 },
    { id: 'xhe', label: 'Helium in the bubble', title: 'Mass fraction of helium in the bubble, the rest air (0.72 in the experiment, where the bubble took in some air). Changing it starts the run again', value: 0.72, min: 0, max: 1, step: 0.01 },
    { id: 'radius', label: 'Bubble radius', title: 'Radius of the bubble, in units of the tube width. Changing it starts the run again', value: 0.15, min: 0.05, max: 0.2, step: 0.01 },
  ],
  init: (_n, _eos, ndim, params) => {
    const ps = postShock(params.mach);
    const xHe = params.xhe;
    const r = params.radius;
    const rhoB = bubbleDensity(xHe);
    // The bubble's left edge a little ahead of the shock, which starts inside the box.
    const cx = -0.25;
    const shock = cx - r - 0.04;
    return (x, y, z) => {
      if (x < shock) return { rho: ps.rho, vx: ps.u, vy: 0, vz: 0, p: ps.p, s: 0, X: [0, 1] };
      const d2 = (x - cx) ** 2 + y * y + (ndim === 3 ? z * z : 0);
      if (d2 < r * r) return { rho: rhoB, vx: 0, vy: 0, vz: 0, p: 1, s: 1, X: [xHe, 1 - xHe] };
      return { rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0, X: [0, 1] };
    };
  },
};
