// A planar shock in air running into a bubble of helium: the shock-tube experiment of Haas &
// Sturtevant (1987), as Quirk & Karni (1996) set it up for computation. The bubble is helium
// with 28% air by mass, as theirs was: at the air's pressure and temperature that gives density
// 0.182 and gamma 1.648 from the mixing rule alone (species/mixture.ts), the values they used.
//
// Air and helium are species that set the equation of state, each with its own gamma (1.4 and
// 5/3). Code units: the air ahead of the shock has density 1 and pressure 1 (sound speed
// sqrt(1.4)); the box is the shock tube's cross-section, walls above and below.
import { momentCoefficients, mixMoments, speciesNuclides } from '../../../species/mixture';
import { composition } from '../../../units/composition';
import type { FlowDef } from '../flow';

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

interface Params {
  mach: number;
  xhe: number;
  radius: number;
}

const shockBubble: FlowDef<Params> = {
  name: 'shock-bubble',
  label: 'Shock hitting a helium bubble',
  tip: 'A planar shock in air (density 1, pressure 1, gamma 1.4) running toward +x into a bubble of helium at the same pressure and temperature, after Haas & Sturtevant (1987). Needs the species he4 and air, in that order, each with its own gamma.',
  wholeBox: true,
  fields: {
    mach: { label: 'Shock Mach number', default: 1.22, min: 1.01, max: 5, step: 0.01, tip: 'Mach number of the shock in the air' },
    xhe: { label: 'Helium in the bubble', default: 0.72, min: 0, max: 1, step: 0.01, tip: 'Mass fraction of helium in the bubble, the rest air (0.72 in the experiment, where the bubble took in some air)' },
    radius: { label: 'Bubble radius', default: 0.15, min: 0.05, max: 0.2, step: 0.01, tip: 'Radius of the bubble, in units of the tube width' },
  },
  init: (ctx, p) => {
    const ndim = ctx.ndim;
    const ps = postShock(p.mach);
    const xHe = p.xhe;
    const r = p.radius;
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

export const flows = [shockBubble];
