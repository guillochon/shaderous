import type { Problem } from './types';
import { sodX, sodY, sodZ, sodDiag, strongShock } from './shocktube';
import { sedov, sedovRad } from './sedov';
import { kelvinHelmholtz, kelvinHelmholtzHeH } from './kelvin_helmholtz';
import { shockBubble } from './shock_bubble';
import { detonationCellular, detonationPulsating, detonationStable } from './detonation';
import { rayleighTaylor } from './rayleigh_taylor';
import { forwardStep } from './forward_step';
import { advection } from './advection';
import { jeans } from './jeans';
import { polytrope } from './polytrope';
import { evrard } from './evrard';
import { tidalDisruption, tidalDisruptionFull, tidalDisruptionN3, tidalDisruptionN3Full } from './tidal_disruption';
import { sphere } from './sphere';
import { binary } from './binary';
import { thermalInstability } from './thermal_instability';
import { isothermalCollapse } from './isothermal';
import { empty } from './empty';
import { wdCollision } from './stellar_collision';
import { whiteDwarfStar } from './white_dwarf_star';
import { alfvenWave, brioWu, fieldLoop, mhdBlast, mhdRotor, orszagTang } from './mhd';

/** A problem in the menu: one or more variants of the same setup, each a preset of its own. */
export interface ProblemFamily {
  id: string;
  label: string;
  variants: { label: string; problem: Problem }[];
}

/** A problem with a single variant, which the menu calls "Standard". */
const single = (p: Problem): ProblemFamily => ({ id: p.id, label: p.label, variants: [{ label: 'Standard', problem: p }] });

const kelvinHelmholtzFamily: ProblemFamily = {
  id: 'kelvin-helmholtz',
  label: 'Kelvin-Helmholtz',
  variants: [
    { label: 'Standard', problem: kelvinHelmholtz },
    { label: 'Helium in hydrogen', problem: kelvinHelmholtzHeH },
  ],
};

/** The problem menu, in loose groups. The first variant of a problem is the one it opens with. */
export const MENU: { group: string; families: ProblemFamily[] }[] = [
  {
    group: 'Instabilities',
    families: [kelvinHelmholtzFamily, single(rayleighTaylor), single(thermalInstability), single(jeans)],
  },
  {
    group: 'Shocks and blasts',
    families: [
      {
        id: 'sedov',
        label: 'Sedov-Taylor blast',
        variants: [
          { label: 'Ideal gas', problem: sedov },
          { label: 'Radiation dominated', problem: sedovRad },
        ],
      },
      {
        id: 'sod',
        label: 'Sod shock tube',
        variants: [
          { label: 'Along x', problem: sodX },
          { label: 'Along y', problem: sodY },
          { label: 'Along z', problem: sodZ },
          { label: 'Diagonal', problem: sodDiag },
        ],
      },
      single(strongShock),
      single(forwardStep),
      single(shockBubble),
      {
        id: 'detonation',
        label: 'Detonation',
        variants: [
          { label: 'Stable', problem: detonationStable },
          { label: 'Pulsating', problem: detonationPulsating },
          { label: 'Cellular (2D)', problem: detonationCellular },
        ],
      },
    ],
  },
  {
    group: 'Self-gravity',
    families: [
      single(polytrope),
      single(whiteDwarfStar),
      single(evrard),
      single(sphere),
      single(isothermalCollapse),
      { id: 'stellar-collision', label: 'Stellar collision', variants: [{ label: 'White dwarfs (C/O)', problem: wdCollision }] },
    ],
  },
  {
    group: 'Point masses',
    families: [
      {
        id: 'tidal-disruption',
        label: 'Tidal disruption',
        variants: [
          { label: 'Partial, γ = 5/3', problem: tidalDisruption },
          { label: 'Full, γ = 5/3', problem: tidalDisruptionFull },
          { label: 'Partial, γ = 4/3', problem: tidalDisruptionN3 },
          { label: 'Full, γ = 4/3', problem: tidalDisruptionN3Full },
        ],
      },
      single(binary),
    ],
  },
  {
    group: 'Magnetic fields',
    families: [single(orszagTang), single(mhdBlast), single(mhdRotor), single(brioWu), single(fieldLoop), single(alfvenWave)],
  },
  {
    group: 'Tests and sandbox',
    families: [single(advection), single(empty)],
  },
];

export const FAMILIES: ProblemFamily[] = MENU.flatMap((g) => g.families);
export const PROBLEMS: Problem[] = FAMILIES.flatMap((f) => f.variants.map((v) => v.problem));

export const problemById = (id: string): Problem => {
  const p = PROBLEMS.find((q) => q.id === id);
  if (!p) throw new Error(`unknown problem: ${id}`);
  return p;
};

/** The menu entry a preset belongs to. */
export const familyOf = (id: string): ProblemFamily => {
  const f = FAMILIES.find((q) => q.variants.some((v) => v.problem.id === id));
  if (!f) throw new Error(`unknown problem: ${id}`);
  return f;
};

/** A menu entry by its id. */
export const familyById = (id: string): ProblemFamily => {
  const f = FAMILIES.find((q) => q.id === id);
  if (!f) throw new Error(`unknown problem: ${id}`);
  return f;
};

export * from './types';
