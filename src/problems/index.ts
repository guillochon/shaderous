import type { Problem } from './types';
import { sodX, sodY, sodZ, sodDiag, strongShock } from './shocktube';
import { sedov, sedovRad } from './sedov';
import { kelvinHelmholtz } from './kelvin_helmholtz';
import { rayleighTaylor } from './rayleigh_taylor';
import { forwardStep } from './forward_step';
import { advection } from './advection';
import { jeans } from './jeans';
import { polytrope } from './polytrope';
import { evrard } from './evrard';
import { tidalDisruption } from './tidal_disruption';
import { sphere } from './sphere';
import { binary } from './binary';
import { thermalInstability } from './thermal_instability';
import { isothermalCollapse } from './isothermal';
import { empty } from './empty';

/** A problem in the menu: one or more variants of the same setup, each a preset of its own. */
export interface ProblemFamily {
  id: string;
  label: string;
  variants: { label: string; problem: Problem }[];
}

/** A problem with a single variant, which the menu calls "Standard". */
const single = (p: Problem): ProblemFamily => ({ id: p.id, label: p.label, variants: [{ label: 'Standard', problem: p }] });

/** The problem menu, in loose groups. The first variant of a problem is the one it opens with. */
export const MENU: { group: string; families: ProblemFamily[] }[] = [
  {
    group: 'Instabilities',
    families: [single(kelvinHelmholtz), single(rayleighTaylor), single(thermalInstability), single(jeans)],
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
    ],
  },
  {
    group: 'Self-gravity',
    families: [single(polytrope), single(evrard), single(sphere), single(isothermalCollapse)],
  },
  {
    group: 'Point masses',
    families: [single(tidalDisruption), single(binary)],
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
