import type { Problem } from './types';
import { sodX, sodY, sodZ, sodDiag, strongShock } from './shocktube';
import { sedov, sedovRad } from './sedov';
import { kelvinHelmholtz } from './kelvin_helmholtz';
import { rayleighTaylor } from './rayleigh_taylor';
import { advection } from './advection';
import { jeans } from './jeans';
import { polytrope } from './polytrope';
import { evrard } from './evrard';
import { tidalDisruption } from './tidal_disruption';
import { sphere } from './sphere';
import { isothermalCollapse } from './isothermal';

export const PROBLEMS: Problem[] = [
  sedov,
  kelvinHelmholtz,
  rayleighTaylor,
  tidalDisruption,
  evrard,
  polytrope,
  isothermalCollapse,
  sedovRad,
  sphere,
  jeans,
  sodX,
  sodY,
  sodZ,
  sodDiag,
  strongShock,
  advection,
];

export const problemById = (id: string): Problem => {
  const p = PROBLEMS.find((q) => q.id === id);
  if (!p) throw new Error(`unknown problem: ${id}`);
  return p;
};

export * from './types';
