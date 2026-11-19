// A setup's physical units, from its `domain.units` (setup/spec.ts UnitsSpec).
import { ASTRO } from '../units/constants';
import { flowUnits, gravityUnits, type UnitSystem } from '../units/units';
import type { UnitsSpec, UnitValue } from './spec';

/** SI value of each unit a setup may name. */
export const UNIT_SI: Record<string, number> = {
  m: 1, cm: 1e-2, km: 1e3, Rearth: ASTRO.Rearth, Rsun: ASTRO.Rsun, au: ASTRO.au, pc: ASTRO.pc,
  kg: 1, g: 1e-3, Mearth: ASTRO.Mearth, Msun: ASTRO.Msun,
  'kg/m3': 1, 'g/cm3': 1e3,
  'm/s': 1, 'km/s': 1e3, 'cm/s': 1e-2,
};

export const LENGTH_UNITS = ['m', 'cm', 'km', 'Rearth', 'Rsun', 'au', 'pc'] as const;
export const MASS_UNITS = ['kg', 'g', 'Mearth', 'Msun'] as const;
export const DENSITY_UNITS = ['kg/m3', 'g/cm3'] as const;
export const VELOCITY_UNITS = ['m/s', 'km/s', 'cm/s'] as const;

/** SI value of one code unit, from a UnitValue; `ref` resolves a part's quantity ("star.mass"). */
function si(v: UnitValue, ref: (name: string) => number): number {
  const unit = UNIT_SI[v.unit];
  if (unit === undefined) throw new Error(`unknown unit: ${v.unit}`);
  const per = v.per === undefined ? 1 : typeof v.per === 'number' ? v.per : ref(v.per);
  // Written so that `per` 1 changes nothing, to the last bit.
  return per === 1 ? v.value * unit : (v.value * unit) / per;
}

export function unitSystem(u: UnitsSpec, ref: (name: string) => number): UnitSystem {
  const name = u.name ?? 'custom';
  const length = si(u.length, ref);
  if (u.mass) return gravityUnits(name, length, si(u.mass, ref));
  if (u.density && u.velocity) return flowUnits(name, length, si(u.density, ref), si(u.velocity, ref));
  throw new Error('units need a length and a mass, or a length, a density and a velocity');
}

/** Whether the units follow a part's quantity (and so the parameters that change it). */
export const unitsFollowParts = (u: UnitsSpec): boolean => [u.length, u.mass, u.density, u.velocity].some((v) => typeof v?.per === 'string');
