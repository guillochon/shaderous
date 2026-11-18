// What the readouts show a code value as: the problem's units in the chosen system.

import { fmt3 } from './diag_plot';
import { meanMolecularWeight, COMPOSITIONS, type Composition } from '../units/composition';
import { comovingFactor } from '../cosmo';
import { DIMS, displayLabel, displayScale, kelvinPerCodeT, type Dim, type DisplaySystem, type Quantity, type UnitSystem } from '../units/units';

/** What a value measures: a quantity, a bare dimension, a temperature, or nothing (a pure number). */
export type Measure = Quantity | Dim | 'temperature' | null;

/** What each view field (problems/types.ts FIELDS, by index) measures; the entropy p / rho^gamma depends on gamma. */
export function fieldMeasure(field: number, gamma: number): Measure {
  switch (field) {
    case 0: return 'density';
    case 1: return 'temperature';
    case 3: return 'pressure';
    case 4: case 5: case 6: case 7: return 'velocity';
    case 9: return [1 - gamma, 3 * gamma - 1, -2];
    case 10: case 30: return 'powerDensity';
    case 11: case 12: return 'rate';
    case 13: return [1, -4, 0];
    case 15: return 'specificEnergy';
    case 16: return 'pressure';
    case 18: return 'time';
    // Radiation: its energy density, temperature and flux (energy per area and time).
    case 31: return 'pressure';
    case 32: return 'temperature';
    case 33: return [1, 0, -3];
    // Passive scalar, Mach number, refinement level, Jeans length in cells, time bin; the
    // magnetic fields, shown in code units until a magnetized preset has physical ones; the
    // reduced flux, a cell's optical depth and r, pure numbers.
    default: return null;
  }
}

export interface Readout {
  /** The problem has units, so the choice of system means something. */
  physical: boolean;
  /** Factor from a code value to the displayed one. */
  scale(m: Measure): number;
  /** Unit label of the displayed value ('' for none). */
  label(m: Measure): string;
  /** The displayed value with its unit. */
  fmt(v: number, m: Measure): string;
}

/**
 * `scale`: comoving coordinates, the scale factor of the state shown. Code values stay comoving;
 * a physical system shows proper ones (src/cosmo.ts comovingFactor).
 */
export function readout(units: UnitSystem | undefined, composition: Composition | undefined, sys: DisplaySystem, scale?: number): Readout {
  const physical = !!units;
  const shown = physical && sys !== 'code';
  const mu = meanMolecularWeight(composition ?? COMPOSITIONS.solar);
  const proper = (m: Quantity | Dim) => (scale === undefined ? 1 : comovingFactor(typeof m === 'string' ? DIMS[m] : m, scale));
  const factor = (m: Measure) => (!shown || m === null ? 1 : m === 'temperature' ? kelvinPerCodeT(units!, mu) : displayScale(units, m, sys) * proper(m));
  const label = (m: Measure) => (!shown || m === null ? '' : m === 'temperature' ? 'K' : displayLabel(units, m, sys));
  return {
    physical,
    scale: factor,
    label,
    fmt: (v, m) => {
      const l = label(m);
      return l ? `${fmt3(v * factor(m))} ${l}` : fmt3(v * factor(m));
    },
  };
}
