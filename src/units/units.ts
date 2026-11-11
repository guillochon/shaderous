// Physical units for problems that stand for something real.
//
// The GPU always works in code units, chosen so that the numbers it handles are near 1:
// single precision keeps 7 digits at any magnitude, but it overflows past 3e38, flushes
// anything below about 1e-38 to zero, and the solvers' floors and epsilons assume values
// near 1. A UnitSystem says what one code unit of length, mass and time is in SI; every other
// unit follows. It touches the run only through the dimensional constants it fixes (G, the
// radiation constant, cooling rates), and otherwise converts on the way in (parameters) and
// on the way out (display, files).
//
// Temperature. Code temperature is p_gas / rho, as every EOS module has it (gas constant 1).
// In kelvin that is T = mu * theta * T_code, with theta = m_u v^2 / k_B for the code velocity
// unit v. theta belongs to the units; mu belongs to the composition, which is why the two are
// kept apart: once mu varies from cell to cell, theta stays one number.

import { SI } from './constants';
import { meanMolecularWeight, type Composition } from './composition';

export interface UnitSystem {
  name: string;
  /** SI value of one code unit of length (m), mass (kg) and time (s) */
  length: number;
  mass: number;
  time: number;
}

/** Exponents of mass, length and time. */
export type Dim = readonly [m: number, l: number, t: number];

export const DIMS = {
  length: [0, 1, 0],
  mass: [1, 0, 0],
  time: [0, 0, 1],
  velocity: [0, 1, -1],
  acceleration: [0, 1, -2],
  density: [1, -3, 0],
  /** also energy density */
  pressure: [1, -1, -2],
  energy: [1, 2, -2],
  /** also the gravitational potential */
  specificEnergy: [0, 2, -2],
  power: [1, 2, -3],
  /** heating or cooling rate per unit volume */
  powerDensity: [1, -1, -3],
  /** kinematic viscosity, thermal diffusivity */
  diffusivity: [0, 2, -1],
  /** G */
  gravConst: [-1, 3, -2],
  /** the radiation constant without its K^-4 */
  arad: [1, -1, -2],
} as const satisfies Record<string, Dim>;

export type Quantity = keyof typeof DIMS;

/** SI value of one code unit of a quantity. */
export function unitOf(u: UnitSystem, q: Quantity | Dim): number {
  const [m, l, t] = typeof q === 'string' ? DIMS[q] : q;
  return u.mass ** m * u.length ** l * u.time ** t;
}

export const toCode = (u: UnitSystem, q: Quantity | Dim, valueSI: number): number => valueSI / unitOf(u, q);
export const fromCode = (u: UnitSystem, q: Quantity | Dim, value: number): number => value * unitOf(u, q);

/** Units from a length and a mass, with the time unit chosen so that G = 1 in code units. */
export function gravityUnits(name: string, length: number, mass: number): UnitSystem {
  return { name, length, mass, time: Math.sqrt(length ** 3 / (SI.G * mass)) };
}

/** Units from a length, a density and a velocity: for gas with no self-gravity to fix the clock. */
export function flowUnits(name: string, length: number, density: number, velocity: number): UnitSystem {
  return { name, length, mass: density * length ** 3, time: length / velocity };
}

/** G in code units. */
export const codeG = (u: UnitSystem): number => toCode(u, 'gravConst', SI.G);

/** Kelvin per code temperature unit for gas of mean molecular weight 1 (theta above). */
export const thetaKelvin = (u: UnitSystem): number => (SI.mu * unitOf(u, 'velocity') ** 2) / SI.kB;

/** Kelvin per code temperature unit for gas of mean molecular weight mu. */
export const kelvinPerCodeT = (u: UnitSystem, mu: number): number => mu * thetaKelvin(u);

/**
 * Gas constant and radiation constant in code units, for EosParams. The gas constant stays 1
 * (code T = p_gas / rho); mu enters the radiation constant through the kelvin scale. When mu
 * varies per cell, the EOS will take theta and mu instead.
 */
export function eosConstants(u: UnitSystem, comp: Composition): { R: number; arad: number } {
  const tk = kelvinPerCodeT(u, meanMolecularWeight(comp));
  return { R: 1, arad: toCode(u, 'arad', SI.arad * tk ** 4) };
}

// ---- Display ----

export type DisplaySystem = 'code' | 'si' | 'cgs';

const SI_LABELS: Record<Quantity, string> = {
  length: 'm', mass: 'kg', time: 's', velocity: 'm/s', acceleration: 'm/s²', density: 'kg/m³',
  pressure: 'Pa', energy: 'J', specificEnergy: 'J/kg', power: 'W', powerDensity: 'W/m³',
  diffusivity: 'm²/s', gravConst: 'm³/(kg s²)', arad: 'J/m³',
};
const CGS_LABELS: Record<Quantity, string> = {
  length: 'cm', mass: 'g', time: 's', velocity: 'cm/s', acceleration: 'cm/s²', density: 'g/cm³',
  pressure: 'dyn/cm²', energy: 'erg', specificEnergy: 'erg/g', power: 'erg/s', powerDensity: 'erg/(cm³ s)',
  diffusivity: 'cm²/s', gravConst: 'cm³/(g s²)', arad: 'erg/cm³',
};

/** SI value to cgs: kg -> 1e3 g, m -> 1e2 cm. */
const siToCgs = ([m, l]: Dim): number => 1e3 ** m * 1e2 ** l;

/**
 * A code value in the chosen system, with its unit label. Without units, or in 'code', the
 * value is returned as it is with no label. Temperature is its own case (displayTemperature).
 */
export function display(u: UnitSystem | undefined, q: Quantity, value: number, sys: DisplaySystem): { value: number; unit: string } {
  if (!u || sys === 'code') return { value, unit: '' };
  const si = fromCode(u, q, value);
  return sys === 'si' ? { value: si, unit: SI_LABELS[q] } : { value: si * siToCgs(DIMS[q]), unit: CGS_LABELS[q] };
}

/** A code temperature in kelvin, given the gas's mean molecular weight (here, or later per cell). */
export function displayTemperature(u: UnitSystem | undefined, mu: number, value: number, sys: DisplaySystem): { value: number; unit: string } {
  if (!u || sys === 'code') return { value, unit: '' };
  return { value: value * kelvinPerCodeT(u, mu), unit: 'K' };
}
