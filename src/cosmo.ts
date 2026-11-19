/**
 * Comoving coordinates (docs/numerics.md section 22): the background expansion, in
 * float64 on the CPU. The GPU advances the scale factor itself, step by step, in the clock pass
 * (hydro/clock.wgsl); this is the reference it is held to, and what converts redshifts to times.
 *
 * Code units: time in 1/H0, so H(a = 1) = 1; the mean comoving density is 1, so the Poisson
 * equation's G is 3 Omega_m / (8 pi). The clock starts at t = 0 at a = aStart.
 */

import { ASTRO, SI } from './units/constants';
import type { UnitSystem } from './units/units';

export interface Cosmology {
  /** matter density today, in units of the critical density */
  omegaM: number;
  /** the cosmological constant's */
  omegaL: number;
  /** scale factor at t = 0 */
  aStart: number;
  /** the Hubble constant in units of 100 km/s/Mpc (physical units only) */
  h?: number;
  /** the box's comoving side in Mpc/h, which gives the run physical units (cosmoUnits) */
  boxMpc?: number;
}

/**
 * The physical units of a run in comoving coordinates whose box has a size (Cosmology.boxMpc):
 * comoving length in box / boxMpc Mpc/h, time in 1/H0, and mass such that a code density of 1
 * is the mean matter density (omegaM times the critical density today), which is what makes
 * G = 3 omegaM / (8 pi) in code units. Velocities (proper, peculiar) are then in units of
 * H0 times the length unit. Densities and pressures in the state are comoving; the display
 * shows proper ones (comovingFactor).
 */
export function cosmoUnits(c: Cosmology, box: number): UnitSystem | undefined {
  if (!c.boxMpc) return undefined;
  const h = c.h ?? 0.7;
  const H0 = (100 * h * 1e3) / (1e6 * ASTRO.pc); // 1/s
  const length = (c.boxMpc * 1e6 * ASTRO.pc) / h / box;
  const rhoCrit = (3 * H0 * H0) / (8 * Math.PI * SI.G);
  return { name: `${c.boxMpc} Mpc/h box`, length, mass: c.omegaM * rhoCrit * length ** 3, time: 1 / H0 };
}

/**
 * The factor from a comoving code value of the given dimensions to the proper one at scale
 * factor a, as the state holds them (docs/numerics.md section 22): lengths are comoving (times
 * a), densities, pressures and energy densities comoving (a^-3 times: those with a mass and a
 * length to the -1 or -3, that is, per unit volume); velocities, specific energies,
 * temperatures, masses and times are as they are.
 */
export function comovingFactor(dim: readonly number[], a: number): number {
  const [m, l, t] = dim;
  if (m === 0 && l === 1 && t === 0) return a; // length
  if (m === 1 && (l === -3 || l === -1)) return a ** -3; // per unit volume: density, pressure, energy density
  return 1;
}

/** Curvature: what makes the densities sum to 1. */
export const omegaK = (c: Cosmology): number => 1 - c.omegaM - c.omegaL;

/** The expansion rate H(a) = da/dt / a, in units of H0. */
export function hubble(c: Cosmology, a: number): number {
  return Math.sqrt(c.omegaM / (a * a * a) + omegaK(c) / (a * a) + c.omegaL);
}

/** G in code units: the mean comoving density is 1. */
export const cosmoG = (c: Cosmology): number => (3 * c.omegaM) / (8 * Math.PI);

/** One RK4 step of da/dt = a H(a), as the clock pass takes it (in float32 there). */
export function rk4(c: Cosmology, a: number, dt: number): number {
  const f = (x: number) => x * hubble(c, x);
  const k1 = f(a);
  const k2 = f(a + 0.5 * dt * k1);
  const k3 = f(a + 0.5 * dt * k2);
  const k4 = f(a + dt * k3);
  return a + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

/**
 * The scale factor at time t, by integrating da/dt = a H(a) from aStart in steps that move a by
 * at most 1e-3 of itself.
 */
export function scaleFactor(c: Cosmology, t: number): number {
  let a = c.aStart;
  let left = t;
  while (left > 0) {
    const dt = Math.min(left, 1e-3 / hubble(c, a));
    a = rk4(c, a, dt);
    left -= dt;
  }
  return a;
}

/** The time at which the scale factor reaches a (from aStart): the integral of da / (a^2 H). */
export function timeOf(c: Cosmology, a: number): number {
  if (a <= c.aStart) return 0;
  // Simpson's rule in ln a, finely enough for float64 agreement with scaleFactor.
  const n = 4096;
  const l0 = Math.log(c.aStart);
  const h = (Math.log(a) - l0) / n;
  const g = (l: number) => 1 / hubble(c, Math.exp(l));
  let s = g(l0) + g(l0 + n * h);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * g(l0 + i * h);
  return (s * h) / 3;
}

/** The time at which a redshift is reached. */
export const timeAtRedshift = (c: Cosmology, z: number): number => timeOf(c, 1 / (1 + z));

/** Einstein-de Sitter: a(t) = (a0^1.5 + 1.5 t)^(2/3) with H0 = 1, exactly. */
export const edsScaleFactor = (aStart: number, t: number): number => Math.pow(Math.pow(aStart, 1.5) + 1.5 * t, 2 / 3);
