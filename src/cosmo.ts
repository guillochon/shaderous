/**
 * Comoving coordinates (plans/coordinate-systems-plan.md, Part C): the background expansion, in
 * float64 on the CPU. The GPU advances the scale factor itself, step by step, in the clock pass
 * (hydro/clock.wgsl); this is the reference it is held to, and what converts redshifts to times.
 *
 * Code units: time in 1/H0, so H(a = 1) = 1; the mean comoving density is 1, so the Poisson
 * equation's G is 3 Omega_m / (8 pi). The clock starts at t = 0 at a = aStart.
 */

export interface Cosmology {
  /** matter density today, in units of the critical density */
  omegaM: number;
  /** the cosmological constant's */
  omegaL: number;
  /** scale factor at t = 0 */
  aStart: number;
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
