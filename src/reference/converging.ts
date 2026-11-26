// Exact solutions of two converging flows of an ideal gas, in planar (k = 1), cylindrical
// (k = 2) and spherical (k = 3) symmetry, as ExactPack has them (noh/noh1.py, noh2/noh2.py).
//
// Noh (1987): cold gas of density rho0 falls at speed u0 toward the center. A shock leaves the
// center at speed u0 (gamma - 1) / 2. Behind it the gas is at rest, at density
// rho0 ((gamma + 1) / (gamma - 1))^k, with specific internal energy u0^2 / 2. Ahead of it the
// gas still falls at u0 and piles up as it converges: rho0 (1 + u0 t / r)^(k - 1).
//
// Uniform collapse (ExactPack's noh2, Coggeshall's problem 1 with b = 0): u = -r / (tau - t),
// so the gas reaches the center all at once at t = tau. The density and the specific internal
// energy stay uniform, rho0 (1 - t / tau)^-k and e0 (1 - t / tau)^-(gamma - 1) k, along the
// adiabat. There is no shock and no pressure gradient.

export interface RadialState {
  rho: number;
  /** radial velocity (negative: inward) */
  u: number;
  p: number;
}

/**
 * The Noh implosion at distance r from the center at time t. `p0` is the starting pressure,
 * which the exact solution takes as zero. Here it follows the falling gas's adiabat, and the
 * post-shock state ignores it, which is good to order p0 / (rho0 u0^2).
 */
export function nohExact(k: 1 | 2 | 3, gamma: number, q: { rho0: number; u0: number; p0?: number }, r: number, t: number): RadialState {
  const rs = (q.u0 * t * (gamma - 1)) / 2;
  if (r < rs) {
    const rho = q.rho0 * ((gamma + 1) / (gamma - 1)) ** k;
    return { rho, u: 0, p: (gamma - 1) * rho * 0.5 * q.u0 * q.u0 };
  }
  const f = (1 + (q.u0 * t) / Math.max(r, 1e-300)) ** (k - 1);
  return { rho: q.rho0 * f, u: -q.u0, p: (q.p0 ?? 0) * f ** gamma };
}

/** Where the Noh shock is at time t. */
export const nohShockRadius = (gamma: number, u0: number, t: number): number => (u0 * t * (gamma - 1)) / 2;

/** The uniform collapse at distance r from the center at time t (t < tau). */
export function uniformCollapseExact(k: 1 | 2 | 3, gamma: number, q: { rho0: number; e0: number; tau: number }, r: number, t: number): RadialState {
  const s = 1 - t / q.tau;
  const rho = q.rho0 * s ** -k;
  const e = q.e0 * s ** (-(gamma - 1) * k);
  return { rho, u: -r / (q.tau - t), p: (gamma - 1) * rho * e };
}
