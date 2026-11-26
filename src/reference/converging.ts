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

/**
 * Three of Coggeshall's similarity solutions of the Euler equations (Coggeshall 1991; ExactPack's
 * cog1, cog2 and cog6), with geometry g = 1, 2, 3 (ExactPack's k is g - 1), pressure in place of
 * ExactPack's Grüneisen Gamma times density times temperature:
 *
 *   1: free homologous expansion, u = r / t, rho = rho0 r^b t^(-b - g), the pressure uniform,
 *      Gamma rho0 T0 t^(-gamma g): each element on its adiabat with nothing pushing it.
 *   2: homologous expansion against a pressure rising outward, u = (2 / c) r / t with
 *      c = 2 + (gamma - 1) g, rho = rho0 r^b t^(-2 (b + g) / c),
 *      p = rho 2 (gamma - 1) g / ((b + 2) c^2) (r / t)^2.
 *   6: Kidder's isentropic compression, which needs gamma = (g + 2) / g (5/3, 2 and 3):
 *      u = -r t / (tau^2 - t^2), rho = rho0 r^b (tau^2 - t^2)^(-(g + b) / 2),
 *      p = rho (tau r / (tau^2 - t^2))^2 / (b + 2). Everything reaches the center at t = tau.
 *
 * `T0` (with `Gamma`, cog1's only) and `tau` (cog6's) are ignored where not used.
 */
export type CogCase = 1 | 2 | 6;
export interface CogParams {
  rho0: number;
  b: number;
  /** cog1: temperature coefficient and Grüneisen parameter (the pressure is their product with rho0) */
  T0?: number;
  Gamma?: number;
  /** cog6: when the gas reaches the center */
  tau?: number;
}
export function coggeshallExact(c: CogCase, g: 1 | 2 | 3, gamma: number, q: CogParams, r: number, t: number): RadialState {
  if (c === 1) {
    return { rho: q.rho0 * r ** q.b * t ** (-q.b - g), u: r / t, p: (q.Gamma ?? 40) * q.rho0 * (q.T0 ?? 1.4) * t ** (-gamma * g) };
  }
  if (c === 2) {
    const cc = 2 + (gamma - 1) * g;
    const rho = q.rho0 * r ** q.b * t ** ((-2 * (q.b + g)) / cc);
    return { rho, u: ((2 / cc) * r) / t, p: (rho * 2 * (gamma - 1) * g * (r / t) ** 2) / ((q.b + 2) * cc * cc) };
  }
  const tau = q.tau ?? 1.25;
  const x1 = tau * tau - t * t;
  const rho = q.rho0 * r ** q.b * x1 ** (-(g + q.b) / 2);
  return { rho, u: (-r * t) / x1, p: (rho * ((tau * r) / x1) ** 2) / (q.b + 2) };
}

/** The adiabatic index Coggeshall's problem 6 needs in geometry g. */
export const kidderGamma = (g: 1 | 2 | 3): number => (g + 2) / g;

/** A one-dimensional state: density, velocity along x, pressure, and what lies there. */
export interface SlabState {
  rho: number;
  u: number;
  p: number;
  what: 'piston' | 'fuel' | 'products' | 'vacuum';
}

/** Fickett's piston-driven detonation: CJ speed, the explosive's density, the piston's speed. */
export interface FickettParams {
  D: number;
  rho0: number;
  up: number;
}

/**
 * The Chapman-Jouguet state of a detonation at speed D into cold explosive (p = 0) of density
 * rho0 that releases q = D^2 / (2 (gamma^2 - 1)) per unit mass: the strong-detonation limit.
 */
export function cjState(gamma: number, D: number, rho0: number) {
  return { p: (rho0 * D * D) / (gamma + 1), rho: (rho0 * (gamma + 1)) / gamma, u: D / (gamma + 1), c: (D * gamma) / (gamma + 1), q: (D * D) / (2 * (gamma * gamma - 1)) };
}

/**
 * Fickett's piston-driven detonation (Fickett & Davis 1979, p. 24), of which the Mader problem
 * is the piston at rest (ExactPack's mader; Timmes et al. 2005), at distance x from where the
 * piston started and time t. The detonation leaves the piston's face at t = 0 at the CJ speed.
 * Behind it is a Taylor wave, u = (2 x/t - D) / (gamma + 1), on the products' isentrope through
 * the CJ state, down to the piston's speed, and behind that a uniform state moving with the
 * piston. The Taylor wave's tail is at x/t = (D + (gamma + 1) u_p) / 2: half the CJ speed for a
 * piston at rest, for any gamma. Valid for as long as the explosive lasts.
 */
export function fickettExact(gamma: number, q: FickettParams, x: number, t: number): SlabState {
  if (x < q.up * t) return { rho: 0, u: q.up, p: 0, what: 'piston' };
  if (x >= q.D * t) return { rho: q.rho0, u: 0, p: 0, what: 'fuel' };
  const cj = cjState(gamma, q.D, q.rho0);
  const u = Math.max((2 * (x / t) - q.D) / (gamma + 1), q.up);
  const c = cj.c + ((gamma - 1) / 2) * (u - cj.u);
  const f = c / cj.c;
  return { rho: cj.rho * f ** (2 / (gamma - 1)), u, p: cj.p * f ** ((2 * gamma) / (gamma - 1)), what: 'products' };
}

/**
 * The escape of detonation products (EHEP: Fickett & Rivard 1974; Dykema 2002; Doebling 2015;
 * ExactPack's ehep), gamma = 3. Fickett's detonation runs through a slab of explosive of length
 * L ahead of a piston at u_p, with vacuum beyond it. When the detonation reaches the end, at
 * t~ = L / D, the products escape into the vacuum, their edge moving on at D, and a rarefaction
 * runs back through them to the piston, from which it reflects. With gamma = 3 the
 * characteristics are straight lines and every region is in closed form. The regions are
 * ExactPack's, cut here by the lines between them:
 *   A, the detonation and then the products' edge, x = D t;
 *   B, the piston, x = u_p t;
 *   C, the Taylor wave's tail, x = (2 u_p + D/2) t;
 *   D, the rarefaction from the end, x + D t / 2 = 3 L / 2;
 *   E, that rarefaction reflected from the piston, x = (2 u_p + D/2) t - 3 L / 2.
 * In every region rho = (16/9) rho0 c / D and p = (16/27) rho0 D^2 (c / D)^3.
 */
export function ehepExact(q: FickettParams & { L: number }, x: number, t: number): SlabState {
  const { D, rho0, up, L } = q;
  const tt = L / D;
  const state = (c: number, u: number): SlabState => ({ rho: (16 / 9) * rho0 * (c / D), u, p: (16 / 27) * rho0 * D * D * (c / D) ** 3, what: 'products' });
  if (x < up * t) return { rho: 0, u: up, p: 0, what: 'piston' };
  if (x >= D * t) return x < L ? { rho: rho0, u: 0, p: 0, what: 'fuel' } : { rho: 0, u: 0, p: 0, what: 'vacuum' };
  const tail = (2 * up + D / 2) * t;
  // V: between the piston and the rarefaction reflected from it
  if (x <= tail - 1.5 * L) return state(((D - up) * tt) / (t - tt), (x - up * tt) / (t - tt));
  if (x + (D * t) / 2 <= 1.5 * L) {
    // not yet reached by the rarefaction from the end: I, the Taylor wave; III, the piston's state
    if (x >= tail) return state(0.5 * (x / t + D / 2), 0.5 * (x / t - D / 2));
    return state(up + D / 2, up);
  }
  // II, the expansion into the vacuum; IV, the rarefaction through the piston's uniform state
  if (x >= tail) {
    const a = x / t;
    const b = (x - L) / (t - tt);
    return state(Math.max(0.5 * (a - b), 0), 0.5 * (a + b));
  }
  const s = (x - L) / (D * t - L);
  return state(up + (D / 2) * (0.5 - s), up + (D / 2) * (0.5 + s));
}
