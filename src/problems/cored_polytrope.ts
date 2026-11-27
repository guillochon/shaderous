// A polytropic envelope around a point-mass core: a giant's envelope about its compact core.
//
// The gas is a polytrope, p = K rho^(1 + 1/n), in balance with its own gravity and with the
// softened pull of a point of mass Mc at its center, whose potential is the point masses' own,
// -G Mc / sqrt(r^2 + eps^2) (gravity/bh.wgsl). With the enthalpy h = (n + 1) K rho^(1/n) = hc theta,
// x = r / R and nu = A m(x) / Menv,
//   theta' = -(nu + A qc x^3 / (x^2 + e^2)^(3/2)) / x^2,    nu' = P x^2 theta^n,
//   A = G Menv / (hc R),  P = 4 pi G R^2 rhoc / hc,  qc = Mc / Menv,  e = eps / R,
// with theta(0) = 1. The envelope ends at the given radius, theta(1) = 0, and has the given
// mass, nu(1) = A: two conditions for A and P. Without a core only the product matters and
// this is the Lane-Emden equation; with one the core's pull sets a scale of its own.
//
// For a given A, P is found by bisection on where theta first reaches zero (more gas, a
// smaller star); A by bisection on nu(1) - A, between none and the most the core alone allows
// (above it the core's pull ends the envelope inside R whatever the gas).

export interface CoredPolytrope {
  n: number;
  /** central density of the gas, and K in p = K rho^(1 + 1/n) */
  rhoc: number;
  K: number;
  /** theta at s = r / R (0 at and beyond the surface): rho = rhoc theta^n */
  theta(s: number): number;
}

const STEPS = 2000;

/** Integrate to x = 1 or the first zero of theta. Returns theta(1) (negative: how far inside the zero fell), nu there, and the table if asked. */
function integrate(n: number, A: number, P: number, qc: number, e: number, table?: Float64Array): { end: number; nu: number } {
  const core = (x: number) => (A * qc * x) / Math.pow(x * x + e * e, 1.5);
  const f = (x: number, th: number, nu: number): [number, number] => [-(x > 0 ? nu / (x * x) : 0) - core(x), P * x * x * Math.pow(Math.max(th, 0), n)];
  const dx = 1 / STEPS;
  let th = 1;
  let nu = 0;
  if (table) table[0] = 1;
  for (let i = 0; i < STEPS; i++) {
    const x = i * dx;
    const [a1, b1] = f(x, th, nu);
    const [a2, b2] = f(x + dx / 2, th + (dx / 2) * a1, nu + (dx / 2) * b1);
    const [a3, b3] = f(x + dx / 2, th + (dx / 2) * a2, nu + (dx / 2) * b2);
    const [a4, b4] = f(x + dx, th + dx * a3, nu + dx * b3);
    const next = th + (dx / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
    nu += (dx / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    if (next <= 0 && i < STEPS - 1) {
      // The envelope ends inside R: by how much.
      const w = th / (th - next);
      return { end: (i + w) * dx - 1, nu };
    }
    th = next;
    if (table) table[i + 1] = Math.max(th, 0);
  }
  return { end: th, nu };
}

const cache = new Map<string, CoredPolytrope>();

/**
 * The root of f between lo and hi (f(lo) > 0 > f(hi) or the reverse) by regula falsi with the
 * Illinois step: as sure as bisection, but each integration counts, and it reaches the root to
 * rounding in about ten where bisection took sixty (a giant's envelope, nested two deep, was 2.4 s
 * of a load; now 0.05).
 */
function root(f: (x: number) => number, lo: number, hi: number, fLo = f(lo), fHi = f(hi)): number {
  let side = 0;
  for (let i = 0; i < 200; i++) {
    const x = fHi !== fLo ? (lo * fHi - hi * fLo) / (fHi - fLo) : 0.5 * (lo + hi);
    const at = x > Math.min(lo, hi) && x < Math.max(lo, hi) ? x : 0.5 * (lo + hi);
    const fx = f(at);
    if (fx === 0 || Math.abs(hi - lo) <= 1e-15 * Math.max(Math.abs(lo), Math.abs(hi))) return at;
    if (fx * fHi > 0) {
      hi = at;
      fHi = fx;
      if (side === 1) fLo /= 2;
      side = 1;
    } else {
      lo = at;
      fLo = fx;
      if (side === -1) fHi /= 2;
      side = -1;
    }
    if (Math.abs(hi - lo) <= 1e-14 * Math.max(Math.abs(lo), Math.abs(hi))) return 0.5 * (lo + hi);
  }
  return 0.5 * (lo + hi);
}

/**
 * The envelope of mass `Menv` and radius `R` of index `n` around a core of mass `Mc` softened
 * by `soft`. Throws if there is none (a core so heavy and compact that its pull alone ends the
 * gas inside R needs a larger softening or a lighter core).
 */
export function coredPolytrope(n: number, Menv: number, R: number, Mc: number, soft: number, G: number): CoredPolytrope {
  const key = JSON.stringify([n, Menv, R, Mc, soft, G]);
  const hit = cache.get(key);
  if (hit) return hit;
  const qc = Mc / Menv;
  const e = soft / R;
  if (!(Menv > 0) || !(Mc > 0) || !(e > 0)) throw new Error('a star with a core needs an envelope mass, a core mass and a core softening above zero');
  // The P that ends the envelope at R, for a given A.
  const solveP = (A: number) => {
    let lo = 0;
    let hi = 1;
    while (integrate(n, A, hi, qc, e).end > 0) {
      hi *= 2;
      if (hi > 1e12) throw new Error('a star with a core: no envelope of that mass and radius');
    }
    return root((P) => integrate(n, A, P, qc, e).end, lo, hi);
  };
  // The most A the core allows: theta(1) = 0 with no gas at all.
  const aMax = 1 / (qc * (1 / e - 1 / Math.sqrt(1 + e * e)));
  const A = root((a) => integrate(n, a, solveP(a), qc, e).nu - a, 1e-9 * aMax, aMax * (1 - 1e-9));
  const P = solveP(A);
  const table = new Float64Array(STEPS + 1);
  integrate(n, A, P, qc, e, table);
  table[STEPS] = 0;
  const hc = (G * Menv) / (A * R);
  const rhoc = (P * hc) / (4 * Math.PI * G * R * R);
  const out: CoredPolytrope = {
    n, rhoc, K: hc / ((n + 1) * Math.pow(rhoc, 1 / n)),
    theta(s) {
      if (!(s < 1)) return 0;
      const x = Math.max(s, 0) * STEPS;
      const i = Math.min(STEPS - 1, Math.floor(x));
      return table[i] + (x - i) * (table[i + 1] - table[i]);
    },
  };
  cache.set(key, out);
  return out;
}
