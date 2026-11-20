// The geometry of a circular binary in its corotating frame: the Roche potential, its inner
// Lagrange point L1, and the size of each body's Roche lobe.
//
// Two masses m1 and m2 a apart turn at Omega^2 = G (m1 + m2) / a^3 about their center of mass.
// In the frame turning with them the gas feels
//   Phi = -G m1 / |x - x1| - G m2 / |x - x2| - Omega^2 R^2 / 2,
// R the distance from the axis through the center of mass. Gas in hydrostatic balance in that
// frame lies on surfaces of constant Phi; the one through L1, where the two bodies' lobes meet,
// is the largest a star can fill and keep all of its gas. A star that overfills it pours gas
// through L1 onto its companion: Roche lobe overflow.

/**
 * Eggleton's (1983) radius of the sphere with the volume of body 1's Roche lobe, over the
 * separation, for q = m1 / m2. Good to 1% for every q.
 */
export function eggleton(q: number): number {
  const q3 = Math.cbrt(q);
  const q23 = q3 * q3;
  return (0.49 * q23) / (0.6 * q23 + Math.log(1 + q3));
}

/**
 * Where L1 lies, as its distance from body 1 over the separation: the root between the two
 * bodies of the pull along the line joining them, in the frame turning with point masses m1, m2.
 */
export function l1Distance(m1: number, m2: number): number {
  const M = m1 + m2;
  const mu2 = m2 / M;
  // Toward body 2 (+x), at distance d from body 1, in units G = a = 1 (so Omega^2 = M). The
  // pull rises monotonically from -infinity to +infinity across the gap: bisection finds it.
  const pull = (d: number) => -m1 / (d * d) + m2 / ((1 - d) * (1 - d)) + M * (d - mu2);
  let lo = 1e-9;
  let hi = 1 - 1e-9;
  for (let i = 0; i < 200 && hi - lo > 1e-15; i++) {
    const mid = 0.5 * (lo + hi);
    if (pull(mid) < 0) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Angular velocity of a circular binary of total G M = gm, a apart, with the pull between the two
 * softened over `soft` (Plummer): the speed binaryMasses (setup/parts/bodies.ts) gives a circular
 * orbit, over the separation.
 */
export function binaryOmega(gm: number, a: number, soft = 0): number {
  return Math.sqrt(gm / (a * a * a)) * Math.pow((a * a) / (a * a + soft * soft), 0.75);
}

/**
 * The Roche potential at (x, y, z), for G m1 at x1 = -m2 / M a and G m2 at x2 = m1 / M a on the
 * x axis and the frame turning about z at Omega (binaryOmega). Body 2's pull is softened over
 * `soft`, as a point-mass companion's is.
 */
export function rochePotential(gm1: number, gm2: number, a: number, x: number, y: number, z: number, soft = 0): number {
  const M = gm1 + gm2;
  const x1 = (-gm2 / M) * a;
  const x2 = (gm1 / M) * a;
  const w = binaryOmega(M, a, soft);
  return -gm1 / Math.hypot(x - x1, y, z) - gm2 / Math.hypot(x - x2, y, z, soft) - 0.5 * w * w * (x * x + y * y);
}

/** Separation at which body 1, of radius R1, is `fill` times the size of its Roche lobe (Eggleton). */
export function fillSeparation(R1: number, m1: number, m2: number, fill: number): number {
  return R1 / (fill * eggleton(m1 / m2));
}
