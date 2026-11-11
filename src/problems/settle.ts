// Starting a star in the code's own hydrostatic balance.
//
// A star set up from its analytic profile is in balance with the analytic potential, not with
// the one the Poisson solver finds for it on the grid, nor with the scheme's discrete form of
// hydrostatic balance (hydro/q.wgsl). The mismatch is small, but the star answers it by
// ringing from the first step. So the star is settled before the run starts, in the manner of
// Zingale et al. (2002) for mapped stellar models: the density is made to satisfy
//   h(rho) + phi = C,    h = (n + 1) K rho^(1/n)  (the enthalpy of a polytrope p = K rho^(1+1/n))
// with phi the grid's own potential and C fixed by the star's mass; the potential is solved
// again for the new density; and the two are repeated until they agree (a self-consistent
// field iteration). For n = 1 this is exactly the trapezoid rule the well-balanced
// reconstruction holds, p_r - p_c = -(rho_c + rho_r) / 2 * (phi_r - phi_c), between any two
// cells; for other n it differs from it by the cube of the enthalpy step between them.

/** A polytropic star at rest, to be settled into the grid's hydrostatic balance. */
export interface StarSpec {
  /** polytropic index, and K in p = K rho^(1 + 1/n) */
  n: number;
  K: number;
  /** density below which a cell is ambient gas, and the ambient density */
  rhoCut: number;
  rhoAmb: number;
}

/**
 * The density that balances the potential `phi`, per cell: zero where the cell is outside the
 * star. `vol(q)` is the volume of cell q and `mass` the mass the star is to have.
 */
export function balancedDensity(star: StarSpec, phi: Float32Array, vol: (q: number) => number, mass: number): Float32Array {
  const hk = (star.n + 1) * star.K;
  const rhoOf = (h: number) => (h > 0 ? Math.pow(h / hk, star.n) : 0);
  let phiMin = Infinity;
  for (let q = 0; q < phi.length; q++) if (phi[q] < phiMin) phiMin = phi[q];
  const massAt = (C: number, cells: Int32Array | null) => {
    let m = 0;
    const n = cells ? cells.length : phi.length;
    for (let i = 0; i < n; i++) {
      const q = cells ? cells[i] : i;
      const r = rhoOf(C - phi[q]);
      if (r > star.rhoCut) m += r * vol(q);
    }
    return m;
  };
  // Bracket the constant: no star at the bottom of the well, too much of one high enough.
  let lo = phiMin;
  let hi = phiMin + Math.abs(phiMin) * 0.25 + 1e-6;
  while (massAt(hi, null) < mass) hi = phiMin + 2 * (hi - phiMin);
  // Only the cells inside that bracket can be part of the star.
  const inside: number[] = [];
  for (let q = 0; q < phi.length; q++) if (phi[q] < hi) inside.push(q);
  const cells = Int32Array.from(inside);
  for (let i = 0; i < 50; i++) {
    const mid = 0.5 * (lo + hi);
    if (massAt(mid, cells) < mass) lo = mid;
    else hi = mid;
  }
  const C = 0.5 * (lo + hi);
  const out = new Float32Array(phi.length);
  for (const q of cells) {
    const r = rhoOf(C - phi[q]);
    if (r > star.rhoCut) out[q] = r;
  }
  return out;
}

/** Conserved state of a settled cell: (rho, E = e_int, e_int, rho * scalar), at rest. */
export function settledCell(star: StarSpec, rho: number, eintFromP: (rho: number, p: number) => number): [number, number, number] {
  const g = 1 + 1 / star.n;
  if (rho > 0) return [rho, eintFromP(rho, star.K * Math.pow(rho, g)), rho];
  // Ambient gas, in rough pressure balance with the star's outermost layer.
  return [star.rhoAmb, eintFromP(star.rhoAmb, star.K * Math.pow(star.rhoCut, g)), 0];
}

/** How many rounds to settle for at most, and the change of density (relative to the largest) that ends them. */
export const SETTLE_ROUNDS = 30;
export const SETTLE_TOL = 1e-5;
