// Starting stars in the code's own hydrostatic balance.
//
// A star set up from its analytic profile is in balance with the analytic potential, not with
// the one the Poisson solver finds for it on the grid, nor with the scheme's discrete form of
// hydrostatic balance (hydro/q.wgsl). The mismatch is small, but the star answers it by
// ringing from the first step (a white dwarf, whose core is nearly relativistic and so only
// just stable, by 20% in central density). So stars are settled before the run starts, in the
// manner of Zingale et al. (2002) for mapped stellar models: each star's density is made to
// satisfy
//   h(rho) + phi = C,    h = int dp / rho  (the enthalpy of its cold, barotropic gas)
// with phi the grid's own potential and C fixed by the star's mass; the potential is solved
// again for the new density; and the two are repeated until they agree (a self-consistent
// field iteration). For a polytrope of n = 1 this is exactly the trapezoid rule the
// well-balanced reconstruction holds, p_r - p_c = -(rho_c + rho_r) / 2 * (phi_r - phi_c),
// between any two cells; otherwise it differs from it by the cube of the enthalpy step.
//
// Several stars settle together, in the same rounds. Each is balanced in its own potential:
// the grid's, less the pull of the others, which outside a spherical star is that of a point
// of its mass, -G M_j / |x - x_j|. Balancing in the full potential would hold a star against
// its companion's pull, which in a star that is falling freely toward it (or orbiting it) is
// not there to be held against. A star that moves keeps its velocity: settling only reshapes
// its density.

/** A polytropic star at rest at the box center, to be settled into the grid's hydrostatic balance. */
export interface StarSpec {
  /** polytropic index, and K in p = K rho^(1 + 1/n) */
  n: number;
  K: number;
  /** density below which a cell is ambient gas, and the ambient density */
  rhoCut: number;
  rhoAmb: number;
}

/** The cold equation of state a star is settled with: its pressure, and its density at enthalpy h (0 for h <= 0). */
export interface Barotrope {
  pressure(rho: number): number;
  rhoOfH(h: number): number;
}

export const polytropeBarotrope = (n: number, K: number): Barotrope => {
  const hk = (n + 1) * K;
  return { pressure: (rho) => K * Math.pow(rho, 1 + 1 / n), rhoOfH: (h) => (h > 0 ? Math.pow(h / hk, n) : 0) };
};

/** One star to settle: its gas, where its center is, how it moves, its size, and the dye it carries. */
export interface SettleStar {
  barotrope: Barotrope;
  center: number[];
  vel: number[];
  /** roughly its radius: cells go to the star they are fewest radii from */
  R: number;
  /** farthest from its center a cell may become part of it (default: anywhere) */
  reach?: number;
  /** passive scalar of its gas */
  scalar: number;
}

/** The stars of the initial conditions, the ambient gas around them, and G. */
export interface Settling {
  stars: SettleStar[];
  /** density below which a cell is ambient gas, and the ambient density and pressure */
  rhoCut: number;
  rhoAmb: number;
  pAmb: number;
  G: number;
  /** change of density (relative to the largest) that ends the rounds (default SETTLE_TOL) */
  tol?: number;
  /** over-relaxation of the rounds' density updates (default 1: none) */
  relax?: number;
}

/** A single polytrope at rest at the box center (Problem.star). */
export function settlingOf(star: StarSpec, G: number): Settling {
  const barotrope = polytropeBarotrope(star.n, star.K);
  return {
    stars: [{ barotrope, center: [0, 0, 0], vel: [0, 0, 0], R: 1, scalar: 1 }],
    rhoCut: star.rhoCut,
    rhoAmb: star.rhoAmb,
    pAmb: barotrope.pressure(star.rhoCut),
    G,
  };
}

/**
 * The density that balances the potential: `phi[k]` at cell `cells[k]` (whose volume is
 * vol[cells[k]]), zero where a cell is outside the star; `mass` is the mass the star is to have.
 * The constant C of h + phi = C is found by the Illinois method (regula falsi that halves a
 * stale end), bracketed around `guess` (the last round's C) if there is one: it moves little
 * from round to round, so a dozen sums over the star's cells find it. Returns the density per
 * listed cell, and C.
 */
function balancedDensity(rhoOfH: (h: number) => number, phi: Float64Array, cells: Int32Array, vol: Float64Array, mass: number, rhoCut: number, guess?: number): { rho: Float32Array; C: number } {
  let phiMin = Infinity;
  for (let k = 0; k < phi.length; k++) if (phi[k] < phiMin) phiMin = phi[k];
  const massAt = (C: number, within: Int32Array) => {
    let m = 0;
    for (let i = 0; i < within.length; i++) {
      const k = within[i];
      const r = rhoOfH(C - phi[k]);
      if (r > rhoCut) m += r * vol[cells[k]];
    }
    return m;
  };
  // Only cells below the bracket's upper end can be part of the star: sum over those alone.
  const below = (top: number) => {
    let n = 0;
    for (let k = 0; k < phi.length; k++) if (phi[k] < top) n++;
    const out = new Int32Array(n);
    for (let k = 0, j = 0; k < phi.length; k++) if (phi[k] < top) out[j++] = k;
    return out;
  };
  // Bracket the constant: no star at the bottom of the well (or below the last C), too much
  // of one high enough.
  const scale = Math.abs(phiMin) + 1e-6;
  let lo = phiMin;
  let hi = phiMin + 0.25 * scale;
  if (guess !== undefined && guess > phiMin) hi = guess + 1e-3 * scale;
  let inside = below(hi);
  let mHi = massAt(hi, inside);
  while (mHi < mass) {
    hi = lo + 2 * (hi - lo);
    inside = below(hi);
    mHi = massAt(hi, inside);
  }
  if (guess !== undefined && guess > phiMin && guess < hi) {
    let d = 1e-3 * scale;
    lo = Math.max(phiMin, guess - d);
    while (lo > phiMin && massAt(lo, inside) > mass) {
      d *= 4;
      lo = Math.max(phiMin, guess - d);
    }
  }
  let fLo = massAt(lo, inside) - mass;
  let fHi = mHi - mass;
  let side = 0;
  let C = hi;
  for (let i = 0; i < 60 && hi - lo > 1e-11 * scale; i++) {
    C = fHi > fLo ? (lo * fHi - hi * fLo) / (fHi - fLo) : 0.5 * (lo + hi);
    if (!(C > lo && C < hi)) C = 0.5 * (lo + hi);
    const f = massAt(C, inside) - mass;
    if (f === 0) break;
    if (f < 0) {
      lo = C;
      fLo = f;
      if (side === -1) fHi *= 0.5;
      side = -1;
    } else {
      hi = C;
      fHi = f;
      if (side === 1) fLo *= 0.5;
      side = 1;
    }
  }
  const out = new Float32Array(phi.length);
  for (const k of inside) {
    const r = rhoOfH(C - phi[k]);
    if (r > rhoCut) out[k] = r;
  }
  return { rho: out, C };
}

/** The cells of a grid and their state, as both simulators read them back. */
export interface SettleCells {
  count: number;
  /** cell center relative to the box center */
  pos(q: number): [number, number, number];
  vol(q: number): number;
  /** conserved state: (rho, momentum), (E, e_int, rho s, mark) */
  u1: Float32Array;
  u2: Float32Array;
}

/**
 * Settles the stars of a Settling on a grid's cells, round by round. A round only reshapes the
 * density (all the potential depends on); finish() then writes each cell's whole state: the
 * star's velocity, the internal energy of its cold gas, the dye, or the ambient gas. Which
 * star a cell can belong to, each star's mass and the others' pull on it are fixed at the start.
 */
export class Settler {
  /** per star: the cells that may become part of it, and its own potential's correction there */
  private cells: Int32Array[] = [];
  private pull: Float64Array[] = [];
  private mass: number[] = [];
  /** per star: its constant C of the last round */
  private C: (number | undefined)[] = [];
  private vol: Float64Array;
  /** star each cell can belong to, or -1 for none (ambient for good) */
  readonly owner: Int32Array;

  constructor(private set: Settling, private grid: SettleCells) {
    const n = grid.count;
    this.owner = new Int32Array(n).fill(-1);
    this.vol = new Float64Array(n);
    const mine: number[][] = set.stars.map(() => []);
    this.mass = set.stars.map(() => 0);
    for (let q = 0; q < n; q++) {
      const [x, y, z] = grid.pos(q);
      this.vol[q] = grid.vol(q);
      // The star the cell is fewest radii from, if it is within that star's reach.
      let best = -1;
      let bd = Infinity;
      set.stars.forEach((s, i) => {
        const d = Math.hypot(x - s.center[0], y - s.center[1], z - s.center[2]);
        if (d / s.R < bd && d <= (s.reach ?? Infinity)) {
          bd = d / s.R;
          best = i;
        }
      });
      this.owner[q] = best;
      if (best < 0) continue;
      mine[best].push(q);
      if (grid.u1[q * 4] > set.rhoCut) this.mass[best] += grid.u1[q * 4] * this.vol[q];
    }
    this.cells = mine.map((m) => Int32Array.from(m));
    // Each star is balanced in its own potential: the grid's, less the others' pull, which
    // outside a spherical star is that of a point of its mass.
    this.pull = this.cells.map((idx, i) => {
      const add = new Float64Array(idx.length);
      set.stars.forEach((o, j) => {
        if (j === i) return;
        idx.forEach((q, k) => {
          const [x, y, z] = grid.pos(q);
          add[k] += (set.G * this.mass[j]) / Math.hypot(x - o.center[0], y - o.center[1], z - o.center[2]);
        });
      });
      return add;
    });
  }

  /** One round: the density that balances each star in the potential `phi`. Returns the largest change. */
  round(phi: Float32Array): number {
    const u1 = this.grid.u1;
    const { rhoCut, rhoAmb } = this.set;
    let change = 0;
    this.set.stars.forEach((s, i) => {
      const idx = this.cells[i];
      const ph = new Float64Array(idx.length);
      for (let k = 0; k < idx.length; k++) ph[k] = phi[idx[k]] + this.pull[i][k];
      const { rho, C } = balancedDensity(s.barotrope.rhoOfH, ph, idx, this.vol, this.mass[i], rhoCut, this.C[i]);
      this.C[i] = C;
      for (let k = 0; k < idx.length; k++) {
        const o = idx[k] * 4;
        const r = rho[k] > 0 ? rho[k] : rhoAmb;
        change = Math.max(change, Math.abs(r - u1[o]));
        // Over-relaxed: the rounds converge geometrically, so going further than asked gets there sooner.
        const w = this.set.relax ?? 1;
        const next = u1[o] + w * (r - u1[o]);
        u1[o] = next > rhoCut ? next : rhoAmb;
      }
    });
    return change;
  }

  /** Write every cell's whole state from its settled density. */
  finish(eintFromP: (rho: number, p: number) => number) {
    const { u1, u2 } = this.grid;
    const { rhoCut, rhoAmb, pAmb } = this.set;
    const eAmb = eintFromP(rhoAmb, pAmb);
    for (let q = 0, o = 0; q < this.grid.count; q++, o += 4) {
      const i = this.owner[q];
      if (i < 0) continue; // never part of a star: as the initial conditions set it
      const r = u1[o];
      if (r > rhoCut) {
        const s = this.set.stars[i];
        const eint = eintFromP(r, s.barotrope.pressure(r));
        const [vx, vy, vz] = s.vel;
        u1[o + 1] = r * vx;
        u1[o + 2] = r * vy;
        u1[o + 3] = r * vz;
        u2[o] = eint + 0.5 * r * (vx * vx + vy * vy + vz * vz);
        u2[o + 1] = eint;
        u2[o + 2] = r * s.scalar;
      } else {
        // Ambient gas at rest, in rough pressure balance with the stars' outermost layers.
        u1[o] = rhoAmb;
        u1[o + 1] = u1[o + 2] = u1[o + 3] = 0;
        u2[o] = eAmb;
        u2[o + 1] = eAmb;
        u2[o + 2] = 0;
      }
    }
  }
}

/** How many rounds to settle for at most, and the change of density (relative to the largest) that ends them. */
export const SETTLE_ROUNDS = 30;
export const SETTLE_TOL = 1e-5;
