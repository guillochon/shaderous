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
//
// x_j is where star j's mass is in the round, its center of mass, so that the pull left out is
// the one in the grid's potential even if the companion has shifted. Nothing holds a star in
// place (moved whole, it is as balanced as before), so any net force left over moves it a
// little every round, and the rounds do not end: two 0.64 M_sun white dwarfs a third of the box
// from its center crept toward each other until the isolated boundary's multipoles went past
// the quadrupole (gravity/multipole_eval.wgsl).
//
// A star in a circular binary is different: it is not falling freely but held, in the frame
// turning with the orbit, against its companion's pull by the centrifugal force. There it is
// balanced in the whole potential, the grid's (both stars' gas) and the point masses' with the
// frame's centrifugal term, the Roche potential (Settling.binary, problems/roche.ts), so that it
// starts tidally stretched toward its companion as a star in a close binary is. Each star then
// stays inside its Roche lobe: a cell where the Roche potential of the two bodies as points is
// higher than at L1 takes none of its gas (Settling.binary.lobe). Bounded only by the sphere of
// its distance to L1, a star that fills its lobe bulged past the lobe around L1, out to that
// sphere, which cut it off flat across the line of centers. But nothing in h + phi = C says where along the line of
// centers the star is, and there the tidal field pulls it away from its place: left alone, the
// rounds slid a star that fills its lobe half its radius toward L1 and cut it off there. So each
// star is held where the orbit put it, as self-consistent-field binaries are (Hachisu 1986, who
// fixes its boundary points): a uniform pull, lambda . (x - x_star), is added to its potential
// every round (holdPull). It cancels the pull of the rest of the binary at the star's center of
// mass (which is what slides it: zero where the orbit put it, growing as it strays), and adds
// gentle proportional and integral feedback on how far that center is from its place, in units
// of G M / R^3. Settled, the proportional part is gone, and what the integral part holds is the
// small force the point-mass orbit leaves unbalanced on an extended star.
//
// In a rotating frame with the centrifugal force (Problem.rotation) the stars spin with the
// frame, at rest in it: each is balanced in the centrifugal potential of its spin about its own
// center as well, -|Omega x (x - x_c)|^2 / 2, which flattens it into a rotating polytrope
// (Settling.omega, inFrame). Its gas ends at the ridge of its effective potential: past it the
// centrifugal potential wins, and the condition above would fill the box's far corners.
//
// On an adaptive mesh the rounds run on the GPU (amr/settle.wgsl, AmrSim.settleOnGpu) when each
// star's gas is one it knows (Barotrope.gpu): only per-block sums come back, where the Settler
// here reads the whole mesh back, some 100 bytes a cell, more than a page may hold on a mesh of
// tens of GB. The two agree to a few parts in a million of the peak density.

import { lobePotential, type RocheBody } from './roche';

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
  /** The same gas for the GPU's settling (amr/settle.wgsl), which knows these two kinds. */
  gpu?: { kind: 'polytrope'; n: number; K: number } | { kind: 'degenerate'; degRho: number; degP: number; /** the cold Coulomb term (problems/white_dwarf.ts), if any */ c1?: number; rho0?: number };
}

export const polytropeBarotrope = (n: number, K: number): Barotrope => {
  const hk = (n + 1) * K;
  return { pressure: (rho) => K * Math.pow(rho, 1 + 1 / n), rhoOfH: (h) => (h > 0 ? Math.pow(h / hk, n) : 0), gpu: { kind: 'polytrope', n, K } };
};

/** One star to settle: its gas, where its center is, how it moves, its size, and the dye it carries. */
export interface SettleStar {
  barotrope: Barotrope;
  center: number[];
  vel: number[];
  /** angular velocity about z through its center, on top of `vel` (a tidally locked star in a binary) */
  spin?: number;
  /** In a binary: G M / R^3, which scales the feedback holding it in place (holdPull) */
  stiff?: number;
  /** In a binary: G times its mass, for its pull on the other star */
  gm?: number;
  /** roughly its radius: cells go to the star they are fewest radii from */
  R: number;
  /** farthest from its center a cell may become part of it (default: anywhere) */
  reach?: number;
  /** In a binary: it stays inside its Roche lobe (Settling.binary.lobe) */
  lobe?: boolean;
  /** passive scalar of its gas */
  scalar: number;
  /**
   * A point-mass core at its center (problems/cored_polytrope.ts): G times its mass and its
   * softening. Point masses are not in the grid's potential, so the core's is added to it: the
   * envelope is balanced in both.
   */
  core?: { gm: number; soft: number };
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
  /**
   * The stars were placed in gas that is already there (the star tool): a cell they do not
   * take keeps the gas it has, and only one a star gives up becomes ambient gas.
   */
  keep?: boolean;
  /**
   * Gas that is not the stars', as softened points: its pull is
   * undone as the other stars' is, so that a star is balanced in its own potential and not held
   * against what it falls freely toward. `gm` is G times the mass.
   */
  others?: { gm: number; at: number[]; soft: number }[];
  /**
   * The stars are in a circular binary (problems/roche.ts): each is balanced in the whole
   * potential of the frame turning with it at `omega` about z through the box center, the grid's
   * (no star's pull is undone) plus these point masses' softened pull (G times the mass; every
   * point mass, the stars' cores among them, which SettleStar.core then leaves out) and the
   * centrifugal term.
   */
  binary?: {
    omega: number;
    masses: { gm: number; at: number[]; soft: number }[];
    /**
     * The binary's Roche lobes (problems/roche.ts): its two bodies as points, and the potential
     * at L1. A star with SettleStar.lobe takes no cell where that potential is higher: it ends
     * on the Roche surface, not on the sphere of its reach.
     */
    lobe?: { bodies: RocheBody[]; phi: number };
  };
  /**
   * The angular velocity of a rotating frame with the centrifugal force (Problem.rotation): each
   * star is balanced in the centrifugal potential of its own spin, -|Omega x (x - x_c)|^2 / 2
   * about its center x_c, as well, and so rotates rigidly with the frame (inFrame).
   */
  omega?: number[];
}

/**
 * Gains of the feedback that holds a star of a binary in its place, per round, in units of
 * G M / R^3. A star's center of mass answers a uniform pull f by moving between f / (G M / R^3)
 * (its surface layers) and a sixth of that (an n = 1.5 star's center) in a round: gains this
 * small cannot overshoot.
 */
const HOLD_P = 0.3;
const HOLD_I = 0.05;

/**
 * A round's uniform pull holding star i of a binary in place: lambda, for the potential term
 * lambda . (x - center). `coms` are the stars' centers of mass, and `sum` the integral of the
 * rounds before (updated). The pull of the rest of the binary at the star's center of mass is
 * cancelled: the point masses', the frame's centrifugal term and the other star's, as a point of
 * its mass at its center of mass.
 */
export function holdPull(set: Settling, i: number, coms: number[][], sum: number[]): number[] {
  const s = set.stars[i];
  const b = set.binary!;
  const com = coms[i];
  const k = s.stiff ?? 0;
  const ext = (x: number, y: number, z: number) => {
    let p = binaryPhi(b, x, y, z);
    coms.forEach((c, j) => {
      if (j !== i) p -= (set.stars[j].gm ?? 0) / Math.hypot(x - c[0], y - c[1], z - c[2]);
    });
    return p;
  };
  const h = 1e-4 * s.R;
  const grad = [0, 1, 2].map((a) => {
    const lo = [...com];
    const hi = [...com];
    lo[a] -= h;
    hi[a] += h;
    return (ext(hi[0], hi[1], hi[2]) - ext(lo[0], lo[1], lo[2])) / (2 * h);
  });
  const d = [0, 1, 2].map((a) => com[a] - (s.center[a] ?? 0));
  for (let a = 0; a < 3; a++) sum[a] += HOLD_I * k * d[a];
  return d.map((v, a) => -grad[a] + HOLD_P * k * v + sum[a]);
}

/** The potential a star in a binary is balanced in, besides the grid's: the point masses' and the centrifugal term. */
export function binaryPhi(b: NonNullable<Settling['binary']>, x: number, y: number, z: number): number {
  let p = -0.5 * b.omega * b.omega * (x * x + y * y);
  for (const m of b.masses) p -= m.gm / Math.hypot(x - m.at[0], y - m.at[1], z - (m.at[2] ?? 0), m.soft);
  return p;
}

/**
 * A settling in a rotating frame (sim.ts rotationOf: Omega, and 1 for the centrifugal force):
 * its stars spin with the frame, at rest in it. The frame's centrifugal potential is
 * -|Omega x x|^2 / 2 about the axis through the box center; about a star's center x_c it is
 * -|Omega x (x - x_c)|^2 / 2 plus a part linear in x - x_c, a uniform force that moves the star
 * whole, as its companion's pull does, and is left out as that is. For a star on the axis the
 * two are the same.
 */
export function inFrame(set: Settling, rot: readonly number[]): Settling {
  // A settling that names its own spin keeps it.
  // A binary is balanced in the frame's whole centrifugal potential already (Settling.binary).
  if (set.omega || set.binary || !rot[3] || !(Math.hypot(rot[0], rot[1], rot[2]) > 0)) return set;
  return { ...set, omega: [rot[0], rot[1], rot[2]] };
}

/**
 * A spinning star's centrifugal potential at d from its center (Settling.omega), or Infinity
 * where no gas of the star can be at rest: past the ridge of its effective potential,
 * -G M / r - Omega^2 w^2 / 2 (w the distance from its axis), which is at w^3 = G M / Omega^2.
 * Beyond it the potential falls again, and the gas held there would be the box's corners'.
 */
export function spinPotential(w: readonly number[], gm: number, dx: number, dy: number, dz: number): number {
  const cx = w[1] * dz - w[2] * dy;
  const cy = w[2] * dx - w[0] * dz;
  const cz = w[0] * dy - w[1] * dx;
  const s = cx * cx + cy * cy + cz * cz;
  const w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  // |Omega x d|^2 = Omega^2 w^2, past the ridge's w^2 = (G M / Omega^2)^(2/3).
  if (s >= w2 * Math.cbrt((gm * gm) / (w2 * w2))) return Infinity;
  return -0.5 * s;
}

/** A polytrope the star tool has just placed: where, how it moves, and its profile's scales. */
export interface PlacedStar {
  n: number;
  K: number;
  rhoc: number;
  /** center relative to the box center, velocity, radius */
  center: number[];
  vel: number[];
  R: number;
}

/** How far from a placed star's center, in radii, its settling may reach. */
const PLACED_REACH = 1.3;
/** Parts per side the box is cut into, for the pull of the gas around a placed star. */
const LUMPS = 6;

/**
 * The settling of a star placed among the gas of a run. The gas just outside it (between 1.05
 * and PLACED_REACH radii) says what is around it: the star ends where it is no denser than
 * the densest of that gas, and a cell it gives up takes that gas's mean density and pressure.
 * The gas farther out, in LUMPS^3 parts of the box (side `box`), is what the star falls toward:
 * its pull is left out of the star's balance. (Point masses are not in the grid's potential.)
 * Returns why not, if the star cannot be settled where it is.
 */
export function placedSettling(
  star: PlacedStar, G: number, cells: SettleCells, pressure: (rho: number, eint: number) => number,
  box = 1,
): Settling | string {
  let peak = 0;
  let vol = 0;
  let mass = 0;
  let pv = 0;
  const lumps = new Float64Array(LUMPS ** 3 * 4);
  const part = (x: number) => Math.min(LUMPS - 1, Math.max(0, Math.floor((x / box + 0.5) * LUMPS)));
  for (let q = 0; q < cells.count; q++) {
    const [x, y, z] = cells.pos(q);
    const d = Math.hypot(x - star.center[0], y - star.center[1], z - star.center[2]) / star.R;
    if (d > PLACED_REACH) {
      const m = cells.u1[q * 4] * cells.vol(q);
      const o = 4 * ((part(z) * LUMPS + part(y)) * LUMPS + part(x));
      lumps[o] += m;
      lumps[o + 1] += m * x;
      lumps[o + 2] += m * y;
      lumps[o + 3] += m * z;
    }
    if (d < 1.05 || d > PLACED_REACH) continue;
    const rho = cells.u1[q * 4];
    const v = cells.vol(q);
    peak = Math.max(peak, rho);
    vol += v;
    mass += rho * v;
    pv += pressure(rho, cells.u2[q * 4 + 1]) * v;
  }
  if (!(vol > 0)) return 'the box has no room around it';
  if (peak > 0.1 * star.rhoc) return 'the gas around it is too dense';
  // A lump is as wide as its part of the box: softened by half of that, it has no sharp center.
  const others: NonNullable<Settling['others']> = [];
  for (let o = 0; o < lumps.length; o += 4) {
    const m = lumps[o];
    if (m > 0) others.push({ gm: G * m, at: [lumps[o + 1] / m, lumps[o + 2] / m, lumps[o + 3] / m], soft: (0.5 * box) / LUMPS });
  }
  return {
    others,
    stars: [{ barotrope: polytropeBarotrope(star.n, star.K), center: star.center, vel: star.vel, R: star.R, reach: PLACED_REACH * star.R, scalar: 1 }],
    rhoCut: peak,
    rhoAmb: mass / vol,
    pAmb: pv / vol,
    G,
    keep: true,
  };
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
 * star a cell can belong to and each star's mass are fixed at the start; the others' pull follows
 * where their mass is.
 */
export class Settler {
  /** per star: the cells that may become part of it */
  private cells: Int32Array[] = [];
  private mass: number[] = [];
  /** per star: its constant C of the last round */
  private C: (number | undefined)[] = [];
  /** per star, in a binary: the integral part of the pull holding it in place (holdPull) */
  readonly held: number[][] = [];
  private vol: Float64Array;
  /** star each cell can belong to, or -1 for none (ambient for good) */
  readonly owner: Int32Array;
  /** Settling.keep: the density of each cell that was not a star's at the start (0 for one that was) */
  private kept: Float32Array | null = null;
  /** the cells finish() wrote */
  readonly rewrote: Uint8Array;

  constructor(private set: Settling, private grid: SettleCells) {
    const n = grid.count;
    this.owner = new Int32Array(n).fill(-1);
    this.vol = new Float64Array(n);
    this.rewrote = new Uint8Array(n);
    if (set.keep) {
      this.kept = new Float32Array(n);
      for (let q = 0; q < n; q++) if (!(grid.u1[q * 4] > set.rhoCut)) this.kept[q] = grid.u1[q * 4];
    }
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
  }

  /** Each star's center of mass as its density stands (its placement if it has none above the cut). */
  private centers(): number[][] {
    const u1 = this.grid.u1;
    return this.set.stars.map((s, i) => {
      let m = 0;
      const c = [0, 0, 0];
      for (const q of this.cells[i]) {
        const r = u1[q * 4];
        if (!(r > this.set.rhoCut)) continue;
        const dm = r * this.vol[q];
        const x = this.grid.pos(q);
        m += dm;
        for (let a = 0; a < 3; a++) c[a] += dm * x[a];
      }
      return m > 0 ? c.map((v) => v / m) : [s.center[0], s.center[1], s.center[2] ?? 0];
    });
  }

  /** One round: the density that balances each star in the potential `phi`. Returns the largest change. */
  round(phi: Float32Array): number {
    const u1 = this.grid.u1;
    const { rhoCut, rhoAmb } = this.set;
    let change = 0;
    // Each star is balanced in its own potential: the grid's, less the others' pull, which
    // outside a spherical star is that of a point of its mass at its center of mass.
    // (In a binary the whole potential: nothing is undone.)
    const bin = this.set.binary;
    const coms = this.set.stars.length > 1 || bin ? this.centers() : [];
    const at = bin ? [] : coms;
    const others = this.set.others ?? [];
    this.set.stars.forEach((s, i) => {
      const idx = this.cells[i];
      const hold = bin ? holdPull(this.set, i, coms, (this.held[i] ??= [0, 0, 0])) : null;
      const lobe = s.lobe && bin?.lobe ? bin.lobe : null;
      const ph = new Float64Array(idx.length);
      // A rotating frame: the star's spin, about its own center, out to the ridge its mass sets.
      // (A binary's Roche potential has the frame's centrifugal term already.)
      const spinW = !bin && this.set.omega?.some((v) => v !== 0) ? this.set.omega : undefined;
      const gm = this.set.G * this.mass[i] + (s.core?.gm ?? 0);
      for (let k = 0; k < idx.length; k++) {
        let p = phi[idx[k]];
        if (at.length || others.length || s.core || bin || spinW) {
          const [x, y, z] = this.grid.pos(idx[k]);
          if (bin) p += binaryPhi(bin, x, y, z) + hold![0] * (x - s.center[0]) + hold![1] * (y - s.center[1]) + hold![2] * (z - (s.center[2] ?? 0));
          else {
            if (spinW) p += spinPotential(spinW, gm, x - s.center[0], y - s.center[1], z - (s.center[2] ?? 0));
            if (s.core) p -= s.core.gm / Math.hypot(x - s.center[0], y - s.center[1], z - (s.center[2] ?? 0), s.core.soft);
          }
          for (let j = 0; j < at.length; j++) if (j !== i) p += (this.set.G * this.mass[j]) / Math.hypot(x - at[j][0], y - at[j][1], z - at[j][2]);
          for (const m of others) p += m.gm / Math.hypot(x - m.at[0], y - m.at[1], z - (m.at[2] ?? 0), m.soft);
        }
        // Outside its Roche lobe, no star: an infinite potential holds no gas.
        if (lobe) {
          const [x, y, z] = this.grid.pos(idx[k]);
          if (lobePotential(lobe.bodies, bin!.omega, x, y, z) > lobe.phi) p = Infinity;
        }
        ph[k] = p;
      }
      const { rho, C } = balancedDensity(s.barotrope.rhoOfH, ph, idx, this.vol, this.mass[i], rhoCut, this.C[i]);
      this.C[i] = C;
      for (let k = 0; k < idx.length; k++) {
        const o = idx[k] * 4;
        // What the cell holds if the star does not: ambient gas, or the gas it had.
        const out = this.kept && this.kept[idx[k]] > 0 ? this.kept[idx[k]] : rhoAmb;
        const r = rho[k] > 0 ? rho[k] : out;
        change = Math.max(change, Math.abs(r - u1[o]));
        // Over-relaxed: the rounds converge geometrically, so going further than asked gets there sooner.
        const w = this.set.relax ?? 1;
        const next = u1[o] + w * (r - u1[o]);
        u1[o] = next > rhoCut ? next : out;
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
      // Gas the star never took stays as it was.
      if (!(r > rhoCut) && this.kept && this.kept[q] > 0) continue;
      this.rewrote[q] = 1;
      if (r > rhoCut) {
        const s = this.set.stars[i];
        const eint = eintFromP(r, s.barotrope.pressure(r));
        let [vx, vy, vz] = s.vel;
        if (s.spin) {
          const [x, y] = this.grid.pos(q);
          vx -= s.spin * (y - s.center[1]);
          vy += s.spin * (x - s.center[0]);
        }
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

/**
 * Put a problem's heat (Problem.heat) into settled cells: each cell it names gets that internal
 * energy density, and its total energy changes by as much. `touched`, if given, marks the cells
 * it changed. Returns how many it changed.
 */
export function heatCells(heat: (x: number, y: number, z: number, rho: number) => number | undefined, cells: SettleCells, touched?: Uint8Array): number {
  const { u1, u2 } = cells;
  let any = 0;
  for (let q = 0; q < cells.count; q++) {
    const [x, y, z] = cells.pos(q);
    const e = heat(x, y, z, u1[4 * q]);
    if (e === undefined || !(e >= 0)) continue;
    u2[4 * q] += e - u2[4 * q + 1];
    u2[4 * q + 1] = e;
    if (touched) touched[q] = 1;
    any++;
  }
  return any;
}
