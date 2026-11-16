import { all, OUTFLOW, type Problem } from '../../src/problems/types';

// A magnetized accretion disk: the magnetorotational instability (MRI) in a global, stratified
// disk around one point mass, on the Cartesian grid.
//
// Units: G M = 1 and r0 = 1, so an orbit at r0 takes 2 pi (tDyn). The gas is globally isothermal
// (barotropic, gamma = 1) with sound speed `cs`: heat from the field's dissipation leaves the
// gas at once, so the disk does not puff up as the turbulence grows. Its scale height
// H = cs / Omega is 0.15 r at r0 and grows as r^(3/2), so H / r flares as r^(1/2).
//
// The disk is in balance with the mass's softened (Plummer) potential: the density is
//   rho(R, z) = rho_mid(R) exp(-(Phi(R, z) - Phi(R, 0)) / cs^2),
// with a midplane density rho_mid = RHO0 R^(-3/2) (a flat surface density) tapered inside
// R_in = 0.75 and outside R_out = 3, over an ambient gas AMB = 1e-5 of it. RHO0 makes the disk
// about a hundredth of the central mass, which feels the gas's pull. For a globally isothermal disk the
// rotation in balance is the same at every height:
//   v_phi^2 = R^2 / (R^2 + eps^2)^(3/2) + cs^2 dln(rho_mid) / dln(R),
// which the tapers make sub- or super-Keplerian at the edges.
//
// The field is poloidal loops inside the disk, with no net flux (as Hawley 2000's, but several
// of them along R): A_phi = a(R) f(z) sin(pi ln(R / R_in) / ln(LOOP)) between R_in and
// R_in LOOP^NLOOP, zero elsewhere, where f = max(exp(-z^2 / 2H^2) - e^-2, 0) closes them at
// |z| = 2H. a(R) sets the vertical field where two loops meet at the midplane (it is strongest
// there, and zero at each loop's center) to sqrt(2 p_mid / beta): the plasma beta there is
// `beta`, and about twice that averaged over a loop. Each loop spans a factor LOOP in radius, about
// 2H across at r0. The field comes from A, so div B starts at zero to round-off.
//
// The MRI's fastest wavelength, 2 pi v_A / Omega, is 1.3 H at beta = 50: some 6 cells at r0 on
// the default 256^3 grid (12 at 512^3), more farther out, where H is larger. Q_z = lambda / dx
// of 8 to 10 resolves it (Hawley, Guan & Krolik 2011); below that the turbulence it drives is
// weaker than it should be.
//
// Field rises out of the disk into the thin corona above it, and a sink leaves the field behind
// as it takes in gas: both would make v_A, and so the timestep, ever more extreme. The Alfven
// ceiling (Problem.fluff.va) adds mass at rest where v_A would pass VA_MAX.
//
// Beyond R = 3.25 the gas is relaxed toward the disk as it started (hydro/damping.wgsl), as
// for the binary disk, so the outer disk is a reservoir and the boundaries see only it. The
// gas dense enough to be the disk (above REFINE_DENSE, about 2H from the midplane at r0) is
// kept at the finest level, and so is the gas around the mass.
export const MRI = {
  cs: 0.15,
  rIn: 0.75,
  rOut: 3,
  rho0: 1e-3,
  amb: 1e-8,
  loop: 1.35,
  nLoop: 4,
  box: 8,
  rDamp: 3.25,
  damp: 3,
  vaMax: 3,
  refineDense: 5e-5,
};

/** A number as a WGSL f32 literal. */
const f32 = (v: number) => (Number.isInteger(v) ? `${v}.0` : String(v));

/** The disk's midplane density at cylindrical radius R. */
export const midplane = (R: number) => MRI.rho0 * R ** -1.5 * Math.exp(-((MRI.rIn / R) ** 8) - (R / MRI.rOut) ** 8);

/** dln(rho_mid) / dln(R). */
const midSlope = (R: number) => -1.5 + 8 * (MRI.rIn / R) ** 8 - 8 * (R / MRI.rOut) ** 8;

/** The softened potential of the mass (G M = 1). */
const phi = (R: number, z: number, eps: number) => -1 / Math.sqrt(R * R + z * z + eps * eps);

/** The disk in balance at (R, z): density and orbital speed. */
export function diskState(R: number, z: number, eps: number): { rho: number; disk: number; v: number } {
  const { cs, amb } = MRI;
  const disk = midplane(R) * Math.exp(-(phi(R, z, eps) - phi(R, 0, eps)) / (cs * cs));
  const rho = disk + amb;
  // The disk's pressure gradient, weighted by its share of the gas: the ambient gas orbits at
  // the Keplerian speed of its radius.
  const v2 = (R * R) / (R * R + eps * eps) ** 1.5 + (cs * cs * midSlope(R) * disk) / rho;
  return { rho, disk, v: Math.sqrt(Math.max(v2, 0)) };
}

/** The scale height cs / Omega_K at R. */
const scaleHeight = (R: number) => MRI.cs * R ** 1.5;

/** The loops' vector potential A_phi at (R, z), for midplane plasma beta `beta`. */
export function loopPotential(R: number, z: number, beta: number): number {
  const { rIn, loop, nLoop, cs } = MRI;
  if (R <= rIn || R >= rIn * loop ** nLoop) return 0;
  const H = scaleHeight(R);
  const f = Math.exp((-z * z) / (2 * H * H)) - Math.exp(-2);
  if (f <= 0) return 0;
  // B_z ~ (1 / R) d(R A_phi) / dR ~ a f pi / (R ln LOOP) where two loops meet, at the midplane.
  const b0 = Math.sqrt((2 * cs * cs * midplane(R)) / beta);
  const a = (b0 * R * Math.log(loop)) / (Math.PI * (1 - Math.exp(-2)));
  return a * f * Math.sin((Math.PI * Math.log(R / rIn)) / Math.log(loop));
}

/** The softening (and sink radius) of the mass: the preset's, but at least two cells. */
const softening = (p: Record<string, number>, dx: number) => Math.max(p.soft, 2 * dx);

/**
 * The same disk, in WGSL, as the damping zone's target. Must match diskState() above. The
 * zone starts at radius P.diff.z; P.diff.w is its rate at the box's edge, per local orbital
 * period. The softening there is far below R, so it is left out.
 */
const DAMPING = `const DAMPING = true;
fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(MRI.box / 2)};
  let r0 = P.diff.z;
  let R = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || R <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((R - r0) / (half - r0), 1.0);
  let edge = min(R, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let cs2 = ${f32(MRI.cs * MRI.cs)};
  let q = pow(${f32(MRI.rIn)} / R, 8.0);
  let o = pow(R / ${f32(MRI.rOut)}, 8.0);
  let mid = ${f32(MRI.rho0)} * pow(R, -1.5) * exp(-q - o);
  let dphi = 1.0 / R - 1.0 / sqrt(R * R + x.z * x.z);
  let disk = mid * exp(-dphi / cs2);
  let rho = disk + ${f32(MRI.amb)};
  let v = sqrt(max(1.0 / R + cs2 * (-1.5 + 8.0 * q - 8.0 * o) * disk / rho, 0.0));
  return DampTarget(rate, rho, vec3<f32>(-v * x.y / R, v * x.x / R, 0.0), cs2 * rho);
}
`;

export const mriDisk: Problem = {
  id: 'mri-disk',
  label: 'Magnetized accretion disk',
  blurb: 'A disk around one point mass threaded by weak field loops: the magnetorotational instability winds them up and turns the flow turbulent within a few orbits, and the turbulence carries gas inward. One orbit at r = 1 is t = 2π.',
  eos: 'barotropic',
  eosParams: { gamma: 1, K: MRI.cs ** 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'off',
  G: 1,
  box: MRI.box,
  tEnd: Infinity,
  checkpointDt: Math.PI,
  tDyn: 2 * Math.PI, // the orbit at r = 1
  rhoFloor: 1e-11,
  supports2d: false,
  grid: { ndim: 3, n: 256, amr: 2 },
  refineMasses: true,
  frame: 'masses',
  refineDense: MRI.refineDense,
  // The corona never asks for refinement on its own.
  refineRho: 1e-6,
  fluff: { rho: 0, u: 0, v: 0, va: MRI.vaMax },
  damping: { wgsl: DAMPING, radius: MRI.rDamp, rate: MRI.damp },
  disk: true,
  view: { field: 0, log: true, mode: 'slice', axis: 1 },
  params: [
    {
      id: 'beta', label: 'Plasma beta at the midplane', value: 50, min: 5, max: 1e5, factor: 2,
      title: 'Ratio of the gas pressure to the magnetic pressure B^2 / 2 at the midplane where two loops meet (the field is strongest there): the larger, the weaker the field, and the shorter the wavelength the MRI grows at (it scales as beta^-1/2, and needs some 8 cells to grow at the right rate). Changing it starts the run again',
    },
    {
      id: 'soft', label: 'Softening and sink radius', value: 0.15, min: 0.02, max: 0.5, step: 0.01, live: true,
      title: 'Radius within which the mass’s pull is softened and it takes in gas (at least two cells). The disk’s inner edge is at r = 0.75. Changes take effect without a restart',
    },
  ],
  masses: (p, _G, dx) => {
    const soft = softening(p, dx);
    return [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft, sinkRadius: soft }];
  },
  mhd: {
    potential: (p) => (x, y, z) => {
      const R = Math.hypot(x, y);
      if (R < 1e-9) return [0, 0, 0];
      const a = loopPotential(R, z, p.beta);
      return [(-a * y) / R, (a * x) / R, 0];
    },
  },
  init: (n, _eos, _ndim, p) => {
    // The softening as the masses get it, on the finest level (refineMasses).
    const eps = softening(p, MRI.box / n);
    return (x, y, z) => {
      const R = Math.max(Math.hypot(x, y), 1e-6);
      const { rho, disk, v } = diskState(R, z, eps);
      return { rho, vx: (-v * y) / R, vy: (v * x) / R, vz: 0, p: MRI.cs ** 2 * rho, s: disk > 10 * MRI.amb ? 1 : 0 };
    };
  },
};
