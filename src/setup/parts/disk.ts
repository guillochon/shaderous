// Gas disks: the initial state, the damping zone's target in WGSL from the same numbers, and
// the poloidal field loops of a magnetized disk.
//
// Two profiles, from the two presets that have disks, kept operation for operation so that
// each starts bit for bit as it did:
//
// 'thin' (binary accretion): a surface density sigma0 (exp(-(rIn / r)^4) exp(-(r / rOut)^8) +
//   floor), on circular orbits about the total mass, flattened inside the orbit of the bodies
//   it surrounds (or inside the softening of a single one). In 3D a Gaussian vertical profile
//   of scale height cs / Omega, at least two cells.
//
// 'balanced' (the magnetized disk): a midplane density sigma0 R^slope tapered inside rIn and
//   outside rOut, in vertical balance with the softened potential of the mass,
//   rho = rho_mid exp(-(Phi(R, z) - Phi(R, 0)) / cs^2), over an ambient gas `floor`; rotation
//   v^2 = G M R^2 / (R^2 + eps^2)^(3/2) + cs^2 dln(rho_mid)/dln(R), the pressure term weighted
//   by the disk's share of the gas.
import type { Prim } from '../../problems/types';
import type { DiskPart } from '../spec';

/** A number as a WGSL f32 literal. Only numbers reach the shader text, never a string from a file. */
export const f32 = (v: number): string => {
  if (!Number.isFinite(v)) throw new Error('damping zone: a number is not finite');
  return Number.isInteger(v) ? `${v}.0` : String(v);
};

export interface DiskContext {
  box: number;
  /** finest cells per side */
  n: number;
  ndim: 2 | 3;
  /** G times the mass the disk orbits */
  gm: number;
  /** softening of that mass as the run has it (two cells at least), or half the separation of the bodies inside */
  soft: number;
  /** half the separation of the binary inside the disk, if it surrounds one */
  flat?: number;
}

export interface BuiltDisk {
  part: DiskPart;
  /** the state at a point relative to the disk's center, and whether it is disk gas (its mass, for sharing momentum) */
  state(x: number, y: number, z: number): Prim;
  /** A_phi of the field loops at (R, z), or null without loops */
  loop: ((R: number, z: number) => number) | null;
  damping: { wgsl: string; radius: number; rate: number } | null;
}

export function buildDisk(part: DiskPart, ctx: DiskContext): BuiltDisk {
  return part.profile === 'thin' ? thinDisk(part, ctx) : balancedDisk(part, ctx);
}

function thinDisk(d: DiskPart, ctx: DiskContext): BuiltDisk {
  const { cs, sigma0, rIn, rOut } = d;
  const floor = d.floor ?? 0;
  const { box, ndim } = ctx;
  const dx = box / ctx.n;
  // Circular speed about the total mass, flattened inside the binary's own orbit.
  const flat2 = ctx.flat !== undefined ? ctx.flat * ctx.flat : ctx.soft * ctx.soft;
  const sgm = ctx.gm === 1 ? 1 : Math.sqrt(ctx.gm);
  const surface = (r: number) => sigma0 * (Math.exp(-((rIn / r) ** 4)) * Math.exp(-((r / rOut) ** 8)) + floor);
  const speed = (r: number) => (sgm === 1 ? r / Math.pow(r * r + flat2, 0.75) : (sgm * r) / Math.pow(r * r + flat2, 0.75));
  const state = (x: number, y: number, z: number): Prim => {
    const r = Math.max(Math.hypot(x, y), 1e-6);
    const sigma = surface(r);
    const v = speed(r);
    let rho = sigma;
    if (ndim === 3) {
      const h = Math.max((cs * r) / v, 2 * dx);
      rho = (sigma / (Math.sqrt(2 * Math.PI) * h)) * Math.exp((-z * z) / (2 * h * h)) + (sigma0 * floor) / box;
    }
    return { rho, vx: (-v * y) / r, vy: (v * x) / r, vz: 0, p: cs * cs * rho, s: sigma > 0.5 * sigma0 ? 1 : 0 };
  };
  // The same disk as the damping zone's target. The zone starts at radius P.diff.z; P.diff.w is
  // its rate at the box's edge, per local orbital period.
  const wgsl = `const DAMPING = true;
fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(box / 2)};
  let r0 = P.diff.z;
  let r = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || r <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((r - r0) / (half - r0), 1.0);
  let edge = min(r, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let sigma = ${f32(sigma0)} * (exp(-pow(${f32(rIn)} / r, 4.0)) * exp(-pow(r / ${f32(rOut)}, 8.0)) + ${f32(floor)});
  let v = ${sgm === 1 ? '' : `${f32(sgm)} * `}r / pow(r * r + ${f32(flat2)}, 0.75);
  var rho = sigma;
  if (P.dims.z > 1u) {
    let h = max(${f32(cs)} * r / v, 2.0 * g_dx);
    rho = sigma / (2.50662827 * h) * exp(-x.z * x.z / (2.0 * h * h)) + ${f32((sigma0 * floor) / box)};
  }
  return DampTarget(rate, rho, vec3<f32>(-v * x.y / r, v * x.x / r, 0.0), ${f32(cs * cs)} * rho);
}
`;
  return { part: d, state, loop: null, damping: d.damp ? { wgsl, radius: d.damp.from, rate: d.damp.rate } : null };
}

function balancedDisk(d: DiskPart, ctx: DiskContext): BuiltDisk {
  const { cs, rIn, rOut } = d;
  const rho0 = d.sigma0;
  const slope = d.slope ?? -1.5;
  const amb = d.floor ?? 0;
  const gm = ctx.gm;
  const eps = ctx.soft;
  /** The disk's midplane density at cylindrical radius R, and dln(rho_mid) / dln(R). */
  const midplane = (R: number) => rho0 * R ** slope * Math.exp(-((rIn / R) ** 8) - (R / rOut) ** 8);
  const midSlope = (R: number) => slope + 8 * (rIn / R) ** 8 - 8 * (R / rOut) ** 8;
  /** The softened potential of the mass. */
  const phi = (R: number, z: number) => -gm / Math.sqrt(R * R + z * z + eps * eps);
  const state = (x: number, y: number, z: number): Prim => {
    const R = Math.max(Math.hypot(x, y), 1e-6);
    const disk = midplane(R) * Math.exp(-(phi(R, z) - phi(R, 0)) / (cs * cs));
    const rho = disk + amb;
    // The disk's pressure gradient, weighted by its share of the gas: the ambient gas orbits
    // at the Keplerian speed of its radius.
    const v2 = (gm * R * R) / (R * R + eps * eps) ** 1.5 + (cs * cs * midSlope(R) * disk) / rho;
    const v = Math.sqrt(Math.max(v2, 0));
    return { rho, vx: (-v * y) / R, vy: (v * x) / R, vz: 0, p: cs ** 2 * rho, s: disk > 10 * amb ? 1 : 0 };
  };
  let loop: BuiltDisk['loop'] = null;
  if (d.loops) {
    const { beta, span, count } = d.loops;
    const sgm = gm === 1 ? 1 : Math.sqrt(gm);
    /** The scale height cs / Omega_K at R. */
    const scaleHeight = (R: number) => (sgm === 1 ? cs * R ** 1.5 : (cs * R ** 1.5) / sgm);
    // Loops with no net flux: A_phi = a(R) f(z) sin(pi ln(R / rIn) / ln(span)) between rIn and
    // rIn span^count, where f = max(exp(-z^2 / 2H^2) - e^-2, 0) closes them at |z| = 2H.
    loop = (R, z) => {
      if (R <= rIn || R >= rIn * span ** count) return 0;
      const H = scaleHeight(R);
      const f = Math.exp((-z * z) / (2 * H * H)) - Math.exp(-2);
      if (f <= 0) return 0;
      // B_z ~ (1 / R) d(R A_phi) / dR ~ a f pi / (R ln span) where two loops meet, at the midplane.
      const b0 = Math.sqrt((2 * cs * cs * midplane(R)) / beta);
      const a = (b0 * R * Math.log(span)) / (Math.PI * (1 - Math.exp(-2)));
      return a * f * Math.sin((Math.PI * Math.log(R / rIn)) / Math.log(span));
    };
  }
  // The same disk as the damping zone's target. The softening there is far below R, so it is
  // left out.
  const g = gm === 1 ? '1.0' : f32(gm);
  const wgsl = `const DAMPING = true;
fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(ctx.box / 2)};
  let r0 = P.diff.z;
  let R = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || R <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((R - r0) / (half - r0), 1.0);
  let edge = min(R, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let cs2 = ${f32(cs * cs)};
  let q = pow(${f32(rIn)} / R, 8.0);
  let o = pow(R / ${f32(rOut)}, 8.0);
  let mid = ${f32(rho0)} * pow(R, ${f32(slope)}) * exp(-q - o);
  let dphi = ${g} / R - ${g} / sqrt(R * R + x.z * x.z);
  let disk = mid * exp(-dphi / cs2);
  let rho = disk + ${f32(amb)};
  let v = sqrt(max(${g} / R + cs2 * (${f32(slope)} + 8.0 * q - 8.0 * o) * disk / rho, 0.0));
  return DampTarget(rate, rho, vec3<f32>(-v * x.y / R, v * x.x / R, 0.0), cs2 * rho);
}
`;
  return { part: d, state, loop, damping: d.damp ? { wgsl, radius: d.damp.from, rate: d.damp.rate } : null };
}

/** Several disks: densities and pressures add, and they share momentum by mass. */
export function sumDisks(states: Prim[]): Prim {
  if (states.length === 1) return states[0];
  let rho = 0, px = 0, py = 0, pz = 0, p = 0, s = 0;
  for (const w of states) {
    rho += w.rho;
    px += w.rho * w.vx;
    py += w.rho * w.vy;
    pz += w.rho * w.vz;
    p += w.p;
    s = Math.max(s, w.s);
  }
  return { rho, vx: px / rho, vy: py / rho, vz: pz / rho, p, s };
}
