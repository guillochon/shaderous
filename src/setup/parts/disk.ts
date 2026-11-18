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
//
// Either profile may take a density of one's own instead, in radius (the surface or midplane
// density) and in height (the density relative to the midplane): as a formula (`radialFn`,
// `verticalFn`; setup/expr.ts parses it, and nothing in it is ever run as code) or as a table of
// points (`radial`, `vertical`). The damping zone's target is generated from the same formula
// or points (a table as a chain of comparisons: a shader has none to look them up in).
import type { Prim } from '../../problems/types';
import { parseFormula } from '../expr';
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
  /** center of mass of what the disk orbits, and its velocity, where they are not the box's own */
  center?: number[];
  vel?: number[];
}

export interface BuiltDisk {
  part: DiskPart;
  /** where it is centered, if not on the box's center */
  center?: number[];
  /** the state at a point relative to the disk's center, and whether it is disk gas (its mass, for sharing momentum) */
  state(x: number, y: number, z: number): Prim;
  /** A_phi of the field loops at (R, z), or null without loops */
  loop: ((R: number, z: number) => number) | null;
  damping: { wgsl: string; radius: number; rate: number } | null;
}

export function buildDisk(part: DiskPart, ctx: DiskContext): BuiltDisk {
  const disk = part.profile === 'thin' ? thinDisk(part, ctx) : balancedDisk(part, ctx);
  const c = ctx.center ?? [0, 0, 0];
  const v = ctx.vel ?? [0, 0, 0];
  if (c.every((x) => x === 0) && v.every((x) => x === 0)) return disk;
  // About a center that is not the box's: the same disk there, moving with it.
  const { state, loop } = disk;
  return {
    ...disk,
    center: c,
    state: (x, y, z) => {
      const w = state(x - c[0], y - c[1], z - c[2]);
      return { ...w, vx: w.vx + v[0], vy: w.vy + v[1], vz: w.vz + v[2] };
    },
    loop,
  };
}

/** A tabulated profile's points, in order, as the functions below read them. */
type Points = [number, number][];
const ordered = (t: Points): Points => [...t].sort((a, b) => a[0] - b[0]);

/**
 * A radial table as a function: the density at r, and its logarithmic slope there. A power
 * law between two points (a straight line where either is zero), nothing outside them.
 */
export function radialTable(table: Points): (r: number) => { value: number; slope: number } {
  const t = ordered(table);
  const n = t.length;
  return (r) => {
    if (n < 2 || r < t[0][0] || r >= t[n - 1][0]) return { value: 0, slope: 0 };
    let i = 0;
    while (i < n - 2 && r >= t[i + 1][0]) i++;
    const [[r0, v0], [r1, v1]] = [t[i], t[i + 1]];
    if (v0 > 0 && v1 > 0 && r0 > 0) {
      const slope = Math.log(v1 / v0) / Math.log(r1 / r0);
      return { value: v0 * Math.pow(r / r0, slope), slope };
    }
    const m = (v1 - v0) / (r1 - r0);
    const value = v0 + m * (r - r0);
    return { value, slope: value > 0 ? (m * r) / value : 0 };
  };
}

/** A vertical table as a function of height above the midplane, and its integral over both sides. */
export function verticalTable(table: Points): { at: (h: number) => number; integral: number } {
  const t = ordered(table.map(([h, v]) => [Math.abs(h), Math.max(v, 0)] as [number, number]));
  const n = t.length;
  const at = (h: number): number => {
    const a = Math.abs(h);
    if (n === 0) return 0;
    if (a <= t[0][0]) return t[0][1];
    if (a >= t[n - 1][0]) return t[n - 1][1];
    let i = 0;
    while (i < n - 2 && a >= t[i + 1][0]) i++;
    return t[i][1] + ((t[i + 1][1] - t[i][1]) * (a - t[i][0])) / (t[i + 1][0] - t[i][0]);
  };
  // Trapezoids between the points, and a flat piece from the midplane to the first. A profile
  // that does not end at zero has no finite integral: it is taken to end at its last point.
  let half = n ? t[0][0] * t[0][1] : 0;
  for (let i = 0; i + 1 < n; i++) half += 0.5 * (t[i][1] + t[i + 1][1]) * (t[i + 1][0] - t[i][0]);
  return { at, integral: 2 * half };
}

/** The radial table as WGSL: `fn disk_radial(r: f32) -> vec2<f32>`, the density and its logarithmic slope. */
function radialWgsl(table: Points): string {
  const t = ordered(table);
  const n = t.length;
  const lines = [`fn disk_radial(r: f32) -> vec2<f32> {`, `  if (r < ${f32(t[0][0])} || r >= ${f32(t[n - 1][0])}) { return vec2<f32>(0.0); }`];
  for (let i = 0; i + 1 < n; i++) {
    const [[r0, v0], [r1, v1]] = [t[i], t[i + 1]];
    const open = i + 2 < n ? `  if (r < ${f32(r1)}) { ` : '  { ';
    if (v0 > 0 && v1 > 0 && r0 > 0) {
      const slope = Math.log(v1 / v0) / Math.log(r1 / r0);
      lines.push(`${open}return vec2<f32>(${f32(v0)} * pow(r / ${f32(r0)}, ${f32(slope)}), ${f32(slope)}); }`);
    } else {
      const m = (v1 - v0) / (r1 - r0);
      lines.push(`${open}let v = ${f32(v0)} + ${f32(m)} * (r - ${f32(r0)}); return vec2<f32>(v, select(0.0, ${f32(m)} * r / v, v > 0.0)); }`);
    }
  }
  return `${lines.join('\n')}\n}\n`;
}

/** The vertical table as WGSL: `fn disk_vertical(h: f32) -> f32`, the density relative to the midplane. */
function verticalWgsl(table: Points): string {
  const t = ordered(table.map(([h, v]) => [Math.abs(h), Math.max(v, 0)] as [number, number]));
  const n = t.length;
  const lines = [`fn disk_vertical(h: f32) -> f32 {`, `  let a = abs(h);`, `  if (a <= ${f32(t[0][0])}) { return ${f32(t[0][1])}; }`];
  for (let i = 0; i + 1 < n; i++) {
    const [[h0, v0], [h1, v1]] = [t[i], t[i + 1]];
    lines.push(`  if (a < ${f32(h1)}) { return ${f32(v0)} + ${f32((v1 - v0) / (h1 - h0))} * (a - ${f32(h0)}); }`);
  }
  lines.push(`  return ${f32(t[n - 1][1])};`);
  return `${lines.join('\n')}\n}\n`;
}

const usable = (t: Points | undefined): t is Points => !!t && t.length >= 2;

/** What a radial formula may name: the radius, the disk's own numbers, and the scale height cs / Omega at that radius. */
export const RADIAL_NAMES = ['r', 'rIn', 'rOut', 'sigma0', 'cs', 'H'] as const;
/** What a vertical formula may name: the height above the midplane (in scale heights, or code lengths). */
export const VERTICAL_NAMES = ['z'] as const;

interface RadialSource {
  at(r: number): { value: number; slope: number };
  /** defines `fn disk_radial(r: f32) -> vec2<f32>`: the density and its logarithmic slope */
  wgsl: string;
}
interface VerticalSource {
  at(h: number): number;
  integral: number;
  /** defines `fn disk_vertical(h: f32) -> f32` */
  wgsl: string;
}

/** Steps either side of r, as a factor, for the slope of a formula: wide enough to survive float32 in the shader. */
const SLOPE_STEP = 1.01;

/** A disk's own radial profile, if it has one: its formula, else its table. `sgm` is the square root of G M of what it orbits. */
function radialSource(d: DiskPart, sgm: number): RadialSource | null {
  if (d.radialFn !== undefined && d.radialFn.trim() !== '') {
    const f = parseFormula(d.radialFn, RADIAL_NAMES);
    const consts = { rIn: d.rIn, rOut: d.rOut, sigma0: d.sigma0, cs: d.cs };
    const value = (r: number) => {
      const v = f.at({ ...consts, r, H: (d.cs * r ** 1.5) / sgm });
      return v > 0 && Number.isFinite(v) ? v : 0;
    };
    const at = (r: number) => {
      const [a, b] = [value(r * SLOPE_STEP), value(r / SLOPE_STEP)];
      return { value: value(r), slope: a > 0 && b > 0 ? Math.log(a / b) / (2 * Math.log(SLOPE_STEP)) : 0 };
    };
    const body = f.wgsl({ r: 'r', rIn: f32(d.rIn), rOut: f32(d.rOut), sigma0: f32(d.sigma0), cs: f32(d.cs), H: `${f32(d.cs / sgm)} * pow(r, 1.5)` });
    // A value that is not a positive number (a logarithm of nothing, a division by zero) is no gas.
    const wgsl = `fn disk_radial_value(r: f32) -> f32 {
  let v = ${body};
  return select(0.0, v, v > 0.0 && v < 3.0e38);
}
fn disk_radial(r: f32) -> vec2<f32> {
  let a = disk_radial_value(r * ${f32(SLOPE_STEP)});
  let b = disk_radial_value(r / ${f32(SLOPE_STEP)});
  return vec2<f32>(disk_radial_value(r), select(0.0, log(a / b) / ${f32(2 * Math.log(SLOPE_STEP))}, a > 0.0 && b > 0.0));
}
`;
    return { at, wgsl };
  }
  return usable(d.radial) ? { at: radialTable(d.radial), wgsl: radialWgsl(d.radial) } : null;
}

/** A disk's own vertical profile, if it has one: its formula, else its table. `reach` is how far up to integrate it. */
function verticalSource(d: DiskPart, reach: number): VerticalSource | null {
  if (d.verticalFn !== undefined && d.verticalFn.trim() !== '') {
    const f = parseFormula(d.verticalFn, VERTICAL_NAMES);
    const at = (h: number) => {
      const v = f.at({ z: Math.abs(h) });
      return v > 0 && Number.isFinite(v) ? v : 0;
    };
    // Simpson's rule from the midplane out, both sides.
    const n = 4000;
    const dz = reach / n;
    let sum = at(0) + at(reach);
    for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * at(i * dz);
    const wgsl = `fn disk_vertical(h: f32) -> f32 {
  let z = abs(h);
  let v = ${f.wgsl({ z: 'z' })};
  return select(0.0, v, v > 0.0 && v < 3.0e38);
}
`;
    return { at, integral: (2 * sum * dz) / 3, wgsl };
  }
  return usable(d.vertical) ? { ...verticalTable(d.vertical), wgsl: verticalWgsl(d.vertical) } : null;
}

/** Check a disk's formulas: throws with what is wrong in the first that does not parse. */
export function checkDiskFormulas(d: DiskPart): void {
  if (d.radialFn?.trim()) parseFormula(d.radialFn, RADIAL_NAMES);
  if (d.verticalFn?.trim()) parseFormula(d.verticalFn, VERTICAL_NAMES);
}

function thinDisk(d: DiskPart, ctx: DiskContext): BuiltDisk {
  const { cs, sigma0, rIn, rOut } = d;
  const floor = d.floor ?? 0;
  const { box, ndim } = ctx;
  const dx = box / ctx.n;
  // Circular speed about the total mass, flattened inside the binary's own orbit.
  const flat2 = ctx.flat !== undefined ? ctx.flat * ctx.flat : ctx.soft * ctx.soft;
  const sgm = ctx.gm === 1 ? 1 : Math.sqrt(ctx.gm);
  const heights = d.verticalIn === 'length';
  const radial = radialSource(d, sgm);
  const vertical = verticalSource(d, heights ? box : 40);
  // A profile of one's own is the surface density; the gas outside the disk is still a share of sigma0.
  const surface = radial
    ? (r: number) => radial.at(r).value + sigma0 * floor
    : (r: number) => sigma0 * (Math.exp(-((rIn / r) ** 4)) * Math.exp(-((r / rOut) ** 8)) + floor);
  const speed = (r: number) => (sgm === 1 ? r / Math.pow(r * r + flat2, 0.75) : (sgm * r) / Math.pow(r * r + flat2, 0.75));
  const state = (x: number, y: number, z: number): Prim => {
    const r = Math.max(Math.hypot(x, y), 1e-6);
    const sigma = surface(r);
    const v = speed(r);
    let rho = sigma;
    if (ndim === 3) {
      const h = Math.max((cs * r) / v, 2 * dx);
      if (vertical) {
        // The table's shape, holding the surface density: over its own integral, in scale heights or lengths.
        rho = (sigma * vertical.at(heights ? z : z / h)) / Math.max(heights ? vertical.integral : h * vertical.integral, 1e-300) + (sigma0 * floor) / box;
      } else rho = (sigma / (Math.sqrt(2 * Math.PI) * h)) * Math.exp((-z * z) / (2 * h * h)) + (sigma0 * floor) / box;
    }
    return { rho, vx: (-v * y) / r, vy: (v * x) / r, vz: 0, p: cs * cs * rho, s: sigma > 0.5 * sigma0 ? 1 : 0 };
  };
  const sigmaWgsl = radial
    ? `disk_radial(r).x + ${f32(sigma0 * floor)}`
    : `${f32(sigma0)} * (exp(-pow(${f32(rIn)} / r, 4.0)) * exp(-pow(r / ${f32(rOut)}, 8.0)) + ${f32(floor)})`;
  const rhoWgsl = vertical
    ? `sigma * disk_vertical(${heights ? 'x.z' : 'x.z / h'}) / ${heights ? f32(Math.max(vertical.integral, 1e-30)) : `(h * ${f32(Math.max(vertical.integral, 1e-30))})`}`
    : `sigma / (2.50662827 * h) * exp(-x.z * x.z / (2.0 * h * h))`;
  const tables = (radial?.wgsl ?? '') + (vertical?.wgsl ?? '');
  // The same disk as the damping zone's target. The zone starts at radius P.diff.z; P.diff.w is
  // its rate at the box's edge, per local orbital period.
  const wgsl = `const DAMPING = true;
${tables}fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(box / 2)};
  let r0 = P.diff.z;
  let r = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || r <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((r - r0) / (half - r0), 1.0);
  let edge = min(r, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let sigma = ${sigmaWgsl};
  let v = ${sgm === 1 ? '' : `${f32(sgm)} * `}r / pow(r * r + ${f32(flat2)}, 0.75);
  var rho = sigma;
  if (P.dims.z > 1u) {
    let h = max(${f32(cs)} * r / v, 2.0 * g_dx);
    rho = ${rhoWgsl} + ${f32((sigma0 * floor) / box)};
  }
  return DampTarget(rate, rho, vec3<f32>(-v * x.y / r, v * x.x / r, 0.0), ${f32(cs * cs)} * rho);
}
`;
  // Field loops on the thin profile's scale height and midplane density as its state has them:
  // floored at two cells of the grid the document names (compile.ts fieldOf), so that a disk
  // thinner than that holds loops as thick as its gas, not sub-cell ones whose radial field
  // overwhelms it. Without a grid (n = 1), the physical height.
  const height = (r: number) => Math.max((cs * r) / speed(r), ctx.n > 1 ? 2 * dx : 0);
  const loop = d.loops ? diskLoops(d.loops, rIn, cs, (r) => (ndim === 3 ? surface(r) / (Math.sqrt(2 * Math.PI) * height(r)) : surface(r)), height) : null;
  return { part: d, state, loop, damping: d.damp ? { wgsl, radius: d.damp.from, rate: d.damp.rate } : null };
}

function balancedDisk(d: DiskPart, ctx: DiskContext): BuiltDisk {
  const { cs, rIn, rOut } = d;
  const rho0 = d.sigma0;
  const slope = d.slope ?? -1.5;
  const amb = d.floor ?? 0;
  const gm = ctx.gm;
  const eps = ctx.soft;
  /** The disk's midplane density at cylindrical radius R, and dln(rho_mid) / dln(R). */
  const heights = d.verticalIn === 'length';
  const sgm0 = gm === 1 ? 1 : Math.sqrt(gm);
  const radial = radialSource(d, sgm0);
  const vertical = verticalSource(d, heights ? ctx.box : 40);
  const midplane = radial ? (R: number) => radial.at(R).value : (R: number) => rho0 * R ** slope * Math.exp(-((rIn / R) ** 8) - (R / rOut) ** 8);
  const midSlope = radial ? (R: number) => radial.at(R).slope : (R: number) => slope + 8 * (rIn / R) ** 8 - 8 * (R / rOut) ** 8;
  /** The scale height cs / Omega_K at R, for a vertical profile given in scale heights. */
  const height = (R: number) => (sgm0 === 1 ? cs * R ** 1.5 : (cs * R ** 1.5) / sgm0);
  /** The softened potential of the mass. */
  const phi = (R: number, z: number) => -gm / Math.sqrt(R * R + z * z + eps * eps);
  const state = (x: number, y: number, z: number): Prim => {
    const R = Math.max(Math.hypot(x, y), 1e-6);
    const disk = vertical ? midplane(R) * vertical.at(heights ? z : z / height(R)) : midplane(R) * Math.exp(-(phi(R, z) - phi(R, 0)) / (cs * cs));
    const rho = disk + amb;
    // The disk's pressure gradient, weighted by its share of the gas: the ambient gas orbits
    // at the Keplerian speed of its radius.
    const v2 = (gm * R * R) / (R * R + eps * eps) ** 1.5 + (disk > 0 ? (cs * cs * midSlope(R) * disk) / rho : 0);
    const v = Math.sqrt(Math.max(v2, 0));
    return { rho, vx: (-v * y) / R, vy: (v * x) / R, vz: 0, p: cs ** 2 * rho, s: disk > 10 * amb ? 1 : 0 };
  };
  const sgm = gm === 1 ? 1 : Math.sqrt(gm);
  /** The scale height cs / Omega_K at R. */
  const scaleHeight = (R: number) => (sgm === 1 ? cs * R ** 1.5 : (cs * R ** 1.5) / sgm);
  const loop = d.loops ? diskLoops(d.loops, rIn, cs, midplane, scaleHeight) : null;
  // The same disk as the damping zone's target. The softening there is far below R, so it is
  // left out.
  const g = gm === 1 ? '1.0' : f32(gm);
  const tables = (radial?.wgsl ?? '') + (vertical?.wgsl ?? '');
  // The midplane density and its logarithmic slope, and the density at height, as the state above has them.
  const midWgsl = radial
    ? `  let rad = disk_radial(R);\n  let mid = rad.x;\n  let dmid = rad.y;`
    : `  let q = pow(${f32(rIn)} / R, 8.0);\n  let o = pow(R / ${f32(rOut)}, 8.0);\n  let mid = ${f32(rho0)} * pow(R, ${f32(slope)}) * exp(-q - o);`;
  const slopeWgsl = radial ? 'dmid' : `(${f32(slope)} + 8.0 * q - 8.0 * o)`;
  const diskWgsl = vertical
    ? `  let disk = mid * disk_vertical(${heights ? 'x.z' : `x.z / (${f32(cs)} * pow(R, 1.5)${sgm0 === 1 ? '' : ` / ${f32(sgm0)}`})`});`
    : `  let dphi = ${g} / R - ${g} / sqrt(R * R + x.z * x.z);\n  let disk = mid * exp(-dphi / cs2);`;
  const wgsl = `const DAMPING = true;
${tables}fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(ctx.box / 2)};
  let r0 = P.diff.z;
  let R = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || R <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((R - r0) / (half - r0), 1.0);
  let edge = min(R, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let cs2 = ${f32(cs * cs)};
${midWgsl}
${diskWgsl}
  let rho = disk + ${f32(amb)};
  let v = sqrt(max(${g} / R + ${radial || vertical ? `select(0.0, cs2 * ${slopeWgsl} * disk / max(rho, 1e-30), disk > 0.0)` : `cs2 * ${slopeWgsl} * disk / rho`}, 0.0));
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

/**
 * The field loops of a disk (DiskPart.loops) with midplane density rho_mid(R) and scale height
 * H(R): loops with no net flux, A_phi = a(R) f(z) sin(pi ln(R / rIn) / ln(span)) between rIn and
 * rIn span^count, where f = max(exp(-z^2 / 2H^2) - e^-2, 0) closes them at |z| = 2H, and a sets
 * the plasma beta where two loops meet at the midplane.
 */
function diskLoops(loops: NonNullable<DiskPart['loops']>, rIn: number, cs: number, rhoMid: (R: number) => number, H: (R: number) => number): (R: number, z: number) => number {
  const { beta, span, count } = loops;
  return (R, z) => {
    if (R <= rIn || R >= rIn * span ** count) return 0;
    const h = H(R);
    const f = Math.exp((-z * z) / (2 * h * h)) - Math.exp(-2);
    if (f <= 0) return 0;
    // B_z ~ (1 / R) d(R A_phi) / dR ~ a f pi / (R ln span) where two loops meet, at the midplane.
    const b0 = Math.sqrt((2 * cs * cs * rhoMid(R)) / beta);
    const a = (b0 * R * Math.log(span)) / (Math.PI * (1 - Math.exp(-2)));
    return a * f * Math.sin((Math.PI * Math.log(R / rIn)) / Math.log(span));
  };
}
