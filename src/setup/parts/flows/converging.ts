// Gas converging on a center, a line or a plane: the Noh implosion and the uniform collapse,
// two problems with exact solutions (src/reference/converging.ts; ExactPack's noh and noh2).
//
// `symmetry` says what the gas converges on: 3 a point (the origin), 2 a line, 1 a plane. In a
// Cartesian run the line is the z axis and the plane x = 0. In an axisymmetric (R, z) or a
// spherical (r, theta) run, whose setup coordinates are (R, z) with the axis along y, the line
// is that axis and the plane z = 0.
import type { FlowContext, FlowDef } from '../flow';

/** The direction away from what the gas converges on, and the distance to it. */
export function radial(ctx: FlowContext, symmetry: number): (x: number, y: number, z: number) => { d: number; n: [number, number, number] } {
  const g = ctx.spec.domain?.geometry;
  const curved = g === 'cylindrical' || g === 'spherical';
  const k = Math.round(symmetry);
  const pick = (x: number, y: number, z: number): [number, number, number] =>
    k >= 3 ? [x, y, z] : k === 2 ? (curved ? [x, 0, z] : [x, y, 0]) : curved ? [0, y, 0] : [x, 0, 0];
  return (x, y, z) => {
    const v = pick(x, y, z);
    const d = Math.hypot(...v);
    // (+ 0: no negative zeros where the gas has no velocity)
    return { d, n: d > 0 ? (v.map((c) => c / d + 0) as [number, number, number]) : [0, 0, 0] };
  };
}

/** A number as a WGSL f32 literal. */
export const f32 = (x: number): string => {
  const s = String(Math.fround(x));
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

/** Cells, from each open face, held at the exact solution: as far as the scheme's stencil reaches, and one more. */
export const HELD_CELLS = 3;

/**
 * A damping zone (hydro/damping.wgsl) that holds the cells next to the box's open faces at the
 * exact solution, set at the end of every step: what the gas beyond the box would do, which an
 * open face (a copy of the cell inside) cannot. `radial` is WGSL for
 * `fn held_radial(d: f32, t: f32) -> vec3<f32>`, the density, the velocity away from the center
 * (negative: inward) and the pressure at distance d from it at time t.
 */
export function heldFaces(ctx: FlowContext, symmetry: number, radial: string): { wgsl: string; radius: number; rate: number; spherical: boolean } {
  const g = ctx.spec.domain?.geometry;
  const curved = g === 'cylindrical' || g === 'spherical';
  const k = Math.round(symmetry);
  const pick = k >= 3 ? 'X' : k === 2 ? (curved ? 'vec3<f32>(X.x, 0.0, X.z)' : 'vec3<f32>(X.x, X.y, 0.0)') : curved ? 'vec3<f32>(0.0, X.y, 0.0)' : 'vec3<f32>(X.x, 0.0, 0.0)';
  // Open faces (outflow, or outflow only), as the setup gives them (absent: outflow).
  const bc = ctx.spec.domain?.bc;
  const open = (side: 'lo' | 'hi', a: number) => [1, 3].includes(bc?.[side]?.[a] ?? 1);
  const tests: string[] = [];
  for (let a = 0; a < 3; a++) {
    const skip = a === 2 ? ' && P.dims.z > 1u' : '';
    if (open('lo', a)) tests.push(`(x[${a}] - lo[${a}] < w${skip})`);
    if (open('hi', a)) tests.push(`(lo[${a}] + side[${a}] - x[${a}] < w${skip})`);
  }
  const zone = tests.length ? tests.join(' || ') : 'false';
  const wgsl = `const DAMPING = true;
${radial}
fn damping_target(x: vec3<f32>, t: f32) -> DampTarget {
  let lo = P.org.xyz;
  let side = box_side();
  let w = ${HELD_CELLS}.0 * g_dx;
  if (!(${zone})) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  // The setup's position: (R, z) in an axisymmetric or spherical run, x itself otherwise.
  var X = x;
  var sn = 0.0;
  var cs = 0.0;
  if (spherical()) {
    let f = (x.x - lo.x) / P.box;
    let r = select(x.x, lo.x * pow((lo.x + P.box) / lo.x, f), P.sph.y != 0.0);
    let th = 3.14159265 * (x.y - lo.y) / P.box;
    sn = sin(th);
    cs = cos(th);
    X = vec3<f32>(r * sn, r * cs, 0.0);
  }
  let v = ${pick};
  let d = length(v);
  let s = held_radial(d, t);
  var vel = select(vec3<f32>(0.0), v * (s.y / max(d, 1e-30)), d > 0.0);
  // A spherical run's velocity is (v_r, v_theta, v_phi).
  if (spherical()) { vel = vec3<f32>(vel.x * sn + vel.y * cs, vel.x * cs - vel.y * sn, vel.z); }
  return DampTarget(1e30, s.x, vel, s.z);
}
`;
  return { wgsl, radius: 0, rate: 0, spherical: true };
}

export const gammaOf = (ctx: FlowContext): number => ctx.spec.physics?.eosParams?.gamma ?? 5 / 3;

/** What is wrong with holding the faces at the exact solution in this setup, or null. */
export function holdIssue(ctx: FlowContext, hold: boolean): string | null {
  if (!hold) return null;
  const eos = ctx.spec.physics?.eos ?? 'ideal';
  return eos === 'ideal' ? null : `its exact solution, which holds the open faces, is for an ideal gas, not "${eos}": turn off “Hold the open faces”`;
}

export const holdField = {
  label: 'Hold the open faces',
  default: true,
  tip: `The cells within ${HELD_CELLS} of each open face are set to the exact solution at the end of every step, as the gas beyond the box would have it. Off: an open face copies the cell inside it, which cannot push gas in as the converging flow does, and its error spreads inward`,
};

const symmetryField = {
  label: 'Symmetry',
  default: 3,
  min: 1,
  max: 3,
  step: 1,
  tip: 'What the gas converges on: 3 a point (spherical), 2 a line (cylindrical), 1 a plane (planar). In a Cartesian run the line is the z axis and the plane x = 0; in an axisymmetric or spherical run, the axis and the plane z = 0',
};

const noh: FlowDef<{ symmetry: number; speed: number; rho: number; p: number; hold: boolean }> = {
  name: 'noh',
  label: 'Noh implosion',
  tip: 'Cold gas of uniform density falling at one speed toward a point, a line or a plane (Noh 1987). A shock leaves the center at a third of that speed (for gamma = 5/3), and the gas behind it is at rest, compressed 64 times in spherical symmetry (16 in cylindrical, 4 in planar). The exact solution is known everywhere, which makes it the standard test of wall heating at the center',
  wholeBox: true,
  fields: {
    symmetry: symmetryField,
    speed: { label: 'Speed', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'The speed at which the gas falls inward' },
    rho: { label: 'Density', default: 1, min: 1e-12, max: 1e12, factor: 1.25, tip: 'Density of the gas at the start' },
    p: { label: 'Pressure', default: 1e-6, min: 0, max: 1e6, factor: 10, tip: 'Pressure of the gas at the start; the exact solution has none, so keep it far below density times speed squared' },
    hold: holdField,
  },
  issues: (ctx, q) => holdIssue(ctx, q.hold),
  damping: (ctx, q) => {
    if (!q.hold) return undefined;
    const g = gammaOf(ctx);
    const k = Math.round(q.symmetry);
    // src/reference/converging.ts nohExact, with the pre-shock pressure on its adiabat
    const radial = `fn held_radial(d: f32, t: f32) -> vec3<f32> {
  if (d < ${f32((q.speed * (g - 1)) / 2)} * t) {
    return vec3<f32>(${f32(q.rho * ((g + 1) / (g - 1)) ** k)}, 0.0, ${f32((g - 1) * q.rho * ((g + 1) / (g - 1)) ** k * 0.5 * q.speed * q.speed)});
  }
  let f = pow(1.0 + ${f32(q.speed)} * t / max(d, 1e-30), ${f32(k - 1)});
  return vec3<f32>(${f32(q.rho)} * f, ${f32(-q.speed)}, ${f32(q.p)} * pow(f, ${f32(g)}));
}`;
    return heldFaces(ctx, q.symmetry, radial);
  },
  init: (ctx, q) => {
    const at = radial(ctx, q.symmetry);
    return (x, y, z) => {
      const { n } = at(x, y, z);
      return { rho: q.rho, vx: -q.speed * n[0] + 0, vy: -q.speed * n[1] + 0, vz: -q.speed * n[2] + 0, p: q.p, s: 0 };
    };
  },
};

const uniformCollapse: FlowDef<{ symmetry: number; tCollapse: number; rho: number; e: number; hold: boolean }> = {
  name: 'uniform-collapse',
  label: 'Uniform collapse',
  tip: 'Gas of uniform density and temperature falling inward at a speed proportional to the distance, u = -r / t_c, so all of it reaches the center at once at t = t_c (ExactPack’s noh2). It stays uniform as it is compressed along its adiabat, with no shock: the density grows as (1 - t / t_c)^-k, for k = 3, 2 or 1 by the symmetry',
  wholeBox: true,
  fields: {
    symmetry: symmetryField,
    tCollapse: { label: 'Collapse time', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'When the gas reaches the center: the distance over the inward speed' },
    rho: { label: 'Density', default: 1, min: 1e-12, max: 1e12, factor: 1.25, tip: 'Density of the gas at the start' },
    e: { label: 'Specific internal energy', default: 1, min: 0, max: 1e6, factor: 1.25, tip: 'Internal energy per unit mass of the gas at the start' },
    hold: holdField,
  },
  issues: (ctx, q) => holdIssue(ctx, q.hold),
  damping: (ctx, q) => {
    if (!q.hold) return undefined;
    const g = gammaOf(ctx);
    const k = Math.round(q.symmetry);
    // src/reference/converging.ts uniformCollapseExact (past the collapse time, the state at it)
    const radial = `fn held_radial(d: f32, t: f32) -> vec3<f32> {
  let s = max(1.0 - t / ${f32(q.tCollapse)}, 1e-6);
  let rho = ${f32(q.rho)} * pow(s, ${f32(-k)});
  return vec3<f32>(rho, -d / (${f32(q.tCollapse)} * s), ${f32((g - 1) * q.e)} * rho * pow(s, ${f32(-(g - 1) * k)}));
}`;
    return heldFaces(ctx, q.symmetry, radial);
  },
  init: (ctx, q) => {
    const at = radial(ctx, q.symmetry);
    const p = ctx.eos.pressure(q.rho, q.rho * q.e);
    return (x, y, z) => {
      const { d, n } = at(x, y, z);
      const u = -d / q.tCollapse;
      return { rho: q.rho, vx: u * n[0] + 0, vy: u * n[1] + 0, vz: u * n[2] + 0, p, s: 0 };
    };
  },
};

export const flows = [noh, uniformCollapse];
