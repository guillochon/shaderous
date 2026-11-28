// Moving bodies on the GPU (plans/wind-tunnel.md 6.2): the `moving` shader slot, generated from
// the setup's moving obstacles. It holds each body's pose as WGSL (its motion and its parents',
// from geometry/motion.ts's definitions, formulas through setup/expr.ts), its shape at the start
// (a sphere or a box), and which marks are moving bodies'. hydro/moving_pose.wgsl evaluates the
// poses once a step; hydro/moving_stamp.wgsl and hydro/moving_exchange.wgsl mark the cells and
// hand the gas of those that change kind to their neighbors.
import { parseFormula } from '../setup/expr';
import type { Motion, Amount } from '../geometry/motion';
import type { Shape } from '../setup/spec';

/** A moving obstacle as the GPU passes take it. */
export interface MovingBody {
  /** its cells' mark: 1 + its body index */
  mark: number;
  shape: Shape;
  motion?: Motion;
  /** index of its parent in the list of moving bodies' chain (any obstacle with a motion or a child that moves), -1 for none */
  parent: number;
}

/** Most moving bodies a run holds (each a body of its own: hydro/forces.ts MAX_BODIES). */
export const MAX_MOVING = 4;

/** The slot without moving bodies: what the passes that look at walls compile against. */
export const MOVING_OFF = 'const MOVING = false;\nconst MOVING_FRACTIONS = false;\nfn moving_mark(m: f32) -> bool { return false; }\nfn gas_fractions(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(1.0); }\n';

const f32 = (x: number) => {
  const s = String(Math.fround(Number(x) || 0));
  return /[.eEn]/.test(s) ? s : `${s}.0`;
};
const v3 = (v: readonly (number | null | undefined)[] | undefined, fill = 0) => `vec3<f32>(${[0, 1, 2].map((a) => f32(v?.[a] ?? fill)).join(', ')})`;

/** An amount (a number or a formula of t) as WGSL in the variable `s`. */
function amount(a: Amount | undefined, s = 's'): string {
  if (a === undefined) return '0.0';
  if (typeof a === 'number') return f32(a);
  return `(${parseFormula(a, ['t']).wgsl({ t: s })})`;
}
const isNumber = (a: Amount | undefined) => a === undefined || typeof a === 'number';

/** The integral from 0 to `t` of an amount, as WGSL: exact for a number, Simpson's rule for a formula. */
function integral(a: Amount | undefined, name: string): { fn: string; call: string } {
  if (isNumber(a)) return { fn: '', call: `${amount(a)} * t` };
  const fn = `fn ${name}(t: f32) -> f32 {
  let n = 64;
  let h = t / f32(n);
  var acc = 0.0;
  for (var i = 0; i <= n; i++) {
    let s = f32(i) * h;
    let w = select(select(2.0, 4.0, (i & 1) == 1), 1.0, i == 0 || i == n);
    acc += w * ${amount(a)};
  }
  return acc * h / 3.0;
}
`;
  return { fn, call: `${name}(t)` };
}

/** WGSL of one motion's own pose at t (from where its part starts), as a function. */
function motionWgsl(m: Motion | undefined, i: number): string {
  let pre = '';
  let body = '  var p = pose_id();\n';
  if (m?.path) {
    const rot = m.path.rotate ?? [0, 0, 0];
    const deg = (a: number) => `${amount(rot[a], 't')} * 0.017453292`;
    body += `  let s = t;\n  let pr = euler_xyz(${deg(0)}, ${deg(1)}, ${deg(2)});\n`;
    const pos = m.path.pos ?? [0, 0, 0];
    body += `  p = compose(Pose(mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)), vec3<f32>(${[0, 1, 2].map((a) => `${amount(pos[a], 't')} - ${amount(pos[a], '0.0')}`).join(', ')})), about(pr, ${v3(m.path.about)}));\n`;
  }
  if (m?.swing) {
    const s = m.swing;
    body += `  p = compose(about(rot_axis(${v3(s.axis)}, ${f32((s.amplitude * Math.PI) / 180)} * sin(${f32((2 * Math.PI) / s.period)} * t + ${f32(((s.phase ?? 0) * Math.PI) / 180)})), ${v3(s.about)}), p);\n`;
  }
  if (m?.spin) {
    const g = integral(m.spin.rate, `spin_angle_${i}`);
    pre += g.fn;
    body += `  p = compose(about(rot_axis(${v3(m.spin.axis)}, ${g.call}), ${v3(m.spin.about)}), p);\n`;
  }
  if (m?.slide) {
    const d = [0, 1, 2].map((a) => integral(m.slide!.velocity[a], `slide_${i}_${a}`));
    pre += d.map((x) => x.fn).join('');
    body += `  p = compose(Pose(mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)), vec3<f32>(${d.map((x) => x.call).join(', ')})), p);\n`;
  }
  return `${pre}fn own_pose_${i}(t: f32) -> Pose {\n${body}  return p;\n}\n`;
}

/**
 * WGSL of a shape's signed distance at the start, `dist_<i>(x)`, in the box's coordinates:
 * negative inside. A box's open sides are far away.
 */
function distWgsl(shape: Shape, i: number, box: number, n: number, ndim: 1 | 2 | 3): string {
  if (shape.type === 'sphere') {
    const r = shape.radiusCells !== undefined ? shape.radiusCells * (box / n) : (shape.radius ?? 0.1 * box);
    const c = shape.center ?? [0, 0, 0];
    const flat = ndim === 2 ? 'q.z = 0.0;\n  ' : '';
    return `fn dist_${i}(x: vec3<f32>) -> f32 {\n  var q = x - ${v3(c)};\n  ${flat}return length(q) - ${f32(r)};\n}\n`;
  }
  if (shape.type === 'box') {
    const lo = [0, 1, 2].map((a) => shape.min?.[a] ?? -1e30);
    const hi = [0, 1, 2].map((a) => shape.max?.[a] ?? 1e30);
    const deg = [0, 1, 2].map((a) => (ndim === 2 && a < 2 ? 0 : Number(shape.rotate?.[a]) || 0));
    const pivot = [0, 1, 2].map((a) => {
      const fl = Number.isFinite(lo[a]) && Math.abs(lo[a]) < 1e29;
      const fh = Number.isFinite(hi[a]) && Math.abs(hi[a]) < 1e29;
      return fl && fh ? (lo[a] + hi[a]) / 2 : fl ? lo[a] : fh ? hi[a] : 0;
    });
    const turned = deg.some((d) => d % 360 !== 0);
    const local = turned
      ? `  let r = euler_xyz(${deg.map((d) => f32((d * Math.PI) / 180)).join(', ')});\n  let q = transpose(r) * (x - ${v3(pivot)}) + ${v3(pivot)};\n`
      : '  let q = x;\n';
    // Per axis, how far outside the slab (negative inside); in 2D z does not count.
    const dz = ndim === 2 ? '-1.0e30' : `max(${f32(lo[2])} - q.z, q.z - ${f32(hi[2])})`;
    return `fn dist_${i}(x: vec3<f32>) -> f32 {\n${local}  let d = vec3<f32>(max(${f32(lo[0])} - q.x, q.x - ${f32(hi[0])}), max(${f32(lo[1])} - q.y, q.y - ${f32(hi[1])}), ${dz});\n  return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);\n}\n`;
  }
  throw new Error(`a moving obstacle must be a sphere or a box for now (not a ${shape.type})`);
}

/**
 * The `moving` slot: MOVING, which marks are moving bodies', and, with `fractions` (the liquid
 * with implicit pressure: plans/wind-tunnel.md 8.1's cheap form), MOVING_FRACTIONS and
 * `gas_fractions(c)`: the cell's gas fraction and its upper faces' open fractions near a moving
 * body (1 elsewhere), which the stamp pass writes (GF, binding 58).
 */
export function movingFlags(bodies: MovingBody[], fractions = false): string {
  if (!bodies.length) return MOVING_OFF;
  const marks = `fn moving_mark(m: f32) -> bool {\n  return ${bodies.map((b) => `abs(m - ${f32(b.mark)}) < 0.25`).join(' || ')};\n}\n`;
  const gf = fractions
    ? '@group(0) @binding(58) var GF: texture_3d<f32>;\nfn gas_fractions(c: vec3<i32>) -> vec4<f32> {\n  return textureLoad(GF, c, 0);\n}\n'
    : 'fn gas_fractions(c: vec3<i32>) -> vec4<f32> {\n  return vec4<f32>(1.0);\n}\n';
  return `const MOVING = true;\nconst MOVING_FRACTIONS = ${fractions};\n${marks}${gf}`;
}

/**
 * The `moving_body` slot for these bodies: poses (`body_pose(i, t)`), the inside tests, and
 * `moving_at(x)`, the mark and velocity a point of the box takes from the bodies (mark 0
 * where none holds it), given the poses the pose pass wrote (POSES).
 */
export function movingBodies(bodies: MovingBody[], ctx: { box: number; n: number; ndim: 1 | 2 | 3 }): string {
  if (bodies.length > MAX_MOVING) throw new Error(`a run holds at most ${MAX_MOVING} moving bodies`);
  let s = `const N_MOVING = ${bodies.length}u;\n`;
  s += bodies.map((b, i) => motionWgsl(b.motion, i)).join('');
  // A body's pose: its parents' poses, then its own.
  s += bodies
    .map((b, i) => {
      const chain: number[] = [];
      for (let k = b.parent, guard = 0; k >= 0 && guard <= bodies.length; k = bodies[k].parent, guard++) chain.push(k);
      let expr = `own_pose_${i}(t)`;
      for (const k of chain) expr = `compose(own_pose_${k}(t), ${expr})`;
      return `fn pose_${i}(t: f32) -> Pose {\n  return ${expr};\n}\n`;
    })
    .join('');
  s += `fn body_pose(i: u32, t: f32) -> Pose {\n${bodies.map((_, i) => `  if (i == ${i}u) { return pose_${i}(t); }\n`).join('')}  return pose_id();\n}\n`;
  s += bodies.map((b, i) => distWgsl(b.shape, i, ctx.box, ctx.n, ctx.ndim)).join('');
  // Each body's distance at a point now (from its start position, the pose's inverse).
  bodies.forEach((_, i) => {
    s += `fn now_dist_${i}(x: vec3<f32>) -> f32 {\n  let p = POSES[${i}];\n  let r = mat3x3<f32>(p.r0.xyz, p.r1.xyz, p.r2.xyz);\n  let d = vec3<f32>(p.r0.w, p.r1.w, p.r2.w);\n  return dist_${i}(r * (x - d));\n}\n`;
  });
  // The signed distance to the nearest body now.
  s += `fn moving_dist(x: vec3<f32>) -> f32 {\n  var d = 1.0e30;\n${bodies.map((_, i) => `  d = min(d, now_dist_${i}(x));\n`).join('')}  return d;\n}\n`;
  // The first body that holds the point, deeper than `thr` (0: its center is inside), its mark
  // and its velocity there; mark 0 where none does.
  s += `fn moving_at(x: vec3<f32>, thr: f32) -> vec4<f32> {\n`;
  bodies.forEach((b, i) => {
    s += `  if (now_dist_${i}(x) < thr) {\n    let p = POSES[${i}];\n    return vec4<f32>(${f32(b.mark)}, cross(p.w.xyz, x) + p.v.xyz);\n  }\n`;
  });
  s += '  return vec4<f32>(0.0);\n}\n';
  return s;
}
