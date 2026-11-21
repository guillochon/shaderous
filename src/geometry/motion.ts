// Bodies that move (plans/wind-tunnel.md 6.1): a part's motion, and its pose at a time as a rigid
// transform from where it starts (x_now = R x_0 + d), chained through its parents (a blade
// pitches on a hub that spins on a nacelle that yaws on a tower). Rates and paths may be
// formulas of t (setup/expr.ts). Velocities are taken from the poses by central differences,
// so that any formula works.
import { parseFormula, type Formula } from '../setup/expr';

type Vec3 = [number, number, number];
/** A number, or a formula of t. */
export type Amount = number | string;

export interface Motion {
  /** turning about `axis` through the point `about` at `rate` radians per unit time */
  spin?: { axis: number[]; about?: number[]; rate: Amount };
  /** moving at `velocity` */
  slide?: { velocity: Amount[] };
  /** turning about `axis` through `about` by `amplitude` degrees, sinusoidally with `period` (and `phase` degrees) */
  swing?: { axis: number[]; about?: number[]; amplitude: number; period: number; phase?: number };
  /** displaced by `pos` and turned by `rotate` degrees (about x, then y, then z, about `about`), each a formula of t */
  path?: { pos?: Amount[]; rotate?: Amount[]; about?: number[] };
}

export interface Pose {
  /** rotation, rows */
  R: number[][];
  d: Vec3;
}

export const POSE_IDENTITY: Pose = { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], d: [0, 0, 0] };

const formulas = new Map<string, Formula>();
/** An amount at time t: a number as it is, a formula of t evaluated. */
export function amountAt(a: Amount | undefined, t: number): number {
  if (a === undefined) return 0;
  if (typeof a === 'number') return a;
  let f = formulas.get(a);
  if (!f) formulas.set(a, (f = parseFormula(a, ['t'])));
  return f.at({ t });
}

/** The integral of an amount from 0 to t (exact for a number; Simpson's rule on 256 pieces for a formula). */
export function integral(a: Amount | undefined, t: number): number {
  if (a === undefined) return 0;
  if (typeof a === 'number') return a * t;
  const n = 256;
  const h = t / n;
  let s = amountAt(a, 0) + amountAt(a, t);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * amountAt(a, i * h);
  return (s * h) / 3;
}

/** The turn by `angle` radians about a unit axis (Rodrigues), as rows. */
export function axisRotation(axis: readonly number[], angle: number): number[][] {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n];
  const c = Math.cos(angle), s = Math.sin(angle), k = 1 - c;
  return [
    [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
    [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
    [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
  ];
}

/** Degrees about x, then y, then z: R = Rz Ry Rx. */
function euler(deg: readonly number[]): number[][] {
  const r = (a: number) => (a * Math.PI) / 180;
  return mulR(axisRotation([0, 0, 1], r(deg[2])), mulR(axisRotation([0, 1, 0], r(deg[1])), axisRotation([1, 0, 0], r(deg[0]))));
}

const mulR = (a: number[][], b: number[][]) => a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
const applyR = (R: number[][], v: readonly number[]): Vec3 => [0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]) as Vec3;

/** A turn R about the point a: x -> R (x - a) + a. */
function about(R: number[][], a: readonly number[] = [0, 0, 0]): Pose {
  const Ra = applyR(R, a);
  return { R, d: [a[0] - Ra[0], a[1] - Ra[1], a[2] - Ra[2]] };
}

/** First q, then p. */
export function compose(p: Pose, q: Pose): Pose {
  const Rd = applyR(p.R, q.d);
  return { R: mulR(p.R, q.R), d: [Rd[0] + p.d[0], Rd[1] + p.d[1], Rd[2] + p.d[2]] };
}

/** Where a point that started at x is, under a pose. */
export const poseApply = (p: Pose, x: readonly number[]): Vec3 => {
  const r = applyR(p.R, x);
  return [r[0] + p.d[0], r[1] + p.d[1], r[2] + p.d[2]];
};

/** Where a point now was at the start: the pose's inverse applied. */
export const poseUnapply = (p: Pose, x: readonly number[]): Vec3 => {
  const y = [x[0] - p.d[0], x[1] - p.d[1], x[2] - p.d[2]];
  return [0, 1, 2].map((i) => p.R[0][i] * y[0] + p.R[1][i] * y[1] + p.R[2][i] * y[2]) as Vec3;
};

/** One motion's own pose at time t (from where its part starts). */
export function motionPose(m: Motion | undefined, t: number): Pose {
  if (!m) return POSE_IDENTITY;
  let p = POSE_IDENTITY;
  if (m.path) {
    const rot = m.path.rotate ? euler([0, 1, 2].map((a) => amountAt(m.path!.rotate![a], t))) : POSE_IDENTITY.R;
    const turn = about(rot, m.path.about);
    const pos = [0, 1, 2].map((a) => amountAt(m.path!.pos?.[a], t) - amountAt(m.path!.pos?.[a], 0));
    p = compose({ R: POSE_IDENTITY.R, d: pos as Vec3 }, turn);
  }
  if (m.swing) {
    const s = m.swing;
    const ang = ((s.amplitude * Math.PI) / 180) * Math.sin((2 * Math.PI * t) / s.period + ((s.phase ?? 0) * Math.PI) / 180);
    p = compose(about(axisRotation(s.axis, ang), s.about), p);
  }
  if (m.spin) p = compose(about(axisRotation(m.spin.axis, integral(m.spin.rate, t)), m.spin.about), p);
  if (m.slide) p = compose({ R: POSE_IDENTITY.R, d: [0, 1, 2].map((a) => integral(m.slide!.velocity[a], t)) as Vec3 }, p);
  return p;
}

/** A part of a chain: its own motion, and its parent's place in the list (-1 for none). */
export interface Moving {
  motion?: Motion;
  parent: number;
}

/** Part i's pose at time t: its parents' poses, then its own (its pivots move with its parent). */
export function chainPose(parts: readonly Moving[], i: number, t: number): Pose {
  let p = motionPose(parts[i].motion, t);
  for (let k = parts[i].parent, guard = 0; k >= 0 && guard < parts.length; k = parts[k].parent, guard++) p = compose(motionPose(parts[k].motion, t), p);
  return p;
}

/**
 * Velocity of the part's material at the point x (where it is now), and its angular velocity,
 * from the poses a little before and after (h of the order of a step).
 */
export function pointVelocity(pose: (t: number) => Pose, x: readonly number[], t: number, h = 1e-4): { v: Vec3; omega: Vec3 } {
  const now = pose(t);
  const x0 = poseUnapply(now, x);
  const a = poseApply(pose(t + h), x0);
  const b = poseApply(pose(t - h), x0);
  const v = [0, 1, 2].map((i) => (a[i] - b[i]) / (2 * h)) as Vec3;
  // The angular velocity: the skew part of dR/dt R^T.
  const Ra = pose(t + h).R, Rb = pose(t - h).R;
  const dR = Ra.map((row, i) => row.map((x2, j) => (x2 - Rb[i][j]) / (2 * h)));
  const W = mulR(dR, [0, 1, 2].map((i) => [now.R[0][i], now.R[1][i], now.R[2][i]]));
  const omega: Vec3 = [(W[2][1] - W[1][2]) / 2, (W[0][2] - W[2][0]) / 2, (W[1][0] - W[0][1]) / 2];
  return { v, omega };
}
