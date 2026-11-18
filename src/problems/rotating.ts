/**
 * Helpers for runs in a rotating frame (Problem.rotation, gravity/rot.wgsl). Velocities in such
 * a run are those relative to the turning frame.
 */
type V3 = [number, number, number];

const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** A velocity v at x as the frame turning at omega sees it: v - omega x x. */
export function toRotating(v: readonly number[], x: readonly number[], omega: readonly number[]): V3 {
  const r = cross(omega, x);
  return [v[0] - r[0], v[1] - r[1], v[2] - r[2]];
}

/** The inverse: a frame velocity v at x as an inertial observer sees it, v + omega x x. */
export function fromRotating(v: readonly number[], x: readonly number[], omega: readonly number[]): V3 {
  const r = cross(omega, x);
  return [v[0] + r[0], v[1] + r[1], v[2] + r[2]];
}

/**
 * The Coriolis force over a time dt, solved exactly: m turned by -2 |omega| dt about omega
 * (Rodrigues' formula), as rot_turn in gravity/rot.wgsl has it.
 */
export function rotTurn(m: readonly number[], omega: readonly number[], dt: number): V3 {
  const w = Math.hypot(omega[0], omega[1], omega[2]);
  if (w === 0) return [m[0], m[1], m[2]];
  const n = [omega[0] / w, omega[1] / w, omega[2] / w];
  const th = -2 * w * dt;
  const h = Math.sin(0.5 * th);
  const nm = cross(n, m);
  const nnm = cross(n, nm);
  const s = Math.sin(th);
  return [0, 1, 2].map((i) => m[i] + s * nm[i] + 2 * h * h * nnm[i]) as V3;
}
