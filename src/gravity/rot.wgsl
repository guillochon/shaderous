// The rotating frame (Params.rot, problems/types.ts Problem.rotation): the frame turns at the
// constant angular velocity Omega = P.rot.xyz about an axis through the origin, and P.rot.w is 1
// when the centrifugal force acts as well as the Coriolis force (0: Coriolis alone, an f-plane).
// The centrifugal part is a potential, which gravity/ext.wgsl adds to the point masses'; the
// Coriolis part turns the momentum, through the frame slot (gravity/frame_rot.wgsl) in the
// hydro passes and directly in the point masses' passes.
// Requires P.

fn rot_on() -> bool {
  return any(P.rot.xyz != vec3<f32>(0.0));
}

// sin(x), to float32 rounding relative to x however small x is: from its series below a few
// tenths (a step turns the frame by at most 0.1 rad), where the GPU's own sin() is accurate in
// absolute terms only. Its error relative to the tiny angle of one step, the same every step,
// unbalanced a shearing box's Coriolis and tidal forces by some 1e-4 of the shear an orbit.
fn rot_sin(x: f32) -> f32 {
  if (abs(x) > 0.5) { return sin(x); }
  let x2 = x * x;
  return x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))));
}

// m turned by -2 |Omega| dt about Omega: the Coriolis force over a time dt, solved exactly
// (Rodrigues' formula). A turn keeps |m|, so the force does no work. 1 - cos is formed as
// 2 sin^2(th / 2), which keeps its precision for small angles.
fn rot_turn(m: vec3<f32>, dt: f32) -> vec3<f32> {
  let w = length(P.rot.xyz);
  if (w == 0.0) { return m; }
  let n = P.rot.xyz / w;
  let th = -2.0 * w * dt;
  let h = rot_sin(0.5 * th);
  let nm = cross(n, m);
  return m + rot_sin(th) * nm + (2.0 * h * h) * cross(n, nm);
}

// The centrifugal force at x, per unit mass: -Omega x (Omega x x).
fn rot_accel(x: vec3<f32>) -> vec3<f32> {
  if (P.rot.w == 0.0) { return vec3<f32>(0.0); }
  return -cross(P.rot.xyz, cross(P.rot.xyz, x));
}

// A shearing box's tidal force (P.shear): -grad of Phi_t = -q Omega^2 x^2 (+ Omega^2 z^2 / 2
// if stratified), the frame turning about z at Omega.
fn tidal_accel(x: vec3<f32>) -> vec3<f32> {
  let w2 = P.shear.x * P.shear.x;
  return vec3<f32>(2.0 * P.shear.y * w2 * x.x, 0.0, -P.shear.z * w2 * x.z);
}
fn tidal_phi(x: vec3<f32>) -> f32 {
  let w2 = P.shear.x * P.shear.x;
  return -P.shear.y * w2 * x.x * x.x + 0.5 * P.shear.z * w2 * x.z * x.z;
}
// The stratification's part of the difference phi(x + d) - phi(x), formed directly. (The radial
// tidal force is balanced by the Coriolis force of the background shear, not by pressure, so the
// well-balanced reconstruction, which this feeds, must not see it.)
fn tidal_dphi_vertical(x: vec3<f32>, d: vec3<f32>) -> f32 {
  let w2 = P.shear.x * P.shear.x;
  return 0.5 * P.shear.z * w2 * (2.0 * x.z + d.z) * d.z;
}

// Its potential, -|Omega x x|^2 / 2, and the exact difference phi(x + d) - phi(x), formed
// directly: -(Omega x x) . (Omega x d) - |Omega x d|^2 / 2.
fn rot_phi(x: vec3<f32>) -> f32 {
  if (P.rot.w == 0.0) { return 0.0; }
  let r = cross(P.rot.xyz, x);
  return -0.5 * dot(r, r);
}
fn rot_dphi(x: vec3<f32>, d: vec3<f32>) -> f32 {
  if (P.rot.w == 0.0) { return 0.0; }
  let r = cross(P.rot.xyz, x);
  let s = cross(P.rot.xyz, d);
  return -dot(r, s) - 0.5 * dot(s, s);
}
