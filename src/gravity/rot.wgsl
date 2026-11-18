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

// m turned by -2 |Omega| dt about Omega: the Coriolis force over a time dt, solved exactly
// (Rodrigues' formula). A turn keeps |m|, so the force does no work. 1 - cos is formed as
// 2 sin^2(th / 2), which keeps its precision for small angles.
fn rot_turn(m: vec3<f32>, dt: f32) -> vec3<f32> {
  let w = length(P.rot.xyz);
  if (w == 0.0) { return m; }
  let n = P.rot.xyz / w;
  let th = -2.0 * w * dt;
  let h = sin(0.5 * th);
  let nm = cross(n, m);
  return m + sin(th) * nm + (2.0 * h * h) * cross(n, nm);
}

// The centrifugal force at x, per unit mass: -Omega x (Omega x x).
fn rot_accel(x: vec3<f32>) -> vec3<f32> {
  if (P.rot.w == 0.0) { return vec3<f32>(0.0); }
  return -cross(P.rot.xyz, cross(P.rot.xyz, x));
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
