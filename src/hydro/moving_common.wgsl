// Rigid poses for moving bodies (hydro/moving.ts, geometry/motion.ts): x_now = r x_0 + d.
struct Pose { r: mat3x3<f32>, d: vec3<f32> }
// A body's pose as the pose pass leaves it for the others: the rows of r with d in w, the
// velocity of the body's material at the box's origin (v), and its angular velocity (w): the
// material at x moves at v + w x x.
struct BodyPose { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, v: vec4<f32>, w: vec4<f32> }

fn pose_id() -> Pose {
  return Pose(mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)), vec3<f32>(0.0));
}
// q first, then p.
fn compose(p: Pose, q: Pose) -> Pose {
  return Pose(p.r * q.r, p.r * q.d + p.d);
}
// The turn r about the point a.
fn about(r: mat3x3<f32>, a: vec3<f32>) -> Pose {
  return Pose(r, a - r * a);
}
// The turn by th radians about an axis (Rodrigues), as a matrix (columns).
fn rot_axis(axis: vec3<f32>, th: f32) -> mat3x3<f32> {
  let n = normalize(axis);
  let c = cos(th);
  let s = sin(th);
  let k = 1.0 - c;
  return mat3x3<f32>(
    vec3<f32>(c + n.x * n.x * k, n.y * n.x * k + n.z * s, n.z * n.x * k - n.y * s),
    vec3<f32>(n.x * n.y * k - n.z * s, c + n.y * n.y * k, n.z * n.y * k + n.x * s),
    vec3<f32>(n.x * n.z * k + n.y * s, n.y * n.z * k - n.x * s, c + n.z * n.z * k),
  );
}
// Radians about x, then y, then z: Rz Ry Rx.
fn euler_xyz(ax: f32, ay: f32, az: f32) -> mat3x3<f32> {
  return rot_axis(vec3<f32>(0.0, 0.0, 1.0), az) * rot_axis(vec3<f32>(0.0, 1.0, 0.0), ay) * rot_axis(vec3<f32>(1.0, 0.0, 0.0), ax);
}
