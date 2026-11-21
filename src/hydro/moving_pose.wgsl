// Moving bodies (plans/wind-tunnel.md 6.2): each body's pose at the end of the step being taken
// (the clock's t + dt; at the start of a run, dt = 0, where it starts), and its velocity, from
// the poses a little before and after: one invocation, once a step.
#include "hydro/common.wgsl"
#include "hydro/moving_common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> POSES: array<BodyPose, 4>;

#include <moving_body>

@compute @workgroup_size(1)
fn main() {
  let t = clock.t + clock.dt;
  let h = 1.0e-3;
  for (var i = 0u; i < N_MOVING; i++) {
    let p = body_pose(i, t);
    let a = body_pose(i, t + h);
    let b = body_pose(i, t - h);
    let s = 0.5 / h;
    let dr = mat3x3<f32>((a.r[0] - b.r[0]) * s, (a.r[1] - b.r[1]) * s, (a.r[2] - b.r[2]) * s);
    let W = dr * transpose(p.r);
    let w = 0.5 * vec3<f32>(W[1][2] - W[2][1], W[2][0] - W[0][2], W[0][1] - W[1][0]);
    let dd = (a.d - b.d) * s;
    // The material at x moves at dd + w x (x - d): at the origin, dd - w x d.
    let v0 = dd - cross(w, p.d);
    POSES[i] = BodyPose(
      vec4<f32>(p.r[0][0], p.r[1][0], p.r[2][0], p.d.x),
      vec4<f32>(p.r[0][1], p.r[1][1], p.r[2][1], p.d.y),
      vec4<f32>(p.r[0][2], p.r[1][2], p.r[2][2], p.d.z),
      vec4<f32>(v0, 0.0),
      vec4<f32>(w, 0.0),
    );
  }
}
