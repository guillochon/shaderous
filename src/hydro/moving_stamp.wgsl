// Moving bodies (plans/wind-tunnel.md 6.2): where each cell is after the step, from the poses
// the pose pass left (hydro/moving_pose.wgsl). MK: the cell's new mark (a moving body's, 0 for
// gas; a held cell's or a static body's as it was) and, in a moving body, the body's velocity
// there.
#include "hydro/common.wgsl"
#include "hydro/moving_common.wgsl"
#include "hydro/moving_flags.wgsl"

@group(0) @binding(2) var<storage, read> POSES: array<BodyPose, 4>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var MK: texture_storage_3d<rgba32float, write>;

#include <moving_body>

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let m = textureLoad(U2, c, 0).w;
  if (m < -0.5 || (m > 0.5 && !moving_mark(m))) {
    textureStore(MK, c, vec4<f32>(m, 0.0, 0.0, 0.0));
    return;
  }
  var x = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  if (P.dims.z == 1u) { x.z = 0.0; }
  textureStore(MK, c, moving_at(x));
}
