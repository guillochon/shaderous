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
// With fractions (plans/wind-tunnel.md 8.1's cheap form): the cell's gas fraction and its upper
// faces' open fractions, from the bodies' signed distances (each clamp(1/2 + d / dx, 0, 1)).
@group(0) @binding(59) var GFO: texture_storage_3d<rgba32float, write>;

#include <moving_body>

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let m = textureLoad(U2, c, 0).w;
  if (m < -0.5 || (m > 0.5 && !moving_mark(m))) {
    textureStore(MK, c, vec4<f32>(m, 0.0, 0.0, 0.0));
    if (MOVING_FRACTIONS) { textureStore(GFO, c, vec4<f32>(1.0)); }
    return;
  }
  var x = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  if (P.dims.z == 1u) { x.z = 0.0; }
  if (!MOVING_FRACTIONS) {
    textureStore(MK, c, moving_at(x, 0.0));
    return;
  }
  // A cell is a body's once its gas fraction is under 1/20: by then the solve has pushed nearly
  // all its liquid out.
  textureStore(MK, c, moving_at(x, -0.45 * P.dx));
  let h = 0.5 * P.dx;
  let fx = clamp(0.5 + moving_dist(x + vec3<f32>(h, 0.0, 0.0)) / P.dx, 0.0, 1.0);
  let fy = clamp(0.5 + moving_dist(x + vec3<f32>(0.0, h, 0.0)) / P.dx, 0.0, 1.0);
  let fz = select(clamp(0.5 + moving_dist(x + vec3<f32>(0.0, 0.0, h)) / P.dx, 0.0, 1.0), 1.0, P.dims.z == 1u);
  textureStore(GFO, c, vec4<f32>(clamp(0.5 + moving_dist(x) / P.dx, 0.0, 1.0), fx, fy, fz));
}
