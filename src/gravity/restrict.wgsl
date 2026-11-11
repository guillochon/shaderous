// Restrict the fine residual to the coarse right-hand side (cell-centered full weighting is the
// average of the eight children) and zero the coarse error. L describes the coarse level.
#include "gravity/mg_common.wgsl"

@group(0) @binding(2) var RES: texture_3d<f32>;
@group(0) @binding(3) var RHSC: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var PHIC: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(L.n))) { return; }
  let c = vec3<i32>(gid);
  let f = 2 * c;
  var sum = 0.0;
  for (var k = 0; k < 8; k++) {
    sum += textureLoad(RES, f + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), 0).x;
  }
  textureStore(RHSC, c, vec4<f32>(0.125 * sum, 0.0, 0.0, 0.0));
  textureStore(PHIC, c, vec4<f32>(0.0));
}
