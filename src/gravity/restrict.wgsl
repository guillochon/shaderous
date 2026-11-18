// Restrict the fine residual to the coarse right-hand side (cell-centered full weighting is the
// average of the eight children) and zero the coarse error. L describes the coarse level.
#include "gravity/mg_common.wgsl"

@group(0) @binding(2) var RES: texture_3d<f32>;
@group(0) @binding(3) var RHSC: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var PHIC: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= lvl_dims())) { return; }
  let c = vec3<i32>(gid);
  let f = 2 * c;
  var sum = 0.0;
  if (L.flat != 0u) {
    // An axisymmetric run: the four children in the plane, weighed by their ring volumes.
    for (var k = 0; k < 4; k++) {
      let o = vec3<i32>(k & 1, (k >> 1) & 1, 0);
      sum += cyl_child_weight(c.x, f.x + o.x) * textureLoad(RES, vec3<i32>(f.xy, 0) + o, 0).x;
    }
    textureStore(RHSC, c, vec4<f32>(0.25 * sum, 0.0, 0.0, 0.0));
    textureStore(PHIC, c, vec4<f32>(0.0));
    return;
  }
  for (var k = 0; k < 8; k++) {
    sum += textureLoad(RES, f + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), 0).x;
  }
  textureStore(RHSC, c, vec4<f32>(0.125 * sum, 0.0, 0.0, 0.0));
  textureStore(PHIC, c, vec4<f32>(0.0));
}
