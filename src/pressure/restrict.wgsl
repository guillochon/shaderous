// Restrict the fine residual to the coarse right-hand side (the average of the children: eight,
// or four in 2D) and zero the coarse error. L describes the coarse level.
#include "pressure/common.wgsl"

@group(0) @binding(1) var RES: texture_3d<f32>;
@group(0) @binding(2) var RHSC: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var PRSC: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= L.dims)) { return; }
  let c = vec3<i32>(gid);
  let flat = L.dims.z == 1u;
  let f = vec3<i32>(2 * c.xy, select(2 * c.z, 0, flat));
  let nk = select(8, 4, flat);
  var sum = 0.0;
  for (var k = 0; k < nk; k++) {
    sum += textureLoad(RES, f + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), 0).x;
  }
  textureStore(RHSC, c, vec4<f32>(sum / f32(nk), 0.0, 0.0, 0.0));
  textureStore(PRSC, c, vec4<f32>(0.0));
}
