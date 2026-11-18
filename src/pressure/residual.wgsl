// Residual r = b - A p.
#include "pressure/common.wgsl"

@group(0) @binding(1) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(2) var RHS: texture_3d<f32>;
@group(0) @binding(3) var COEF: texture_3d<f32>;
@group(0) @binding(4) var RES: texture_storage_3d<r32float, write>;

#include "pressure/stencil.wgsl"

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= L.dims)) { return; }
  let c = vec3<i32>(gid);
  let s = op_sum(c);
  var r = 0.0;
  if (s.y > 0.0) { r = textureLoad(RHS, c, 0).x - (s.y * textureLoad(PRS, c).x - s.x) / (L.h * L.h); }
  textureStore(RES, c, vec4<f32>(r, 0.0, 0.0, 0.0));
}
