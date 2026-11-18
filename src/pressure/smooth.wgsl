// One color of a red-black Gauss-Seidel sweep, in place. A cell only reads the other color, so
// the read-write texture is race-free; the dispatch covers every other cell along x.
#include "pressure/common.wgsl"

@group(0) @binding(1) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var RHS: texture_3d<f32>;
@group(0) @binding(3) var COEF: texture_3d<f32>;

#include "pressure/stencil.wgsl"

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = 2u * gid.x + ((gid.y + gid.z + L.color) & 1u);
  if (x >= L.dims.x || gid.y >= L.dims.y || gid.z >= L.dims.z) { return; }
  let c = vec3<i32>(i32(x), i32(gid.y), i32(gid.z));
  let s = op_sum(c);
  // A cell nothing reaches (inside an obstacle) keeps its value.
  if (s.y <= 0.0) { return; }
  textureStore(PRS, c, vec4<f32>((s.x + L.h * L.h * textureLoad(RHS, c, 0).x) / s.y, 0.0, 0.0, 0.0));
}
