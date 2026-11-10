// Residual r = f - Lap(phi).
#include "gravity/mg_common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var RES: texture_storage_3d<r32float, write>;

#include "gravity/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(L.n))) { return; }
  let c = vec3<i32>(gid);
  let s = nb_sum(c);
  let lap = (s.x - s.y * textureLoad(PHI, c).x) / (L.h * L.h);
  textureStore(RES, c, vec4<f32>(textureLoad(RHS, c, 0).x - lap, 0.0, 0.0, 0.0));
}
