// One colour of a red-black Gauss-Seidel sweep, in place. A cell only reads the other colour,
// so the read-write texture is race-free.
#include "gravity/mg_common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var RHS: texture_3d<f32>;

#include "gravity/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(L.n))) { return; }
  if (((gid.x + gid.y + gid.z) & 1u) != L.color) { return; }
  let c = vec3<i32>(gid);
  let s = nb_sum(c);
  let f = textureLoad(RHS, c, 0).x;
  textureStore(PHI, c, vec4<f32>((s.x - L.h * L.h * f) / s.y, 0.0, 0.0, 0.0));
}
