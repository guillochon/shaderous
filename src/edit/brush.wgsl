// Applies the editing brush on the uniform grid: reads the current state, writes the other
// parity (every cell, so the two can be swapped).
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "edit/brush_core.wgsl"

@group(0) @binding(2) var N1: texture_3d<f32>;
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let x = (vec3<f32>(gid) + 0.5) / vec3<f32>(P.dims.xyz);
  comp_use(comp_tex(c));
  let out = brush_apply(textureLoad(N1, c, 0), textureLoad(N2, c, 0), x);
  textureStore(O1, c, out.u1);
  textureStore(O2, c, out.u2);
}
