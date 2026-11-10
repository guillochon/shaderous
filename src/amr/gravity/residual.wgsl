// Residual r = f - Lap(phi) on the blocks of one tree level.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var RES: texture_storage_3d<r32float, write>;

#include "amr/gravity/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = g_inv(gid);
  if (!inv.ok) { return; }
  let o = g_origin(inv.slot);
  let s = g_sum(inv.slot, o, inv.l);
  let h = g_h(inv.slot);
  let lap = (s.x - s.y * textureLoad(PHI, o + inv.l).x) / (h * h);
  textureStore(RES, o + inv.l, vec4<f32>(textureLoad(RHS, o + inv.l, 0).x - lap, 0.0, 0.0, 0.0));
}
