// Applies the editing brush on the block grid: every leaf cell, from the current state into
// the other parity (see edit/brush.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "edit/brush_core.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var N1: texture_3d<f32>;
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  g_dx = blk_dx(inv.slot);
  let c = atl(blk_origin(inv.slot), inv.l);
  let x = blk_pos(inv.slot, inv.l) / P.box + 0.5;
  let out = brush_apply(textureLoad(N1, c, 0), textureLoad(N2, c, 0), brush_weight(x));
  textureStore(O1, c, out.u1);
  textureStore(O2, c, out.u2);
}
