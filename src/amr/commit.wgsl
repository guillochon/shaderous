// Per-block timesteps without a gravity field: the update writes the blocks that stepped into
// the other state texture, and this copies them back, so the blocks that did not step keep
// their place.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var N1: texture_3d<f32>;
@group(0) @binding(2) var N2: texture_3d<f32>;
@group(0) @binding(3) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(O1, c, textureLoad(N1, c, 0));
  textureStore(O2, c, textureLoad(N2, c, 0));
}
