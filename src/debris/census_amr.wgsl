// Debris census on the block grid (debris/debris.wgsl): every leaf cell into the core or its
// energy bin. Dispatched over the leaf list like the update.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "debris/debris_on.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
// The gas's own potential (x), where there is one.
@group(0) @binding(3) var PHI: texture_3d<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let dx = blk_dx(inv.slot);
  let dv = select(dx * dx * dx, dx * dx, is2d());
  let phi = select(0.0, textureLoad(PHI, c, 0).x, has_potential());
  deb_census(blk_pos(inv.slot, inv.l), textureLoad(U1, c, 0), textureLoad(U2, c, 0), phi, dv);
}
