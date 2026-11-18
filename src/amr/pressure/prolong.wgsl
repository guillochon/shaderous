// Prolongation into the blocks of a tree level: each cell takes the change of its parent cell
// since the restriction (its pressure now less the restricted one AUXP kept), piecewise
// constant. (With restriction by averaging that is enough for a second-order operator.)
#include "amr/pressure/common.wgsl"

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(6) var AUXP: texture_3d<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || !(clock.dt > 0.0)) { return; }
  let parent = bt(inv.slot, 5u);
  if (parent < 0) { return; }
  let o = blk_coord(inv.slot) & vec3<i32>(1);
  var lp = (inv.l >> vec3<u32>(1u)) + o * (BS / 2);
  if (is2d()) { lp.z = 0; }
  let cp = atl(blk_origin(parent), lp);
  let c = atl(blk_origin(inv.slot), inv.l);
  let e = textureLoad(PRS, cp).x - textureLoad(AUXP, cp, 0).x;
  textureStore(PRS, c, vec4<f32>(textureLoad(PRS, c).x + e, 0.0, 0.0, 0.0));
}
