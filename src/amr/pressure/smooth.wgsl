// One color of a red-black Gauss-Seidel sweep of the plain operator over the blocks of a tree
// level, in place. Colors are by a cell's global parity (a block has an even number of cells
// per side), so a cell reads only the other color, here and in neighboring blocks, and the
// coarser level, which this pass does not write.
#include "amr/pressure/common.wgsl"

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var TAU: texture_3d<f32>;

override COLOR: u32 = 0u;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
#include "amr/pressure/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || !(clock.dt > 0.0)) { return; }
  let l = inv.l;
  if (u32(l.x + l.y + l.z) % 2u != COLOR) { return; }
  if (ap_held(inv.slot, l)) { return; }
  let c = atl(blk_origin(inv.slot), l);
  let h = blk_dx(inv.slot);
  let s = ap_plain(inv.slot, l);
  let b = textureLoad(RHS, c, 0).x + textureLoad(TAU, c, 0).x;
  textureStore(PRS, c, vec4<f32>((s.x + h * h * b) / (s.y + ap_alpha() * h * h), 0.0, 0.0, 0.0));
}
