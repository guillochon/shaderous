// First guess for the potential in a block: the value of the parent cell that covers each of
// its cells. The solve that follows refines it.
//   ALL = false  only blocks the last re-mesh created (their slots hold stale values)
//   ALL = true   every block in the list: a cold start, level by level, so each level begins
//                from the coarser solution instead of from zero
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

override ALL: bool = false;

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  if (!ALL && RG[u32(inv.slot) * 12u] != 2) { return; }
  let parent = bt(inv.slot, 5u);
  if (parent < 0) { return; }
  let lc = (inv.l >> vec3<u32>(1u)) + (BS / 2) * (blk_coord(inv.slot) & vec3<i32>(1));
  let v = textureLoad(PHI, blk_origin(parent) + lc).x;
  textureStore(PHI, blk_origin(inv.slot) + inv.l, vec4<f32>(v, 0.0, 0.0, 0.0));
}
