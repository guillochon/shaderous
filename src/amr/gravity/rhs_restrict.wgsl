// Source on parent blocks: the average of their children's, in place. Run on each level in
// turn, finest parents first, so every level ends up with the restricted density.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var RHS: texture_storage_3d<r32float, read_write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || blk_leaf(inv.slot)) { return; }
  let h = vec3<i32>(inv.l >= vec3<i32>(BS / 2));
  let ch = bt(inv.slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  let o = blk_origin(ch);
  let f = 2 * (inv.l - (BS / 2) * h);
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    sum += textureLoad(RHS, o + f + vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1)).x;
  }
  textureStore(RHS, blk_origin(inv.slot) + inv.l, vec4<f32>(0.125 * sum, 0.0, 0.0, 0.0));
}
