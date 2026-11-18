// Source on parent blocks: the average of their children's, in place (in an axisymmetric run
// weighed by ring volume, so that a parent holds its children's mass). Run on each level in
// turn, finest parents first, so every level ends up with the restricted density.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var RHS: texture_storage_3d<r32float, read_write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || blk_leaf(inv.slot)) { return; }
  if (is2d()) {
    // Four children in the plane, weighed by their ring volumes in an axisymmetric run.
    let h2 = vec2<i32>(inv.l.xy >= vec2<i32>(BS / 2));
    let ch2 = bt(inv.slot, 12u + u32(h2.x | (h2.y << 1u)));
    let o2 = blk_origin(ch2);
    let f2 = 2 * (inv.l.xy - (BS / 2) * h2);
    let pi = blk_coord(inv.slot).x * BS + inv.l.x;
    let ci = blk_coord(ch2).x * BS + f2.x;
    var s2 = 0.0;
    for (var i = 0; i < 4; i++) {
      let a = vec3<i32>(f2 + vec2<i32>(i & 1, (i >> 1u) & 1), 0);
      s2 += cyl_child_weight(pi, ci + (i & 1)) * textureLoad(RHS, o2 + a).x;
    }
    textureStore(RHS, blk_origin(inv.slot) + inv.l, vec4<f32>(0.25 * s2, 0.0, 0.0, 0.0));
    return;
  }
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
