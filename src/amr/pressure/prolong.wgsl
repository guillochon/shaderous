// Prolongation into the blocks of a tree level: each cell takes the change of the coarser level
// since the restriction (a parent cell's pressure now less the restricted one AUXP kept),
// interpolated linearly between the parent cell and its neighbors toward this cell (bilinear in
// 2D). Where a neighbor is not a parent's cell (a coarser leaf, or beyond the box) there is no
// change to interpolate with, and the parent cell's own stands in.
//
// (Piecewise constant, each cell simply taking its parent's change, was the first version: a
// cycle then cut the residual by only 0.7 at Mach 0.1 and 0.85 at Mach 0.01 on a three-level
// mesh. The liquid did not show it, because its solve starts from the last step's pressure.)
#include "amr/pressure/common.wgsl"

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(6) var AUXP: texture_3d<f32>;

// The change at cell l of the parent block `slot`, where l may lie one cell outside it: then the
// same-level neighbor's cell if that block is a parent too, else `own`.
fn change(slot0: i32, l0: vec3<i32>, own: f32) -> f32 {
  var slot = slot0;
  var l = l0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    if (l[a] < 0 || l[a] >= BS) {
      let up = l[a] >= BS;
      let nb = bt(slot, 6u + u32(2 * a) + u32(up));
      if (nb < 0 || blk_leaf(nb)) { return own; }
      slot = nb;
      l[a] += select(BS, -BS, up);
    }
  }
  let c = atl(blk_origin(slot), l);
  return textureLoad(PRS, c).x - textureLoad(AUXP, c, 0).x;
}

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
  let e0 = textureLoad(PRS, cp).x - textureLoad(AUXP, cp, 0).x;
  // Toward the second-nearest parent cell along each axis: weights 3/4 and 1/4.
  let d = 2 * (inv.l & vec3<i32>(1)) - 1;
  let nk = select(8, 4, is2d());
  var e = 0.0;
  for (var k = 0; k < nk; k++) {
    let q = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    var w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(q));
    if (is2d()) { w.z = 1.0; }
    var ek = e0;
    if (k > 0) { ek = change(parent, lp + q * d, e0); }
    e += w.x * w.y * w.z * ek;
  }
  textureStore(PRS, c, vec4<f32>(textureLoad(PRS, c).x + e, 0.0, 0.0, 0.0));
}
