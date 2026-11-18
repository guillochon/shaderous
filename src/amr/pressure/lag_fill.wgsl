// Implicit pressure for a gas with an energy on blocks: the carried pressure of every leaf, its
// two ghost layers included (PF), from the new pressure of the leaves' own cells (PN, which the
// correction or lag_init wrote). A ghost takes the cell it stands for: a same-level neighbor's,
// the cell's image beyond a wall or its edge cell beyond an open face, the mean of the children
// where the neighbor is refined, and where it is coarser a value on the line from the
// block's own cell through the coarse cell's center. (The predictor takes this field's
// gradient, which is all the ghosts beyond a coarser face are for; the solve forms its own
// values across levels, amr/pressure/faces_gas.wgsl.)
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(25) var PF: texture_storage_3d<r32float, write>;
@group(0) @binding(29) var PN: texture_3d<f32>;

fn pn(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PN, atl(blk_origin(slot), l), 0).x;
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let inv = inv_flat(wid, lix, NGH);
  if (!inv.ok) { return; }
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }
  var p = 0.0;
  if (s.kind == 1) {
    // The coarser leaf's cell that holds this ghost.
    var lc = ((blk_coord(inv.slot) * BS + inv.l) >> vec3<u32>(1u)) - blk_coord(s.slot) * BS;
    for (var a = 0; a < 3; a++) {
      if (inv.l[a] < 0) { lc[a] = BS - 1 - ((-1 - inv.l[a]) >> 1u); }
      if (inv.l[a] >= BS) { lc[a] = (inv.l[a] - BS) >> 1u; }
    }
    if (is2d()) { lc.z = 0; }
    // On the line from the block's cell beside the face through the coarse cell's center, which
    // lies one fine cell beyond the face: the first ghost is 2/3 of the way, the second 4/3.
    let li = clamp(inv.l, vec3<i32>(0), vec3<i32>(BS - 1));
    let out = max(max(-inv.l, inv.l - (BS - 1)), vec3<i32>(0));
    let depth = max(out.x, max(out.y, out.z));
    let pi = pn(inv.slot, select(li, vec3<i32>(li.xy, 0), is2d()));
    p = pi + f32(depth) * 0.6666667 * (pn(s.slot, lc) - pi);
  } else if (blk_leaf(s.slot)) {
    p = pn(s.slot, s.l);
  } else {
    // A refined neighbor: the mean of the children of its cell.
    let o = s.l / (BS / 2);
    let child = bt(s.slot, 12u + u32(o.x | (o.y << 1u) | (o.z << 2u)));
    let f0 = 2 * (s.l - o * (BS / 2));
    let nk = select(8, 4, is2d());
    for (var k = 0; k < nk; k++) { p += pn(child, f0 + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1)); }
    p /= f32(nk);
  }
  textureStore(PF, atl(blk_origin(inv.slot), inv.l), vec4<f32>(p, 0.0, 0.0, 0.0));
}
