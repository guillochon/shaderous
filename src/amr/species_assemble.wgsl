// Species on the block grid, first pass: the mass fractions of every leaf's cells and ghosts,
// SX, from rho X wherever it lives, as amr/assemble.wgsl builds the primitives: the cell
// itself, a same-level neighbor's cell or its image at a boundary, the average of a finer
// block's cells (their species masses over their sum), or the coarser cell beyond the face.
// The last is taken as it is, not interpolated: that keeps a ghost's fractions within those of
// the cells around it and summing to one, at the price of a first-order ghost next to a
// coarser block.
#include "amr/species_common.wgsl"

@group(0) @binding(3) var S: texture_3d<f32>;
@group(0) @binding(4) var SX: texture_storage_3d<rgba32float, write>;

fn u_leaf(slot: i32, l: vec3<i32>) -> array<vec4<f32>, NG> {
  let c = atl(blk_origin(slot), l);
  var u: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { u[g] = textureLoad(S, sp(c, g), 0); }
  return u;
}

// Fractions of interior cell l of any block: a leaf stores it, a parent's is its children's.
fn x_at(slot: i32, l: vec3<i32>) -> array<vec4<f32>, NG> {
  if (blk_leaf(slot)) { return fractions(u_leaf(slot, l)); }
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let ch = bt(slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  var u: array<vec4<f32>, NG>;
  if (ch < 0 || !blk_leaf(ch)) {
    // Refined further: 2:1 balance keeps every leaf's stencil from reaching here.
    u[0].x = 1.0;
    return u;
  }
  let f = 2 * (l - (BS / 2) * h);
  let nz = select(2, 1, is2d());
  for (var k = 0; k < nz; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let c = u_leaf(ch, f + vec3<i32>(i, j, k));
        for (var g = 0u; g < NG; g++) { u[g] += c[g]; }
      }
    }
  }
  return fractions(u);
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let inv = inv_flat(wid, lix, NGH);
  if (!inv.ok) { return; }
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }
  var x: array<vec4<f32>, NG>;
  if (s.kind == 0) {
    x = x_at(s.slot, s.l);
  } else {
    // The coarser leaf's cell that covers this ghost (amr/assemble.wgsl from_coarse).
    let gf = blk_coord(inv.slot) * BS + inv.l;
    let nc = i32(P.amr.y) * BS * (1 << u32(bt(s.slot, 0u)));
    var gc = gf >> vec3<u32>(1u);
    gc = ((gc % nc) + nc) % nc;
    var lc = gc - blk_coord(s.slot) * BS;
    if (is2d()) { lc.z = 0; }
    x = x_at(s.slot, lc);
  }
  let c = atl(blk_origin(inv.slot), inv.l);
  for (var g = 0u; g < NG; g++) { textureStore(SX, sp(c, g), x[g]); }
}
