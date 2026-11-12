// Species on the block grid, first pass: the mass fractions of every leaf's cells and ghosts,
// SX, from rho X wherever it lives, as amr/assemble.wgsl builds the primitives: the cell
// itself, a same-level neighbor's cell or its image at a boundary, the average of a finer
// block's cells (their species masses over their sum), or the coarser cell beyond the face,
// interpolated linearly with the slopes of hydro/species_slopes.wgsl: limited for each species,
// summing to zero, so the ghost's fractions sum to one and stay close to the range of the
// coarse cells around it (clamped to [0, 1] in case the three axes' shifts add up past it).
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

struct MaybeX { x: array<vec4<f32>, NG>, ok: bool }

// Fractions of a cell next to the interior of block `slot` (inside it or beyond a face), if a
// block at its level holds it.
fn x_near(slot: i32, l: vec3<i32>) -> MaybeX {
  let s = ghost_src(slot, l);
  var x: array<vec4<f32>, NG>;
  if (s.kind != 0) { return MaybeX(x, false); }
  return MaybeX(x_at(s.slot, s.l), true);
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
    // Its slope along each axis, from the coarse cells on both sides (none where either side is
    // not at the coarse block's level), toward the quarter of it the ghost sits in.
    let off = vec3<f32>(gf & vec3<i32>(1)) - 0.5;
    let axes = select(3, 2, is2d());
    var y = x;
    for (var a = 0; a < axes; a++) {
      let m = x_near(s.slot, lc - unit(a));
      let p = x_near(s.slot, lc + unit(a));
      if (!m.ok || !p.ok) { continue; }
      let d = slopes(m.x, x, p.x);
      for (var g = 0u; g < NG; g++) { y[g] += 0.5 * off[a] * d[g]; }
    }
    x = face_fractions(y);
  }
  let c = atl(blk_origin(inv.slot), inv.l);
  for (var g = 0u; g < NG; g++) { textureStore(SX, sp(c, g), x[g]); }
}
