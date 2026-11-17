// The field brushes on the block grid (edit/field_brush_core.wgsl), in two passes over the
// leaves from the current parity into the other:
//   faces   texels [0, BS] of every leaf: the field plus the brush's change, and the
//           compensations as they were;
//   energy  every leaf cell: the state as it was, its total energy changed by that of the field.
// A face is changed from its global position alone, so the copies two leaves hold of a shared
// face get the same bits, and a coarse face the mean of the fine changes on it.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "edit/brush_core.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var BF: texture_3d<f32>;   // faces before
@group(0) @binding(3) var CF: texture_3d<f32>;   // their compensations
@group(0) @binding(4) var BO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var CO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var BN: texture_3d<f32>;   // faces after (energy pass)
@group(0) @binding(7) var N1: texture_3d<f32>;
@group(0) @binding(8) var N2: texture_3d<f32>;
@group(0) @binding(9) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block

fn fb_cells() -> f32 {
  return f32(i32(P.amr.y << P.amr.z) * BS);
}

// The cell-centered field of a leaf cell from a face texture: the mean of its two faces along
// each axis, from the leaf's own faces (texel BS holds its upper ones).
fn centered(F: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
  let lo = textureLoad(F, c, 0).xyz;
  var hi = lo;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && is2d()) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    hi[a] = textureLoad(F, c + e, 0)[a];
  }
  return 0.5 * (lo + hi);
}

// The field of the leaf cell under x, as amr/bsample.wgsl finds it.
fn fb_field(x: vec3<f32>) -> vec3<f32> {
  let roots = i32(P.amr.y << P.amr.z);
  let slot = i32(textureLoad(IDX, min(vec3<i32>(x * f32(roots)), vec3<i32>(roots - 1)), 0).x);
  let ncell = i32(P.amr.y) * (1 << u32(bt(slot, 0u))) * BS;
  var l = clamp(vec3<i32>(x * f32(ncell)) - blk_coord(slot) * BS, vec3<i32>(0), vec3<i32>(BS - 1));
  if (is2d()) { l.z = 0; }
  return centered(BF, atl(blk_origin(slot), l));
}

#include "edit/field_brush_core.wgsl"

@compute @workgroup_size(FLAT_WG)
fn faces(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = flat_row(wid);
  let w = BS + 1;
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  if (row >= LIST[0] || i >= n) { return; }
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { l.z = 0; }
  let slot = i32(LIST[1u + row]);
  let x = atl(blk_origin(slot), l);
  let level = u32(bt(slot, 0u));
  let h = 1.0 / f32(i32(P.amr.y << level) * BS);
  var o = vec3<f32>(blk_coord(slot) * BS + l) * h;
  if (is2d()) { o.z = 0.0; }
  let m = 1 << (P.amr.z - level);
  var b = textureLoad(BF, x, 0);
  for (var a = 0; a < loop_count(3); a++) {
    b[a] += fb_face(a, o, h, m);
  }
  textureStore(BO, x, b);
  textureStore(CO, x, textureLoad(CF, x, 0));
}

@compute @workgroup_size(8, 8, 4)
fn energy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let bo = centered(BF, c);
  let bn = centered(BN, c);
  var u2 = textureLoad(N2, c, 0);
  u2.x += 0.5 * dot(bn - bo, bn + bo);
  textureStore(O1, c, textureLoad(N1, c, 0));
  textureStore(O2, c, u2);
}
