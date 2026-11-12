// Species on the block grid, second pass: the fractions half a step ahead, SH, for every leaf's
// cells and first ghost layer, X - dt/2 v . grad X, as amr/predict.wgsl predicts the
// primitives. A ghost standing for a same-level cell gets that cell's own prediction, made
// from its own block's data, so the two blocks sharing a face reconstruct identical species
// fluxes there. A ghost next to a coarser block is predicted in place, without the transverse
// slopes that would need edge ghosts.
#include "amr/species_common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;   // (rho, v), the density negative in a solid
@group(0) @binding(4) var SX: texture_3d<f32>;
@group(0) @binding(5) var SH: texture_storage_3d<rgba32float, write>;

fn load_sx(c: vec3<i32>) -> array<vec4<f32>, NG> {
  var x: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { x[g] = textureLoad(SX, sp(c, g), 0); }
  return x;
}

fn predict_x(slot: i32, l: vec3<i32>, mask: vec3<f32>) -> array<vec4<f32>, NG> {
  let c = atl(blk_origin(slot), l);
  let x = load_sx(c);
  var xh = x;
  let w = textureLoad(W1, c, 0);
  if (w.x < 0.0) { return x; }
  let h = 0.5 * clock.dt / blk_dx(slot);
  let nax = select(3, 2, is2d());
  for (var a = 0; a < nax; a++) {
    if (mask[a] == 0.0) { continue; }
    var e = vec3<i32>(0);
    e[a] = 1;
    let s = slopes(load_sx(c - e), x, load_sx(c + e));
    for (var g = 0u; g < NG; g++) { xh[g] -= h * w.yzw[a] * s[g]; }
  }
  return xh;
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let inv = inv_flat(wid, lix, 1);
  if (!inv.ok) { return; }
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }
  var xh: array<vec4<f32>, NG>;
  if (s.kind == 0) {
    xh = predict_x(s.slot, s.l, vec3<f32>(1.0));
  } else {
    let rim = (inv.l == vec3<i32>(0)) | (inv.l == vec3<i32>(BS - 1));
    xh = predict_x(inv.slot, inv.l, select(vec3<f32>(1.0), vec3<f32>(0.0), rim));
  }
  let c = atl(blk_origin(inv.slot), inv.l);
  for (var g = 0u; g < NG; g++) { textureStore(SH, sp(c, g), xh[g]); }
}
