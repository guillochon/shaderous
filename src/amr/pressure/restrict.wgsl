// Restriction onto the parents of a tree level, in two stages over that level's blocks.
//
// STAGE 0 (parents only): a parent cell takes the mean of its children's pressures; AUXR keeps
// the mean of their residuals and AUXP that restricted pressure.
// STAGE 1: the full-approximation right-hand sides. A parent's is the restricted residual plus
// the plain operator on the restricted pressure, written to RHS. A leaf's stays what the source
// pass made it, and TAU takes the plain operator less the composite one where a face of the
// cell is refined (zero elsewhere): with it the plain operator, which the level relaxes,
// reproduces the composite one for the fine pressures as they are now.
#include "amr/pressure/common.wgsl"
#include <ap_coef>

override STAGE: u32 = 0u;

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var RHS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var TAU: texture_storage_3d<r32float, read_write>;
@group(0) @binding(5) var AUXR: texture_storage_3d<r32float, read_write>;
@group(0) @binding(6) var AUXP: texture_storage_3d<r32float, read_write>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
#include "amr/pressure/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || !(clock.dt > 0.0)) { return; }
  let slot = inv.slot;
  let l = inv.l;
  let c = atl(blk_origin(slot), l);
  let h = blk_dx(slot);
  let ah2 = ap_alpha_at(slot, l) * h * h;
  let leaf = blk_leaf(slot);
  if (STAGE == 0u) {
    if (leaf) { return; }
    // The child block that holds this cell's children, and the first of them.
    let o = l / (BS / 2);
    let child = bt(slot, 12u + u32(o.x | (o.y << 1u) | (o.z << 2u)));
    let f0 = 2 * (l - o * (BS / 2));
    let hf = 0.5 * h;
    let nk = select(8, 4, is2d());
    var p = 0.0;
    var r = 0.0;
    for (var k = 0; k < nk; k++) {
      let lf = f0 + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
      let cf = atl(blk_origin(child), lf);
      let pf = textureLoad(PRS, cf).x;
      p += pf;
      // (A child in an obstacle has no equation, and no residual.)
      if (ap_held(child, lf)) { continue; }
      let s = ap_plain(child, lf);
      r += textureLoad(RHS, cf).x + textureLoad(TAU, cf).x - ((s.y + ap_alpha_at(child, lf) * hf * hf) * pf - s.x) / (hf * hf);
    }
    p /= f32(nk);
    textureStore(PRS, c, vec4<f32>(p, 0.0, 0.0, 0.0));
    textureStore(AUXR, c, vec4<f32>(r / f32(nk), 0.0, 0.0, 0.0));
    textureStore(AUXP, c, vec4<f32>(p, 0.0, 0.0, 0.0));
    return;
  }
  let p = textureLoad(PRS, c).x;
  let s = ap_both(slot, l);
  if (leaf) {
    if (ap_held(slot, l)) {
      textureStore(TAU, c, vec4<f32>(0.0));
      return;
    }
    // h^2 (plain - composite): the alpha terms cancel.
    textureStore(TAU, c, vec4<f32>(((s.w * p - s.z) - (s.y * p - s.x)) / (h * h), 0.0, 0.0, 0.0));
    return;
  }
  textureStore(RHS, c, vec4<f32>(textureLoad(AUXR, c).x + ((s.w + ah2) * p - s.z) / (h * h), 0.0, 0.0, 0.0));
  textureStore(TAU, c, vec4<f32>(0.0));
}
