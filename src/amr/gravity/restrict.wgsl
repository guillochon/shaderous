// Residual r = f - Lap(phi) of a sublevel, averaged straight onto the next coarser one (the
// eight fine cells a coarse cell covers, four in 2D, all lie in the same block), where it
// becomes the source for the error; the error there starts from zero. GL describes the fine
// sublevel; its "other" fields describe the coarse one, whose cells this pass is dispatched over.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var RHSC: texture_storage_3d<r32float, write>;
@group(0) @binding(5) var PHIC: texture_storage_3d<r32float, write>;

#include "amr/gravity/stencil.wgsl"

// 2D: the four fine cells under a coarse one, weighed by their ring volumes in an axisymmetric
// run (cyl_child_weight; 1 otherwise).
fn restrict2d(gid: vec3<u32>) {
  let inv = g_inv2d(gid, GL.obsize);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let lc = inv.l;
  let o = g_origin(slot);
  let h = g_h(slot);
  let pi = blk_coord(slot).x * GL.obsize + lc.x;
  var sum = 0.0;
  for (var i = 0; i < 4; i++) {
    let l = 2 * lc + vec3<i32>(i & 1, (i >> 1u) & 1, 0);
    let s = g_sum(slot, o, l);
    let r = textureLoad(RHS, o + l, 0).x - (s.x - s.y * textureLoad(PHI, o + l).x) / (h * h);
    sum += cyl_child_weight(pi, 2 * pi + (i & 1)) * r;
  }
  let c = g_origin_other(slot) + lc;
  textureStore(RHSC, c, vec4<f32>(0.25 * sum, 0.0, 0.0, 0.0));
  textureStore(PHIC, c, vec4<f32>(0.0));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (is2d()) {
    restrict2d(gid);
    return;
  }
  let span = u32(max(GL.obsize, 8));
  let zs = g_zspan(GL.obsize);
  let b = gid.x / span + (gid.z / zs) * SLAB_BLOCKS;
  let lc = vec3<i32>(i32(gid.x % span), i32(gid.y), i32(gid.z % zs));
  if (b >= LIST[0] || any(lc >= vec3<i32>(GL.obsize))) { return; }
  let slot = i32(LIST[1u + b]);
  let o = g_origin(slot);
  let h = g_h(slot);
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let l = 2 * lc + vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1);
    let s = g_sum(slot, o, l);
    sum += textureLoad(RHS, o + l, 0).x - (s.x - s.y * textureLoad(PHI, o + l).x) / (h * h);
  }
  let c = g_origin_other(slot) + lc;
  textureStore(RHSC, c, vec4<f32>(0.125 * sum, 0.0, 0.0, 0.0));
  textureStore(PHIC, c, vec4<f32>(0.0));
}
