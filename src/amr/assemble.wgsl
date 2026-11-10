// Primitives for every block, ghosts included, in one pass from the conserved state.
//
// Nothing is filled in place: each texel of W, interior or ghost, is computed from wherever
// its value lives in U. That is the cell itself, a neighbouring block's cell, a boundary
// image, the restriction of a finer block, or an interpolation from a coarser one. (WebGPU
// cannot read and write one rgba32float texture in the same pass, so ghost zones cannot be
// copied within a texture.)
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var W2: texture_storage_3d<rgba32float, write>;

#include "amr/cons.wgsl"

// Fine ghost cell l of block `slot`, interpolated from the coarser leaf `cs` beyond the face.
fn from_coarse(slot: i32, l: vec3<i32>, cs: i32) -> Cons {
  let gf = blk_coord(slot) * BS + l;                 // fine global cell index
  let nc = i32(P.amr.y) * BS * (1 << u32(bt(cs, 0u))); // coarse cells across the domain
  var gc = gf >> vec3<u32>(1u);                       // floor(gf / 2)
  gc = ((gc % nc) + nc) % nc;                         // periodic wrap
  var lc = gc - blk_coord(cs) * BS;
  if (is2d()) { lc.z = 0; }
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);
  return prolong(
    u_at(cs, lc),
    u_near(cs, lc - ex), u_near(cs, lc + ex),
    u_near(cs, lc - ey), u_near(cs, lc + ey),
    u_near(cs, lc - ez), u_near(cs, lc + ez),
    gf & vec3<i32>(1),
  );
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, NGH);
  if (!inv.ok) { return; }
  g_dx = blk_dx(inv.slot);
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }

  var u: Cons;
  if (s.kind == 0) {
    u = u_at(s.slot, s.l);
  } else {
    u = from_coarse(inv.slot, inv.l, s.slot);
  }
  var w = prim_from_cons(u.a, u.b);
  w.w1 = vec4<f32>(w.w1.x, src_velocity(w.w1.yzw, s));

  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(W1, c, w.w1);
  textureStore(W2, c, w.w2);
}
