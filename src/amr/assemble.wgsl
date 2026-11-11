// Primitives for every block, ghosts included, in one pass from the conserved state.
//
// Nothing is filled in place: each texel of W, interior or ghost, is computed from wherever
// its value lives in U. That is the cell itself, a neighbouring block's cell, a boundary
// image, the restriction of a finer block, or an interpolation from a coarser one. (WebGPU
// cannot read and write one rgba32float texture in the same pass, so ghost zones cannot be
// copied within a texture.)
//
// The pass also does the first stage of the timestep reduction: each workgroup folds the
// signal rate and acceleration (per cell size) of the leaf cells it covers into shared
// atomics, and writes one record for the later stages of gpu/reduce.wgsl. Both are
// non-negative, so their bit patterns order as integers the way the floats do. Records of
// workgroups beyond the current block list are stale; the driver clears them when the list
// changes.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/diff_rate.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var W2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var G: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> RATES: array<f32>;   // (max rate / dx, max |g| / dx) per workgroup

var<workgroup> wg_max: array<atomic<u32>, 2>;

#include "amr/cons.wgsl"
#include <bins>
#include <mid>

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
  if (mid_block(cs)) {
    // The coarser block is half way through its step: interpolate its half-step prediction.
    return prolong(
      mid_cons(cs, lc),
      MaybeCons(mid_cons(cs, lc - ex), true), MaybeCons(mid_cons(cs, lc + ex), true),
      MaybeCons(mid_cons(cs, lc - ey), true), MaybeCons(mid_cons(cs, lc + ey), true),
      MaybeCons(mid_cons(cs, lc - ez), true), MaybeCons(mid_cons(cs, lc + ez), true),
      gf & vec3<i32>(1),
    );
  }
  return prolong(
    u_at(cs, lc),
    u_near(cs, lc - ex), u_near(cs, lc + ex),
    u_near(cs, lc - ey), u_near(cs, lc + ey),
    u_near(cs, lc - ez), u_near(cs, lc + ez),
    gf & vec3<i32>(1),
  );
}

@compute @workgroup_size(8, 8, 4)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lix: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let inv = inv_padded(gid, NGH);
  if (inv.ok) {
    g_dx = blk_dx(inv.slot);
    let s = ghost_src(inv.slot, inv.l);
    if (s.kind != 2) {
      var u: Cons;
      if (s.kind == 0) {
        if (mid_block(s.slot)) { u = mid_cons(s.slot, s.l); } else { u = u_at(s.slot, s.l); }
      } else {
        u = from_coarse(inv.slot, inv.l, s.slot);
      }
      var w = prim_from_cons(u.a, u.b);
      // The primitives have no spare component: a solid cell is marked by a negative density.
      let solid = u.b.w > 0.5;
      w.w1 = vec4<f32>(select(w.w1.x, -w.w1.x, solid), src_velocity(w.w1.yzw, s));

      let c = atl(blk_origin(inv.slot), inv.l);
      textureStore(W1, c, w.w1);
      textureStore(W2, c, w.w2);

      let depth = select(BS, 1, is2d());
      if (all(inv.l >= vec3<i32>(0)) && all(inv.l < vec3<i32>(BS, BS, depth)) && blk_leaf(inv.slot) && !solid) {
        // Unsplit scheme: the stable step is set by the sum of the per-axis signal speeds.
        let rate = abs(w.w1.y) + abs(w.w1.z) + abs(w.w1.w) + P.geom.y * w.w2.y + diff_speed(abs(w.w1.x), w.w2.w);
        var g = P.gext.xyz;
        if (P.grav_on != 0u) { g += textureLoad(G, c, 0).xyz; }
        atomicMax(&wg_max[0], bitcast<u32>(rate / g_dx));
        atomicMax(&wg_max[1], bitcast<u32>(length(g) / g_dx));
      }
    }
  }
  workgroupBarrier();
  if (lix == 0u) {
    // Record grid: 3 workgroups per slot along x, by 3, by 5 (1 in 2D).
    let a = u32(P.amr.x);
    let nx = 3u * a * a * select(a, 1u, is2d());
    let base = ((wid.z * 3u + wid.y) * nx + wid.x) * 2u;
    RATES[base] = bitcast<f32>(atomicLoad(&wg_max[0]));
    RATES[base + 1u] = bitcast<f32>(atomicLoad(&wg_max[1]));
  }
}
