// MHD on the block grid: assembly of one stage's state (docs/mhd-plan.md, section 4b). For
// every leaf, its cells, face ghosts and edge ghosts get their primitives (W1, W2), cell-centered
// field (W3) and faces (FA), from wherever they live: the block itself, a neighbor at the same
// level, the restriction of a finer one, or the prolongation of a coarser one, all found
// through the leaf index (mhd/amr_common.wgsl). The passes that follow read only the block's
// own padded storage.
//
// The region: two layers of ghosts beyond each face, and the edge ghosts where one coordinate
// is up to two cells out and a second one cell out, which the edge fields at the block's edges
// and the fluxes of the ghost rows beside them need. Corner ghosts are never needed.
//
// ASM_STAGE says which (mhd/amr_common.wgsl): the start of each block's step (1) or its
// midpoint (2), with every neighbor read at that time (blended between its stores with time bins).
// With RATES the pass also does the first stage of the timestep reduction, as amr/assemble.wgsl
// does: the signal rate per cell size over each workgroup's leaf cells, with the fastest
// magnetosonic speed.
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
@group(0) @binding(5) var W3: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var FA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var BF: texture_3d<f32>;
@group(0) @binding(16) var HF: texture_3d<f32>;   // the half-step stores (mhd/amr_common.wgsl's blend)
@group(0) @binding(17) var H1: texture_3d<f32>;
@group(0) @binding(18) var H2: texture_3d<f32>;
@group(0) @binding(28) var IDX: texture_3d<u32>;
@group(0) @binding(12) var<storage, read_write> RATES: array<f32>;

override RATES_ON: bool = true;

var<workgroup> wg_max: array<atomic<u32>, 2>;

#include "amr/cons.wgsl"
#include <bins>
#include "mhd/amr_common.wgsl"

@compute @workgroup_size(FLAT_WG)
fn main(
  @builtin(local_invocation_index) lix: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  if (lix == 0u) {
    atomicStore(&wg_max[0], 0u);
    atomicStore(&wg_max[1], 0u);
  }
  workgroupBarrier();
  let inv = inv_flat(wid, lix, NGH);
  if (inv.ok) {
    let slot = inv.slot;
    let l = inv.l;
    let depth = select(BS, 1, is2d());
    let top = vec3<i32>(BS, BS, depth);
    let below = l < vec3<i32>(0);
    let above = l >= top;
    let out = vec3<i32>(below) + vec3<i32>(above);
    // How far out each coordinate lies.
    let far = max(-l, l - top + 1) * out;
    let count = out.x + out.y + out.z;
    let deep = i32(far.x > 1) + i32(far.y > 1) + i32(far.z > 1);
    if (count <= 1 || (count == 2 && deep <= 1)) {
      g_dx = blk_dx(slot);
      g_recv_bin = blk_bin(slot);
      let level = u32(bt(slot, 0u));
      let g = blk_coord(slot) * BS + l;
      var u: Cons;
      if (count == 0) { u = u_leaf_m(slot, l); } else { u = cons_at(level, g); }
      let w = prim_from_cons(u.a, u.b);
      let cf = faces_at(level, g);
      let B = cf.B;
      let f = cf.lo;
      let c = atl(blk_origin(slot), l);
      textureStore(W1, c, w.w1);
      textureStore(W2, c, w.w2);
      // The fourth channel carries the shock flag the half-step state holds in U2.w (stage 2).
      textureStore(W3, c, vec4<f32>(B, u.b.w));
      textureStore(FA, c, vec4<f32>(f, 0.0));
      if (RATES_ON && count == 0) {
        let cf = sqrt(w.w2.y * w.w2.y + dot(B, B) / w.w1.x);
        let rate = abs(w.w1.y) + abs(w.w1.z) + abs(w.w1.w) + P.geom.y * cf;
        atomicMax(&wg_max[0], bitcast<u32>(rate / g_dx));
      }
    }
  }
  workgroupBarrier();
  if (RATES_ON && lix == 0u) {
    let base = (flat_row(wid) * num_wg.x + wid.x) * 2u;
    RATES[base] = bitcast<f32>(atomicLoad(&wg_max[0]));
    RATES[base + 1u] = 0.0;
  }
}
