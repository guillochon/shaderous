// How far a V-cycle left the potential from solving Poisson's equation: the largest residual
// |f - Lap(phi)| over the full-resolution cells of the blocks in the list, and the largest
// |f| to measure it against, folded into RES (as the bit patterns of non-negative floats,
// which order as the floats do). amr/gravity/gate.wgsl reads them to decide whether another
// cycle runs. GL describes sublevel 0.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> RES: array<atomic<u32>, 4>;

#include "amr/gravity/stencil.wgsl"

var<workgroup> wg_max: array<atomic<u32>, 2>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  if (lix == 0u) {
    atomicStore(&wg_max[0], 0u);
    atomicStore(&wg_max[1], 0u);
  }
  workgroupBarrier();
  let inv = g_inv(gid);
  if (inv.ok) {
    let o = g_origin(inv.slot);
    let h = g_h(inv.slot);
    let s = g_sum(inv.slot, o, inv.l);
    let f = textureLoad(RHS, o + inv.l, 0).x;
    let r = f - (s.x - s.y * textureLoad(PHI, o + inv.l).x) / (h * h);
    atomicMax(&wg_max[0], bitcast<u32>(abs(r)));
    atomicMax(&wg_max[1], bitcast<u32>(abs(f)));
  }
  workgroupBarrier();
  if (lix == 0u) {
    atomicMax(&RES[0], atomicLoad(&wg_max[0]));
    atomicMax(&RES[1], atomicLoad(&wg_max[1]));
  }
}
