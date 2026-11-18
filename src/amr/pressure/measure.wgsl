// The residual of the composite equation over the leaves, for the tally (pressure/tally.wgsl):
// the largest |b - A p| and the largest |b|, as float bits.
#include "amr/pressure/common.wgsl"

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(9) var<storage, read_write> RES: array<atomic<u32>, 2>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
#include "amr/pressure/stencil.wgsl"

var<workgroup> wg: array<atomic<u32>, 2>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let inv = inv_interior(gid);
  if (inv.ok && clock.dt > 0.0) {
    let c = atl(blk_origin(inv.slot), inv.l);
    let h = blk_dx(inv.slot);
    let s = ap_both(inv.slot, inv.l);
    let b = textureLoad(RHS, c, 0).x;
    let r = b - ((s.y + ap_alpha() * h * h) * textureLoad(PRS, c).x - s.x) / (h * h);
    atomicMax(&wg[0], bitcast<u32>(abs(r)));
    atomicMax(&wg[1], bitcast<u32>(abs(b)));
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicMax(&RES[0], atomicLoad(&wg[0]));
    atomicMax(&RES[1], atomicLoad(&wg[1]));
  }
}
