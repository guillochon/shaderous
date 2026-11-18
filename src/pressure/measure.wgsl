// The residual of the finest level, measured for the gate (pressure/gate.wgsl): the largest
// |b - A p| and the largest |b| over the grid, as float bits (non-negative floats order as their
// bits do). A workgroup folds its cells into shared atomics and one invocation hands the result
// on, so the buffer sees one update per workgroup.
#include "pressure/common.wgsl"

@group(0) @binding(1) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(2) var RHS: texture_3d<f32>;
@group(0) @binding(3) var COEF: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> RES: array<atomic<u32>, 2>;

#include "pressure/stencil.wgsl"

var<workgroup> wg: array<atomic<u32>, 2>;

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (all(gid < L.dims)) {
    let c = vec3<i32>(gid);
    let s = op_sum(c);
    if (s.y > 0.0) {
      let b = textureLoad(RHS, c, 0).x;
      let r = b - (s.y * textureLoad(PRS, c).x - s.x) / (L.h * L.h);
      atomicMax(&wg[0], bitcast<u32>(abs(r)));
      atomicMax(&wg[1], bitcast<u32>(abs(b)));
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicMax(&RES[0], atomicLoad(&wg[0]));
    atomicMax(&RES[1], atomicLoad(&wg[1]));
  }
}
