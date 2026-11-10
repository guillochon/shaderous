// Refinement indicator: for every leaf block, the largest normalised jump in density or
// pressure between a cell's two neighbours along any axis, |q+ - q-| / (q+ + q-).
// One value per slot, accumulated with an atomic max on the float's bit pattern (which orders
// the same way as the float for non-negative values).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var W1: texture_3d<f32>;
@group(0) @binding(2) var W2: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> FLAGS: array<atomic<u32>>;

fn jump(c: vec3<i32>, e: vec3<i32>) -> f32 {
  let rp = textureLoad(W1, c + e, 0).x;
  let rm = textureLoad(W1, c - e, 0).x;
  let pp = textureLoad(W2, c + e, 0).x;
  let pm = textureLoad(W2, c - e, 0).x;
  return max(abs(rp - rm) / (rp + rm), abs(pp - pm) / (pp + pm));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  var ind = max(jump(c, vec3<i32>(1, 0, 0)), jump(c, vec3<i32>(0, 1, 0)));
  if (!is2d()) { ind = max(ind, jump(c, vec3<i32>(0, 0, 1))); }
  atomicMax(&FLAGS[u32(inv.slot)], bitcast<u32>(max(ind, 0.0)));
}
