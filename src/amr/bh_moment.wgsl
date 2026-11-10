// Reduction source for the perturber on the block grid: gas mass and the perturber's total
// pull on it, (m, m a), over leaf cells.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Perturber;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { var r0: Red; r0[0] = 0.0; r0[1] = 0.0; r0[2] = 0.0; r0[3] = 0.0; return r0; }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let dx = blk_dx(slot);
  let m = textureLoad(U1, atl(blk_origin(slot), l), 0).x * dx * dx * dx;
  let a = bh_accel(blk_pos(slot, l), BH.pos.xyz);
  var r1: Red;
  r1[0] = m;
  r1[1] = m * a.x;
  r1[2] = m * a.y;
  r1[3] = m * a.z;
  return r1;
}
