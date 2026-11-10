// Multipole reduction source on the block grid: mass and its first and second moments about
// the box centre, over leaf cells.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { var r0: Red; r0[0] = 0.0; r0[1] = 0.0; r0[2] = 0.0; r0[3] = 0.0; r0[4] = 0.0; r0[5] = 0.0; r0[6] = 0.0; r0[7] = 0.0; r0[8] = 0.0; r0[9] = 0.0; return r0; }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let dx = blk_dx(slot);
  let m = textureLoad(U1, atl(blk_origin(slot), l), 0).x * dx * dx * dx;
  let x = blk_pos(slot, l);
  var r1: Red;
  r1[0] = m;
  r1[1] = m * x.x;
  r1[2] = m * x.y;
  r1[3] = m * x.z;
  r1[4] = m * x.x * x.x;
  r1[5] = m * x.y * x.y;
  r1[6] = m * x.z * x.z;
  r1[7] = m * x.x * x.y;
  r1[8] = m * x.x * x.z;
  r1[9] = m * x.y * x.z;
  return r1;
}
