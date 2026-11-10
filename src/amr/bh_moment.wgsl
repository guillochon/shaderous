// Reduction source for the point masses on the block grid, over leaf cells: gas mass, each
// mass's total pull on the gas, and what each sink takes this step (gravity/bh_cell.wgsl).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Masses;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> clock: Clock;

#include "gravity/bh_cell.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let at = atl(blk_origin(slot), l);
  var u1 = textureLoad(U1, at, 0);
  if (!blk_leaf(slot) || abs(textureLoad(U2, at, 0).w) > 0.5) { u1 = vec4<f32>(0.0); }
  let dx = blk_dx(slot);
  return bh_cell(u1, dx * dx * dx, blk_pos(slot, l), clock.dt);
}
