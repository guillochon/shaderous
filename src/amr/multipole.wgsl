// Multipole reduction source on the block grid: the mass and its moments about the box center,
// over leaf cells, to the order of the boundary potential (gravity/moments.wgsl).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "gravity/moments.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) {
    var none: Red;
    return none;
  }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let dx = blk_dx(slot);
  let m = textureLoad(U1, atl(blk_origin(slot), l), 0).x * dx * dx * dx;
  return mp_moments(m, blk_pos(slot, l));
}
