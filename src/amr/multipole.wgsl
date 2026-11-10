// Multipole reduction source on the block grid: mass and its first and second moments about
// the box centre, over leaf cells.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return Red(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let dx = blk_dx(slot);
  let m = textureLoad(U1, atl(blk_origin(slot), l), 0).x * dx * dx * dx;
  let x = blk_pos(slot, l);
  return Red(
    m, m * x.x, m * x.y, m * x.z,
    m * x.x * x.x, m * x.y * x.y, m * x.z * x.z,
    m * x.x * x.y, m * x.x * x.z, m * x.y * x.z,
  );
}
