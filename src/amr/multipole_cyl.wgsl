// Multipole reduction source on the block grid of an axisymmetric (R, z) run: each leaf cell is
// a ring, of volume 2 pi R dx^2, whose moments are filled in by symmetry about the axis
// (gravity/moments.wgsl, mp_ring_moments; gravity/multipole.wgsl on the uniform grid). Kept
// apart from amr/multipole.wgsl so that 3D runs compile the same code as before.
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
  let l = vec3<i32>(c.x % BS, c.y, 0);
  let dx = blk_dx(slot);
  let x = blk_pos(slot, l);
  return mp_ring_moments(textureLoad(U1, atl(blk_origin(slot), l), 0).x * 6.2831853 * x.x * dx * dx, x.x, x.y);
}
