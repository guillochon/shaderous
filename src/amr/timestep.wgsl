// Timestep reduction source on the block grid: (max signal rate / dx, max |g| / dx) over leaf
// cells. The reduction runs over every slot, laid end to end along x; slots that are empty or
// hold a parent contribute the identity.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var W1: texture_3d<f32>;
@group(0) @binding(2) var W2: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return Red(-3.0e38, -3.0e38); }
  let a = atl(blk_origin(slot), vec3<i32>(c.x % BS, c.y, c.z));
  let w1 = textureLoad(W1, a, 0);
  let cs = textureLoad(W2, a, 0).y;
  let dx = blk_dx(slot);
  // Unsplit scheme: the stable step is set by the sum of the per-axis signal speeds.
  let rate = abs(w1.y) + abs(w1.z) + abs(w1.w) + P.geom.y * cs;
  return Red(rate / dx, length(P.gext.xyz) / dx);
}
