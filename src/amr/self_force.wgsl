// Self-gravity's net force on the gas (amr/amr_sim.ts encodeSelfForce), a reduction over the
// leaf cells: the mass m and m g, g the field atlas's self-gravity as the field pass left it
// (with the uniform acceleration already taken off it, amr/gravity/fix_on.wgsl).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  var r = red_identity();
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return r; }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let dx = blk_dx(slot);
  let a = atl(blk_origin(slot), l);
  // (A solid cell carries a negative density: it is not gas.)
  let m = max(textureLoad(U1, a, 0).x, 0.0) * dx * dx * dx;
  let g = textureLoad(G, a, 0).xyz;
  r[0] = m;
  r[1] = m * g.x;
  r[2] = m * g.y;
  r[3] = m * g.z;
  return r;
}
