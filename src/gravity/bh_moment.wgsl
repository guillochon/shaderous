// Reduction source for the point masses: gas mass, each mass's total pull on the gas, and what
// each sink takes this step (gravity/bh_cell.wgsl).
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Masses;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> clock: Clock;

#include "gravity/bh_cell.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  var u1 = textureLoad(U1, c, 0);
  // Solid and fixed cells are not gas the masses can pull on or take.
  if (abs(textureLoad(U2, c, 0).w) > 0.5) { u1 = vec4<f32>(0.0); }
  // (An axisymmetric run's cell is a ring of volume 2 pi R dx^2.)
  let vol = select(P.geom.x, 6.2831853 * (f32(c.x) + 0.5) * P.dx * P.dx * P.dx, cylindrical());
  return bh_cell(u1, vol, cell_pos(c), clock.dt);
}
