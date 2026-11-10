// Reduction source for the perturber: gas mass and the perturber's total pull on it, (m, m a).
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Perturber;

fn red_cell(c: vec3<i32>) -> Red {
  let m = textureLoad(U1, c, 0).x * P.geom.x;
  let a = bh_accel(cell_pos(c), BH.pos.xyz);
  var r0: Red;
  r0[0] = m;
  r0[1] = m * a.x;
  r0[2] = m * a.y;
  r0[3] = m * a.z;
  return r0;
}
