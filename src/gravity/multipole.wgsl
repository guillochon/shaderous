// Multipole reduction source: mass and its first and second moments about the box centre.
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let m = textureLoad(U1, c, 0).x * P.dx * P.dx * P.dx;
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  var r0: Red;
  r0[0] = m;
  r0[1] = m * x.x;
  r0[2] = m * x.y;
  r0[3] = m * x.z;
  r0[4] = m * x.x * x.x;
  r0[5] = m * x.y * x.y;
  r0[6] = m * x.z * x.z;
  r0[7] = m * x.x * x.y;
  r0[8] = m * x.x * x.z;
  r0[9] = m * x.y * x.z;
  return r0;
}
