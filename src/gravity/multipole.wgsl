// Multipole reduction source: mass and its first and second moments about the box centre.
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let m = textureLoad(U1, c, 0).x * P.dx * P.dx * P.dx;
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  return Red(
    m, m * x.x, m * x.y, m * x.z,
    m * x.x * x.x, m * x.y * x.y, m * x.z * x.z,
    m * x.x * x.y, m * x.x * x.z, m * x.y * x.z,
  );
}
