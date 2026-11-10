// Reduction source for the perturber: gas mass and the perturber's total pull on it, (m, m a).
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Perturber;

fn red_cell(c: vec3<i32>) -> Red {
  let m = textureLoad(U1, c, 0).x * P.geom.x;
  let a = bh_accel(cell_pos(c), BH.pos.xyz);
  return Red(m, m * a.x, m * a.y, m * a.z);
}
