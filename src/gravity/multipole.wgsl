// Multipole reduction source: the mass and its moments about the box center, to the order of
// the boundary potential (gravity/moments.wgsl, gravity/multipole_eval.wgsl).
#include "hydro/common.wgsl"
#include "gravity/moments.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let m = textureLoad(U1, c, 0).x * P.dx * P.dx * P.dx;
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  return mp_moments(m, x);
}
