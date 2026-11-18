// Multipole reduction source: the mass and its moments about the box center, to the order of
// the boundary potential (gravity/moments.wgsl, gravity/multipole_eval.wgsl).
#include "hydro/common.wgsl"
#include "gravity/moments.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let x = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  // (An axisymmetric run: the cell is a ring, of volume 2 pi R dx^2.)
  if (cylindrical()) { return mp_ring_moments(textureLoad(U1, c, 0).x * 6.2831853 * x.x * P.dx * P.dx, x.x, x.y); }
  let m = textureLoad(U1, c, 0).x * P.dx * P.dx * P.dx;
  return mp_moments(m, x);
}
