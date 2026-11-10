// Uniform grid: primitive loads through the boundary mapping. Requires P, the limiter, and
// bindings W1, W2. (Block grids define load_q themselves: their ghost cells are stored.)
#include "hydro/q.wgsl"

fn load_q(c: vec3<i32>) -> Q {
  let m = bc_map(c);
  let w1 = textureLoad(W1, m.c, 0);
  let w2 = textureLoad(W2, m.c, 0);
  return Q(vec4<f32>(w1.x, bc_velocity(w1.yzw, m)), vec4<f32>(w2.x, w2.z, w2.y, 0.0));
}
