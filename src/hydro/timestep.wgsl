// Timestep reduction source: (max signal rate, max |g|).
#include "hydro/common.wgsl"

@group(0) @binding(1) var W1: texture_3d<f32>;
@group(0) @binding(2) var W2: texture_3d<f32>;
@group(0) @binding(3) var G: texture_3d<f32>;

#include "gravity/gfield.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  let w1 = textureLoad(W1, c, 0);
  let cs = textureLoad(W2, c, 0).y;
  // Unsplit scheme: the stable step is set by the sum of the per-axis signal speeds.
  let rate = abs(w1.y) + abs(w1.z) + abs(w1.w) + 3.0 * cs;
  return Red(rate, length(g_load(c)));
}
