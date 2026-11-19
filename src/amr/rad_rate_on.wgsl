// With radiation on the block grid, for the timestep (amr/assemble.wgsl): radiation/rate_on.wgsl's
// term, from the radiation as amr/rad_assemble.wgsl built it as the step started. Requires the
// equation of state, comp and g_dx (the block's cell size).
#include <opacity>

@group(0) @binding(28) var RADR: texture_3d<f32>;

const RAD_RATE = true;
fn rad_cs2(c: vec3<i32>, rho: f32, eint: f32) -> f32 {
  let E = max(textureLoad(RADR, c, 0).x, 0.0);
  let T = eos_temperature(rho, eint);
  let tau = kappa_rosseland(rho, T) * rho * g_dx;
  return (4.0 / 9.0) * min(1.0, tau) * E / rho;
}
