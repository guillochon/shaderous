// With radiation (radiation/radiation.ts), for the timestep (hydro/rates.wgsl): what the
// radiation's push adds to the square of the sound speed in gas it is trapped in,
// (4/9) min(1, tau_c) E / rho. Its force reaches the gas as a source term in the sweeps
// (radiation/sweep.wgsl), which the Riemann solver does not know of; the step must.
// Requires the equation of state, comp and <opacity>.
#include <opacity>

// (41: implicit pressure's lagged pressure takes 30, pressure/predict_on.wgsl, and the two combine.)
@group(0) @binding(41) var RADR: texture_3d<f32>;   // radiation (E, F) as the step starts

const RAD_RATE = true;
fn rad_cs2(c: vec3<i32>, rho: f32, eint: f32) -> f32 {
  let E = max(textureLoad(RADR, c, 0).x, 0.0);
  let T = eos_temperature(rho, eint);
  let tau = kappa_rosseland(rho, T) * rho * P.dx;
  return (4.0 / 9.0) * min(1.0, tau) * E / rho;
}
