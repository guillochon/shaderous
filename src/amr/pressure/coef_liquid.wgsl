// The block solver's coefficients for the liquid (amr/pressure/stencil.wgsl): the operator is
// alpha p - lap p, with one alpha for every cell. An open face holds p = 0.
const AP_OPEN = 2.0;

fn ap_beta(slot: i32, l: vec3<i32>, a: i32, up: bool) -> f32 {
  return 1.0;
}
// Time-centered (amr/pressure/theta.wgsl): alpha / theta^2, one theta for the mesh.
#include "amr/pressure/theta.wgsl"
fn ap_alpha_at(slot: i32, l: vec3<i32>) -> f32 {
  let th = ap_theta_liquid();
  return ap_alpha() / (th * th);
}
