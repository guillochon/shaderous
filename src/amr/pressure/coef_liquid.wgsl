// The block solver's coefficients for the liquid (amr/pressure/stencil.wgsl): the operator is
// alpha p - lap p, with one alpha for every cell. An open face holds p = 0.
const AP_OPEN = 2.0;

fn ap_beta(slot: i32, l: vec3<i32>, a: i32, up: bool) -> f32 {
  return 1.0;
}
fn ap_alpha_at(slot: i32, l: vec3<i32>) -> f32 {
  return ap_alpha();
}
