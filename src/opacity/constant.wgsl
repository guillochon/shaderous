// Constant opacities (opacity/opacity.ts): P.kap = (absorption, scattering, 0, 0), per unit mass.
fn kappa_planck(rho: f32, T: f32) -> f32 {
  return P.kap.x;
}
fn kappa_rosseland(rho: f32, T: f32) -> f32 {
  return P.kap.x + P.kap.y;
}
