// Kramers' free-free opacity plus electron scattering (opacity/opacity.ts):
// P.kap = (K0, kappa_es, least temperature, 0); kappa_ff = K0 rho T^-3.5 is the absorption.
// The power is taken as exp(-3.5 ln T): T^-3.5 of a hot cell is far below float32's range of
// T^4 and the product would underflow before the density multiplies it.
fn kramers_ff(rho: f32, T: f32) -> f32 {
  let t = max(max(T, P.kap.z), 1.0e-30);
  return P.kap.x * rho * exp(-3.5 * log(t));
}
fn kappa_planck(rho: f32, T: f32) -> f32 {
  return kramers_ff(rho, T);
}
fn kappa_rosseland(rho: f32, T: f32) -> f32 {
  return kramers_ff(rho, T) + P.kap.y;
}
