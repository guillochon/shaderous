// Electron scattering (opacity/opacity.ts): P.kap = (a small true absorption, the scattering
// opacity 0.2 (1 + X) cm^2/g in code units, 0, 0). Scattering drags on the flux; only the
// absorption exchanges heat with the gas.
fn kappa_planck(rho: f32, T: f32) -> f32 {
  return P.kap.x;
}
fn kappa_rosseland(rho: f32, T: f32) -> f32 {
  return P.kap.x + P.kap.y;
}
