// Barotropic: p = K rho^gamma. gamma = 1 is isothermal with K = c_s^2.
// The energy equation is ignored; e_int is slaved to the density.
const EOS_HAS_ENERGY = false;
const EOS_KEEP_TOTAL = false;

// The polytropic index, as the other modules name their adiabatic index. The composition does
// not reach a barotropic gas.
fn eos_gamma() -> f32 {
  return P.eos.x;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  if (P.eos.x == 1.0) { return P.eos.y * rho; }
  return P.eos.y * pow(rho, P.eos.x);
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  return sqrt(P.eos.x * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  // Diagnostic only. Isothermal gas is given the monatomic value.
  if (P.eos.x < 1.001) { return 1.5 * p; }
  return p / (P.eos.x - 1.0);
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return eos_pressure(rho, eint) / rho;
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  // The temperature is not free here: it follows the density.
  return eos_eint_from_p(rho, eos_pressure(rho, 0.0));
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
