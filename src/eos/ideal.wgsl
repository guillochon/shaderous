// Ideal gas, p = (gamma - 1) e_int. Temperature in units where p = rho T.
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, rebuild E from the advected e_int (hydro/dual_energy.wgsl).
const EOS_KEEP_TOTAL = false;

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  return (P.eos.x - 1.0) * eint;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  return sqrt(P.eos.x * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  return p / (P.eos.x - 1.0);
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return (P.eos.x - 1.0) * eint / rho;
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  return rho * T / (P.eos.x - 1.0);
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
