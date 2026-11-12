// Ideal gas, p = (gamma - 1) e_int. Temperature in units where p = rho T for the reference
// composition. Where the composition varies (eos/comp_on), p = x rho T and e_int = w rho T with
// x = g_comp.x the particles per unit mass relative to the reference's and w = g_comp.w the heat
// capacity likewise (species/mixture.ts): gamma - 1 = x / w.
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, rebuild E from the advected e_int (hydro/dual_energy.wgsl).
const EOS_KEEP_TOTAL = false;

// The gas's adiabatic index, and gamma - 1. Where the composition varies (eos/comp_on) they are
// the cell's: 1 + x / w from its particles x and heat capacity w (species/mixture.ts).
fn eos_gamma() -> f32 {
  if (COMP) { return 1.0 + g_comp.x / g_comp.w; }
  return P.eos.x;
}
fn eos_gm1() -> f32 {
  if (COMP) { return g_comp.x / g_comp.w; }
  return P.eos.x - 1.0;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  return eos_gm1() * eint;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  return sqrt(eos_gamma() * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  return p / eos_gm1();
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  if (COMP) { return eint / (rho * g_comp.w); }
  return eos_gm1() * eint / rho;
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  if (COMP) { return rho * g_comp.w * T; }
  return rho * T / eos_gm1();
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
