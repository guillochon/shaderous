// Ideal gas plus radiation pressure in local thermodynamic equilibrium:
//   p     = R rho T + a T^4 / 3
//   e_int = R rho T / (gamma - 1) + a T^4
// P.eos = (gamma, unused, R, a). Temperature comes from a fixed-count Newton iteration. Where
// the composition varies (eos/comp_on), R is scaled by the cell's g_comp.x.
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, rebuild E from the advected e_int (hydro/dual_energy.wgsl).
const EOS_KEEP_TOTAL = false;

#include "eos/quartic.wgsl"

fn eos_R() -> f32 {
  if (COMP) { return P.eos.z * g_comp.x; }
  return P.eos.z;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let T = eos_quartic(P.eos.w, eos_R() * rho / (P.eos.x - 1.0), eint);
  let T2 = T * T;
  return eos_R() * rho * T + P.eos.w * T2 * T2 / 3.0;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, eos_R() * rho, p);
  let beta = clamp(eos_R() * rho * T / p, 0.0, 1.0);   // gas pressure fraction
  return sqrt(eos_gamma1_gas_rad(beta, P.eos.x) * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, eos_R() * rho, p);
  let T2 = T * T;
  return eos_R() * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return eos_quartic(P.eos.w, eos_R() * rho / (P.eos.x - 1.0), eint);
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  let T2 = T * T;
  return eos_R() * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
