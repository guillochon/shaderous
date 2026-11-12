// Ideal gas plus radiation pressure in local thermodynamic equilibrium:
//   p     = R rho T + a T^4 / 3
//   e_int = R rho T / (gamma - 1) + a T^4
// P.eos = (gamma, unused, R, a). Temperature comes from a fixed-count Newton iteration.
const EOS_HAS_ENERGY = true;

#include "eos/quartic.wgsl"

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let T = eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), eint);
  let T2 = T * T;
  return P.eos.z * rho * T + P.eos.w * T2 * T2 / 3.0;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, p);
  let beta = clamp(P.eos.z * rho * T / p, 0.0, 1.0);   // gas pressure fraction
  return sqrt(eos_gamma1_gas_rad(beta, P.eos.x) * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, p);
  let T2 = T * T;
  return P.eos.z * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), eint);
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  let T2 = T * T;
  return P.eos.z * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure floor.
fn eos_eint_floor(rho: f32) -> f32 {
  return eos_eint_from_p(rho, P.p_floor);
}
