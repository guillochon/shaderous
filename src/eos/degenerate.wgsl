// Ideal gas plus radiation plus the pressure of fully degenerate electrons (white dwarfs):
//   p     = P_deg(rho) + R rho T + a T^4 / 3
//   e_int = E_deg(rho) + R rho T / (gamma - 1) + a T^4
// The degenerate part is the zero-temperature electron gas at any degree of relativity
// (Chandrasekhar 1939): with x = p_F / (m_e c) = (rho / rho_d)^(1/3),
//   P_deg = A f(x),  f(x) = x (2 x^2 - 3) sqrt(1 + x^2) + 3 asinh(x)
//   E_deg = A g(x),  g(x) = 8 x^3 (sqrt(1 + x^2) - 1) - f(x)
// where rho_d = m_u (8 pi / 3) (m_e c / h)^3 / Ye and A = pi m_e^4 c^5 / (3 h^3), in code units.
// The thermal part (ions and electrons, as an ideal gas of the composition's mu) is added on
// top. Each piece is thermodynamically consistent on its own, so their sum is: along an
// adiabat the thermal gas follows its own adiabat and P_deg depends on the density alone, so
// c^2 = dP_deg/drho + Gamma_1 p_th / rho. Counting the electrons in both parts overstates the
// pressure only where kT is comparable to the Fermi energy; well away from that, one of the
// two is negligible.
//
// P.eos = (gamma, unused, R, a), P.eos2 = (rho_d, A, 0, 0). In single precision the thermal
// part is resolved only where it is more than about 1e-6 of the degenerate pressure.
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, keep E as the conservative update made it and take the
// pressure from the advected e_int (hydro/dual_energy.wgsl). The pressure is always defined,
// P_deg at the least: E - KE below zero does not leave the gas without one, as it would an
// ideal gas. Rebuilding E there made energy: the one cell ahead of each strong shock in a white
// dwarf collision has its E - KE dip below zero (a velocity overshoot in cold gas falling into
// a deep well), and filling it back in each step added 2.5% of the total at the bounce.
const EOS_KEEP_TOTAL = true;

#include "eos/quartic.wgsl"

// (f(x), g(x)). Below x = 1/2 both are differences of nearly equal terms, so they are summed as
// series instead: f = 8 sum_k c_k x^(5+2k) / (5+2k) with c_k the coefficients of (1+x^2)^(-1/2),
// and g = sum_k (8 b_(k+1) - 8 c_k / (5+2k)) x^(5+2k) with b_k those of (1+x^2)^(1/2).
fn deg_fg(x: f32) -> vec2<f32> {
  if (x < 0.5) {
    let x2 = x * x;
    var xp = x2 * x2 * x;
    var c = 1.0;
    var b = 0.5;
    var f = 0.0;
    var g = 0.0;
    for (var k = 0; k < 12; k++) {
      let n = f32(5 + 2 * k);
      f += 8.0 * c * xp / n;
      g += (8.0 * b - 8.0 * c / n) * xp;
      let kf = f32(k);
      c *= -(2.0 * kf + 1.0) / (2.0 * kf + 2.0);
      b *= (0.5 - (kf + 1.0)) / (kf + 2.0);
      xp *= x2;
    }
    return vec2<f32>(f, g);
  }
  let s = sqrt(1.0 + x * x);
  let f = x * (2.0 * x * x - 3.0) * s + 3.0 * asinh(x);
  return vec2<f32>(f, 8.0 * x * x * x * (s - 1.0) - f);
}

fn deg_x(rho: f32) -> f32 {
  return pow(rho / P.eos2.x, 1.0 / 3.0);
}

// (P_deg, E_deg) at density rho.
fn deg_pe(rho: f32) -> vec2<f32> {
  return P.eos2.y * deg_fg(deg_x(rho));
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let d = deg_pe(rho);
  let T = eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), max(eint - d.y, 0.0));
  let T2 = T * T;
  return d.x + P.eos.z * rho * T + P.eos.w * T2 * T2 / 3.0;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let x = deg_x(rho);
  let pd = P.eos2.y * deg_fg(x).x;
  let x2 = x * x;
  // dP_deg / drho = A f'(x) dx/drho, with f'(x) = 8 x^4 / sqrt(1 + x^2) and dx/drho = x / (3 rho).
  let cd2 = P.eos2.y * 8.0 * x2 * x2 * x / (3.0 * rho * sqrt(1.0 + x2));
  let pth = max(p - pd, 0.0);
  var ct2 = 0.0;
  if (pth > 0.0) {
    let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, pth);
    let beta = clamp(P.eos.z * rho * T / pth, 0.0, 1.0);
    ct2 = eos_gamma1_gas_rad(beta, P.eos.x) * pth / rho;
  }
  return sqrt(cd2 + ct2);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let d = deg_pe(rho);
  let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, max(p - d.x, 0.0));
  let T2 = T * T;
  return d.y + P.eos.z * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), max(eint - deg_pe(rho).y, 0.0));
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  let T2 = T * T;
  return deg_pe(rho).y + P.eos.z * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl). Not the zero-temperature
// energy E_deg: cold gas sits right on it, so truncation errors take e_int just below it as
// often as just above, and flooring them back up would add energy every step (13% of the total
// across a white dwarf collision). Below E_deg the pressure is simply P_deg and T is 0, so the
// total energy stays as the conservative update made it.
fn eos_eint_floor(rho: f32) -> f32 {
  return 0.0;
}
