// Ideal gas plus radiation pressure in local thermodynamic equilibrium:
//   p     = R rho T + a T^4 / 3
//   e_int = R rho T / (gamma - 1) + a T^4
// P.eos = (gamma, unused, R, a). Temperature comes from a fixed-count Newton iteration.
const EOS_HAS_ENERGY = true;

// Root of A T^4 + B T = C for A >= 0, B > 0, C >= 0. The start is the smaller of the two
// one-term solutions, which is an upper bound on the root; Newton then descends monotonically.
fn eos_quartic(A: f32, B: f32, C: f32) -> f32 {
  var T = C / B;
  if (A > 0.0) { T = min(T, sqrt(sqrt(C / A))); }
  for (var i = 0; i < 8; i++) {
    let T3 = T * T * T;
    T = (3.0 * A * T3 * T + C) / (4.0 * A * T3 + B);
  }
  return T;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let T = eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), eint);
  let T2 = T * T;
  return P.eos.z * rho * T + P.eos.w * T2 * T2 / 3.0;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, p);
  let beta = clamp(P.eos.z * rho * T / p, 0.0, 1.0);   // gas pressure fraction
  let g1 = P.eos.x - 1.0;
  let k = 4.0 - 3.0 * beta;
  let gamma1 = beta + k * k * g1 / (beta + 12.0 * g1 * (1.0 - beta));
  return sqrt(gamma1 * p / rho);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let T = eos_quartic(P.eos.w / 3.0, P.eos.z * rho, p);
  let T2 = T * T;
  return P.eos.z * rho * T / (P.eos.x - 1.0) + P.eos.w * T2 * T2;
}
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return eos_quartic(P.eos.w, P.eos.z * rho / (P.eos.x - 1.0), eint);
}
