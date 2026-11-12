// Root of A T^4 + B T = C for A >= 0, B > 0, C >= 0: the temperature of gas plus radiation.
// The start is the smaller of the two one-term solutions, which is an upper bound on the root;
// Newton then descends monotonically. Mirrored by quartic() in src/eos/eos.ts.
fn eos_quartic(A: f32, B: f32, C: f32) -> f32 {
  var T = C / B;
  if (A > 0.0) { T = min(T, sqrt(sqrt(C / A))); }
  for (var i = 0; i < 8; i++) {
    let T3 = T * T * T;
    T = (3.0 * A * T3 * T + C) / (4.0 * A * T3 + B);
  }
  return T;
}

// Adiabatic index Gamma_1 of ideal gas plus radiation, from the gas share beta of the pressure
// (gamma is the gas's own). Between gamma (no radiation) and 4/3 (radiation only).
fn eos_gamma1_gas_rad(beta: f32, gamma: f32) -> f32 {
  let g1 = gamma - 1.0;
  let k = 4.0 - 3.0 * beta;
  return beta + k * k * g1 / (beta + 12.0 * g1 * (1.0 - beta));
}
