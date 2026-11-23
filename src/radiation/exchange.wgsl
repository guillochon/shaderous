// The local implicit exchange of radiation and gas in one cell over a sweep of length dt
// (plan section 3.3), after the transport has made E* and F*. Float64 mirror: exchange() in
// radiation/radiation.ts. Requires the equation of state and radiation/common.wgsl.
//
// 1. The flux: F' = F* / (1 + theta_F), theta_F = r c kappa_R rho dt; the gas gains
//    (F* - F') / (r c^2) of momentum.
// 2. The temperature: backward Euler for the pair (rho e, E) leaves one equation,
//    rho e(T) + A a T^4 = rho e + A E*, A = theta / (r (1 + theta)), theta = r c kappa_P rho dt,
//    a sum of positive terms whatever the optical depth: nothing subtracts a T^4 from E. Then
//    E' = E* / (1 + theta) + a T^4 theta / (1 + theta).
// E_gas + E / r is kept exactly: the larger of the two takes the rounding of the root.

struct RadExchange { E: f32, F: vec3<f32>, dmom: vec3<f32>, eint: f32, T: f32 }

// The root T of e(rho, T) + K T^4 = C (K >= 0): Newton from the smaller of the two one-term
// solutions, an upper bound from which it descends monotonically (as solveTemperature).
fn rad_temperature(rho: f32, K: f32, C: f32) -> f32 {
  if (!(C > 0.0)) { return 0.0; }
  var T = eos_temperature(rho, C);
  if (K > 0.0) { T = min(T, sqrt(sqrt(C / K))); }
  for (var i = 0; i < 12; i++) {
    let e = eos_eint_from_T(rho, T);
    // The heat capacity from a small step (exact for a linear e(T)).
    let h = max(T * 1.0e-3, 1.0e-30);
    let cv = (eos_eint_from_T(rho, T + h) - e) / h;
    let T3 = T * T * T;
    let df = cv + 4.0 * K * T3;
    if (!(df > 0.0)) { break; }
    T = max(T - (e + K * T3 * T - C) / df, 0.0);
  }
  return T;
}

fn rad_exchange(rho: f32, eint: f32, Es: f32, Fs: vec3<f32>, kP: f32, kR: f32, r: f32, dt: f32) -> RadExchange {
  let c = rad_c();
  let a = rad_a();
  var x: RadExchange;
  let thF = r * c * kR * rho * dt;
  x.F = Fs / (1.0 + thF);
  // The share of the flux the gas takes, thF / (1 + thF), not F* - F': in thin gas thF is below
  // the rounding of 1 (1e-8 in rho = 1e-6 at c = 100), 1 + thF rounds to 1 and the difference to
  // nothing, and a little denser it kept a few bits of rounding. The gas there got no push, or a
  // random one that a low density turned into a large speed: rad-shadow came out on one side or
  // the other of passing by the rounding of the GPU. Momentum is then kept to the rounding of F.
  x.dmom = Fs * (thF / (1.0 + thF)) / (r * c * c);
  let th = r * c * kP * rho * dt;
  let w = th / (1.0 + th);
  let A = w / r;
  x.T = rad_temperature(rho, A * a, eint + A * Es);
  let T2 = x.T * x.T;
  x.E = Es / (1.0 + th) + a * T2 * T2 * w;
  x.eint = eos_eint_from_T(rho, x.T);
  // Conservation exactly, not to the rounding of the root: the larger of the two energies takes
  // what the root's rounding left over (a change of at most a rounding of the smaller one, and
  // so of itself). Over 1e4 sweeps of moving gas the rounding had drifted the total by 7e-5.
  if (x.eint >= x.E / r) {
    x.eint = eint + (Es - x.E) / r;
  } else {
    x.E = Es + r * (eint - x.eint);
  }
  return x;
}
