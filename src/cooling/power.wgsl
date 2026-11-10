// Optically thin cooling with a power-law cooling function, and uniform heating:
//   de/dt = rho heat - rho^2 Lambda0 T^alpha
// P.cool = (Lambda0, alpha, heat, temperature floor). Gas at or below the floor does not cool.
// alpha = 1/2 is bremsstrahlung. Float64 mirror: src/cooling/cooling.ts.
const COOLING = true;

// Net heating rate per unit volume (negative: the gas cools).
fn cool_rate(rho: f32, T: f32) -> f32 {
  var lam = 0.0;
  if (T > P.cool.w && T > 0.0) { lam = P.cool.x * pow(T, P.cool.y); }
  return rho * (P.cool.z - rho * lam);
}
