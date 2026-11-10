// Pressure floors, and the barotropic closure built on them. Requires P and the equation of state.

// Jeans (Truelove) floor: the pressure at which the Jeans length spans the requested number of
// cells. P.jeans_floor is (cells / 4)^2, or 0 when the floor is off.
fn jeans_pressure(rho: f32) -> f32 {
  return P.jeans_floor * 16.0 * P.G * rho * rho * P.dx * P.dx / 3.14159265;
}

fn pressure_floor(rho: f32) -> f32 {
  return max(P.p_floor, jeans_pressure(rho));
}

// Barotropic gas: (pressure, sound speed) as functions of density alone, floors included.
// Every pass must use this same closure. In particular the sound speed has to be the true
// dp/drho of whichever branch is active: on the Jeans floor p ~ rho^2, so c^2 = 2 p / rho.
// Using the unfloored c there underestimates the signal speed, which makes HLL under-dissipative
// and lets self-gravitating cores gain energy.
fn baro_state(rho: f32) -> vec2<f32> {
  let pe = eos_pressure(rho, 0.0);
  let pj = jeans_pressure(rho);
  if (pj > pe && pj >= P.p_floor) {
    return vec2<f32>(pj, sqrt(2.0 * pj / rho));
  }
  let p = max(pe, P.p_floor);
  return vec2<f32>(p, eos_sound_speed(rho, p));
}
