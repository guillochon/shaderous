// Optically thin cooling and heating, applied to a cell's internal energy at the end of its
// update, at fixed density. Requires P, the equation of state, <cooling> and hydro/advance.wgsl
// (NewState). Float64 mirror: integrateCooling() in src/cooling/cooling.ts.
//
// The rate can be stiff: where the cooling time is far below the timestep an explicit step
// would overshoot. So the step is split into up to COOL_SUBSTEPS second-order (Heun) substeps,
// none changing the energy by more than a tenth. If the rate changes sign across a substep, the
// gas reaches thermal balance within it: the energy goes to the balance point instead (by the
// secant through the two rates, refined by the substeps that are left). A cell whose cooling
// time is so short that the substeps do not cover the step stops there, having lost at most
// 70% of its energy, and carries on in the next step; the timestep is never shortened for
// cooling.
const COOL_SUBSTEPS = 12;

fn cool_edot(rho: f32, eint: f32) -> f32 {
  return cool_rate(rho, eos_temperature(rho, eint));
}

fn apply_cooling(st: NewState, dt: f32) -> NewState {
  if (!COOLING || !EOS_HAS_ENERGY) { return st; }
  let rho = st.u1.x;
  let e0 = st.u2.y;
  // Never below the pressure floor, nor (by cooling) below the temperature floor.
  let emin = eos_eint_from_p(rho, P.p_floor);
  let efloor = max(emin, eos_eint_from_T(rho, P.cool.w));
  var e = e0;
  var left = dt;
  for (var i = 0; i < COOL_SUBSTEPS; i++) {
    if (left <= 0.0) { break; }
    let k1 = cool_edot(rho, e);
    if (k1 == 0.0) { break; }
    let h = min(left, 0.1 * e / abs(k1));
    let e1 = max(e + h * k1, emin);
    let k2 = cool_edot(rho, e1);
    if (k1 * k2 < 0.0) {
      e += (e1 - e) * k1 / (k1 - k2);
      continue;
    }
    var en = e + 0.5 * h * (k1 + k2);
    if (en < e) { en = max(en, min(e, efloor)); }
    e = max(en, emin);
    left -= h;
  }
  if (!(e == e)) { return st; }
  return NewState(st.u1, vec4<f32>(st.u2.x + (e - e0), e, st.u2.zw));
}
