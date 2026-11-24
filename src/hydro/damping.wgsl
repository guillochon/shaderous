// Damping zone: where a preset asks for it (Problem.damping), the gas is relaxed toward a fixed
// state, as wave-damping zones in disk codes do (de Val-Borro et al. 2006). An accretion disk
// keeps its outer edge as a reservoir that viscosity draws on, and the boundaries see the
// state they started with instead of whatever has spread out to them.
//
// Start-up damping (Problem.startDamping, P.calm): for a while after the start every cell's
// velocity in the frame is damped at a rate that falls smoothly to zero,
//   rate(t) = rate0 (1 + cos(pi t / T)) / 2   for t < T, and 0 after,
// so that what a settled star was still out of balance with on the grid can start to move and
// is taken away while it is small, instead of ringing through the run. Its kinetic energy is
// taken off the total energy: the gas is not heated.
//
// Applied to a cell's state at the end of its update, after sources and sinks.
// Requires P, g_dx, the equation of state, hydro/dual_energy.wgsl, hydro/advance.wgsl
// (NewState) and the <damping> slot, which defines
//   const DAMPING: bool;
//   fn damping_target(x: vec3<f32>) -> DampTarget;   // x relative to the box center
struct DampTarget {
  rate: f32,         // relaxation rate (per unit time); 0 outside the zone
  rho: f32,
  vel: vec3<f32>,
  p: f32,
}

#include <damping>

// The start-up damping's rate at time `t` (0 once it is over, or without one).
fn start_damping_rate(t: f32) -> f32 {
  let rate = P.calm.x;
  let span = P.calm.y;
  if (!(rate > 0.0) || !(t < span)) { return 0.0; }
  return 0.5 * rate * (1.0 + cos(3.14159265 * max(t, 0.0) / span));
}

// Whether apply_damping can change anything: a damping zone, or start-up damping.
fn damping_on() -> bool {
  return DAMPING || P.calm.x > 0.0;
}

// The state of the cell at `x` (in fractions of the box) after relaxing for a step of length
// dt that starts at time t.
fn apply_damping(st: NewState, x: vec3<f32>, dt: f32, t: f32) -> NewState {
  if (st.u2.w > 0.5) { return st; }   // solid cells keep their gas as it is
  var s = st;
  let calm = start_damping_rate(t + 0.5 * dt);
  if (calm > 0.0) {
    let keep = exp(-calm * dt);
    let rho = max(s.u1.x, P.rho_floor);
    let ke = 0.5 * dot(s.u1.yzw, s.u1.yzw) / rho;
    s = NewState(vec4<f32>(s.u1.x, s.u1.yzw * keep), vec4<f32>(s.u2.x - ke * (1.0 - keep * keep), s.u2.yzw));
  }
  if (!DAMPING) { return s; }
  var pos = frac_pos(x);
  if (P.dims.z == 1u) { pos.z = 0.0; }
  let tg = damping_target(pos);
  if (!(tg.rate > 0.0)) { return s; }
  let f = 1.0 - exp(-tg.rate * dt);
  let rho = max(s.u1.x, P.rho_floor);
  let rn = max(rho + (tg.rho - rho) * f, P.rho_floor);
  let v = s.u1.yzw / rho;
  let mom = rn * (v + (tg.vel - v) * f);
  // The specific internal energy relaxes too; a barotropic gas's follows its density anyway.
  let e = s.u2.y / rho;
  let et = eos_eint_from_p(tg.rho, tg.p) / tg.rho;
  var eint = rn * (e + (et - e) * f);
  let ke = 0.5 * dot(mom, mom) / rn;
  var E = ke + eint;
  let en = dual_energy(rn, ke, E, eint);
  E = en.x;
  eint = en.y;
  return NewState(vec4<f32>(rn, mom), vec4<f32>(E, eint, s.u2.z * rn / rho, s.u2.w));
}
