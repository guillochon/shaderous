// Damping zone: where a preset asks for it (Problem.damping), the gas is relaxed toward a fixed
// state, as wave-damping zones in disk codes do (de Val-Borro et al. 2006). An accretion disk
// keeps its outer edge as a reservoir that viscosity draws on, and the boundaries see the
// state they started with instead of whatever has spread out to them.
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

// The state of the cell at `x` (in fractions of the box) after relaxing for a step of length dt.
fn apply_damping(st: NewState, x: vec3<f32>, dt: f32) -> NewState {
  if (!DAMPING || st.u2.w > 0.5) { return st; }   // solid cells keep their gas as it is
  var pos = (x - 0.5) * P.box;
  if (P.dims.z == 1u) { pos.z = 0.0; }
  let t = damping_target(pos);
  if (!(t.rate > 0.0)) { return st; }
  let f = 1.0 - exp(-t.rate * dt);
  let rho = max(st.u1.x, P.rho_floor);
  let rn = max(rho + (t.rho - rho) * f, P.rho_floor);
  let v = st.u1.yzw / rho;
  let mom = rn * (v + (t.vel - v) * f);
  // The specific internal energy relaxes too; a barotropic gas's follows its density anyway.
  let e = st.u2.y / rho;
  let et = eos_eint_from_p(t.rho, t.p) / t.rho;
  var eint = rn * (e + (et - e) * f);
  let ke = 0.5 * dot(mom, mom) / rn;
  var E = ke + eint;
  let en = dual_energy(rn, ke, E, eint);
  E = en.x;
  eint = en.y;
  return NewState(vec4<f32>(rn, mom), vec4<f32>(E, eint, st.u2.z * rn / rho, st.u2.w));
}
