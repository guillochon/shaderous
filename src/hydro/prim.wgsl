// Conserved -> primitive, one equation-of-state call per cell.
//   w1 = (rho, vx, vy, vz)    w2 = (p, c_s, scalar, e_int)
// Requires P, the equation of state, hydro/pfloor.wgsl, and g_dx set.
#include <refdev>
struct Prims { w1: vec4<f32>, w2: vec4<f32> }

// The same for a gas stored as its deviation from the reference r = ref_at(c)
// (hydro/ref_on.wgsl): w1 = (rho - rho_ref, v), w2 = (p - p_ref, c_s, scalar, rho e - e_ref).
// The ideal gas alone (SimConfig.reference): p - p_ref = (gamma - 1)(rho e - e_ref), exactly
// the deviation, with no large pressure formed on the way.
fn prim_dev(u1: vec4<f32>, u2: vec4<f32>, r: vec4<f32>) -> Prims {
  let rho = max(u1.x + r.x, P.rho_floor);
  // (The stored deviation itself, not rho - rho_ref, which would round to the size of rho.)
  let rd = select(u1.x, rho - r.x, u1.x + r.x < P.rho_floor);
  let v = u1.yzw / rho;
  var pd = (P.eos.x - 1.0) * u2.y;
  var p = r.y + pd;
  let pmin = pressure_floor(rho);
  if (!(p >= pmin)) {
    p = pmin;
    pd = pmin - r.y;
  }
  return Prims(vec4<f32>(rd, v), vec4<f32>(pd, eos_sound_speed(rho, p), u2.z / rho, u2.y));
}

fn prim_from_cons(u1: vec4<f32>, u2: vec4<f32>) -> Prims {
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  var eint = u2.y;
  var p: f32;
  var cs: f32;
  if (EOS_HAS_ENERGY) {
    // Pressure and sound speed together: an equation of state that solves for its temperature
    // does so once for both.
    let pc = eos_prim(rho, eint);
    p = pc.x;
    cs = pc.y;
    // Pressure floor, optionally raised so the Jeans length stays resolved (Truelove).
    let pmin = pressure_floor(rho);
    if (!(p >= pmin)) {
      p = pmin;
      eint = eos_eint_from_p(rho, p);
      cs = eos_sound_speed(rho, p);
    }
  } else {
    let st = baro_state(rho);
    p = st.x;
    cs = st.y;
  }
  return Prims(vec4<f32>(rho, v), vec4<f32>(p, cs, u2.z / rho, eint));
}
