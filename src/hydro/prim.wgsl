// Conserved -> primitive, one equation-of-state call per cell.
//   w1 = (rho, vx, vy, vz)    w2 = (p, c_s, scalar, e_int)
// Requires P, the equation of state, hydro/pfloor.wgsl, and g_dx set.
struct Prims { w1: vec4<f32>, w2: vec4<f32> }

fn prim_from_cons(u1: vec4<f32>, u2: vec4<f32>) -> Prims {
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  var eint = u2.y;
  var p: f32;
  var cs: f32;
  if (EOS_HAS_ENERGY) {
    p = eos_pressure(rho, eint);
    // Pressure floor, optionally raised so the Jeans length stays resolved (Truelove).
    let pmin = pressure_floor(rho);
    if (!(p >= pmin)) {
      p = pmin;
      eint = eos_eint_from_p(rho, p);
    }
    cs = eos_sound_speed(rho, p);
  } else {
    let st = baro_state(rho);
    p = st.x;
    cs = st.y;
  }
  return Prims(vec4<f32>(rho, v), vec4<f32>(p, cs, u2.z / rho, eint));
}
