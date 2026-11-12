// Diagnostics reduction source. Channel order must match DIAG in src/sim.ts.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  if (u2.w > 0.5) { return red_identity(); }   // solid: no gas
  comp_use(comp_tex(c));
  let dv = P.geom.x;
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  let eint = u2.y;
  let p = max(eos_pressure(rho, eint), P.p_floor);
  let cs = eos_sound_speed(rho, p);
  let speed = length(v);
  let T = eos_temperature(rho, eint);
  let s = u2.z / rho;

  var egrav = 0.0;
  var jeans = 0.0;
  if (has_potential()) {
    let phi = textureLoad(PHI, c, 0).x;
    // Periodic potentials are defined up to a constant; weighting by (rho - mean) removes it.
    let w = select(rho, rho - P.rho_mean, P.grav_bc == 0u);
    egrav = 0.5 * w * phi * dv;
    // Truelove: Jeans length under 4 cells.
    let lj = cs * sqrt(3.14159265 / (P.G * rho));
    jeans = select(0.0, 1.0, lj < 4.0 * P.dx);
  }
  // Potential energy in the uniform external field, measured from the box center.
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  egrav -= rho * dot(P.gext.xyz, x) * dv;
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  var r0: Red;
  r0[0] = rho * dv;
  r0[1] = u1.y * dv;
  r0[2] = u1.z * dv;
  r0[3] = u1.w * dv;
  r0[4] = 0.5 * rho * dot(v, v) * dv;
  // Internal energy as the total holds it, E - KE (the same as e_int wherever the two are
  // synchronized; where an equation of state keeps E apart from e_int, the total is E).
  r0[5] = (u2.x - 0.5 * rho * dot(v, v)) * dv;
  r0[6] = egrav;
  r0[7] = speed / cs;
  r0[8] = floored;
  r0[9] = jeans;
  r0[10] = rho;
  r0[11] = rho;
  r0[12] = T;
  r0[13] = T;
  r0[14] = s;
  r0[15] = s;
  r0[16] = p;
  r0[17] = p;
  r0[18] = speed;
  r0[19] = speed;
  r0[20] = rho * dv * x.x;
  r0[21] = rho * dv * x.y;
  r0[22] = rho * dv * x.z;
  return r0;
}
