// Diagnostics reduction source. Channel order must match DIAG in src/sim.ts.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
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
  if (P.grav_on != 0u) {
    let phi = textureLoad(PHI, c, 0).x;
    // Periodic potentials are defined up to a constant; weighting by (rho - mean) removes it.
    let w = select(rho, rho - P.rho_mean, P.grav_bc == 0u);
    egrav = 0.5 * w * phi * dv;
    // Truelove: Jeans length under 4 cells.
    let lj = cs * sqrt(3.14159265 / (P.G * rho));
    jeans = select(0.0, 1.0, lj < 4.0 * P.dx);
  }
  // Potential energy in the uniform external field, measured from the box centre.
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  egrav -= rho * dot(P.gext.xyz, x) * dv;
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  return Red(
    rho * dv, u1.y * dv, u1.z * dv, u1.w * dv,
    0.5 * rho * dot(v, v) * dv, eint * dv, egrav,
    speed / cs, floored, jeans,
    rho, rho, T, T, s, s, p, p, speed, speed,
    rho * dv * x.x, rho * dv * x.y, rho * dv * x.z,
  );
}
