// Diagnostics reduction source. Channel order must match DIAG in src/sim.ts.
#include "hydro/common.wgsl"
#include <eos>
#include "gravity/rot.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_3d<f32>;
@group(0) @binding(4) var HEAT: texture_3d<f32>;   // burning's heating rate over the last step (one texel of 0 without)

fn red_cell(c: vec3<i32>) -> Red {
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  if (u2.w > 0.5) { return red_identity(); }   // solid: no gas
  comp_use(comp_tex(c));
  // (An axisymmetric run's cell is a ring: its true volume, 2 pi R dx^2, so masses are true ones.)
  var dv = select(P.geom.x, 6.2831853 * (f32(c.x) + 0.5) * P.dx * P.dx * P.dx, cylindrical());
  // (A spherical run's cell is a ring too, of volume 2 pi dr^3/3 dcos(theta): sph_volume.)
  if (spherical()) { dv = sph_volume(c.x, c.y); }
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
    let lj = cs * sqrt(3.14159265 / (P.G * rho)) * jeans_comoving();
    jeans = select(0.0, 1.0, lj < 4.0 * P.dx);
  }
  // Potential energy in the uniform external field, measured from the box center. A 2D run
  // lies in the plane z = 0 (as gravity/bh.wgsl's cell_pos).
  let xb = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  var x = select(xb, vec3<f32>(xb.xy, 0.0), P.dims.z == 1u);
  // A spherical run: positions and momenta in (R, z, phi), as an axisymmetric run has them.
  var mom = u1.yzw;
  if (spherical()) {
    let g = sph_cell(c.x, c.y);
    let ct = cos((f32(c.y) + 0.5) * P.sph.z);
    x = g.r * P.dx * vec3<f32>(g.st, ct, 0.0);
    mom = vec3<f32>(u1.y * g.st + u1.z * ct, u1.y * ct - u1.z * g.st, u1.w);
  }
  egrav -= rho * dot(P.gext.xyz, x) * dv;
  // A rotating frame's centrifugal potential: the total energy is then the Jacobi integral,
  // which is what the frame conserves.
  if (P.rot.w != 0.0) { egrav += rho * rot_phi(x) * dv; }
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  var r0: Red;
  r0[0] = rho * dv;
  r0[1] = mom.x * dv;
  r0[2] = mom.y * dv;
  r0[3] = mom.z * dv;
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
  r0[23] = textureLoad(HEAT, c, 0).x * dv;
  return r0;
}
