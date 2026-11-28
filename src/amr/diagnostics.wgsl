// Diagnostics reduction source on the block grid, over leaf cells only. Same channels as
// hydro/diagnostics.wgsl; empty slots and parents contribute the identity of each channel.
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "gravity/rot.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_3d<f32>;
@group(0) @binding(4) var HEAT: texture_3d<f32>;   // burning's heating rate over the last step (one texel of 0 without)

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  let big = 3.0e38;
  if (!blk_leaf(slot)) {
    var r0: Red;
    r0[0] = 0.0;
    r0[1] = 0.0;
    r0[2] = 0.0;
    r0[3] = 0.0;
    r0[4] = 0.0;
    r0[5] = 0.0;
    r0[6] = 0.0;
    r0[7] = -big;
    r0[8] = 0.0;
    r0[9] = 0.0;
    r0[10] = big;
    r0[11] = -big;
    r0[12] = big;
    r0[13] = -big;
    r0[14] = big;
    r0[15] = -big;
    r0[16] = big;
    r0[17] = -big;
    r0[18] = big;
    r0[19] = -big;
    r0[20] = 0.0;
    r0[21] = 0.0;
    r0[22] = 0.0;
    r0[23] = 0.0;
    return r0;
  }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let a = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, a, 0);
  let u2 = textureLoad(U2, a, 0);
  if (u2.w > 0.5) { return red_identity(); }   // solid: no gas
  comp_use(comp_tex(a));
  let dx = blk_dx(slot);
  // (An axisymmetric run's cell is a ring of volume 2 pi R dx^2.)
  // (A 1D run's cell is per unit area across x: its reduction takes one row of each block.)
  var dv = select(select(dx * dx * dx, dx * dx, is2d()), dx, is1d());
  if (cylindrical()) { dv = 6.2831853 * (f32(blk_coord(slot).x * BS + l.x) + 0.5) * dx * dx * select(dx, 1.0, is1d()); }
  // (A spherical run's cell is a ring of volume 2 pi dr^3/3 dcos(theta): amr/grid.wgsl blk_sph_volume.)
  if (spherical()) { dv = blk_sph_volume(slot, l); }
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  let eint = u2.y;
  let p = max(eos_pressure(rho, eint), P.p_floor);
  let cs = eos_sound_speed(rho, p);
  let speed = length(v);
  let T = eos_temperature(rho, eint);
  let s = u2.z / rho;
  var x = blk_pos(slot, l);
  // A spherical run: positions and momenta in (R, z, phi), as hydro/diagnostics.wgsl has them.
  var mom = u1.yzw;
  if (spherical()) {
    let g = blk_sph(slot, l);
    let q = blk_sph_grid(slot);
    let ct = cos((f32(blk_coord(slot).y * BS + l.y) + 0.5) * q.dt);
    x = g.r * dx * vec3<f32>(g.st, ct, 0.0);
    mom = vec3<f32>(u1.y * g.st + u1.z * ct, u1.y * ct - u1.z * g.st, u1.w);
  }
  // Potential energy in the uniform external field, measured from the box center.
  var egrav = -rho * dot(P.gext.xyz, x) * dv;
  // A rotating frame's centrifugal potential (hydro/diagnostics.wgsl).
  if (P.rot.w != 0.0) { egrav += rho * rot_phi(x) * dv; }
  var jeans = 0.0;
  if (has_potential()) {
    // Periodic potentials are defined up to a constant; weighting by (rho - mean) removes it.
    let w = select(rho, rho - P.rho_mean, P.grav_bc == 0u);
    egrav += 0.5 * w * textureLoad(PHI, a, 0).x * dv;
    // Truelove: Jeans length under 4 cells.
    jeans = select(0.0, 1.0, cs * sqrt(3.14159265 / (P.G * rho)) * jeans_comoving() < 4.0 * dx);
  }
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  var r1: Red;
  r1[0] = rho * dv;
  r1[1] = mom.x * dv;
  r1[2] = mom.y * dv;
  r1[3] = mom.z * dv;
  r1[4] = 0.5 * rho * dot(v, v) * dv;
  // Internal energy as the total holds it, E - KE (the same as e_int wherever the two are
  // synchronized; where an equation of state keeps E apart from e_int, the total is E).
  r1[5] = (u2.x - 0.5 * rho * dot(v, v)) * dv;
  r1[6] = egrav;
  r1[7] = speed / cs;
  r1[8] = floored;
  r1[9] = jeans;
  r1[10] = rho;
  r1[11] = rho;
  r1[12] = T;
  r1[13] = T;
  r1[14] = s;
  r1[15] = s;
  r1[16] = p;
  r1[17] = p;
  r1[18] = speed;
  r1[19] = speed;
  r1[20] = rho * dv * x.x;
  r1[21] = rho * dv * x.y;
  r1[22] = rho * dv * x.z;
  r1[23] = textureLoad(HEAT, a, 0).x * dv;
  return r1;
}
