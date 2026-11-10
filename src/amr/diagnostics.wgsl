// Diagnostics reduction source on the block grid, over leaf cells only. Same channels as
// hydro/diagnostics.wgsl; empty slots and parents contribute the identity of each channel.
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_3d<f32>;

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
    return r0;
  }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let a = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, a, 0);
  let u2 = textureLoad(U2, a, 0);
  if (u2.w > 0.5) { return red_identity(); }   // solid: no gas
  let dx = blk_dx(slot);
  let dv = select(dx * dx * dx, dx * dx, is2d());
  let rho = max(u1.x, P.rho_floor);
  let v = u1.yzw / rho;
  let eint = u2.y;
  let p = max(eos_pressure(rho, eint), P.p_floor);
  let cs = eos_sound_speed(rho, p);
  let speed = length(v);
  let T = eos_temperature(rho, eint);
  let s = u2.z / rho;
  let x = blk_pos(slot, l);
  // Potential energy in the uniform external field, measured from the box centre.
  var egrav = -rho * dot(P.gext.xyz, x) * dv;
  var jeans = 0.0;
  if (has_potential()) {
    // Periodic potentials are defined up to a constant; weighting by (rho - mean) removes it.
    let w = select(rho, rho - P.rho_mean, P.grav_bc == 0u);
    egrav += 0.5 * w * textureLoad(PHI, a, 0).x * dv;
    // Truelove: Jeans length under 4 cells.
    jeans = select(0.0, 1.0, cs * sqrt(3.14159265 / (P.G * rho)) < 4.0 * dx);
  }
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  var r1: Red;
  r1[0] = rho * dv;
  r1[1] = u1.y * dv;
  r1[2] = u1.z * dv;
  r1[3] = u1.w * dv;
  r1[4] = 0.5 * rho * dot(v, v) * dv;
  r1[5] = eint * dv;
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
  return r1;
}
