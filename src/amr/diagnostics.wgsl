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
    return Red(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -big, 0.0, 0.0, big, -big, big, -big, big, -big, big, -big, big, -big, 0.0, 0.0, 0.0);
  }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let a = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, a, 0);
  let u2 = textureLoad(U2, a, 0);
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
  if (P.grav_on != 0u) {
    // Periodic potentials are defined up to a constant; weighting by (rho - mean) removes it.
    let w = select(rho, rho - P.rho_mean, P.grav_bc == 0u);
    egrav += 0.5 * w * textureLoad(PHI, a, 0).x * dv;
    // Truelove: Jeans length under 4 cells.
    jeans = select(0.0, 1.0, cs * sqrt(3.14159265 / (P.G * rho)) < 4.0 * dx);
  }
  let floored = select(0.0, 1.0, u1.x <= P.rho_floor * 1.0001 || p <= P.p_floor * 1.0001);

  return Red(
    rho * dv, u1.y * dv, u1.z * dv, u1.w * dv,
    0.5 * rho * dot(v, v) * dv, eint * dv, egrav,
    speed / cs, floored, jeans,
    rho, rho, T, T, s, s, p, p, speed, speed,
    rho * dv * x.x, rho * dv * x.y, rho * dv * x.z,
  );
}
