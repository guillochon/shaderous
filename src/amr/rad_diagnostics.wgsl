// Radiation diagnostics on the block grid, over leaf cells: the channels of
// radiation/diagnostics.wgsl (radiation/radiation.ts RAD_DIAG_OPS). The luminosity is the
// outward flux through the open faces of the domain, of whatever level's cells lie on them.
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var RR: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return red_identity(); }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let a = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, a, 0);
  let u2 = textureLoad(U2, a, 0);
  if (u2.w > 0.5) { return red_identity(); }
  comp_use(comp_tex(a));
  let R = textureLoad(RR, a, 0);
  let dx = blk_dx(slot);
  // (A 1D run's cells are per unit area across x, and their faces are of unit area.)
  let dv = select(select(dx * dx * dx, dx * dx, is2d()), dx, is1d());
  let area = select(select(dx * dx, dx, is2d()), 1.0, is1d());
  let rho = max(u1.x, P.rho_floor);
  let T = eos_temperature(rho, u2.y);
  let r = rad_reduction(kappa_rosseland(rho, T) * rho * dx);
  let E = max(R.x, 0.0);
  let f = select(0.0, length(R.yzw) / (rad_c() * E), E > 0.0);
  var lum = 0.0;
  for (var k = 0; k < 3; k++) {
    if (k >= blk_axes()) { break; }
    if (l[k] == 0) {
      let nb = bt(slot, 6u + u32(2 * k));
      if (nb == -2 || nb == -4) { lum += max(-R.yzw[k], 0.0) * area; }
    }
    if (l[k] == BS - 1) {
      let nb = bt(slot, 7u + u32(2 * k));
      if (nb == -2 || nb == -4) { lum += max(R.yzw[k], 0.0) * area; }
    }
  }
  let p = max(eos_pressure(rho, u2.y), P.p_floor);
  let speed = length(u1.yzw) / rho + eos_sound_speed(rho, p);
  var o: Red;
  o[0] = E * dv;
  o[1] = E / r * dv;
  o[2] = select(0.0, E * dv, r < 1.0);
  o[3] = lum;
  o[4] = select(0.0, 1.0, f >= 0.999);
  o[5] = select(0.0, 1.0, speed > rad_chat() / 3.0);
  o[6] = f;
  o[7] = sqrt(sqrt(E / rad_a()));
  return o;
}
