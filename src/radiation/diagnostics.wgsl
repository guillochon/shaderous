// Radiation diagnostics on the uniform grid (radiation/radiation.ts RAD_DIAG_OPS): the reduction's
// first stage. Channels:
//   0  total radiation energy, sum E dV
//   1  sum E / r dV: with the gas's energy, what the scheme conserves
//   2  radiation energy in cells whose speed of light is reduced (r < 1)
//   3  luminosity through the box's open faces (outflow), sum of the outward F . n dA
//   4  cells whose flux is at the streaming limit, |F| >= 0.999 c E
//   5  cells where |v| + c_s exceeds chat / 3: the reduced speed of light is too slow there
//   6  largest reduced flux |F| / (c E)
//   7  largest radiation temperature (E / a)^(1/4)
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var RR: texture_3d<f32>;

fn open_face(k: u32) -> bool {
  return k == 1u || k == 3u;
}

fn red_cell(c: vec3<i32>) -> Red {
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  if (u2.w > 0.5) { return red_identity(); }
  comp_use(comp_tex(c));
  let R = textureLoad(RR, c, 0);
  let flat = P.dims.z == 1u;
  let dv = P.geom.x;
  // (A 1D run's faces are of unit area: its cells are per unit area across x.)
  let area = select(select(P.dx * P.dx, P.dx, flat), 1.0, P.dims.y == 1u);
  let rho = max(u1.x, P.rho_floor);
  let T = eos_temperature(rho, u2.y);
  let r = rad_reduction(kappa_rosseland(rho, T) * rho * P.dx);
  let E = max(R.x, 0.0);
  let f = select(0.0, length(R.yzw) / (rad_c() * E), E > 0.0);
  var lum = 0.0;
  let n = vec3<i32>(P.dims.xyz);
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    if (c[a] == 0 && open_face(P.bc_lo[a])) { lum += max(-R.yzw[a], 0.0) * area; }
    if (c[a] == n[a] - 1 && open_face(P.bc_hi[a])) { lum += max(R.yzw[a], 0.0) * area; }
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
