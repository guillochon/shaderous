// Conserved -> primitive, one equation-of-state call per cell.
//   W1 = (rho, vx, vy, vz)    W2 = (p, c_s, scalar, e_int)
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var W2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);

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

  textureStore(W1, c, vec4<f32>(rho, v));
  textureStore(W2, c, vec4<f32>(p, cs, u2.z / rho, eint));
}
