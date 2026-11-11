// Gravity field texture, written once per step after the Poisson solve:
//   xyz = acceleration: self-gravity from the new potential, plus the point masses' pull minus
//         the frame's own acceleration when there are any
//   w   = potential extrapolated half a step ahead, which the next update uses for its
//         time-centered energy source (1.5 phi - 0.5 phi_old for self-gravity)
// The hydro passes then need one load per cell instead of six potential loads.
#include "hydro/common.wgsl"
#include "gravity/accel.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var PHI: texture_3d<f32>;
@group(0) @binding(2) var PHI_OLD: texture_3d<f32>;
@group(0) @binding(3) var GOUT: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read> BH: Masses;
@group(0) @binding(5) var<storage, read> clock: Clock;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  var g = vec3<f32>(0.0);
  var ph = 0.0;
  if (has_potential()) {
    g = gravity(PHI, c);
    ph = 1.5 * textureLoad(PHI, c, 0).x - 0.5 * textureLoad(PHI_OLD, c, 0).x;
  }
  if (P.bh.z != 0.0) {
    let x = cell_pos(c);
    // The frame's acceleration a acts as a uniform field -a, whose potential is a . x.
    g -= BH.frame.xyz;
    ph += dot(BH.frame.xyz, x);
    for (var i = 0u; i < mass_count(); i++) {
      g += bh_accel(x, BH.list[i]);
      ph += bh_phi(x, BH.list[i], 0.5 * clock.dt);
    }
  }
  textureStore(GOUT, c, vec4<f32>(g, ph));
}
