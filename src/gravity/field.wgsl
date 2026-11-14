// Gravity field texture, written once per step after the Poisson solve:
//   xyz = self-gravity's acceleration, from the new potential
//   w   = self-gravity's potential extrapolated half a step ahead (1.5 phi - 0.5 phi_old),
//         which the next update uses for its time-centered energy source
// The hydro passes then need one load per cell instead of six potential loads. The point
// masses and the frame are not in it: each pass adds them itself (gravity/ext.wgsl).
#include "hydro/common.wgsl"
#include "gravity/accel.wgsl"

@group(0) @binding(1) var PHI: texture_3d<f32>;
@group(0) @binding(2) var PHI_OLD: texture_3d<f32>;
@group(0) @binding(3) var GOUT: texture_storage_3d<rgba32float, write>;

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
  textureStore(GOUT, c, vec4<f32>(g, ph));
}
