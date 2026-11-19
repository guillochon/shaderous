// Gravity field texture, written once per step after the Poisson solve:
//   xyz = self-gravity's acceleration, from the new potential
//   w   = the potential the next update charges self-gravity's flux-form work with: the new
//         potential itself where gravity/correct.wgsl then moves that work to the mean of this
//         solve and the next (P.flags bit 6), else extrapolated half a step ahead (1.5 phi -
//         0.5 phi_old); with implicit pressure the potential itself
// The hydro passes then need one load per cell instead of six potential loads. The point
// masses and the frame are not in it: each pass adds them itself (gravity/ext.wgsl).
#include "hydro/common.wgsl"
#include "gravity/accel.wgsl"

@group(0) @binding(1) var PHI: texture_3d<f32>;
@group(0) @binding(2) var PHI_OLD: texture_3d<f32>;
@group(0) @binding(3) var GOUT: texture_storage_3d<rgba32float, write>;
// With comoving coordinates the field stored is the proper acceleration, -grad phi / a.
@group(0) @binding(4) var<storage, read> clock: Clock;
// The last step's dt, for the next solve's guess (gravity/advance.wgsl); a step of dt = 0 leaves it.
@group(0) @binding(5) var<storage, read_write> DTP: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (all(gid == vec3<u32>(0u)) && clock.dt > 0.0) { DTP[0] = clock.dt; }
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  var g = vec3<f32>(0.0);
  var ph = 0.0;
  if (has_potential()) {
    g = gravity(PHI, c) / select(1.0, clock.a1, cosmo_on());
    // (With implicit pressure, P.ohm.y, the potential is the advected density's already, solved
    // before the pressure passes that kick the faces with it: src/sim.ts.)
    let ahead = select(0.5, 0.0, P.ohm.y > 0.0 || (P.flags & 64u) != 0u);
    ph = (1.0 + ahead) * textureLoad(PHI, c, 0).x - ahead * textureLoad(PHI_OLD, c, 0).x;
  }
  textureStore(GOUT, c, vec4<f32>(g, ph));
}
