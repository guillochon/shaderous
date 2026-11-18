// Poisson source 4 pi G rho. Periodic boxes subtract the mean density so the problem is solvable.
// With comoving coordinates the source is 4 pi G (rho - mean) / a, at the end of the step the
// solve is for (src/cosmo.ts).
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<storage, read> clock: Clock;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  var rho = textureLoad(U1, c, 0).x;
  if (P.grav_bc == 0u) { rho -= P.rho_mean; }
  textureStore(RHS, c, vec4<f32>(4.0 * 3.14159265 * P.G * rho / select(1.0, clock.a1, cosmo_on()), 0.0, 0.0, 0.0));
}
