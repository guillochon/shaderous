// Poisson source 4 pi G rho. Periodic boxes subtract the mean density so the problem is solvable.
// With comoving coordinates the source is 4 pi G (rho - mean) / a, at the end of the step the
// solve is for (src/cosmo.ts).
//
// With the flux-form energy's correction (P.flags bit 6, docs/numerics.md section 14) it also
// keeps, for gravity/correct.wgsl, the step's change of density: the state the step started from
// (U1_OLD) is overwritten by the time the correction runs.
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<storage, read> clock: Clock;
@group(0) @binding(4) var U1_OLD: texture_3d<f32>;
@group(0) @binding(5) var AUX: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let rho1 = textureLoad(U1, c, 0).x;
  var rho = rho1;
  if (P.grav_bc == 0u) { rho -= P.rho_mean; }
  textureStore(RHS, c, vec4<f32>(4.0 * 3.14159265 * P.G * rho / select(1.0, clock.a1, cosmo_on()), 0.0, 0.0, 0.0));
  if ((P.flags & 64u) != 0u) {
    textureStore(AUX, c, vec4<f32>(rho1 - textureLoad(U1_OLD, c, 0).x, 0.0, 0.0, 0.0));
  }
}
