// Poisson source 4 pi G rho on the leaf blocks. Periodic boxes subtract the mean density.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var RHS: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  var rho = textureLoad(U1, c, 0).x;
  if (P.grav_bc == 0u) { rho -= P.rho_mean; }
  textureStore(RHS, c, vec4<f32>(4.0 * 3.14159265 * P.G * rho, 0.0, 0.0, 0.0));
}
