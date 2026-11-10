// Conserved -> primitive on a uniform grid.
//   W1 = (rho, vx, vy, vz)    W2 = (p, c_s, scalar, e_int)
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var W2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  g_dx = P.dx;
  let c = vec3<i32>(gid);
  let w = prim_from_cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
  textureStore(W1, c, w.w1);
  textureStore(W2, c, w.w2);
}
