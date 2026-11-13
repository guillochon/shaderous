// Debris census on a uniform grid (debris/debris.wgsl): every cell into the core or its
// energy bin.
#include "hydro/common.wgsl"
#include "debris/debris_on.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
// The gas's own potential (x), where there is one.
@group(0) @binding(3) var PHI: texture_3d<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let phi = select(0.0, textureLoad(PHI, c, 0).x, has_potential());
  deb_census(cell_pos(c), textureLoad(U1, c, 0), textureLoad(U2, c, 0), phi, P.geom.x);
}
