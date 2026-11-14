// Field lines (render/fieldlines.wgsl), uniform grid: the cell-centered magnetic field and the
// velocity resampled onto the field-line volume, one voxel per cell or a cell under the
// center of each voxel when the volume is coarser than the grid, and the largest |B| in it.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(13) var BF: texture_3d<f32>;
@group(0) @binding(14) var BV: texture_storage_3d<rgba16float, write>;   // B, |B|
@group(0) @binding(15) var VV: texture_storage_3d<rgba16float, write>;   // velocity, density
@group(0) @binding(16) var<storage, read_write> FR: array<atomic<u32>, 4>;

#include "mhd/common.wgsl"

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = textureDimensions(BV);
  if (any(gid >= m)) { return; }
  let n = vec3<f32>(P.dims.xyz);
  let c = clamp(vec3<i32>(floor((vec3<f32>(gid) + 0.5) / vec3<f32>(m) * n)), vec3<i32>(0), vec3<i32>(P.dims.xyz) - 1);
  let B = bcell(c);
  let u = textureLoad(U1, c, 0);
  let rho = max(u.x, P.rho_floor);
  let b = length(B);
  textureStore(BV, vec3<i32>(gid), vec4<f32>(B, b));
  textureStore(VV, vec3<i32>(gid), vec4<f32>(u.yzw / rho, rho));
  atomicMax(&FR[0], bitcast<u32>(b));
}
