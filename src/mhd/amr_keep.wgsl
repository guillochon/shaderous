// MHD on the block grid: what stage 2's first-order flux correction (mhd/face_flux.wgsl) needs of
// stage 1's assembly, kept after it, before stage 2's overwrites it (mhd/amr_mhd.ts). Per texel
// of a leaf's assembled region, ghosts included:
//   K0 = (rho, v)                         as W1 holds them
//   K1 = (p, scalar, b_x, b_y)            p and the scalar from W2, the lower faces from FA
//   K2 = (b_z, cell-centered field)       the last face, and W3's field
// The cell-centered field is kept as the assembly made it, not rebuilt from the kept faces: a ghost
// cell the correction reads can have an upper face at a corner texel the assembly never fills.
// Dispatched as the assembly is, over the same list.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var FA: texture_3d<f32>;
@group(0) @binding(8) var W3: texture_3d<f32>;
@group(0) @binding(5) var K0: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var K1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var K2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_flat(wid, lix, NGH);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let w2 = textureLoad(W2, c, 0);
  let f = textureLoad(FA, c, 0);
  textureStore(K0, c, textureLoad(W1, c, 0));
  textureStore(K1, c, vec4<f32>(w2.x, w2.z, f.x, f.y));
  textureStore(K2, c, vec4<f32>(f.z, textureLoad(W3, c, 0).xyz));
}
