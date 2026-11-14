// Per-block timesteps with a magnetic field: the update wrote the faces (and compensations) of
// the blocks that stepped or took a deposit into the other parity; this copies them back, so
// the blocks that did not keep their place (amr/commit.wgsl does the same for the cells). Over
// the texels [0, BS] of each block of the list.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var NB: texture_3d<f32>;
@group(0) @binding(2) var NC: texture_3d<f32>;
@group(0) @binding(3) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var OC: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = flat_row(wid);
  let w = BS + 1;
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  if (row >= LIST[0] || i >= n) { return; }
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { l.z = 0; }
  let x = atl(blk_origin(i32(LIST[1u + row])), l);
  textureStore(OB, x, textureLoad(NB, x, 0));
  textureStore(OC, x, textureLoad(NC, x, 0));
}
