// Per-block timesteps, for the blocks whose potential has just been solved at the end of a
// step: the rate at which it changed over that step (see amr/gravity/half_on.wgsl), and the
// potential itself as the reference for the next one. Cells and both ghost layers, which
// the field pass also reads.
//
// FORGET starts the history afresh instead, on a new mesh or a new run: no rate, and the
// potential as it is for reference.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/bins_on.wgsl"
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var PHI: texture_3d<f32>;
@group(0) @binding(3) var PSOLVED: texture_storage_3d<r32float, read_write>;   // potential at the block's last solve
@group(0) @binding(4) var DPHI: texture_storage_3d<r32float, write>;

override FORGET: bool = false;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 2);
  if (!inv.ok) { return; }
  let c = blk_origin(inv.slot) + inv.l;
  let now = textureLoad(PHI, c, 0).x;
  var rate = 0.0;
  if (!FORGET) {
    let dt = blk_dt(inv.slot);
    if (dt > 0.0) { rate = (now - textureLoad(PSOLVED, c).x) / dt; }
  }
  textureStore(DPHI, c, vec4<f32>(rate, 0.0, 0.0, 0.0));
  textureStore(PSOLVED, c, vec4<f32>(now, 0.0, 0.0, 0.0));
}
