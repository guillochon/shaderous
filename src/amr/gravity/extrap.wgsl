// Per-block timesteps: carry the potential one substep forward on the blocks that are not
// starting a step, in place, at the rate it last changed there.
//
// Between the full solves at the ends of cycles, the potential is solved only on the blocks
// that start a step, and what they see across their faces are these extrapolated values.
// The far field, which changes smoothly, is what a slow block's potential mostly is, and that
// is what extrapolation gets right.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/bins_on.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;

#include "amr/gravity/half_on.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || blk_active(inv.slot)) { return; }
  let c = blk_origin(inv.slot) + inv.l;
  let v = textureLoad(PHI, c).x + clock.dt * textureLoad(DPHI, c, 0).x;
  textureStore(PHI, c, vec4<f32>(v, 0.0, 0.0, 0.0));
}
