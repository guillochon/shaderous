// Gravity field atlas, written once per step after the Poisson solve, for every block:
//   xyz = self-gravity's acceleration, from the new potential (interior cells)
//   w   = self-gravity's potential extrapolated half a step ahead, for the update's
//         time-centered energy source (interior cells and the first ghost layer) and the
//         well-balanced pressure reconstruction (both ghost layers)
// The point masses and the frame are not in it: each pass adds them itself (gravity/ext.wgsl).
// With per-block timesteps the half step is the block's own, and the pass runs over the
// blocks about to start a step.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var PHI: texture_3d<f32>;
@group(0) @binding(2) var PHI_OLD: texture_3d<f32>;
@group(0) @binding(3) var GOUT: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var<storage, read> clock: Clock;

fn phi(c: vec3<i32>) -> f32 {
  return textureLoad(PHI, c, 0).x;
}

#include <half>

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 2);
  if (!inv.ok) { return; }
  let l = inv.l;
  let out = vec3<i32>(l < vec3<i32>(0)) + vec3<i32>(l >= vec3<i32>(BS));
  let nout = out.x + out.y + out.z;
  if (nout > 1) { return; }
  let c = atl(blk_origin(inv.slot), l);
  var g = vec3<f32>(0.0);
  let self_gravity = has_potential();
  if (nout == 0 && self_gravity) {
    let h = blk_dx(inv.slot);
    g = -vec3<f32>(
      phi(c + vec3<i32>(1, 0, 0)) - phi(c - vec3<i32>(1, 0, 0)),
      phi(c + vec3<i32>(0, 1, 0)) - phi(c - vec3<i32>(0, 1, 0)),
      phi(c + vec3<i32>(0, 0, 1)) - phi(c - vec3<i32>(0, 0, 1)),
    ) * (0.5 / h);
  }
  var ph = 0.0;
  if (self_gravity) { ph = phi_half(c, inv.slot); }
  textureStore(GOUT, c, vec4<f32>(g, ph));
}
