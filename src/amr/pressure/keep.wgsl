// Implicit pressure for a gas with an energy on blocks, before the advected state's primitives
// replace those of the state the step started from: for each leaf cell, the change advection
// makes of that state's pressure over the step, -dt v . grad p by first-order upwind differences
// (PD; amr/pressure/source_gas.wgsl).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(15) var W1: texture_3d<f32>;
@group(0) @binding(22) var W2: texture_3d<f32>;
@group(0) @binding(24) var PD: texture_storage_3d<r32float, write>;

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let inv = inv_flat(wid, lix, 0);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let p = textureLoad(W2, c, 0).x;
  if (any(inv.l < vec3<i32>(0)) || any(inv.l >= vec3<i32>(BS)) ) { return; }
  let v = textureLoad(W1, c, 0).yzw;
  var adv = 0.0;
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    adv += max(v[a], 0.0) * (p - textureLoad(W2, c - e, 0).x) + min(v[a], 0.0) * (textureLoad(W2, c + e, 0).x - p);
  }
  textureStore(PD, c, vec4<f32>(-clock.dt / blk_dx(inv.slot) * adv, 0.0, 0.0, 0.0));
}
