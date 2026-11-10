// Restrict the residual to the next coarser sublevel (average of the eight cells it covers,
// all within the same block) and zero the error there. GL describes the coarse sublevel; its
// "other" fields describe the fine one.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"

@group(0) @binding(2) var RES: texture_3d<f32>;
@group(0) @binding(3) var RHSC: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var PHIC: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = g_inv(gid);
  if (!inv.ok) { return; }
  let fine = g_origin_other(inv.slot);
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    sum += textureLoad(RES, fine + 2 * inv.l + vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1), 0).x;
  }
  let c = g_origin(inv.slot) + inv.l;
  textureStore(RHSC, c, vec4<f32>(0.125 * sum, 0.0, 0.0, 0.0));
  textureStore(PHIC, c, vec4<f32>(0.0));
}
