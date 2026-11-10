// Trilinear prolongation of the coarse error, added to the finer sublevel in place. GL
// describes the fine sublevel; its "other" fields describe the coarse one. Coarse cells beyond
// the block are found through its neighbours; off the level the error is taken as zero.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"

@group(0) @binding(2) var PHIC: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, read_write>;

fn coarse(slot: i32, q: vec3<i32>) -> f32 {
  let r = g_resolve(slot, q, GL.obsize);
  if (!r.ok) { return 0.0; }
  return textureLoad(PHIC, g_origin_other(r.slot) + r.l, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = g_inv(gid);
  if (!inv.ok) { return; }
  let cc = inv.l / 2;
  let d = 2 * (inv.l % 2) - 1;   // towards the second-nearest coarse cell along each axis
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let s = vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1);
    let w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(s));
    sum += w.x * w.y * w.z * coarse(inv.slot, cc + s * d);
  }
  let c = g_origin(inv.slot) + inv.l;
  textureStore(PHI, c, vec4<f32>(textureLoad(PHI, c).x + sum, 0.0, 0.0, 0.0));
}
