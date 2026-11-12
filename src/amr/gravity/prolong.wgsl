// Trilinear prolongation of the coarse error, added to the finer sublevel in place. GL
// describes the fine sublevel; its "other" fields describe the coarse one. Coarse cells beyond
// the block are found through its neighbors; off the level the error is taken as zero.
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

// The coarse cells a workgroup's 8 x 8 x 4 fine cells interpolate from: a 6 x 6 x 4 patch
// (4 x 4 x 2 under the fine cells and one more on every side), loaded once into workgroup
// memory instead of eight texture reads per fine cell. A workgroup never spans two blocks.
var<workgroup> tile: array<f32, 144>;

@compute @workgroup_size(8, 8, 4)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lix: u32,
) {
  let span = u32(max(GL.bsize, 8));
  let b = (wid.x * 8u) / span;
  let l0 = vec3<i32>(i32((wid.x * 8u) % span), i32(wid.y * 8u), i32(wid.z * 4u));
  let cb = l0 / 2 - vec3<i32>(1);   // coarse cell at tile index (0, 0, 0)
  if (b < LIST[0] && lix < 144u) {
    let slot = i32(LIST[1u + b]);
    let q = cb + vec3<i32>(i32(lix % 6u), i32((lix / 6u) % 6u), i32(lix / 36u));
    var v: f32;
    if (all(q >= vec3<i32>(0)) && all(q < vec3<i32>(GL.obsize))) {
      v = textureLoad(PHIC, g_origin_other(slot) + q, 0).x;
    } else {
      v = coarse(slot, q);
    }
    tile[lix] = v;
  }
  workgroupBarrier();

  let inv = g_inv(gid);
  if (!inv.ok) { return; }
  let cc = inv.l / 2;
  let d = 2 * (inv.l % 2) - 1;   // toward the second-nearest coarse cell along each axis
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let s = vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1);
    let w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(s));
    let t = cc + s * d - cb;
    sum += w.x * w.y * w.z * tile[t.x + 6 * t.y + 36 * t.z];
  }
  let c = g_origin(inv.slot) + inv.l;
  textureStore(PHI, c, vec4<f32>(textureLoad(PHI, c).x + sum, 0.0, 0.0, 0.0));
}
