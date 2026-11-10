// Trilinear prolongation of the coarse error, added to the fine potential in place.
// L describes the fine level.
#include "gravity/mg_common.wgsl"

@group(0) @binding(2) var PHIC: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, read_write>;

// Coarse error with boundaries: periodic wraps; isolated has zero error at the face, so a
// ghost is minus its mirror image.
fn coarse(q: vec3<i32>) -> f32 {
  let nc = i32(L.n) / 2;
  if (P.grav_bc == 0u) {
    return textureLoad(PHIC, (q + nc) % nc, 0).x;
  }
  let cl = clamp(q, vec3<i32>(0), vec3<i32>(nc - 1));
  let outside = vec3<i32>(q != cl);
  let sgn = 1.0 - 2.0 * f32((outside.x + outside.y + outside.z) & 1);
  return sgn * textureLoad(PHIC, cl, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(L.n))) { return; }
  let c = vec3<i32>(gid);
  let cc = c / 2;
  let d = 2 * (c % 2) - 1;   // direction of the second-nearest coarse cell along each axis
  var sum = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(o));
    sum += w.x * w.y * w.z * coarse(cc + o * d);
  }
  textureStore(PHI, c, vec4<f32>(textureLoad(PHI, c).x + sum, 0.0, 0.0, 0.0));
}
