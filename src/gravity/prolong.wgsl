// Trilinear prolongation of the coarse error, added to the fine potential in place.
// L describes the fine level.
#include "gravity/mg_common.wgsl"

@group(0) @binding(2) var PHIC: texture_3d<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, read_write>;

// Coarse error with boundaries: periodic wraps; isolated has zero error at the face, so a
// ghost is minus its mirror image; across a plane of mirror symmetry it is the image itself.
fn coarse(q: vec3<i32>) -> f32 {
  // The coarse level's cells along each axis (one along z for an axisymmetric run).
  let nc = max(vec3<i32>(lvl_dims()) / 2, vec3<i32>(1));
  if (P.grav_bc == 0u) {
    return textureLoad(PHIC, (q + nc) % nc, 0).x;
  }
  let cl = clamp(q, vec3<i32>(0), nc - 1);
  let mirror = (q < vec3<i32>(0)) & vec3<bool>(sym_lo(0), sym_lo(1), sym_lo(2));
  let outside = select(vec3<i32>(q != cl), vec3<i32>(0), mirror);
  let sgn = 1.0 - 2.0 * f32((outside.x + outside.y + outside.z) & 1);
  return sgn * textureLoad(PHIC, cl, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= lvl_dims())) { return; }
  let c = vec3<i32>(gid);
  let cc = c / 2;
  let d = 2 * (c % 2) - 1;   // direction of the second-nearest coarse cell along each axis
  var sum = 0.0;
  // (An axisymmetric run's levels are one cell deep: bilinear in the plane.)
  let nk = select(8, 4, L.flat != 0u);
  for (var k = 0; k < nk; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    var w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(o));
    if (L.flat != 0u) { w.z = 1.0; }
    sum += w.x * w.y * w.z * coarse(vec3<i32>((cc + o * d).xy, select(cc.z + o.z * d.z, 0, L.flat != 0u)));
  }
  textureStore(PHI, c, vec4<f32>(textureLoad(PHI, c).x + sum, 0.0, 0.0, 0.0));
}
