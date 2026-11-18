// Trilinear (bilinear in 2D) prolongation of the coarse error, added to the fine pressure in
// place. L describes the fine level. Beyond a closed face the error is that of the cell inside
// (zero gradient), beyond an open one its negative (zero at the face); periodic wraps.
#include "pressure/common.wgsl"

@group(0) @binding(1) var PRSC: texture_3d<f32>;
@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var COEF: texture_3d<f32>;

fn coarse(q0: vec3<i32>) -> f32 {
  let nc = max(vec3<i32>(L.dims) / 2, vec3<i32>(1));
  var q = q0;
  var sgn = 1.0;
  for (var a = 0; a < 3; a++) {
    if (q[a] < 0) {
      if (L.lo[a] == PF_PERIODIC) { q[a] += nc[a]; } else { q[a] = 0; if (L.lo[a] == PF_OPEN) { sgn = -sgn; } }
    } else if (q[a] >= nc[a]) {
      if (L.hi[a] == PF_PERIODIC) { q[a] -= nc[a]; } else { q[a] = nc[a] - 1; if (L.hi[a] == PF_OPEN) { sgn = -sgn; } }
    }
  }
  return sgn * textureLoad(PRSC, q, 0).x;
}

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= L.dims)) { return; }
  let c = vec3<i32>(gid);
  // A cell inside an obstacle takes no correction.
  let k = textureLoad(COEF, c, 0);
  let flat = L.dims.z == 1u;
  let cc = vec3<i32>(c.xy / 2, select(c.z / 2, 0, flat));
  let d = 2 * (c % 2) - 1;   // direction of the second-nearest coarse cell along each axis
  let nk = select(8, 4, flat);
  var sum = 0.0;
  for (var i = 0; i < nk; i++) {
    let o = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    var w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(o));
    if (flat) { w.z = 1.0; }
    sum += w.x * w.y * w.z * coarse(vec3<i32>((cc + o * d).xy, select(cc.z + o.z * d.z, 0, flat)));
  }
  if (k.w <= 0.0 && all(k.xyz == vec3<f32>(0.0))) { return; }
  textureStore(PRS, c, vec4<f32>(textureLoad(PRS, c).x + sum, 0.0, 0.0, 0.0));
}
