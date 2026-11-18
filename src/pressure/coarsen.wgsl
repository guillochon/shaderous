// A coarse level's coefficients from the next finer level's: a coarse face takes the mean of
// the fine faces that tile it (four, or two in 2D), and alpha the mean of the children. A wall
// that covers part of a coarse face leaves it partly open, in proportion. L describes the
// coarse level.
#include "pressure/common.wgsl"

@group(0) @binding(1) var COEF: texture_3d<f32>;
@group(0) @binding(2) var COEFC: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, PWZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= L.dims)) { return; }
  let c = vec3<i32>(gid);
  let flat = L.dims.z == 1u;
  let f = vec3<i32>(2 * c.xy, select(2 * c.z, 0, flat));
  let nk = select(8, 4, flat);
  var out = vec4<f32>(0.0);
  for (var k = 0; k < nk; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let v = textureLoad(COEF, f + o, 0);
    // A child's upper face along an axis lies on the coarse cell's upper face if the child is
    // the upper one along that axis.
    out += vec4<f32>(v.xyz * vec3<f32>(o), v.w);
  }
  let faces = f32(nk) * 0.5;
  textureStore(COEFC, c, vec4<f32>(out.xyz / faces, out.w / f32(nk)));
}
