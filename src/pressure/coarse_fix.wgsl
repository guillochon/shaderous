// The coarsest level, after its sweeps: where no face of the box is open, the operator's flux
// terms sum to zero over the level, so the sum of the equation fixes the mean of the pressure,
//   sum(alpha p) = sum(b).
// Gauss-Seidel moves that mean only at the rate alpha h^2, which is nothing when the sound
// crosses many cells in a step; this sets it outright. One invocation: the level has a few cells.
#include "pressure/common.wgsl"

@group(0) @binding(1) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var RHS: texture_3d<f32>;
@group(0) @binding(3) var COEF: texture_3d<f32>;

@compute @workgroup_size(1)
fn main() {
  if (L.fix_mean == 0u) { return; }
  let d = vec3<i32>(L.dims);
  var sa = 0.0;
  var sb = 0.0;
  for (var z = 0; z < d.z; z++) {
    for (var y = 0; y < d.y; y++) {
      for (var x = 0; x < d.x; x++) {
        let c = vec3<i32>(x, y, z);
        let a = textureLoad(COEF, c, 0).w;
        sa += a;
        sb += textureLoad(RHS, c, 0).x - a * textureLoad(PRS, c).x;
      }
    }
  }
  if (sa <= 0.0) { return; }
  let shift = sb / sa;
  for (var z = 0; z < d.z; z++) {
    for (var y = 0; y < d.y; y++) {
      for (var x = 0; x < d.x; x++) {
        let c = vec3<i32>(x, y, z);
        if (textureLoad(COEF, c, 0).w > 0.0) { textureStore(PRS, c, vec4<f32>(textureLoad(PRS, c).x + shift, 0.0, 0.0, 0.0)); }
      }
    }
  }
}
