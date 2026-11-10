// Axis-aligned slice, one screen texel block per cell (no interpolation), for comparing
// against reference plots.
#include "render/camera.wgsl"

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // Largest centred square that fits, with a small margin.
  let fit = 0.94;
  var st = in.uv;
  if (C.p.y > 1.0) { st.x *= C.p.y; } else { st.y /= C.p.y; }
  st = st / fit * 0.5 + 0.5;
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(BG, 1.0); }

  let n = C.q.z;
  let ij = vec2<i32>(floor(st * n));
  let k = i32(clamp(floor(C.q.y * n), 0.0, n - 1.0));
  var c = vec3<i32>(ij.x, ij.y, k);            // z-normal: x right, y up
  let axis = i32(C.q.x);
  if (axis == 0) { c = vec3<i32>(k, ij.x, ij.y); }   // x-normal: y right, z up
  if (axis == 1) { c = vec3<i32>(ij.x, k, ij.y); }   // y-normal: x right, z up
  let v = textureLoad(VOL, c, 0).r;
  return vec4<f32>(textureSampleLevel(TF, S, vec2<f32>(v, 0.5), 0.0).rgb, 1.0);
}
