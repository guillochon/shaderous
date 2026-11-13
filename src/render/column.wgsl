// Projection view: the map the projection pass (render/project.wgsl) made, one screen texel
// block per column, laid out, zoomed and panned like the slice view.
#include "render/camera.wgsl"

@group(0) @binding(4) var PROJ: texture_2d<f32>;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  let st = st0 / C.r.y + C.r.zw;
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(BG, 1.0); }
  let t = textureLoad(PROJ, vec2<i32>(floor(st * C.q.z)), 0).r;
  var col = textureSampleLevel(TF, S, vec2<f32>(t, 0.5), 0.0).rgb;
  if (t < -0.5) { col = SOLID; }
  return vec4<f32>(col, 1.0);
}
