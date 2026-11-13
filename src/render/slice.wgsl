// Slice and projection views: the 2D map the slice or projection pass made (render/project.wgsl),
// one screen texel block per cell of the finest level (no interpolation), for comparing against
// reference plots.
#include "render/camera.wgsl"

// r: the value as the transfer function reads it (-1 for a solid); g: 1 on a block's outline.
@group(0) @binding(4) var MAP: texture_2d<f32>;

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // The square laid out by the app (sliceLayout), in units of its width from its middle.
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  // Zoom and pan: C.r.zw is the point of the slice at the middle of that square.
  let st = st0 / C.r.y + C.r.zw;
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(BG, 1.0); }
  let m = textureLoad(MAP, vec2<i32>(floor(st * f32(textureDimensions(MAP).x))), 0);
  var col = textureSampleLevel(TF, S, vec2<f32>(m.r, 0.5), 0.0).rgb;
  if (m.r < -0.5) { col = SOLID; }
  // Block outlines of an adaptive mesh.
  col = mix(col, vec3<f32>(0.75, 0.8, 0.9), 0.3 * m.g * C.r.x);
  return vec4<f32>(col, 1.0);
}
