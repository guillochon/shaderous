// Axis-aligned slice, one screen texel block per cell (no interpolation), for comparing
// against reference plots.
#include "render/camera.wgsl"

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // The square laid out by the app (sliceLayout), in units of its width from its middle.
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  // Zoom and pan: C.r.zw is the point of the slice at the middle of that square.
  let st = st0 / C.r.y + C.r.zw;
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(BG, 1.0); }

  let n = C.q.z;
  let ij = vec2<i32>(floor(st * n));
  let axis = i32(C.q.x);
  // Depth of the grid along the slice normal (1 along z in a 2D run).
  let depth = select(n, C.q.w, axis == 2);
  let k = i32(clamp(floor(C.q.y * depth), 0.0, depth - 1.0));
  var c = vec3<i32>(ij.x, ij.y, k);            // z-normal: x right, y up
  if (axis == 0) { c = vec3<i32>(k, ij.x, ij.y); }   // x-normal: y right, z up
  if (axis == 1) { c = vec3<i32>(ij.x, k, ij.y); }   // y-normal: x right, z up
  let t = textureLoad(VOL, c, 0);
  var col = textureSampleLevel(TF, S, vec2<f32>(t.r, 0.5), 0.0).rgb;
  if (t.r < -0.5) { col = SOLID; }
  // Block outlines of an adaptive mesh: g, b, a mark block edges for z-, x- and y-normal slices.
  var edge = t.g;
  if (axis == 0) { edge = t.b; }
  if (axis == 1) { edge = t.a; }
  col = mix(col, vec3<f32>(0.75, 0.8, 0.9), 0.3 * edge * C.r.x);
  return vec4<f32>(col, 1.0);
}
