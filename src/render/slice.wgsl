// Slice and projection views: the 2D map the slice or projection pass made (render/project.wgsl),
// one screen texel block per cell of the finest level (no interpolation), for comparing against
// reference plots.
#include "render/camera.wgsl"

// r: the value as the transfer function reads it (-1 for a solid); g: 1 on a block's outline;
// b: log10 |value|.
@group(0) @binding(4) var MAP: texture_2d<f32>;

// A contour line at a pixel: how strongly the pixel lies on it, and the ink it is drawn in.
struct Line {
  a: f32,
  ink: vec3<f32>,
}

// The ink of the line at level k: dark if the level's own color is light, light if it is dark,
// so a line keeps one ink along its whole length (choosing by the cell under each pixel flips
// it from cell to cell where the colors sit near the threshold). `neg`: the line is where the
// value is minus the level (a contour of |value| in a field of either sign).
fn ink(k: f32, neg: bool) -> vec3<f32> {
  let lg = C.k.y + k * (C.k.z - C.k.y) / (C.k.x + 1.0);
  var x = select(pow(10.0, lg), -pow(10.0, lg), neg);
  if (C.sc.x != 0.0) { x = lg; }
  let t = clamp((x - C.sc.y) / max(C.sc.z - C.sc.y, 1.0e-20), 0.0, 1.0);
  let col = textureSampleLevel(TF, S, vec2<f32>(t, 0.5), 0.0).rgb;
  return select(vec3<f32>(0.95), vec3<f32>(0.05), dot(col, vec3<f32>(0.299, 0.587, 0.114)) > 0.5);
}

// How strongly a pixel lies on a contour line: C.k.x levels, evenly spaced in log10 |value| from
// C.k.y to C.k.z, the end levels left out (so n lines fall inside the range). log10 |value| is
// interpolated between the four nearest cell centers, and the line is about a pixel wide at any
// zoom. Lines are left out next to a solid, and fade where levels crowd within a few pixels.
// Called in uniform control flow (it takes derivatives).
fn contour(st: vec2<f32>) -> Line {
  let side = i32(textureDimensions(MAP).x);
  let g = st * f32(side) - 0.5;
  let i0 = vec2<i32>(floor(g));
  let fr = g - floor(g);
  let top = vec2<i32>(side - 1);
  let a = textureLoad(MAP, clamp(i0, vec2<i32>(0), top), 0);
  let b = textureLoad(MAP, clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0), top), 0);
  let c = textureLoad(MAP, clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0), top), 0);
  let d = textureLoad(MAP, clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0), top), 0);
  let lg = mix(mix(a.b, b.b, fr.x), mix(c.b, d.b, fr.x), fr.y);
  // In units of the spacing between levels: level k lies at f = k.
  let f = (lg - C.k.y) / max(C.k.z - C.k.y, 1.0e-6) * (C.k.x + 1.0);
  let fw = fwidth(f);
  let k = round(f);
  let px = abs(f - k) / max(fw, 1.0e-6);   // pixels to the nearest level
  let gas = min(min(a.r, b.r), min(c.r, d.r)) > -0.5;
  let on = gas && C.k.x > 0.0 && k >= 1.0 && k <= C.k.x;
  // Which side of zero (on a linear scale, where the field may take either sign).
  let t = mix(mix(a.r, b.r, fr.x), mix(c.r, d.r, fr.x), fr.y);
  let zero = -C.sc.y / max(C.sc.z - C.sc.y, 1.0e-20);
  let neg = C.sc.x == 0.0 && t < zero;
  let alpha = select(0.0, (1.0 - smoothstep(0.5, 1.25, px)) * (1.0 - smoothstep(0.2, 0.4, fw)), on);
  return Line(alpha, ink(k, neg));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // The square laid out by the app (sliceLayout), in units of its width from its middle.
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  // Zoom and pan: C.r.zw is the point of the slice at the middle of that square.
  let st = st0 / C.r.y + C.r.zw;
  let line = contour(st);
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(BG, 1.0); }
  let m = textureLoad(MAP, vec2<i32>(floor(st * f32(textureDimensions(MAP).x))), 0);
  var col = textureSampleLevel(TF, S, vec2<f32>(m.r, 0.5), 0.0).rgb;
  if (m.r < -0.5) { col = SOLID; }
  // Block outlines of an adaptive mesh.
  col = mix(col, vec3<f32>(0.75, 0.8, 0.9), 0.3 * m.g * C.r.x);
  col = mix(col, line.ink, 0.85 * line.a);
  return vec4<f32>(col, 1.0);
}
