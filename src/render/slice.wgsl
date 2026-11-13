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

// The contour under the pointer (Renderer.hoverLevel): drawn wider, in a color of its own.
const HOVER_INK = vec3<f32>(0.2, 0.9, 1.0);

// The ink of the line at level k: dark if the level's own color is light, light if it is dark,
// so a line keeps one ink along its whole length (choosing by the cell under each pixel flips
// it from cell to cell where the colors sit near the threshold). `neg`: the line is where the
// value is minus the level (a contour of |value| in a field of either sign).
fn ink(k: f32, neg: bool) -> vec3<f32> {
  let col = level_color(k, neg);
  return select(vec3<f32>(0.95), vec3<f32>(0.05), dot(col, vec3<f32>(0.299, 0.587, 0.114)) > 0.5);
}

// How strongly a pixel lies on a contour line: C.k.x levels, evenly spaced in log10 |value| from
// C.k.y to C.k.z, the end levels left out (so n lines fall inside the range). log10 |value| is
// smoothed over the 4 x 4 nearest cell centers with a cubic B-spline, whose slope is continuous
// across cell edges (bilinear interpolation bends every contour at each edge, scalloping it) and
// which never overshoots (so no ghost lines beside a shock); it may move a line by a fraction of
// a cell and smooth away bumps one cell wide. The cells are those of the grid, not the map's
// texels: on an adaptive mesh a coarse cell covers many texels of one value (the map's a holds
// its side), and the spline reads the centers of the cells the size of the one under the pixel,
// so a line may kink slightly where the level changes. The line is about a pixel wide at any
// zoom. Lines are left out within two cells of a solid, and fade where levels crowd within a few
// pixels. Called in uniform control flow (it takes derivatives). Renderer.contourAt() mirrors it.
fn bspline(t: f32) -> vec4<f32> {
  let s = 1.0 - t;
  return vec4<f32>(s * s * s, 3.0 * t * t * t - 6.0 * t * t + 4.0, -3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0, t * t * t) / 6.0;
}

fn contour(st: vec2<f32>) -> Line {
  let side = i32(textureDimensions(MAP).x);
  let top = vec2<i32>(side - 1);
  // The side of the cell under the pixel, in texels.
  let cell = max(textureLoad(MAP, clamp(vec2<i32>(floor(st * f32(side))), vec2<i32>(0), top), 0).a, 1.0);
  // Position in cells of that size, from the first cell's center.
  let g = st * f32(side) / cell - 0.5;
  let i0 = floor(g);
  let fr = g - i0;
  let wx = bspline(fr.x);
  let wy = bspline(fr.y);
  var lg = 0.0;
  var t = 0.0;     // the color scale's value, smoothed the same way
  var low = 0.0;   // the least r of the 16: below -0.5 if any is solid
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      // The texel at the center of cell (i0 + (i - 1, j - 1)).
      let at = vec2<i32>(floor((i0 + vec2<f32>(f32(i - 1), f32(j - 1)) + 0.5) * cell));
      let m = textureLoad(MAP, clamp(at, vec2<i32>(0), top), 0);
      lg += wx[i] * wy[j] * m.b;
      t += wx[i] * wy[j] * m.r;
      low = min(low, m.r);
    }
  }
  // In units of the spacing between levels: level k lies at f = k.
  let f = (lg - C.k.y) / max(C.k.z - C.k.y, 1.0e-6) * (C.k.x + 1.0);
  let fw = fwidth(f);
  let k = round(f);
  let px = abs(f - k) / max(fw, 1.0e-6);   // pixels to the nearest level
  let gas = low > -0.5;
  let on = gas && C.k.x > 0.0 && k >= 1.0 && k <= C.k.x;
  // Which side of zero (on a linear scale, where the field may take either sign).
  let zero = -C.sc.y / max(C.sc.z - C.sc.y, 1.0e-20);
  let neg = C.sc.x == 0.0 && t < zero;
  let hot = k == C.k.w;
  let width = select(vec2<f32>(0.5, 1.25), vec2<f32>(1.0, 2.0), hot);
  let crowd = select(1.0 - smoothstep(0.2, 0.4, fw), 1.0, hot);
  let alpha = select(0.0, (1.0 - smoothstep(width.x, width.y, px)) * crowd, on);
  return Line(select(alpha, alpha / 0.85, hot), select(ink(k, neg), HOVER_INK, hot));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // The square laid out by the app (sliceLayout), in units of its width from its middle.
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  // Zoom and pan: C.r.zw is the point of the slice at the middle of that square.
  let st = st0 / C.r.y + C.r.zw;
  let line = contour(st);
  if (any(st < vec2<f32>(0.0)) || any(st >= vec2<f32>(1.0))) { return vec4<f32>(C.bg.rgb, 1.0); }
  let m = textureLoad(MAP, vec2<i32>(floor(st * f32(textureDimensions(MAP).x))), 0);
  var col = textureSampleLevel(TF, S, vec2<f32>(m.r, 0.5), 0.0).rgb;
  if (m.r < -0.5) { col = SOLID; }
  // Block outlines of an adaptive mesh.
  col = mix(col, vec3<f32>(0.75, 0.8, 0.9), 0.3 * m.g * C.r.x);
  col = mix(col, line.ink, 0.85 * line.a);
  return vec4<f32>(col, 1.0);
}
