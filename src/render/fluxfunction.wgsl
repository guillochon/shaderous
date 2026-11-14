// Field lines of a 2D run (render/fieldlines.ts): contours of the flux function A_z, with
// B_x = dA/dy and B_y = -dA/dx. Between two contours flows the same magnetic flux, so the lines
// crowd where the field is strong, and in ideal MHD A_z moves with the gas: the lines follow it
// with nothing to choose or carry. A_z comes from the face field (render/aflux.wgsl), at the
// corners of an M x M grid. This draw pass (`vs`, `fs`) lays the lines over the slice view, as
// flat tubes colored by |B| (render/tube_shade.wgsl): unlike the view's contour lines, which are
// thin and flat.
#include "render/camera.wgsl"
#include "render/tube_shade.wgsl"

// The field-line volume (render/bsample.wgsl, amr/bsample.wgsl): |B| for the lines' color.
@group(0) @binding(4) var BV: texture_3d<f32>;
// A_z at corner (i, j): A[j * (M + 1) + i].
@group(0) @binding(5) var<storage, read> A: array<f32>;
// The least and greatest A_z, as ordered keys (render/aflux.wgsl, key()).
@group(0) @binding(6) var<storage, read> AR: array<u32, 2>;

struct Draw {
  n: vec4<f32>,    // x: contour levels; y: line width (pixels); z: 1 on a light background; w: M
}
@group(0) @binding(7) var<uniform> D: Draw;
// The largest |B| in the volume (render/bsample.wgsl), as float bits.
@group(0) @binding(8) var<storage, read> FR: array<u32, 4>;

fn unkey(k: u32) -> f32 {
  return bitcast<f32>(select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u));
}

// A_z at a point of the slice (st in [0, 1]^2), bilinear between corners.
fn a_at(st: vec2<f32>) -> f32 {
  let mi = u32(D.n.w);
  let w = mi + 1u;
  let g = clamp(st, vec2<f32>(0.0), vec2<f32>(1.0)) * f32(mi);
  let i0 = min(vec2<u32>(floor(g)), vec2<u32>(mi - 1u));
  let f = g - vec2<f32>(i0);
  let a00 = A[i0.y * w + i0.x];
  let a10 = A[i0.y * w + i0.x + 1u];
  let a01 = A[(i0.y + 1u) * w + i0.x];
  let a11 = A[(i0.y + 1u) * w + i0.x + 1u];
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  // As the slice view maps the canvas (render/slice.wgsl).
  let st0 = vec2<f32>(in.uv.x * C.right.w + C.eye.w, in.uv.y * C.up.w);
  let st = st0 / C.r.y + C.r.zw;
  let inside = all(st >= vec2<f32>(0.0)) && all(st <= vec2<f32>(1.0));
  let lo = unkey(AR[0]);
  let hi = unkey(AR[1]);
  let n = max(D.n.x, 1.0);
  // Levels n inside the range, its ends left out: level k at f = k.
  let f = (a_at(st) - lo) / max(hi - lo, 1.0e-30) * (n + 1.0);
  let fw = fwidth(f);
  let k = round(f);
  let px = abs(f - k) / max(fw, 1.0e-9);
  // (Derivatives are taken first: outside the map is decided after them, in uniform control flow.)
  let on = inside && k >= 1.0 && k <= n && hi > lo;
  // Across the line: signed pixels from its middle, and the direction on the screen.
  let g = vec2<f32>(dpdx(f), dpdy(f));
  let gl = max(length(g), 1.0e-9);
  let across = (f - k) / gl;
  let r = 0.5 * D.n.y;
  // Strength: |B| here, over two decades below the largest.
  let m = textureDimensions(BV).x;
  let cell = min(vec2<u32>(clamp(st, vec2<f32>(0.0), vec2<f32>(1.0)) * f32(m)), vec2<u32>(m - 1u));
  let b = length(textureLoad(BV, vec3<u32>(cell, 0u), 0).xy);
  let t = clamp(1.0 + 0.5 * log(max(b / max(bitcast<f32>(FR[0]), 1.0e-30), 1.0e-30)) * 0.4342944819, 0.0, 1.0);
  var c = flat_tube(across / r, g / gl, tube_color(t), abs(across) - r, D.n.z != 0.0);
  // Faded out where levels crowd closer than about twice the width (no haze of overlapping tubes).
  c.a *= 1.0 - smoothstep(0.1, 0.18, fw);
  return vec4<f32>(c.rgb, select(0.0, c.a, on));
}
