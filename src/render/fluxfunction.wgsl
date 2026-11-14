// Field lines of a 2D run (render/fieldlines.ts): contours of the flux function A_z, with
// B_x = dA/dy and B_y = -dA/dx. Between two contours flows the same magnetic flux, so the lines
// crowd where the field is strong, and in ideal MHD A_z moves with the gas: the lines follow it
// with nothing to choose or carry.
//
// A_z is integrated from the field-line volume's cell-centered field (render/bsample.wgsl,
// amr/bsample.wgsl) at the corners of its M x M cells, in units of the box side: first along the
// bottom row (`row`, one invocation), then up each column of corners (`cols`). The draw pass
// (`vs`, `fs`) lays the lines over the slice view, as flat tubes colored by |B|
// (render/tube_shade.wgsl): unlike the view's contour lines, which are thin and flat.
#include "render/camera.wgsl"
#include "render/tube_shade.wgsl"

@group(0) @binding(4) var BV: texture_3d<f32>;
// A_z at corner (i, j) of the volume's cells: A[j * (M + 1) + i].
@group(0) @binding(5) var<storage, read_write> A: array<f32>;
// The least and greatest A_z, as ordered keys (key() below).
@group(0) @binding(6) var<storage, read_write> AR: array<atomic<u32>, 2>;

struct Draw {
  n: vec4<f32>,    // x: contour levels; y: line width (pixels); z: 1 on a light background
}
@group(0) @binding(7) var<uniform> D: Draw;
// The largest |B| in the volume (render/bsample.wgsl), as float bits.
@group(0) @binding(8) var<storage, read> FR: array<u32, 4>;

fn key(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  return select(~b, b | 0x80000000u, (b & 0x80000000u) == 0u);
}
fn unkey(k: u32) -> f32 {
  return bitcast<f32>(select(~k, k & 0x7fffffffu, (k & 0x80000000u) != 0u));
}

fn side() -> u32 {
  return textureDimensions(BV).x;
}
fn bcell(i: u32, j: u32) -> vec2<f32> {
  let m = side();
  return textureLoad(BV, vec3<u32>(min(i, m - 1u), min(j, m - 1u), 0u), 0).xy;
}

fn note(a: f32) {
  atomicMin(&AR[0], key(a));
  atomicMax(&AR[1], key(a));
}

@compute @workgroup_size(1)
fn row() {
  let m = side();
  let h = 1.0 / f32(m);
  var a = 0.0;
  A[0] = 0.0;
  note(0.0);
  for (var i = 0u; i < m; i++) {
    // B_y on the bottom edge between corners i and i + 1.
    a -= bcell(i, 0u).y * h;
    A[i + 1u] = a;
    note(a);
  }
}

@compute @workgroup_size(64)
fn cols(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = side();
  let i = gid.x;
  if (i > m) { return; }
  let h = 1.0 / f32(m);
  let w = m + 1u;
  var a = A[i];
  for (var j = 0u; j < m; j++) {
    // B_x on the vertical edge from corner (i, j) to (i, j + 1): the mean of the cells beside it.
    let bx = 0.5 * (bcell(select(i - 1u, 0u, i == 0u), j).x + bcell(i, j).x);
    a += bx * h;
    A[(j + 1u) * w + i] = a;
    note(a);
  }
}

// A_z at a point of the slice (st in [0, 1]^2), bilinear between corners.
fn a_at(st: vec2<f32>) -> f32 {
  let m = f32(side());
  let w = side() + 1u;
  let g = clamp(st, vec2<f32>(0.0), vec2<f32>(1.0)) * m;
  let i0 = min(vec2<u32>(floor(g)), vec2<u32>(side() - 1u));
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
  let lo = unkey(atomicLoad(&AR[0]));
  let hi = unkey(atomicLoad(&AR[1]));
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
  let m = side();
  let cell = min(vec2<u32>(clamp(st, vec2<f32>(0.0), vec2<f32>(1.0)) * f32(m)), vec2<u32>(m - 1u));
  let b = length(textureLoad(BV, vec3<u32>(cell, 0u), 0).xy);
  let t = clamp(1.0 + 0.5 * log(max(b / max(bitcast<f32>(FR[0]), 1.0e-30), 1.0e-30)) * 0.4342944819, 0.0, 1.0);
  var c = flat_tube(across / r, g / gl, tube_color(t), abs(across) - r, D.n.z != 0.0);
  // Faded out where levels crowd closer than about twice the width (no haze of overlapping tubes).
  c.a *= 1.0 - smoothstep(0.1, 0.18, fw);
  return vec4<f32>(c.rgb, select(0.0, c.a, on));
}
