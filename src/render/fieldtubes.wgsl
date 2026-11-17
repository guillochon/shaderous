// Drawing the field lines traced by render/fieldlines.wgsl.
//
// Volume view: each segment between two points of a line is a capsule (a cylinder with round
// ends, so a line bends without gaps), drawn on a camera-facing quad that covers it. The
// fragment finds where the eye's ray meets the capsule, shades it with the isosurfaces' lights,
// and writes the depth for the tubes among themselves, and the distance along the ray for the
// ray-marcher (render/raymarch.wgsl), which draws the gas in front of a tube over it.
//
// Slice view of a 3D run: each segment is a flat tube a few pixels wide over the map, colored and
// shaded as the volume view's tubes are (render/tube_shade.wgsl).
#include "render/camera.wgsl"
#include "render/capsule.wgsl"
#include "render/tube_shade.wgsl"

struct Draw {
  n: vec4<f32>,    // x: points per line; y: tube radius (box units); z: line width (pixels); w: plane's normal axis
  px: vec4<f32>,   // canvas size in pixels; z: 1 on a light background; w: device pixels per CSS pixel
}
@group(0) @binding(4) var<storage, read> PTS: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> FR: array<u32, 4>;
@group(0) @binding(6) var<uniform> D: Draw;

// The two ends of segment s (its line's point k and k + 1), or ok = false past a line's ends.
struct Seg {
  ok: bool,
  a: vec4<f32>,
  b: vec4<f32>,
}
fn segment(s: u32) -> Seg {
  let n = u32(D.n.x);
  let k = s % n;
  if (k + 1u >= n) { return Seg(false, vec4<f32>(0.0), vec4<f32>(0.0)); }
  let a = PTS[s];
  let b = PTS[s + 1u];
  return Seg(a.w >= 0.0 && b.w >= 0.0, a, b);
}

// |B| as a fraction of the way from the least to the greatest on the lines, in log (the same
// fraction in log B^2), over at most TUBE_DECADES of B^2 below the greatest
// (render/tube_shade.wgsl).
fn strength(b: f32) -> f32 {
  let hi = log(max(bitcast<f32>(FR[2]), 1.0e-30));
  let lo = max(log(max(bitcast<f32>(FR[1]), 1.0e-30)), hi - 0.5 * TUBE_DECADES * 2.302585093);
  return clamp((log(max(b, 1.0e-30)) - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0);
}

struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) @interpolate(flat) a: vec4<f32>,
  @location(2) @interpolate(flat) b: vec4<f32>,
  @location(3) @interpolate(flat) r: f32,
}

// Least tube radius: 0.6 of a CSS pixel at the segment's distance, so a far or thin tube stays
// a continuous line rather than breaking into the pixels whose centers it happens to cover.
fn tube_radius(mid: vec3<f32>) -> f32 {
  let w = max(dot(mid - C.eye.xyz, C.fwd.xyz), 1.0e-6);
  let pixel = 2.0 * w * C.p.x / max(D.px.y, 1.0);
  return max(D.n.y, 0.6 * max(D.px.w, 1.0) * pixel);
}

@vertex
fn tube_vs(@builtin(vertex_index) v: u32, @builtin(instance_index) s: u32) -> TOut {
  let g = segment(s);
  if (!g.ok) { return TOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec3<f32>(0.0), g.a, g.b, 0.0); }
  let mid = 0.5 * (g.a.xyz + g.b.xyz);
  let r = tube_radius(mid);
  let view = normalize(mid - C.eye.xyz);
  // The segment's direction, in the plane facing the camera; the quad spans it plus a radius
  // (with a margin for perspective) on every side.
  let d = g.b.xyz - g.a.xyz;
  var along = d - view * dot(d, view);
  if (length(along) < 1.0e-9) { along = C.right.xyz; }
  along = normalize(along);
  let side = normalize(cross(view, along));
  let q = QUAD[v];
  let e = 1.3 * r;
  let p = mix(g.a.xyz - along * e, g.b.xyz + along * e, q.x) + side * e * q.y;
  return TOut(project(p), p, g.a, g.b, r);
}

struct TFrag {
  @location(0) color: vec4<f32>,
  @location(1) dist: vec4<f32>,   // distance along the ray, in r
  @builtin(frag_depth) depth: f32,
}

@fragment
fn tube_fs(in: TOut) -> TFrag {
  let ro = C.eye.xyz;
  let rd = normalize(in.world - ro);
  let t = capsule(ro, rd, in.a.xyz, in.b.xyz, in.r);
  if (t <= 0.0) { discard; }
  let p = ro + rd * t;
  let ba = in.b.xyz - in.a.xyz;
  let y = clamp(dot(p - in.a.xyz, ba) / max(dot(ba, ba), 1.0e-20), 0.0, 1.0);
  let nrm = normalize(p - (in.a.xyz + y * ba));
  let base = tube_color(strength(mix(in.a.w, in.b.w, y)));
  let key = normalize(-rd + 0.6 * C.up.xyz + 0.4 * C.right.xyz);
  let diff = max(dot(nrm, key), 0.0);
  let fill = max(dot(nrm, -rd), 0.0);
  let spec = pow(max(dot(nrm, normalize(key - rd)), 0.0), 40.0);
  let col = base * (0.2 + 0.6 * diff + 0.3 * fill) + 0.3 * spec;
  let w = dot(p - ro, C.fwd.xyz);
  return TFrag(vec4<f32>(col, 1.0), vec4<f32>(t, 0.0, 0.0, 0.0), w / (w + 1.0));
}

// Slice view: a point of the plane (box units) on the canvas, in clip coordinates. Inverts the
// slice view's mapping (render/slice.wgsl, fs).
fn slice_clip(p: vec3<f32>) -> vec2<f32> {
  let ax = i32(D.n.w);
  var ab = vec2<i32>(0, 1);
  if (ax == 0) { ab = vec2<i32>(1, 2); }
  if (ax == 1) { ab = vec2<i32>(0, 2); }
  let st = vec2<f32>(p[ab.x], p[ab.y]) + 0.5;
  let st0 = (st - C.r.zw) * C.r.y;
  return vec2<f32>((st0.x - C.eye.w) / C.right.w, st0.y / C.up.w);
}

struct LOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) across: f32,   // pixels from the line's middle, signed
  @location(1) b: f32,        // |B|
  @location(2) @interpolate(flat) n: vec2<f32>,   // across the line, on the screen (y down)
}

@vertex
fn flat_vs(@builtin(vertex_index) v: u32, @builtin(instance_index) s: u32) -> LOut {
  let g = segment(s);
  if (!g.ok) { return LOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), 0.0, 0.0, vec2<f32>(0.0)); }
  let a = slice_clip(g.a.xyz);
  let b = slice_clip(g.b.xyz);
  // In pixels, so the width is the same in every direction.
  let half_px = 0.5 * D.px.xy;
  let d = (b - a) * half_px;
  let len = length(d);
  let dir = select(vec2<f32>(1.0, 0.0), d / len, len > 1.0e-6);
  let nrm = vec2<f32>(-dir.y, dir.x);
  // Half the tube's width plus room for its outline.
  let w = 0.5 * D.n.z + 2.0;
  let q = QUAD[v];
  // Extended by half a width at each end, so consecutive segments overlap at the joints.
  let px = mix(a * half_px - dir * 0.5 * w, b * half_px + dir * 0.5 * w, q.x) + nrm * w * q.y;
  return LOut(vec4<f32>(px / half_px, 0.0, 1.0), q.y * w, mix(g.a.w, g.b.w, q.x), vec2<f32>(nrm.x, -nrm.y));
}

@fragment
fn flat_fs(in: LOut) -> @location(0) vec4<f32> {
  let r = 0.5 * D.n.z;
  return flat_tube(in.across / r, in.n, tube_color(strength(in.b)), abs(in.across) - r, D.px.z != 0.0);
}
