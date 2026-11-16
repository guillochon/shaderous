// Volume view: the path a point mass being placed would take (gravity/preview.ts), as a tube
// with a cone for an arrowhead. Drawn like the field lines' tubes (render/fieldtubes.wgsl) and
// into the same targets: color, distance along the ray for the ray-marcher (which draws the gas
// in front of the tube over it), and depth, which sorts it among the field lines.
#include "render/camera.wgsl"
#include "render/capsule.wgsl"

struct Path {
  n: vec4<f32>,      // x: points of the tube; y: its radius (box units); z: radius of the cone's base
  base: vec4<f32>,   // center of the cone's base, where the tube ends
  tip: vec4<f32>,    // the cone's tip: the end of the path
  color: vec4<f32>,
}
// xyz in box units ([-0.5, 0.5]^3); w: how far along the whole path, 0 to 1. After the path
// come those of the run's other masses, drawn thinner and dimmer: their points have w = 2, and
// a point with w = -1 stands between one path and the next (no tube is drawn to or from it).
@group(0) @binding(4) var<storage, read> PTS: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> D: Path;

struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) @interpolate(flat) a: vec4<f32>,
  @location(2) @interpolate(flat) b: vec4<f32>,
}

// A camera-facing quad covering the stretch from a to b and `r` to every side of it (with a
// margin for perspective).
fn cover(a: vec3<f32>, b: vec3<f32>, r: f32, v: u32) -> vec3<f32> {
  let view = normalize(0.5 * (a + b) - C.eye.xyz);
  let d = b - a;
  var along = d - view * dot(d, view);
  if (length(along) < 1.0e-9) { along = C.right.xyz; }
  along = normalize(along);
  let side = normalize(cross(view, along));
  let q = QUAD[v];
  let e = 1.3 * r;
  return mix(a - along * e, b + along * e, q.x) + side * e * q.y;
}

// The radius of the tube through a point: the path's own, or less for another mass's.
fn tube_radius(w: f32) -> f32 {
  return select(D.n.y, 0.45 * D.n.y, w >= 2.0);
}

@vertex
fn seg_vs(@builtin(vertex_index) v: u32, @builtin(instance_index) s: u32) -> TOut {
  let a = PTS[s];
  let b = PTS[s + 1u];
  if (a.w < 0.0 || b.w < 0.0) { return TOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec3<f32>(0.0), a, b); }
  let p = cover(a.xyz, b.xyz, tube_radius(a.w), v);
  return TOut(project(p), p, a, b);
}

@vertex
fn head_vs(@builtin(vertex_index) v: u32) -> TOut {
  let p = cover(D.base.xyz, D.tip.xyz, D.n.z, v);
  return TOut(project(p), p, D.base, D.tip);
}

struct TFrag {
  @location(0) color: vec4<f32>,
  @location(1) dist: vec4<f32>,   // distance along the ray, in r
  @builtin(frag_depth) depth: f32,
}

// Lit as the field lines' tubes are; the path dims toward its far end.
fn shade(ro: vec3<f32>, rd: vec3<f32>, t: f32, nrm: vec3<f32>, along: f32) -> TFrag {
  let base = D.color.rgb * mix(1.0, 0.6, along);
  let key = normalize(-rd + 0.6 * C.up.xyz + 0.4 * C.right.xyz);
  let diff = max(dot(nrm, key), 0.0);
  let fill = max(dot(nrm, -rd), 0.0);
  let spec = pow(max(dot(nrm, normalize(key - rd)), 0.0), 40.0);
  let col = base * (0.25 + 0.6 * diff + 0.3 * fill) + 0.25 * spec;
  let w = dot(rd * t, C.fwd.xyz);
  return TFrag(vec4<f32>(col, 1.0), vec4<f32>(t, 0.0, 0.0, 0.0), w / (w + 1.0));
}

@fragment
fn seg_fs(in: TOut) -> TFrag {
  let ro = C.eye.xyz;
  let rd = normalize(in.world - ro);
  let t = capsule(ro, rd, in.a.xyz, in.b.xyz, tube_radius(in.a.w));
  if (t <= 0.0) { discard; }
  let p = ro + rd * t;
  let ba = in.b.xyz - in.a.xyz;
  let y = clamp(dot(p - in.a.xyz, ba) / max(dot(ba, ba), 1.0e-20), 0.0, 1.0);
  // Another mass's path is dimmer than the far end of the path itself.
  let along = select(mix(in.a.w, in.b.w, y), 1.6, in.a.w >= 2.0);
  return shade(ro, rd, t, normalize(p - (in.a.xyz + y * ba)), along);
}

// Where the ray from ro along rd (unit) first meets the cone with its base (radius ra) at pa
// and its tip at pb: the distance in w (-1 for a miss) and the surface's normal there. In
// units of the cone's length, so that the squares below stay within float32's reach.
fn cone(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, ra: f32) -> vec4<f32> {
  let len = length(pb - pa);
  let ba = (pb - pa) / len;   // unit: the lengths below are in cone lengths
  let oa = (ro - pa) / len;
  let r = ra / len;
  let m1 = dot(oa, ba);
  let m2 = dot(rd, ba);
  let m3 = dot(rd, oa);
  let m5 = dot(oa, oa);
  // The base: seen from behind it.
  if (m1 < 0.0 && m2 > 0.0) {
    let h = oa * m2 - rd * m1;
    if (dot(h, h) < r * r * m2 * m2) { return vec4<f32>(-ba, -m1 / m2 * len); }
  }
  // The side.
  let hy = 1.0 + r * r;
  let k2 = 1.0 - m2 * m2 * hy;
  let k1 = m3 - m1 * m2 * hy + r * r * m2;
  let k0 = m5 - m1 * m1 * hy + r * (r * m1 * 2.0 - r);
  let h = k1 * k1 - k2 * k0;
  if (h < 0.0 || abs(k2) < 1.0e-12) { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
  let t = (-k1 - sqrt(h)) / k2;
  let y = m1 + t * m2;
  if (t <= 0.0 || y < 0.0 || y > 1.0) { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }
  return vec4<f32>(normalize(oa + t * rd + r * r * ba - ba * hy * y), t * len);
}

@fragment
fn head_fs(in: TOut) -> TFrag {
  let ro = C.eye.xyz;
  let rd = normalize(in.world - ro);
  let hit = cone(ro, rd, in.a.xyz, in.b.xyz, D.n.z);
  if (hit.w <= 0.0) { discard; }
  return shade(ro, rd, hit.w, hit.xyz, 1.0);
}
