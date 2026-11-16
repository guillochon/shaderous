// Tubes in the volume view (render/fieldtubes.wgsl, render/pathtube.wgsl): each stretch of a
// line is a capsule drawn on a camera-facing quad that covers it. Requires C (render/camera.wgsl).

// The camera's projection (render/lines.wgsl), with a depth inside the clip range.
fn project(p: vec3<f32>) -> vec4<f32> {
  let d = p - C.eye.xyz;
  let w = dot(d, C.fwd.xyz);
  return vec4<f32>(dot(d, C.right.xyz) / (C.p.x * C.p.y), dot(d, C.up.xyz) / C.p.x, 0.5 * w, w);
}

var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));

// Where the ray from ro along rd (unit) first meets the capsule from pa to pb of radius r, or -1.
fn capsule(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, r: f32) -> f32 {
  let ba = pb - pa;
  let oa = ro - pa;
  let baba = dot(ba, ba);
  let bard = dot(ba, rd);
  let baoa = dot(ba, oa);
  let rdoa = dot(rd, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - r * r * baba;
  var h = b * b - a * c;
  if (h >= 0.0 && a > 1.0e-20) {
    let t = (-b - sqrt(h)) / a;
    let y = baoa + t * bard;
    if (y > 0.0 && y < baba) { return t; }
  }
  // The round ends.
  var best = -1.0;
  for (var e = 0; e < 2; e++) {
    let oc = select(ro - pb, oa, e == 0);
    let b2 = dot(rd, oc);
    let c2 = dot(oc, oc) - r * r;
    let h2 = b2 * b2 - c2;
    if (h2 > 0.0) {
      let t = -b2 - sqrt(h2);
      if (t > 0.0 && (best < 0.0 || t < best)) { best = t; }
    }
  }
  return best;
}
