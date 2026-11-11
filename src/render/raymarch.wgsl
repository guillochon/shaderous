// Emission-absorption ray-marcher through the unit box, front to back, with a 1D transfer
// function (color and opacity against the normalized field value).
#include "render/camera.wgsl"

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// 1 where the point lies on an edge of the box (two coordinates at the faces).
fn edge(p: vec3<f32>, w: f32) -> f32 {
  let a = step(vec3<f32>(0.5 - w), abs(p));
  return step(1.5, a.x + a.y + a.z);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let ro = C.eye.xyz;
  let rd = normalize(C.fwd.xyz + in.uv.x * C.p.y * C.p.x * C.right.xyz + in.uv.y * C.p.x * C.up.xyz);

  // Slab intersection with [-0.5, 0.5]^3.
  let inv = 1.0 / rd;
  let ta = (vec3<f32>(-0.5) - ro) * inv;
  let tb = (vec3<f32>(0.5) - ro) * inv;
  let tlo = min(ta, tb);
  let thi = max(ta, tb);
  let t0 = max(max(tlo.x, max(tlo.y, tlo.z)), 0.0);
  let t1 = min(thi.x, min(thi.y, thi.z));
  if (t1 <= t0) { return vec4<f32>(BG, 1.0); }

  let ds = 1.0 / C.p.w;
  let count = min(i32(ceil((t1 - t0) / ds)), 1024);
  let jitter = hash(in.pos.xy);
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  for (var i = 0; i < count; i++) {
    let p = ro + rd * (t0 + (f32(i) + jitter) * ds);
    let v = textureSampleLevel(VOL, S, p + 0.5, 0.0).r;
    if (v < -0.5) {
      // The surface of an obstacle: opaque, darker with depth so its shape reads.
      col += (1.0 - alpha) * SOLID * (1.15 - 0.6 * f32(i) / f32(count));
      alpha = 1.0;
      break;
    }
    let tf = textureSampleLevel(TF, S, vec2<f32>(v, 0.5), 0.0);
    let a = 1.0 - exp(-tf.a * C.p.z * ds);
    col += (1.0 - alpha) * a * tf.rgb;
    alpha += (1.0 - alpha) * a;
    if (alpha > 0.995) { break; }
  }
  col += (1.0 - alpha) * BG;

  // Box outline: far edges are dimmed by the gas in front of them.
  let w = 0.004;
  let near = edge(ro + rd * t0, w);
  let far = edge(ro + rd * t1, w) * (1.0 - alpha);
  col = mix(col, vec3<f32>(0.45, 0.5, 0.6), 0.6 * max(near, far));
  return vec4<f32>(col, 1.0);
}
