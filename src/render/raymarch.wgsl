// Emission-absorption ray-marcher through the unit box, front to back, with a 1D transfer
// function (color and opacity against the normalized field value). With contours on it also
// draws isosurfaces at their levels (C.k, as the slice view's lines), lit, in the color of their
// level, C.fwd.w opaque; C.sc.w 0 leaves the fog out, so the surfaces show alone.
// What it marches through comes from the <volume> slot: the uniform display volume
// (render/volume_uniform.wgsl) or an adaptive mesh's leaf blocks (render/volume_atlas.wgsl).
#include "render/camera.wgsl"
#include <volume>

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// 1 where the point lies on an edge of the box (two coordinates at the faces).
fn edge(p: vec3<f32>, w: f32) -> f32 {
  let a = step(vec3<f32>(0.5 - w), abs(p));
  return step(1.5, a.x + a.y + a.z);
}

// Volume view's isosurfaces: in units of the spacing between levels, level k lies at f = k.
fn iso_f(p: vec3<f32>) -> f32 {
  return (vol_look(p).s.b - C.k.y) * (C.k.x + 1.0) / max(C.k.z - C.k.y, 1.0e-6);
}

// The isosurface through p (in the box, [-0.5, 0.5]^3) seen along rd, lit: a key light above
// and to the right of the camera, a fill from the camera, and a highlight.
fn iso_shade(p: vec3<f32>, rd: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
  // Differences across the cell the surface passes through.
  let e = vol_look(p).cell;
  let gr = vec3<f32>(
    iso_f(p + vec3<f32>(e, 0.0, 0.0)) - iso_f(p - vec3<f32>(e, 0.0, 0.0)),
    iso_f(p + vec3<f32>(0.0, e, 0.0)) - iso_f(p - vec3<f32>(0.0, e, 0.0)),
    iso_f(p + vec3<f32>(0.0, 0.0, e)) - iso_f(p - vec3<f32>(0.0, 0.0, e)),
  );
  var n = normalize(gr + vec3<f32>(1.0e-12));
  if (dot(n, rd) > 0.0) { n = -n; }
  let key = normalize(-rd + 0.6 * C.up.xyz + 0.4 * C.right.xyz);
  let diff = max(dot(n, key), 0.0);
  let fill = max(dot(n, -rd), 0.0);
  let spec = pow(max(dot(n, normalize(key - rd)), 0.0), 40.0);
  // Lifted a little so the levels at the dark end of a color map still read against the background.
  let c = mix(base, vec3<f32>(1.0), 0.12);
  return c * (0.18 + 0.62 * diff + 0.25 * fill) + 0.35 * spec;
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
  if (t1 <= t0) { return vec4<f32>(C.bg.rgb, 1.0); }

  let jitter = hash(in.pos.xy);
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let iso = C.k.x > 0.0;
  let zero = -C.sc.y / max(C.sc.z - C.sc.y, 1.0e-20);   // the color scale's value at zero
  var fprev = 0.0;
  var pprev = vec3<f32>(0.0);
  var have = false;   // fprev, pprev hold the last sample
  // Each sample stands for the stretch of ray up to the next one, whose length depends on the
  // cell it is in (vol_step); the first is placed a random part of a step in, against banding.
  var t = t0 + jitter * vol_step(vol_look(ro + rd * t0));
  for (var i = 0; i < VOL_MAX_SAMPLES && t < t1; i++) {
    let p = ro + rd * t;
    let look = vol_look(p);
    let s = look.s;
    let ds = vol_step(look);
    let v = s.r;
    if (v < -0.5) {
      // The surface of an obstacle: opaque, darker with depth so its shape reads.
      col += (1.0 - alpha) * SOLID * (1.15 - 0.6 * (t - t0) / (t1 - t0));
      alpha = 1.0;
      break;
    }
    if (iso) {
      // A level crossed since the last sample (the first, if several), in shown gas: found to a
      // thirty-second of the step by bisection, then lit. It lies in front of this sample's fog.
      let f = (s.b - C.k.y) * (C.k.x + 1.0) / max(C.k.z - C.k.y, 1.0e-6);
      let up = f > fprev;
      let k = select(floor(fprev), floor(fprev) + 1.0, up);
      if (have && s.g > 0.5 && floor(f) != floor(fprev) && k >= 1.0 && k <= C.k.x) {
        var a = pprev;
        var b = p;
        for (var j = 0; j < 5; j++) {
          let m = 0.5 * (a + b);
          if ((iso_f(m) < k) == up) { a = m; } else { b = m; }
        }
        let h = 0.5 * (a + b);
        let neg = C.sc.x == 0.0 && v < zero;
        let sa = C.fwd.w;
        col += (1.0 - alpha) * sa * iso_shade(h, rd, level_color(k, neg));
        alpha += (1.0 - alpha) * sa;
      }
      fprev = f;
      pprev = p;
      have = true;
    }
    if (C.sc.w != 0.0) {
      let tf = textureSampleLevel(TF, S, vec2<f32>(v, 0.5), 0.0);
      // g: 1 where the gas is at least the minimum density, 0 where it is hidden (render/field.wgsl).
      let a = (1.0 - exp(-tf.a * C.p.z * ds)) * s.g;
      col += (1.0 - alpha) * a * tf.rgb;
      alpha += (1.0 - alpha) * a;
    }
    if (alpha > 0.995) { break; }
    t += ds;
  }
  col += (1.0 - alpha) * C.bg.rgb;

  // Box outline, unless C.bg.w is 0: far edges are dimmed by the gas in front of them. On a light
  // background the outline is darker than the background, on a dark one lighter.
  let w = 0.004;
  let near = edge(ro + rd * t0, w);
  let far = edge(ro + rd * t1, w) * (1.0 - alpha);
  let ink = select(vec3<f32>(0.45, 0.5, 0.6), vec3<f32>(0.35, 0.38, 0.45), dot(C.bg.rgb, vec3<f32>(1.0)) > 1.5);
  col = mix(col, ink, 0.6 * max(near, far) * C.bg.w);
  return vec4<f32>(col, 1.0);
}
