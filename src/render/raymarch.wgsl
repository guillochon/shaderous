// Emission-absorption ray-marcher through the unit box, front to back, with a 1D transfer
// function (color and opacity against the normalized field value). With contours on it also
// draws isosurfaces at their levels (C.k, as the slice view's lines), lit, in the color of their
// level, C.fwd.w opaque; C.sc.w 0 leaves the fog out, so the surfaces show alone.
// What it marches through comes from the <volume> slot: the uniform display volume
// (render/volume_uniform.wgsl) or an adaptive mesh's leaf blocks (render/volume_atlas.wgsl).
#include "render/camera.wgsl"
#include <volume>

// Field lines drawn as tubes before the gas (render/fieldtubes.wgsl): their color, and how far
// along the ray they lie, in r (0 where there is none). The gas in front of a tube is drawn over it.
@group(0) @binding(9) var TUBE: texture_2d<f32>;
@group(0) @binding(10) var TUBE_T: texture_2d<f32>;

// Isosurfaces closer together than this many cells, across the nearer one, fade into it: after a
// jump the cells do not resolve (a level that near it is not told apart from it), and after any
// other surface.
const ISO_GAP_JUMP = 3.0;
const ISO_GAP = 1.0;
// A change of this many decades across a few cells is a jump the cells do not resolve
// (smooth isosurfaces: its levels are drawn as one, at its middle).
const JUMP_DECADES = 3.0;
// How far either side of a crossing, in steps (two thirds of a cell), the gas is read to tell a
// jump and where its middle lies: past the reach of the smoothing.
const JW = 4.5;
// Most levels a step handles (a sharp edge may cross many: the rest are the same surface, or lie
// too close to show apart).
const MAX_LEVELS_PER_STEP = 4;

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// 1 where the point lies on an edge of the box (two coordinates at the faces).
fn edge(p: vec3<f32>, w: f32) -> f32 {
  let a = step(vec3<f32>(0.5 - w), abs(p));
  return step(1.5, a.x + a.y + a.z);
}

// Volume view's isosurfaces: in units of the spacing between levels, level k lies at f = k.
// From the log of the value (the volume's b), trilinear between cell centers: the exact surfaces,
// and how the march finds a level crossed.
fn iso_f(p: vec3<f32>) -> f32 {
  return (vol_look(p).s.b - C.k.y) * (C.k.x + 1.0) / max(C.k.z - C.k.y, 1.0e-6);
}

// Level units from the value itself, as a share of the color range's top (the volume's a).
fn iso_of_lin(x: f32) -> f32 {
  let top = select(log(max(abs(C.sc.z), 1.0e-30)) * 0.4342944819, C.sc.z, C.sc.x != 0.0);
  return (log(max(x, 1.0e-30)) * 0.4342944819 + top - C.k.y) * (C.k.x + 1.0) / max(C.k.z - C.k.y, 1.0e-6);
}

// The same from the value (not its log) smoothed by a cubic B-spline across the cells around p,
// from eight trilinear lookups (Sigg & Hadwiger 2005, GPU Gems 2 ch. 20), each placed between two
// cells so that its blend of them carries their B-spline weights. The value, not its log: a cell
// a surface crosses holds the share of it inside (a planet's edge), which places the surface
// within the cell only in the value. The B-spline then takes away the kinks of the trilinear
// field's gradient at each cell's faces, which show in the lighting as ripples a cell apart. The
// grid's phase is taken from the cell at p (lookups across a change of level blend whatever
// cells lie there).
fn lin_smooth(p: vec3<f32>) -> f32 {
  let c = vol_look(p).cell;
  let t = (p + 0.5) / c - 0.5;   // in cells, cell centers at whole numbers
  let i = floor(t);
  let f = t - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  // The two lookups along each axis, back in the box.
  let q0 = (i - 1.0 + w1 / g0 + 0.5) * c - 0.5;
  let q1 = (i + 1.0 + w3 / g1 + 0.5) * c - 0.5;
  var sum = 0.0;
  for (var j = 0u; j < 8u; j++) {
    let hi = vec3<bool>((j & 1u) != 0u, (j & 2u) != 0u, (j & 4u) != 0u);
    let w = select(g0, g1, hi);
    sum += w.x * w.y * w.z * vol_look(select(q0, q1, hi)).s.a;
  }
  return sum;
}

fn iso_smooth(p: vec3<f32>) -> f32 {
  return iso_of_lin(lin_smooth(p));
}

// The smoothed field's gradient at p, normalized, from the corners of a tetrahedron half a cell
// (c) out: four lookups of it rather than six.
fn smooth_normal(p: vec3<f32>, c: f32) -> vec3<f32> {
  let h = 0.5 * c;
  let a = vec3<f32>(1.0, -1.0, -1.0);
  let b = vec3<f32>(-1.0, -1.0, 1.0);
  let cc = vec3<f32>(-1.0, 1.0, -1.0);
  let d = vec3<f32>(1.0, 1.0, 1.0);
  let g = a * iso_smooth(p + h * a) + b * iso_smooth(p + h * b) + cc * iso_smooth(p + h * cc) + d * iso_smooth(p + h * d);
  return normalize(g + vec3<f32>(1.0e-12));
}

// The isosurface through p (in the box, [-0.5, 0.5]^3) seen along rd, lit: a key light above
// and to the right of the camera, a fill from the camera, and a highlight.
fn iso_shade(p: vec3<f32>, rd: vec3<f32>, base: vec3<f32>, exact: bool) -> vec3<f32> {
  let e = vol_look(p).cell;
  var gr: vec3<f32>;
  if (exact) {
    // Differences across the cell the surface passes through.
    gr = vec3<f32>(
      iso_f(p + vec3<f32>(e, 0.0, 0.0)) - iso_f(p - vec3<f32>(e, 0.0, 0.0)),
      iso_f(p + vec3<f32>(0.0, e, 0.0)) - iso_f(p - vec3<f32>(0.0, e, 0.0)),
      iso_f(p + vec3<f32>(0.0, 0.0, e)) - iso_f(p - vec3<f32>(0.0, 0.0, e)),
    );
  } else {
    gr = smooth_normal(p, e);
  }
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
  let tp = min(vec2<i32>(in.pos.xy), vec2<i32>(textureDimensions(TUBE_T)) - 1);
  let tube_t = textureLoad(TUBE_T, tp, 0).r;
  let tube = tube_t > 0.0;
  if (t1 <= t0) {
    // The ray misses the box: the background, behind any tube that lies out there (the path of
    // a mass being placed may leave the box).
    var bg = C.bg.rgb;
    if (tube) {
      let tc = textureLoad(TUBE, tp, 0);
      bg = mix(bg, tc.rgb, tc.a);
    }
    return vec4<f32>(bg, 1.0);
  }
  let tend = select(t1, min(t1, tube_t), tube);
  let jitter = hash(in.pos.xy);
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let iso = C.k.x > 0.0;
  let zero = -C.sc.y / max(C.sc.z - C.sc.y, 1.0e-20);   // the color scale's value at zero
  var fprev = 0.0;
  var pprev = vec3<f32>(0.0);
  var have = false;   // fprev, pprev hold the last sample
  // The last isosurface drawn: where, and its normal.
  var hlast = vec3<f32>(1.0e30);
  var nlast = vec3<f32>(0.0);
  var gaplast = 0.0;   // and how close behind it, across it, another fades out
  // Each sample stands for the stretch of ray up to the next one, whose length depends on the
  // cell it is in (vol_step); the first is placed a random part of a step in, against banding.
  var t = t0 + jitter * vol_step(vol_look(ro + rd * t0));
  for (var i = 0; i < VOL_MAX_SAMPLES && t < tend; i++) {
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
      // Levels crossed since the last sample, in shown gas: each found by
      // bisection, then lit. It lies in front of this sample's fog.
      // (Smooth surfaces: on the value interpolated linearly, as they are placed, else a level
      // crossed in the log may lie well away from where the smoothed value crosses it.)
      let f = select(iso_of_lin(s.a), (s.b - C.k.y) * (C.k.x + 1.0) / max(C.k.z - C.k.y, 1.0e-6), C.o.x != 0.0);
      let up = f > fprev;
      // Each level crossed since the last sample, in the direction f went, nearest first (a step
      // across a sharp edge may cross several, and may start far outside the range the levels
      // divide).
      var fcur = fprev;
      for (var n = 0; n < MAX_LEVELS_PER_STEP; n++) {
        let k = select(min(floor(fcur), C.k.x), max(floor(fcur) + 1.0, 1.0), up);
        let crossed = select((f < k), (f >= k), up);
        if (!(have && s.g > 0.5 && crossed && k >= 1.0 && k <= C.k.x)) { break; }
        // Past this one, for the next.
        fcur = select(k - 0.5, k, up);
        // Unless exact, the smoothed surface is sought instead, in the step either side as well
        // (smoothing moves it by up to about half a cell).
        var exact = C.o.x != 0.0;
        var a = pprev;
        var b = p;
        var dir = up;
        var kt = k;   // the value sought, in level units
        var lin = false;   // bisect on the trilinear value rather than the smoothed one
        if (!exact) {
          let d = p - pprev;
          let fa = iso_smooth(pprev - d);
          let fb = iso_smooth(p + d);
          // A jump across several levels within a few cells (a planet's edge, several decades
          // within one): the levels below its middle are not placed by the cells, whose shares
          // of it place only the half-way value; drawn there, they would follow the outline of
          // the cells it touches. Each of its levels is drawn at the value half way between the
          // gas either side of it (read farther out, where the smoothing has let go), so they
          // fall together and are drawn once (ISO_GAP).
          let la = lin_smooth(pprev - JW * d);
          let lb = lin_smooth(p + JW * d);
          let fa3 = iso_of_lin(la);
          let fb3 = iso_of_lin(lb);
          // (In levels, or in decades: with few levels a cliff of many decades may span less
          // than two of them.)
          let decades = abs(log(max(lb, 1.0e-30)) - log(max(la, 1.0e-30))) * 0.4342944819;
          let jump = abs(fb3 - fa3) >= 2.0 || decades >= JUMP_DECADES;
          if (jump) {
            // (It lies between the two reads, which may be farther than a step from this one.)
            // Its other levels, crossed in the steps to come, are this same surface.
            kt = iso_of_lin(0.5 * (la + lb));
            a = pprev - JW * d;
            b = p + JW * d;
            dir = lb > la;
          } else if ((fa < kt) != (fb < kt)) {
            a = pprev - d;
            b = p + d;
            dir = fb > fa;
          } else if ((fa3 < kt) != (fb3 < kt)) {
            // Smoothing moved it farther than a step.
            a = pprev - JW * d;
            b = p + JW * d;
            dir = fb3 > fa3;
          } else {
            // The smoothed field does not cross the level near here (a ray grazing an edge the
            // smoothing has moved): the linearly interpolated one, on which the crossing was
            // found, does within the step.
            lin = true;
          }
        }
        for (var j = 0; j < select(8, 5, exact); j++) {
          let m = 0.5 * (a + b);
          var fm: f32;
          if (exact) { fm = iso_f(m); } else if (lin) { fm = iso_of_lin(vol_look(m).s.a); } else { fm = iso_smooth(m); }
          if ((fm < kt) == dir) { a = m; } else { b = m; }
        }
        let h = 0.5 * (a + b);
        // A surface close behind the last one drawn, across it (levels packed into one jump, or
        // one lying just under it), fades out: at the gap (ISO_GAP, ISO_GAP_JUMP) it is drawn
        // in full, at under half that not at all. Cut off sharply, which of them a pixel shows would turn on where
        // its samples fall, and speckle. Across the surface, not along the ray, so the same
        // from every side.
        let c = vol_look(h).cell;
        let nh = smooth_normal(h, c);
        let dh = h - hlast;
        var keep = smoothstep(0.5 * gaplast, gaplast, abs(dot(dh, nlast)));
        if (dot(dh, dh) > 16.0 * gaplast * gaplast) { keep = 1.0; }
        if (keep > 0.0) {
          if (keep > 0.5) {
            hlast = h;
            nlast = nh;
            gaplast = select(ISO_GAP, ISO_GAP_JUMP, kt != k) * c;
          }
          let neg = C.sc.x == 0.0 && v < zero;
          let sa = C.fwd.w * keep;
          col += (1.0 - alpha) * sa * iso_shade(h, rd, level_color(k, neg), C.o.x != 0.0);
          alpha += (1.0 - alpha) * sa;
        }
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
  if (tube) {
    let tc = textureLoad(TUBE, tp, 0);
    col += (1.0 - alpha) * tc.a * tc.rgb;
    alpha += (1.0 - alpha) * tc.a;
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
