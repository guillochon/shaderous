// The look of field lines (render/fieldtubes.wgsl, render/fluxfunction.wgsl): colored by the
// field's strength and shaded as tubes, in the volume view and in the plane alike, so they read
// as field lines and not as the view's contour lines (thin, flat, one ink per level).

// Weak field to strong: deep teal, cyan, near white. Cool, so they stand apart from the warm
// default color map (inferno) of the gas behind them.
fn tube_color(t: f32) -> vec3<f32> {
  let a = vec3<f32>(0.1, 0.38, 0.55);
  let m = vec3<f32>(0.3, 0.78, 0.95);
  let c = vec3<f32>(0.9, 0.98, 1.0);
  return select(mix(m, c, 2.0 * t - 1.0), mix(a, m, 2.0 * t), t < 0.5);
}

// A tube drawn flat on the screen: `s` is the signed position across it, -1 to 1 from one edge
// to the other, and `n` the direction across it on the screen (y down). Lit from the upper left,
// with a highlight; ringed by an outline a pixel wide, dark (a little lighter on a light
// background). `edge`: pixels from the tube's side outward (negative inside it). Returns the
// color and coverage.
fn flat_tube(s: f32, n: vec2<f32>, base: vec3<f32>, edge: f32, light_bg: bool) -> vec4<f32> {
  let x = clamp(s, -1.0, 1.0);
  let nrm = normalize(vec3<f32>(x * n.x, -x * n.y, sqrt(max(1.0 - x * x, 0.0))));
  let key = normalize(vec3<f32>(-0.45, 0.6, 0.9));
  let diff = max(dot(nrm, key), 0.0);
  let spec = pow(max(dot(nrm, normalize(key + vec3<f32>(0.0, 0.0, 1.0))), 0.0), 30.0);
  let col = base * (0.3 + 0.75 * diff) + 0.35 * spec;
  let rim = select(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.25, 0.25, 0.28), light_bg);
  let inside = 1.0 - smoothstep(-0.5, 0.5, edge);
  let cov = 1.0 - smoothstep(1.0, 2.0, edge);
  return vec4<f32>(mix(rim, col, inside), cov);
}
