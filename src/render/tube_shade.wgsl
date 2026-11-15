// The look of field lines (render/fieldtubes.wgsl, render/fluxfunction.wgsl): thin, shaded as
// tubes, and colored by the magnetic pressure on a log scale, as Guillochon & McCourt (2017)
// drew theirs, in the volume view and in the plane alike. The colors alone set them apart from
// the view's contour lines (one ink per level).

// Fraction t of the way from the weakest to the strongest field shown, in log B^2: a rainbow
// in hue from red (weak) through orange, yellow, green, cyan and blue to magenta (strong).
fn tube_color(t: f32) -> vec3<f32> {
  let h = 5.0 * clamp(t, 0.0, 1.0);   // sixths of the hue circle: 0 red ... 5 magenta
  let rgb = clamp(abs(fract(vec3<f32>(h / 6.0) + vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
  return rgb;
}

// Decades of magnetic pressure B^2 the colors span below the strongest field shown, at most:
// those of Guillochon & McCourt's figures (1e3 to 1e9 erg cm^-3).
const TUBE_DECADES: f32 = 6.0;

// A tube drawn flat on the screen: `s` is the signed position across it, -1 to 1 from one edge
// to the other, and `n` the direction across it on the screen (y down). Lit from the upper left,
// gently (a thin line keeps its color). `edge`: pixels from the tube's side outward (negative
// inside it); the edge is antialiased over a pixel. Returns the color and coverage.
fn flat_tube(s: f32, n: vec2<f32>, base: vec3<f32>, edge: f32, light_bg: bool) -> vec4<f32> {
  let x = clamp(s, -1.0, 1.0);
  let nrm = normalize(vec3<f32>(x * n.x, -x * n.y, sqrt(max(1.0 - x * x, 0.0))));
  let key = normalize(vec3<f32>(-0.45, 0.6, 0.9));
  let diff = max(dot(nrm, key), 0.0);
  // On a light background, a little darker, so yellow and cyan still show.
  let col = base * select(0.65 + 0.35 * diff, 0.5 + 0.3 * diff, light_bg);
  let cov = 1.0 - smoothstep(-0.5, 0.5, edge);
  return vec4<f32>(col, cov);
}
