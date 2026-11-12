// Slopes of the mass fractions, for the species passes of both grids (hydro/species_common.wgsl,
// amr/species_common.wgsl). Requires NG.

// Differences below this are rounding (X is rho X over its sum in single precision) and change
// sign from cell to cell: counted, a species that is the same everywhere would stop every
// slope. A species that varies by less than that may overshoot by as little.
const SLOPE_FLOOR: f32 = 1e-6;

// Monotonized-central slope.
fn mc(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let c = 0.5 * (a + b);
  let s = sign(c) * min(min(2.0 * abs(a), 2.0 * abs(b)), abs(c));
  return select(vec4<f32>(0.0), s, a * b > vec4<f32>(0.0));
}

// Slopes of every species of a cell with fractions o, between its neighbors m and p. Each
// species gets its own monotonized-central slope, so a sharp front in one species does not
// flatten the others. Those slopes need not sum to zero, and face values must sum to one: the
// excess is taken back from the slopes of the sign that has too much, each in proportion to
// its size. That only shrinks slopes (each stays within its own limit, so no face value
// leaves the range of its neighbors), and a species with no slope, such as one that is the
// same everywhere, is left alone. (Scaling face values back to sum to one, as consistent
// multi-fluid advection does (Plewa & Mueller 1999), would change such a species.)
fn slopes(m: array<vec4<f32>, NG>, o: array<vec4<f32>, NG>, p: array<vec4<f32>, NG>) -> array<vec4<f32>, NG> {
  var s: array<vec4<f32>, NG>;
  var up = 0.0;     // sum of the positive slopes
  var down = 0.0;   // sum of the negative ones, as a positive number
  for (var g = 0u; g < NG; g++) {
    let a = o[g] - m[g];
    let b = p[g] - o[g];
    let real = max(abs(a), abs(b)) > vec4<f32>(SLOPE_FLOOR);
    s[g] = select(vec4<f32>(0.0), mc(a, b), real);
    up += dot(max(s[g], vec4<f32>(0.0)), vec4<f32>(1.0));
    down -= dot(min(s[g], vec4<f32>(0.0)), vec4<f32>(1.0));
  }
  // Scale the larger side down to the smaller one.
  let kp = select(1.0, down / up, up > down);
  let kn = select(1.0, up / down, down > up);
  for (var g = 0u; g < NG; g++) { s[g] = select(s[g] * kn, s[g] * kp, s[g] > vec4<f32>(0.0)); }
  return s;
}

// Mass fractions from rho X: each over their sum, so they sum to one even where the density
// was changed by something other than the species pass (settling, a brush, a sink, a floor).
fn fractions(u: array<vec4<f32>, NG>) -> array<vec4<f32>, NG> {
  var sum = 0.0;
  for (var g = 0u; g < NG; g++) { sum += dot(u[g], vec4<f32>(1.0)); }
  var x: array<vec4<f32>, NG>;
  let inv = 1.0 / max(sum, 1e-30);
  for (var g = 0u; g < NG; g++) { x[g] = u[g] * inv; }
  return x;
}

// Face fractions clamped to [0, 1] and scaled to sum to one (both only undo rounding when the
// slopes are as above).
fn face_fractions(x: array<vec4<f32>, NG>) -> array<vec4<f32>, NG> {
  var c: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { c[g] = clamp(x[g], vec4<f32>(0.0), vec4<f32>(1.0)); }
  return fractions(c);
}

fn unit(axis: i32) -> vec3<i32> {
  var e = vec3<i32>(0);
  e[axis] = 1;
  return e;
}
