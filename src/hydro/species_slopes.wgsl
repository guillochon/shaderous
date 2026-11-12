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

// Slopes of every species of a cell with fractions o, between its neighbors m and p, with one
// limiter for all: the central slopes, scaled by the smallest ratio of any species'
// monotonized-central slope to its central one. Central slopes sum to zero (the fractions sum
// to one in every cell), so the scaled ones do too, and each species stays within its own
// limit. (Limiting each species on its own leaves face values that do not sum to one; scaling
// them back, as consistent multi-fluid advection does (Plewa & Mueller 1999), changes a
// species that is the same everywhere.)
fn slopes(m: array<vec4<f32>, NG>, o: array<vec4<f32>, NG>, p: array<vec4<f32>, NG>) -> array<vec4<f32>, NG> {
  var phi = 1.0;
  for (var g = 0u; g < NG; g++) {
    let a = o[g] - m[g];
    let b = p[g] - o[g];
    let c = 0.5 * (a + b);
    let real = max(abs(a), abs(b)) > vec4<f32>(SLOPE_FLOOR);
    let r = select(vec4<f32>(1.0), mc(a, b) / c, real & (abs(c) > vec4<f32>(1e-30)));
    phi = min(phi, min(min(r.x, r.y), min(r.z, r.w)));
  }
  var s: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { s[g] = 0.5 * phi * (p[g] - m[g]); }
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
