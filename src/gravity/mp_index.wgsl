// The multipole moments of isolated boundaries: m x^a y^b z^c summed over the gas, for
// a + b + c <= MP_ORDER (35 of them), ordered by degree, then a and b descending.
const MP_ORDER = 4;
const MP_COUNT = 35u;

fn mp_index(a: i32, b: i32, c: i32) -> u32 {
  let d = a + b + c;
  return u32(d * (d + 1) * (d + 2) / 6 + (d - a) * (d - a + 1) / 2 + (d - a - b));
}
fn mp_pow(v: f32, n: i32) -> f32 {
  var p = 1.0;
  for (var k = 0; k < n; k++) { p *= v; }
  return p;
}
fn mp_fact(n: i32) -> f32 {
  var f = 1.0;
  for (var k = 2; k <= n; k++) { f *= f32(k); }
  return f;
}
fn mp_binom(n: i32, k: i32) -> f32 {
  return mp_fact(n) / (mp_fact(k) * mp_fact(n - k));
}
