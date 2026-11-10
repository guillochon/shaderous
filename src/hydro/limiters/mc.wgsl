// Monotonized-central limiter.
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let m = min(min(2.0 * abs(a), 2.0 * abs(b)), 0.5 * abs(a + b));
  return select(vec4<f32>(0.0), sign(a) * m, a * b > vec4<f32>(0.0));
}
