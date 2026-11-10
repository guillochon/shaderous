// Minmod limiter (most diffusive).
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return select(vec4<f32>(0.0), sign(a) * min(abs(a), abs(b)), a * b > vec4<f32>(0.0));
}
