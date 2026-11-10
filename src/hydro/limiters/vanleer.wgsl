// van Leer (harmonic mean) limiter.
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let ab = a * b;
  return select(vec4<f32>(0.0), 2.0 * ab / (a + b), ab > vec4<f32>(0.0));
}
