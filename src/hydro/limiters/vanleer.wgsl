// van Leer (harmonic mean) limiter.
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let ab = a * b;
  return select(vec4<f32>(0.0), 2.0 * ab / (a + b), ab > vec4<f32>(0.0));
}
// The same slope serves both faces of a cell.
const LIMITER_SYMMETRIC = true;
fn edge4(far: vec4<f32>, near: vec4<f32>, scale: vec4<f32>) -> vec4<f32> {
  return limit4(far, near);
}
