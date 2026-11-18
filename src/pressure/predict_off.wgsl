// The explicit scheme takes its pressure gradient from the state (hydro/predictor.wgsl).
fn pp_gradient(c: vec3<i32>) -> vec3<f32> {
  return vec3<f32>(0.0);
}
