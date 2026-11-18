// The pressure the block predictor takes the gradient of with implicit pressure: a liquid's
// rides in its state's internal-energy slot (amr/pressure/correct.wgsl), so the primitives
// hold it, ghosts and all.
fn ap_lagp(c: vec3<i32>) -> f32 {
  return textureLoad(W2, c, 0).w;
}
