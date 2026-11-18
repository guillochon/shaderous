// The pressure the block predictor takes the gradient of with implicit pressure, for a gas with
// an energy: the solved pressure carried beside the state (amr/pressure/lag_fill.wgsl).
@group(0) @binding(47) var PFA: texture_3d<f32>;

fn ap_lagp(c: vec3<i32>) -> f32 {
  return textureLoad(PFA, c, 0).x;
}
