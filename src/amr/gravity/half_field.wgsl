// Potential half a step ahead with per-block timesteps: the block's own half step, at the
// rate its potential last changed. Requires phi(), amr/gravity/half_on.wgsl and blk_dt().
fn phi_half(c: vec3<i32>, slot: i32) -> f32 {
  return phi(c) + 0.5 * blk_dt(slot) * textureLoad(DPHI, c, 0).x;
}
