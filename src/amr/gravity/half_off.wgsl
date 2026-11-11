// Potential half a step ahead, with one timestep for all blocks: linear extrapolation from the
// potentials of this step and the last. Requires phi() and PHI_OLD.
fn phi_half(c: vec3<i32>, slot: i32) -> f32 {
  return 1.5 * phi(c) - 0.5 * textureLoad(PHI_OLD, c, 0).x;
}
