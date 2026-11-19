// Potential half a step ahead, with one timestep for all blocks: linear extrapolation from the
// potentials of this step and the last. Requires phi() and PHI_OLD.
// With implicit pressure (P.ohm.y) the potential is the advected density's already, solved before
// the pressure passes that kick the faces with it (gravity/field.wgsl): it is taken as it is.
fn phi_half(c: vec3<i32>, slot: i32) -> f32 {
  let ahead = select(0.5, 0.0, P.ohm.y > 0.0);
  return (1.0 + ahead) * phi(c) - ahead * textureLoad(PHI_OLD, c, 0).x;
}
