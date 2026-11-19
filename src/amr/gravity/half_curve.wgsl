// Potential half a step ahead with per-block timesteps and self-gravity: what conserves the
// energy is the mean of the potentials at the two ends of the block's step (the work of the
// mass fluxes against it is then the change in the potential energy), and in a collapse that
// mean runs ahead of the linear extrapolation by the potential's curvature in time. From the
// rates over the block's last two steps (DPHI and DPHI0, amr/gravity/sync.wgsl), taken as of
// equal length: phi + (dt / 2) (2 r1 - r0), the mean of phi and a quadratic's next value.
// Requires phi(), amr/gravity/half_on.wgsl and blk_dt().
@group(0) @binding(7) var DPHI0: texture_3d<f32>;
fn phi_half(c: vec3<i32>, slot: i32) -> f32 {
  let r1 = textureLoad(DPHI, c, 0).x;
  let r0 = textureLoad(DPHI0, c, 0).x;
  let r = select(2.0 * r1 - r0, r1, r0 >= 3.0e38);   // (sync.wgsl's NONE)
  return phi(c) + 0.5 * blk_dt(slot) * r;
}
