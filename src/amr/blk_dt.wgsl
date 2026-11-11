// Length of a block's own step. Requires the clock and <bins>.
fn blk_dt(slot: i32) -> f32 {
  return clock.dt * f32(1u << blk_bin(slot));
}
