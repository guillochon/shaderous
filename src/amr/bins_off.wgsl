// Per-block timesteps are off: every block takes every step, with the clock's dt.
const BINS_ON = false;
fn blk_bin(slot: i32) -> u32 { return 0u; }
fn blk_active(slot: i32) -> bool { return true; }
fn blk_half(slot: i32) -> bool { return false; }
