// Per-block timesteps. Time advances in cycles of 2^K substeps of the clock's dt, the step of
// the fastest block. A block in bin k steps 2^k substeps at a time, starting on the substeps
// that are multiples of 2^k, so every block is at the same time when a cycle ends. Bins are
// assigned at the start of each cycle (amr/bins_init.wgsl, amr/bins_relax.wgsl) such that
// face neighbors are at most one bin apart and a block is never slower than a coarser
// neighbor; a parent's bin is the fastest of its children's.
//   BIN[0]         the substep of the cycle now being taken
//   BIN[1 + slot]  the block's bin
@group(0) @binding(29) var<storage, read> BIN: array<u32>;

const BINS_ON = true;
fn blk_bin(slot: i32) -> u32 {
  return BIN[1u + u32(slot)];
}
// The block starts a step on this substep.
fn blk_active(slot: i32) -> bool {
  return (BIN[0] & ((1u << blk_bin(slot)) - 1u)) == 0u;
}
// The block is exactly half way through its step: the substep on which a neighbor one bin
// faster starts its second step.
fn blk_half(slot: i32) -> bool {
  let k = blk_bin(slot);
  return k > 0u && (BIN[0] & ((1u << k) - 1u)) == (1u << (k - 1u));
}
