// Bookkeeping of a substep with per-block timesteps (amr/bins_on.wgsl), in two parts:
//   MODE 0, before the substep: bank the one just taken on the clock.
//   MODE 1, after it: move the substep counter on, and empty the block lists, which are then
//           rebuilt for the blocks that start a step next (amr/bins_lists.wgsl).
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read_write> clock: Clock;
@group(0) @binding(2) var<storage, read_write> BIN: array<u32>;
@group(0) @binding(3) var<storage, read_write> CNT: array<atomic<u32>>;

override MODE: u32 = 0u;
override CYCLE: u32 = 8u;   // substeps per cycle

@compute @workgroup_size(1)
fn main() {
  if (MODE == 0u) {
    // Kahan summation keeps t accurate in float32 (as in hydro/clock.wgsl).
    let prev = clock.dt;
    let y = prev - clock.tc;
    let tt = clock.t + y;
    clock.tc = (tt - clock.t) - y;
    clock.t = tt;
    if (prev > 0.0) { clock.steps += 1.0; }
  } else {
    BIN[0] = (BIN[0] + 1u) % CYCLE;
    atomicStore(&CNT[0], 0u);
    atomicStore(&CNT[1], 0u);
    atomicStore(&CNT[2], 0u);
  }
}
