// The blocks the next substep works on, gathered from the list of all blocks:
//   LA  every block that starts a step then (parents included): assembled and predicted, and
//       with self-gravity the blocks whose potential is solved for that time
//   LL  the leaves among them: they have just finished a step, and get its closing gravity kick
//   LU  those leaves, plus the leaves at the midpoint of their step that have a faster
//       neighbor and so receive the second half of its fluxes: updated and committed
// The order within a list is whatever the atomic counter hands out; no result depends on it.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/bins_on.wgsl"

@group(0) @binding(1) var<storage, read_write> LA: array<u32>;
@group(0) @binding(2) var<storage, read_write> LU: array<u32>;
@group(0) @binding(3) var<storage, read_write> CNT: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> LL: array<u32>;

// Whether a leaf has a face neighbor (a leaf at its level, or finer leaves) in a faster bin.
fn faster_neighbor(slot: i32) -> bool {
  let k = blk_bin(slot);
  let nf = select(6u, 4u, is2d());
  for (var f = 0u; f < nf; f++) {
    let nb = bt(slot, 6u + f);
    if (nb < 0 || nb == slot) { continue; }
    if (blk_leaf(nb)) {
      if (blk_bin(nb) < k) { return true; }
      continue;
    }
    // Refined: the children that touch the shared face.
    let axis = f >> 1u;
    let want = 1u - (f & 1u);   // beyond our lower face they are the upper children
    for (var ch = 0u; ch < 8u; ch++) {
      if (((ch >> axis) & 1u) != want) { continue; }
      let c = bt(nb, 12u + ch);
      if (c >= 0 && blk_leaf(c) && blk_bin(c) < k) { return true; }
    }
  }
  return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= LIST[0]) { return; }
  let slot = i32(LIST[1u + i]);
  let on = blk_active(slot);
  let leaf = blk_leaf(slot);
  if (on) {
    let n = atomicAdd(&CNT[0], 1u);
    LA[1u + n] = u32(slot);
  }
  if (on && leaf) {
    let n = atomicAdd(&CNT[2], 1u);
    LL[1u + n] = u32(slot);
  }
  if (leaf && (on || (blk_half(slot) && faster_neighbor(slot)))) {
    let n = atomicAdd(&CNT[1], 1u);
    LU[1u + n] = u32(slot);
  }
}
