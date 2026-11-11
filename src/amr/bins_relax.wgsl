// One relaxation sweep of the bins towards the rules the stepping relies on:
//   - face neighbours are at most one bin apart, so a slower block's faster neighbour starts
//     steps only at its start and at its midpoint;
//   - a block is never slower than a coarser neighbour, so the finer side is the one whose
//     fluxes are used at a refinement boundary, as with a single timestep;
//   - a parent's bin is the fastest of its children's, which stands for them towards a
//     coarser neighbour.
// The first sweep (DILATE) instead brings every block down to its fastest neighbour's bin: a
// block next to fast gas may be reached by it before the cycle ends, and steps as if it had.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> KIN: array<u32>;
@group(0) @binding(2) var<storage, read_write> KOUT: array<u32>;

override KMAX: u32 = 3u;
override DILATE: bool = false;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= LIST[0]) { return; }
  let slot = i32(LIST[1u + i]);
  var k = KIN[1u + u32(slot)];
  if (!blk_leaf(slot)) {
    k = KMAX;
    for (var ch = 0u; ch < 8u; ch++) {
      let c = bt(slot, 12u + ch);
      if (c >= 0) { k = min(k, KIN[1u + u32(c)]); }
    }
  } else {
    let nf = select(6u, 4u, is2d());
    for (var f = 0u; f < nf; f++) {
      let nb = bt(slot, 6u + f);
      if (nb >= 0 && nb != slot) {
        k = min(k, KIN[1u + u32(nb)] + select(1u, 0u, DILATE));
      } else if (nb == -1) {
        k = min(k, KIN[1u + u32(bt(slot, 20u + f))]);
      }
    }
  }
  KOUT[1u + u32(slot)] = k;
}
