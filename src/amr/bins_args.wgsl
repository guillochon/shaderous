// Closes the lists of a substep (amr/bins_lists.wgsl): their counts, the workgroup counts of
// the passes dispatched over them, and running totals of the blocks worked on.
//
// ARGS holds one (x, y, z) triple per kind of dispatch; the offsets are BIN_ARGS in
// src/amr/amr_sim.ts and must match:
//   0  padded pass over LA          1  interior pass over LU       2  interior pass over LL
//   3..7   multigrid, all cells of LA at sublevels of 16, 8, 4, 2, 1 cells per block side
//   8..12  the same for one color of a red-black sweep
// followed by the totals: ARGS[60] block updates, ARGS[61] blocks assembled.
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read_write> LA: array<u32>;
@group(0) @binding(2) var<storage, read_write> LU: array<u32>;
@group(0) @binding(3) var<storage, read_write> CNT: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ARGS: array<u32>;
@group(0) @binding(5) var<storage, read_write> LL: array<u32>;

fn put(i: u32, x: u32, y: u32, z: u32) {
  ARGS[3u * i] = x;
  ARGS[3u * i + 1u] = y;
  ARGS[3u * i + 2u] = z;
}

@compute @workgroup_size(1)
fn main() {
  let na = atomicLoad(&CNT[0]);
  let nu = atomicLoad(&CNT[1]);
  let nl = atomicLoad(&CNT[2]);
  let flat = P.dims.z == 1u;
  LA[0] = na;
  LU[0] = nu;
  LL[0] = nl;
  // Padded passes: 3 x 3 x 5 workgroups per block; interior passes: 2 x 2 x 4.
  put(0u, na * 3u, 3u, select(5u, 1u, flat));
  put(1u, nu * 2u, 2u, select(4u, 1u, flat));
  put(2u, nl * 2u, 2u, select(4u, 1u, flat));
  // Multigrid (src/amr/gravity.ts, cells() and color()).
  put(3u, na * 2u, 2u, 4u);
  put(4u, na, 1u, 2u);
  put(5u, na, 1u, 1u);
  put(6u, na, 1u, 1u);
  put(7u, na, 1u, 1u);
  put(8u, na, 2u, 4u);
  put(9u, (na + 1u) / 2u, 1u, 2u);
  put(10u, (na + 3u) / 4u, 1u, 1u);
  put(11u, (na + 7u) / 8u, 1u, 1u);
  put(12u, (na + 7u) / 8u, 1u, 1u);
  ARGS[60] += nu;
  ARGS[61] += na;
}
