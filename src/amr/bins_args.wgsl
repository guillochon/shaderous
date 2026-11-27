// Closes the lists of a substep (amr/bins_lists.wgsl): their counts, the workgroup counts of
// the passes dispatched over them, and running totals of the blocks worked on.
//
// ARGS holds one (x, y, z) triple per kind of dispatch; the offsets are BIN_ARGS in
// src/amr/amr_sim.ts and must match:
//   0  padded pass over LA          1  interior pass over LU       2  interior pass over LL
//   3..7   multigrid, all cells of LA at sublevels of 16, 8, 4, 2, 1 cells per block side
//   8..12  the same for one color of a red-black sweep
//   13 flat pass (amr/list.wgsl, inv_flat) over LL with one ghost layer: the prediction
//   14 flat pass over LA with two ghost layers: the assembly
//   15 face ghosts of LA (amr/gravity/ghost.wgsl): 6 x 2 x 16 x 16 cells in rows of 64
//   16  the magnetic field's assembly over LL (two ghost layers; mhd/amr_mhd.ts, whose
//       fluxes, edges and update over LL are sized per chunk by mhd/amr_chunk_args.wgsl)
//   17  boxes of 17 texels per side over LU (the field's faces back)
//   18  the same over LD (the field's deposits)
// followed by the totals: ARGS[60] block updates, ARGS[61] blocks assembled.
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read_write> LA: array<u32>;
@group(0) @binding(2) var<storage, read_write> LU: array<u32>;
@group(0) @binding(3) var<storage, read_write> CNT: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ARGS: array<u32>;
@group(0) @binding(5) var<storage, read_write> LL: array<u32>;
@group(0) @binding(6) var<storage, read_write> LD: array<u32>;
@group(0) @binding(7) var<storage, read> clock: Clock;

// Slabs of a long list (amr/list.wgsl): must match SLAB_QUADS, SLAB_BLOCKS and FLAT_ROWS there.
const SLAB_QUADS = 16384u;
const SLAB_BLOCKS = 16384u;
const FLAT_ROWS = 32768u;

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
  let nd = atomicLoad(&CNT[3]);
  LD[0] = nd;
  let flat = P.dims.z == 1u;
  LA[0] = na;
  LU[0] = nu;
  LL[0] = nl;
  // Padded passes: 3 x 3 x 5 workgroups per block; interior passes: 2 x 2 x 4. In 2D a
  // workgroup covers four blocks (amr/list.wgsl): 3 x 3 or 2 x 2 per four.
  if (flat) {
    let qa = (na + 3u) / 4u;
    let qu = (nu + 3u) / 4u;
    let ql = (nl + 3u) / 4u;
    put(0u, min(qa, SLAB_QUADS) * 3u, 3u, (qa + SLAB_QUADS - 1u) / SLAB_QUADS);
    put(1u, min(qu, SLAB_QUADS) * 2u, 2u, (qu + SLAB_QUADS - 1u) / SLAB_QUADS);
    put(2u, min(ql, SLAB_QUADS) * 2u, 2u, (ql + SLAB_QUADS - 1u) / SLAB_QUADS);
  } else {
    // In slabs of SLAB_BLOCKS stacked along z, a block's depth of workgroups each (amr/dispatch.ts).
    let sa = (na + SLAB_BLOCKS - 1u) / SLAB_BLOCKS;
    let su = (nu + SLAB_BLOCKS - 1u) / SLAB_BLOCKS;
    let sl = (nl + SLAB_BLOCKS - 1u) / SLAB_BLOCKS;
    put(0u, min(na, SLAB_BLOCKS) * 3u, 3u, 5u * sa);
    put(1u, min(nu, SLAB_BLOCKS) * 2u, 2u, 4u * su);
    put(2u, min(nl, SLAB_BLOCKS) * 2u, 2u, 4u * sl);
  }
  // Multigrid (src/amr/gravity.ts, cells() and color(); amr/dispatch.ts gravityCells and
  // gravityColor), in slabs of SLAB_BLOCKS along z, 4 ceil(bsize / 4) invocations each.
  let ma = min(na, SLAB_BLOCKS);
  let ga = (na + SLAB_BLOCKS - 1u) / SLAB_BLOCKS;
  put(3u, ma * 2u, 2u, 4u * ga);
  put(4u, ma, 1u, 2u * ga);
  put(5u, ma, 1u, ga);
  put(6u, ma, 1u, ga);
  put(7u, ma, 1u, ga);
  put(8u, ma, 2u, 4u * ga);
  put(9u, (ma + 1u) / 2u, 1u, 2u * ga);
  put(10u, (ma + 3u) / 4u, 1u, ga);
  put(11u, (ma + 7u) / 8u, 1u, ga);
  put(12u, (ma + 7u) / 8u, 1u, ga);
  let w = 18u; // block side (16) plus a ghost layer on each face
  put(13u, (select(w * w * w, w * w, flat) + 63u) / 64u, min(nl, FLAT_ROWS), (nl + FLAT_ROWS - 1u) / FLAT_ROWS);
  let w2 = 20u; // and two
  put(14u, (select(w2 * w2 * w2, w2 * w2, flat) + 63u) / 64u, min(na, FLAT_ROWS), (na + FLAT_ROWS - 1u) / FLAT_ROWS);
  put(15u, 48u, min(na, FLAT_ROWS), (na + FLAT_ROWS - 1u) / FLAT_ROWS);
  let box17 = (select(17u * 17u * 17u, 17u * 17u, flat) + 63u) / 64u;
  put(16u, (select(w2 * w2 * w2, w2 * w2, flat) + 63u) / 64u, min(nl, FLAT_ROWS), (nl + FLAT_ROWS - 1u) / FLAT_ROWS);
  // (rows in slabs of FLAT_ROWS, as above: a 2D mesh with a field can have more leaves than an
  // axis may have workgroups)
  put(17u, box17, min(nu, FLAT_ROWS), (nu + FLAT_ROWS - 1u) / FLAT_ROWS);
  put(18u, box17, min(nd, FLAT_ROWS), (nd + FLAT_ROWS - 1u) / FLAT_ROWS);
  // (Not a cycle the clock holds, at a movie's frame time or a stop: taken at dt = 0, it does
  // nothing, and a recorded run's totals would differ from the unrecorded one's.)
  if (clock.dt > 0.0) {
    ARGS[60] += nu;
    ARGS[61] += na;
  }
}
