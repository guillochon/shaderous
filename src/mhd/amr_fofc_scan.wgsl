// The first-order flux correction after the update on the block grid (mhd/amr_fofc.wgsl): after
// stage 2's update, list the leaves to correct: each leaf of the step (LIST: every leaf, or a
// substep's leaves that start a step) that has a cell stage 2 all but emptied, and its face
// neighbors that start a step too. A leaf reads its own mark alone, so a step with no marks costs
// one texel per leaf; SEEN (the step's stamp per slot) lists each leaf once. As it lists, it sizes
// the dispatches of the fluxes and update over the list, chunk by chunk (ARGS, as
// mhd/amr_chunk_args.wgsl lays them out: rows here, the rest set when the buffer is made).
// reset_main empties the list and the rows after their passes.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(28) var IDX: texture_3d<u32>;
@group(0) @binding(35) var NF: texture_3d<f32>;   // the step's new faces, which carry the marks
@group(0) @binding(36) var<storage, read_write> RL: array<atomic<u32>>;   // count, then slots
@group(0) @binding(37) var<storage, read_write> ARGS: array<atomic<u32>>;
@group(0) @binding(38) var<storage, read_write> SEEN: array<atomic<u32>>;

override PER_CHUNK: u32 = 1u;
override CHUNKS: u32 = 1u;

#include "mhd/amr_lookup.wgsl"
#include "mhd/amr_fofc.wgsl"

// List leaf `slot`, unless this step has already.
fn list_leaf(slot: i32) {
  let stamp = u32(fofc_stamp());
  if (atomicExchange(&SEEN[u32(slot)], stamp) == stamp) { return; }
  let at = atomicAdd(&RL[0], 1u);
  atomicStore(&RL[1u + at], u32(slot));
  // Its chunk dispatches at least this many rows.
  let c = at / PER_CHUNK;
  let rows = at % PER_CHUNK + 1u;
  atomicMax(&ARGS[6u * c + 1u], rows);
  atomicMax(&ARGS[6u * c + 4u], rows);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= LIST[0]) { return; }
  let slot = i32(LIST[1u + i]);
  if (!fofc_block_marked(slot)) { return; }
  list_leaf(slot);
  // Its face neighbors: the leaves at points one cell beyond each face, at the middles of its
  // quarters (halves in 2D), so that a neighbor one level finer is found at each of its blocks.
  let level = u32(bt(slot, 0u));
  let base = blk_coord(slot) * BS;
  for (var k = 0; k < loop_count(6 * 4); k++) {
    let a = k / 8;
    if (a >= select(3, 2, is2d())) { break; }
    let up = ((k / 4) & 1) == 1;
    let q = k & 3;
    let b = (a + 1) % 3;
    let c = (a + 2) % 3;
    var l = vec3<i32>(0);
    l[a] = select(-1, BS, up);
    l[b] = select(BS / 4, 3 * BS / 4, (q & 1) == 1);
    l[c] = select(BS / 4, 3 * BS / 4, (q & 2) == 2);
    if (is2d()) {
      if (c == 2 && q >= 2) { continue; }
      if (b == 2 && (q & 1) == 1) { continue; }
      l.z = 0;
    }
    let g = base + l;
    if (fofc_beyond_wall(level, g)) { continue; }
    let s = fofc_leaf_at(level, map_cell(level, g).g);
    // (With time bins, a neighbor at its midpoint takes the corrected register in its deposit.)
    if (s >= 0 && s != slot && blk_active(s)) { list_leaf(s); }
  }
}

@compute @workgroup_size(1)
fn reset_main() {
  atomicStore(&RL[0], 0u);
  for (var c = 0u; c < CHUNKS; c++) {
    atomicStore(&ARGS[6u * c + 1u], 0u);
    atomicStore(&ARGS[6u * c + 4u], 0u);
  }
}
