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
#include "mhd/amr_fofc_list.wgsl"

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= LIST[0]) { return; }
  let slot = i32(LIST[1u + i]);
  if (!fofc_block_marked(slot)) { return; }
  list_leaf(slot);
  list_neighbors(slot);
}

// After the correction: the list empty, and both rounds' dispatches (ARGS, and ARGS1 the
// recheck's: mhd/amr_fofc_recheck.wgsl) with no rows.
@group(0) @binding(40) var<storage, read_write> ARGS1: array<atomic<u32>>;
@compute @workgroup_size(1)
fn reset_main() {
  atomicStore(&RL[0], 0u);
  for (var c = 0u; c < CHUNKS; c++) {
    atomicStore(&ARGS[6u * c + 1u], 0u);
    atomicStore(&ARGS[6u * c + 4u], 0u);
    atomicStore(&ARGS1[6u * c + 1u], 0u);
    atomicStore(&ARGS1[6u * c + 4u], 0u);
  }
}
