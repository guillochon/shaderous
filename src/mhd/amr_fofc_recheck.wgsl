// The first-order flux correction after the update on the block grid, between its two rounds
// (mhd/amr_fofc.wgsl): over the leaves round 0 updated again (RL, its list, by chunk), mark the
// cells it left all but emptied in their turn, a face of a marked cell having taken more of them
// at first order than it did at second, and add their leaves' face neighbors to the list. Round 1
// then corrects the whole list, as round 0 did the update's marks. The list is round 0's and grows:
// its sizes for round 1 (ARGS, which this pass is not dispatched from) cover the leaves already on
// it and those it adds.
//
// The verdict reads the state the redo wrote against the start of the step: less than FOFC_KEEP of
// the density left. (Not the internal energy, as the update's own check does: this sees it after
// its floor, so every cold cell at the floor would be marked step after step; marked so, the cold
// gas over the poles of a magnetized disk on blocks lay at the pressure floor in a thousand cells
// and more, against a few dozen without.) It marks the cell and its block in M1, round 1's marks
// (the faces carry round 0's, and a texture of four floats a texel cannot be read and written in
// one pass).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // the start of the step
@group(0) @binding(9) var O1: texture_3d<f32>;   // what the redo left
@group(0) @binding(11) var M1: texture_storage_3d<r32float, write>;
@group(0) @binding(28) var IDX: texture_3d<u32>;
@group(0) @binding(36) var<storage, read_write> RL: array<atomic<u32>>;   // count, then slots
@group(0) @binding(37) var<storage, read_write> ARGS: array<atomic<u32>>;   // round 1's sizes
@group(0) @binding(38) var<storage, read_write> SEEN: array<atomic<u32>>;
// A chunk of the list (mhd/amr_block.wgsl's, which this pass does not include: it reads none of
// the assembled state).
struct Chunk { first: u32, count: u32, unused0: u32, unused1: u32 }
@group(0) @binding(14) var<uniform> CHUNK: Chunk;

override PER_CHUNK: u32 = 1u;

#include "mhd/amr_lookup.wgsl"
#include "mhd/amr_fofc_core.wgsl"
#include "mhd/amr_fofc_list.wgsl"

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = flat_row(wid);
  if (row >= CHUNK.count) { return; }
  let at = CHUNK.first + row;
  let slot = i32(atomicLoad(&RL[1u + at]));
  // Round 1 goes through every leaf round 0 did.
  if (wid.x == 0u && lix == 0u) { list_rows(at); }
  let w = BS + 1;
  let i = i32(wid.x * FLAT_WG + lix);
  if (i >= select(w * w * w, w * w, is2d())) { return; }
  var t = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { t.z = 0; }
  if (any(t >= vec3<i32>(BS, BS, select(BS, 1, is2d())))) { return; }
  let o = blk_origin(slot);
  let x = atl(o, t);
  if (textureLoad(O1, x, 0).x >= FOFC_KEEP * textureLoad(U1, x, 0).x) { return; }
  textureStore(M1, x, vec4<f32>(fofc_stamp()));
  textureStore(M1, fofc_corner(slot), vec4<f32>(fofc_stamp()));
  list_neighbors(slot);
}
