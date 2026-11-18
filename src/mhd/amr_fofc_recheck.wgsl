// The first-order flux correction after the update on the block grid, between its two rounds
// (mhd/amr_fofc.wgsl): over the leaves round 0 updated again (LIST: its list, by chunk), mark the
// cells it left all but emptied in their turn, a face of a marked cell having taken more of them
// at first order than it did at second. Round 1 then corrects them as round 0 did the update's.
// The verdict reads the state the redo wrote against the start of the step: less than FOFC_KEEP of
// the density left. (Not the internal energy, as the update's own check does: this sees it after
// its floor, so every cold cell at the floor would be marked step after step; marked so, the cold
// gas over the poles of a magnetized disk on blocks lay at the pressure floor in a thousand cells
// and more, against a few dozen without), and
// marks the cell and its block in M1, round 1's marks (the faces carry round 0's, and a texture
// of four floats a texel cannot be read and written in one pass).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // the start of the step
@group(0) @binding(9) var O1: texture_3d<f32>;   // what the redo left
@group(0) @binding(11) var M1: texture_storage_3d<r32float, write>;

// A chunk of the list, and its leaves' boxes of texels, as mhd/amr_block.wgsl has them (which
// this pass does not include: it reads none of the assembled state).
struct Chunk { first: u32, count: u32, unused0: u32, unused1: u32 }
@group(0) @binding(14) var<uniform> CHUNK: Chunk;
struct BoxInv { ok: bool, slot: i32, l: vec3<i32> }
fn inv_box(wid: vec3<u32>, lix: u32, w: i32) -> BoxInv {
  let row = flat_row(wid);
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { l.z = 0; }
  if (row >= CHUNK.count || CHUNK.first + row >= LIST[0] || i >= n) { return BoxInv(false, 0, l); }
  return BoxInv(true, i32(LIST[1u + CHUNK.first + row]), l);
}

const FOFC_KEEP = 0.125;
fn fofc_stamp() -> f32 {
  return clock.steps + 1.0;
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_box(wid, lix, BS + 1);
  if (!inv.ok) { return; }
  let t = inv.l;
  if (any(t >= vec3<i32>(BS, BS, select(BS, 1, is2d())))) { return; }
  let o = blk_origin(inv.slot);
  let x = atl(o, t);
  let start = textureLoad(U1, x, 0).x;
  let u1 = textureLoad(O1, x, 0);
  if (u1.x >= FOFC_KEEP * start) { return; }
  textureStore(M1, x, vec4<f32>(fofc_stamp()));
  textureStore(M1, atl(o, vec3<i32>(-NGH, -NGH, select(-NGH, 0, is2d()))), vec4<f32>(fofc_stamp()));
}
