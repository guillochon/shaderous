// Whether a refinement check's answer will re-mesh, decided on the GPU as soon as the check is
// measured, so that the page may queue steps behind the check without waiting for its answer
// (AmrSim.encodeSteps, speculate). If it will, the clock is held (Clock.hold, hydro/clock.wgsl):
// those steps do nothing, and the re-mesh lands before the next step taken, as when the page waits.
//
// BlockTree.adapt, without refinement zones or obstacles, changes the mesh if a leaf below the
// level limit reads above refineHi (it is refined: REF lists those leaves), or else if a parent
// it could merge has no "hot" child: none above refineLo, none at or above 0 with a face
// neighbor above refineHi (CAND: per such parent, each child's slot and what it reads of its
// neighbors, BlockTree.adaptWatch). An indicator of exactly 0 counts as -1 (AmrSim's readFlags,
// adapt's indicator); a neighbor that is no leaf reads 0. The thresholds are rounded down to
// float32 (U.hi, U.lo): for a float32 indicator x, x > hi in double precision is x > U.hi here.
// (A full block budget can leave a wanted refinement undone: then the clock is held for nothing.)
//   MODE 0, over REF then CAND: any refinement or merge is counted.
//   MODE 1, one invocation: hold the clock if there was any.
#include "hydro/common.wgsl"

override MODE: u32 = 0u;
override NCH: u32 = 4u;    // children of a block
override NFACE: u32 = 4u;  // faces of a block

struct Spec {
  hi: f32,
  lo: f32,
  n_ref: u32,
  n_cand: u32,
}

@group(0) @binding(1) var<storage, read> FLAGS: array<f32>;
@group(0) @binding(2) var<storage, read> REF: array<u32>;
@group(0) @binding(3) var<storage, read_write> DIFF: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> clock: Clock;
@group(0) @binding(5) var<uniform> U: Spec;
// Per parent NCH children of 1 + NFACE entries: the child's slot, then each face's neighbor:
// its slot, -2 for a block that is no leaf, -1 past the box's edge. (AmrSim.adaptWatch packs it.)
@group(0) @binding(6) var<storage, read> CAND: array<i32>;

// A leaf's indicator as adapt reads it. (AmrSim.adaptWould mirrors all of this.)
fn ind(slot: i32) -> f32 {
  let raw = FLAGS[slot];
  return select(raw, -1.0, raw == 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (MODE == 1u) {
    if (atomicLoad(&DIFF[0]) != 0u) { clock.hold = 1.0; }
    return;
  }
  let i = gid.x;
  if (i < U.n_ref) {
    if (ind(i32(REF[i])) > U.hi) { atomicOr(&DIFF[0], 1u); }
    return;
  }
  let j = i - U.n_ref;
  if (j >= U.n_cand) { return; }
  let base = j * NCH * (1u + NFACE);
  for (var c = 0u; c < NCH; c++) {
    let o = base + c * (1u + NFACE);
    let x = ind(CAND[o]);
    if (x > U.lo) { return; }
    if (x < 0.0) { continue; }
    for (var f = 0u; f < NFACE; f++) {
      let nb = CAND[o + 1u + f];
      if (nb == -1) { continue; }
      let y = select(ind(nb), 0.0, nb == -2);
      if (y > U.hi) { return; }
    }
  }
  atomicOr(&DIFF[0], 1u);
}
