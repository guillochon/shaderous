// Fills the two ghost layers of the potential, in place (r32float can be read and written in
// one pass). MODE selects which faces:
//   0  faces whose neighbor is one level coarser: trilinear interpolation of the coarser
//      level's potential to the ghost-cell center. Run on a level before it is solved; these
//      values are its boundary condition there.
//   1  every other face, after all levels are solved: copies from same-level neighbors, and
//      linear extrapolation at isolated domain boundaries (matching a one-sided gradient).
// The hydro passes then difference the potential without knowing about block boundaries.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"

override MODE: u32 = 0u;

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;

// Coarse-level potential at coarse cell q of block `cs` or beyond it; `fallback` if unreachable.
fn coarse_phi(cs: i32, q: vec3<i32>, fallback: f32) -> f32 {
  let r = g_resolve(cs, q, BS);
  if (!r.ok) { return fallback; }
  return textureLoad(PHI, blk_origin(r.slot) + r.l).x;
}

// Only the face ghosts get an invocation: per block, 6 faces x 2 layers x BS x BS cells, in
// rows of 64 (workgroup x), one block of the list per workgroup row (y). Edge and corner ghosts
// are never used.
const FACE_GHOSTS = 6 * 2 * BS * BS;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let i = i32(wid.x * 64u + lix);
  if (wid.y >= LIST[0] || i >= FACE_GHOSTS) { return; }
  let slot = i32(LIST[1u + wid.y]);
  let face = i / (2 * BS * BS);          // -x, +x, -y, +y, -z, +z
  let r = i % (2 * BS * BS);
  let layer = r / (BS * BS);             // 0 next to the block, 1 beyond
  let axis = face / 2;
  let upper = (face & 1) == 1;
  var l = vec3<i32>(0);
  l[(axis + 1) % 3] = r % BS;
  l[(axis + 2) % 3] = (r / BS) % BS;
  l[axis] = select(-1 - layer, BS + layer, upper);
  let inv = Inv(true, slot, l);
  let nb = bt(inv.slot, 6u + u32(2 * axis) + u32(upper));
  let c = blk_origin(inv.slot) + l;

  if (MODE == 0u) {
    if (nb != -1) { return; }
    let cs = bt(inv.slot, 20u + u32(2 * axis) + u32(upper));
    let gf = blk_coord(inv.slot) * BS + l;
    let nc = i32(P.amr.y) * BS * (1 << u32(bt(cs, 0u)));
    var gc = gf >> vec3<u32>(1u);
    gc = ((gc % nc) + nc) % nc;
    let lc = gc - blk_coord(cs) * BS;
    // The ghost center sits a quarter of a coarse cell off the center of coarse cell lc, toward
    // the neighbor given by the fine index's parity.
    let d = 2 * (gf & vec3<i32>(1)) - 1;
    let base = textureLoad(PHI, blk_origin(cs) + lc).x;
    var sum = 0.0;
    for (var i = 0; i < 8; i++) {
      let s = vec3<i32>(i & 1, (i >> 1u) & 1, (i >> 2u) & 1);
      let w = mix(vec3<f32>(0.75), vec3<f32>(0.25), vec3<f32>(s));
      sum += w.x * w.y * w.z * coarse_phi(cs, lc + s * d, base);
    }
    textureStore(PHI, c, vec4<f32>(sum, 0.0, 0.0, 0.0));
    return;
  }

  if (nb == -1) { return; }
  var e = vec3<i32>(0);
  e[axis] = select(1, -1, upper);   // inward
  var v = 0.0;
  if (nb >= 0) {
    var q = l;
    q[axis] += select(BS, -BS, upper);
    v = textureLoad(PHI, blk_origin(nb) + q).x;
  } else {
    // How far outside: 1 or 2 cells. The first interior cell and the one after it.
    let depth = select(-l[axis], l[axis] - BS + 1, upper);
    let p0 = textureLoad(PHI, c + depth * e).x;
    let p1 = textureLoad(PHI, c + (depth + 1) * e).x;
    if (depth == 1) { v = 2.0 * p0 - p1; } else { v = 3.0 * p0 - 2.0 * p1; }
  }
  textureStore(PHI, c, vec4<f32>(v, 0.0, 0.0, 0.0));
}
