// Fills the first ghost layer of the potential, in place (r32float can be read and written in
// one pass). MODE selects which faces:
//   0  faces whose neighbour is one level coarser: trilinear interpolation of the coarser
//      level's potential to the ghost-cell centre. Run on a level before it is solved; these
//      values are its boundary condition there.
//   1  every other face, after all levels are solved: copies from same-level neighbours, and
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

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 1);
  if (!inv.ok) { return; }
  let l = inv.l;
  let out = vec3<i32>(l < vec3<i32>(0)) + vec3<i32>(l >= vec3<i32>(BS));
  if (out.x + out.y + out.z != 1) { return; }
  var axis = 0;
  if (out.y != 0) { axis = 1; }
  if (out.z != 0) { axis = 2; }
  let upper = l[axis] >= BS;
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
    // The ghost centre sits a quarter of a coarse cell off the centre of coarse cell lc, towards
    // the neighbour given by the fine index's parity.
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
    v = 2.0 * textureLoad(PHI, c + e).x - textureLoad(PHI, c + 2 * e).x;
  }
  textureStore(PHI, c, vec4<f32>(v, 0.0, 0.0, 0.0));
}
