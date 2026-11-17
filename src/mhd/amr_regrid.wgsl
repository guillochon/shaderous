// Re-meshing the magnetic field (docs/numerics.md section 21, Re-meshing): every face of every leaf of the
// new tree, from the old tree's faces, divergence-free. Runs beside amr/regrid.wgsl, which does
// the cells, with the same regrid table (RG); BT and IDX here are the OLD tree's block table and
// leaf index (copied before the new tree was uploaded), and LIST the new leaves.
//
//   copy      the block is unchanged: its faces and their compensations as they were
//   restrict  a merged block: each coarse face the mean of the fine faces on it, which keeps every
//             coarse cell's divergence the sum of its children's (zero)
//   prolong   a new child: for each coarse cell of the old parent, first the fine faces on its six
//             faces (copied where an old leaf at the fine level held them: never interpolate where
//             fine data exist), otherwise the coarse face with limited slopes across it, keeping its
//             mean; then the faces inside it, from the mean of the two outer faces in line with
//             each, corrected by the least change that makes every fine cell divergence-free
//
// The correction: with r the divergences of the 2^d fine cells, the interior faces change by
// delta = -A^T L^+ r, A the incidence of interior faces on cells and L = A A^T the Laplacian
// of the hypercube of fine cells. Its eigenvectors are the Walsh functions (-1)^(s . c), with
// eigenvalues 2|s|, so L^+ is a transform, a division and the inverse transform. The fine cells'
// divergence sums to the coarse cell's (zero), so the system is consistent and each fine cell
// ends divergence-free to rounding. (The construction differs from Toth & Roe 2002's explicit
// formulas but shares their aims: exactly divergence-free, conservative, existing fine faces
// kept.)
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;
@group(0) @binding(9) var BF: texture_3d<f32>;    // old faces
@group(0) @binding(10) var BC: texture_3d<f32>;   // their compensations
@group(0) @binding(11) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(13) var OC: texture_storage_3d<rgba32float, write>;
@group(0) @binding(28) var IDX: texture_3d<u32>;
// The half-step stores the shared readers declare (read with weight 0 here: no time bins at a
// re-mesh, which comes at the end of a cycle).
@group(0) @binding(16) var HF: texture_3d<f32>;
@group(0) @binding(17) var H1: texture_3d<f32>;
@group(0) @binding(18) var H2: texture_3d<f32>;

#include "amr/cons.wgsl"
#include "amr/bins_off.wgsl"
#include "mhd/amr_common.wgsl"

const RG_STRIDE = 12u;

// The fine face (level lf, index gf) on a face of a coarse cell: an old leaf's at that level if
// one held it, else the coarse face prolonged with slopes. Where it is a plain copy, g_copied
// says so and g_copied_at where its compensation is.
fn outer_face(lf: u32, a: i32, gf: vec3<i32>) -> f32 {
  let held = face_lvl(lf, a, gf);
  if (held.y > 0.0) { return held.x; }
  var gc = gf >> vec3<u32>(1u);
  if (is2d()) { gc.z = 0; }
  let v = coarse_face_sloped(lf, a, gc, gf & vec3<i32>(1));
  // Prolonged, not copied (the coarse faces read on the way set g_copied for themselves).
  g_copied = false;
  return v;
}

// The face along a below fine cell gf (level lf, inside the old parent leaf), when gf's parity
// along a is 1: the face inside the coarse cell, from the projection described above.
fn inner_face(lf: u32, a: i32, gf: vec3<i32>) -> f32 {
  let base = (gf >> vec3<u32>(1u)) << vec3<u32>(1u);   // the coarse cell's first fine cell
  let d = select(3, 2, is2d());
  let nc = 1 << u32(d);
  // Outer faces, lower (0) and upper (1) side of each axis, by fine position across: index
  // ((a * 2 + side) * 2 + s1) * 2 + s2, s1 along a + 1 and s2 along a + 2.
  var outer: array<f32, 24>;
  var inner: array<f32, 12>;
  // Every outer face in one loop, (axis, side, s1, s2) from its index, so outer_face is compiled
  // into it once (see loop_count, mhd/amr_common.wgsl).
  for (var i = 0; i < loop_count(d * 8); i++) {
    let ax = i / 8;
    let side = (i / 4) & 1;
    let s1 = (i / 2) & 1;
    let s2 = i & 1;
    let t1 = (ax + 1) % 3;
    let t2 = (ax + 2) % 3;
    if (is2d() && ((t1 == 2 && s1 == 1) || (t2 == 2 && s2 == 1))) { continue; }
    var g = base;
    g[t1] += s1;
    g[t2] += s2;
    g[ax] += 2 * side;
    if (is2d()) { g.z = 0; }
    outer[((ax * 2 + side) * 2 + s1) * 2 + s2] = outer_face(lf, ax, g);
  }
  for (var ax = 0; ax < d; ax++) {
    for (var s = 0; s < 4; s++) {
      let lo = outer[(ax * 2 + 0) * 4 + s];
      let hi = outer[(ax * 2 + 1) * 4 + s];
      inner[ax * 4 + s] = 0.5 * (lo + hi);
    }
  }
  // Divergence of each fine cell c (bit k of c: its position along axis k) with those faces.
  var r: array<f32, 8>;
  for (var c = 0; c < nc; c++) {
    var div = 0.0;
    for (var ax = 0; ax < d; ax++) {
      let t1 = (ax + 1) % 3;
      let t2 = (ax + 2) % 3;
      let s1 = select(0, (c >> u32(t1)) & 1, t1 < d);
      let s2 = select(0, (c >> u32(t2)) & 1, t2 < d);
      let side = (c >> u32(ax)) & 1;
      let mid = inner[(ax * 2 + s1) * 2 + s2];
      if (side == 0) {
        div += mid - outer[((ax * 2 + 0) * 2 + s1) * 2 + s2];
      } else {
        div += outer[((ax * 2 + 1) * 2 + s1) * 2 + s2] - mid;
      }
    }
    r[c] = div;
  }
  // y = L^+ r through the Walsh transform.
  var y: array<f32, 8>;
  for (var c = 0; c < nc; c++) { y[c] = 0.0; }
  for (var s = 1; s < nc; s++) {
    var coef = 0.0;
    for (var c = 0; c < nc; c++) {
      coef += select(r[c], -r[c], (countOneBits(u32(s & c)) & 1u) == 1u);
    }
    coef /= f32(nc) * 2.0 * f32(countOneBits(u32(s)));
    for (var c = 0; c < nc; c++) {
      y[c] += select(coef, -coef, (countOneBits(u32(s & c)) & 1u) == 1u);
    }
  }
  // The face asked for: between the fine cell below (parity 0 along a) and the one above.
  let t1 = (a + 1) % 3;
  let t2 = (a + 2) % 3;
  let p = gf & vec3<i32>(1);
  let s1 = select(0, p[t1], t1 < d);
  let s2 = select(0, p[t2], t2 < d);
  var lo_c = 0;
  if (t1 < d) { lo_c |= s1 << u32(t1); }
  if (t2 < d) { lo_c |= s2 << u32(t2); }
  let hi_c = lo_c | (1 << u32(a));
  return inner[(a * 2 + s1) * 2 + s2] + y[hi_c] - y[lo_c];
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let row = flat_row(wid);
  let w = BS + 1;
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  if (row >= LIST[0] || i >= n) { return; }
  let slot = i32(LIST[1u + row]);
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w));
  if (is2d()) { l.z = 0; }
  let top = vec3<i32>(BS, BS, select(BS, 1, is2d()));
  let r = u32(slot) * RG_STRIDE;
  let kind = RG[r];
  let src = RG[r + 1u];
  var b = vec3<f32>(0.0);
  var comp = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var ok = l[a] <= top[a] && !(is2d() && a == 2 && l.z != 0);
    for (var k = 0; k < 3; k++) {
      if (k != a && l[k] >= top[k]) { ok = false; }
    }
    if (!ok) { continue; }
    if (kind == 2) {
      // A new child of the old leaf `src`, at the next level.
      let oct = RG[r + 2u];
      let lf = u32(bt(src, 0u)) + 1u;
      let cc = 2 * blk_coord(src) + vec3<i32>(oct & 1, (oct >> 1u) & 1, (oct >> 2u) & 1);
      var gf = cc * BS + l;
      if (is2d()) { gf.z = 0; }
      if ((gf[a] & 1) == 0 || (is2d() && a == 2)) {
        b[a] = outer_face(lf, a, gf);
        // A face copied from an old leaf takes its compensation too, so the two copies stay one pair.
        if (g_copied) { comp[a] = textureLoad(BC, g_copied_at, 0)[a]; }
      } else {
        b[a] = inner_face(lf, a, gf);
      }
    } else if (kind == 3) {
      // A merged block: the mean of its old children's faces on each of its faces.
      let h = vec3<i32>(2 * l >= vec3<i32>(BS)) * vec3<i32>(1, 1, select(1, 0, is2d()));
      let ch = RG[r + 4u + u32(h.x | (h.y << 1u) | (h.z << 2u))];
      let o = blk_origin(ch);
      let f = 2 * l - BS * h;
      let t1 = (a + 1) % 3;
      let t2 = (a + 2) % 3;
      let m1 = select(2, 1, is2d() && t1 == 2);
      let m2 = select(2, 1, is2d() && t2 == 2);
      var sum = 0.0;
      for (var q = 0; q < m2; q++) {
        for (var p = 0; p < m1; p++) {
          var x = f;
          x[t1] += p;
          x[t2] += q;
          sum += textureLoad(BF, atl(o, x), 0)[a];
        }
      }
      b[a] = sum / f32(m1 * m2);
      // On the block's own lower or upper face, where an old leaf at the merged block's level
      // holds that face beyond it, its copy: the two then agree bit for bit (the mean of the
      // fine faces equals it only to rounding).
      if (l[a] == 0 || l[a] == BS) {
        let lc = u32(bt(ch, 0u)) - 1u;
        let g = (blk_coord(ch) >> vec3<u32>(1u)) * BS + l;
        var out = g;
        out[a] = select(g[a] - 1, g[a], l[a] == BS);
        let n = n_at(lc);
        let inside = out[a] >= 0 && out[a] < n[a];
        if (inside || P.bc_lo[a] == 0u) {
          let mc = map_cell(lc, out);
          let hb = holder(lc, mc.g);
          if (hb.rel == 0) {
            var lb = local_of(hb.slot, mc.g);
            if (l[a] == 0) { lb[a] += 1; }
            b[a] = face_own(hb.slot, a, lb);
            comp[a] = textureLoad(BC, atl(blk_origin(hb.slot), lb), 0)[a];
          }
        }
      }
    } else {
      let x = atl(blk_origin(src), l);
      b[a] = textureLoad(BF, x, 0)[a];
      comp[a] = textureLoad(BC, x, 0)[a];
    }
  }
  let x = atl(blk_origin(slot), l);
  textureStore(OB, x, vec4<f32>(b, 0.0));
  textureStore(OC, x, vec4<f32>(comp, 0.0));
}
