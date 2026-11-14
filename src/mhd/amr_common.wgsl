// MHD on the block grid: values of cells and faces anywhere, at a given level, whatever leaf
// holds them (docs/mhd-plan.md, section 4b).
//
// A ghost of a block is found from its global position: each axis is mapped through the
// domain's boundary (wrapped, mirrored or clamped), and the leaf index IDX (the leaf slot per
// finest-level block) gives the leaf that covers it. With 2:1 balance across faces and edges
// that leaf is at the same level (copy), one finer (restriction, through the same-level
// parent), or one coarser (prolongation).
//
// Faces: texel l of a block holds (b_x on the lower x face of cell l, b_y on its lower y face,
// b_z on its lower z face), for l in [0, BS] along the face's normal and [0, BS) across it;
// texel BS holds the block's upper faces. Prolongation is linear along each component's own
// axis and constant across it, which keeps a fine cell divergence-free inside a
// divergence-free coarse one; restriction is the area mean.
//
// Requires P, amr/grid.wgsl, amr/cons.wgsl (u_leaf, u_at, u_near, prolong), and bindings U1, U2
// (conserved state), BF (faces) and IDX (texture_3d<u32>).

// Cells per side of the domain at a level (1 along z in 2D).
fn n_at(level: u32) -> vec3<i32> {
  let n = i32(P.amr.y) * BS * (1 << level);
  return select(vec3<i32>(n), vec3<i32>(n, n, 1), is2d());
}

// A cell mapped into the domain: its index, the sign each velocity and field component takes
// (mirrored at a reflecting face), and for outflow-only faces the side (-1 / +1) to clamp the
// normal velocity toward.
struct GCell { g: vec3<i32>, flip: vec3<f32>, side: vec3<f32> }

fn map_cell(level: u32, g0: vec3<i32>) -> GCell {
  let n = n_at(level);
  var o = GCell(g0, vec3<f32>(1.0), vec3<f32>(0.0));
  for (var a = 0; a < 3; a++) {
    let i = g0[a];
    if (i >= 0 && i < n[a]) { continue; }
    let up = i >= n[a];
    let bc = select(P.bc_lo[a], P.bc_hi[a], up);
    if (bc == 0u) {
      o.g[a] = ((i % n[a]) + n[a]) % n[a];
    } else if (bc == 2u) {
      o.g[a] = select(-1 - i, 2 * n[a] - 1 - i, up);
      o.flip[a] = -1.0;
    } else {
      o.g[a] = select(0, n[a] - 1, up);
      if (bc == 3u) { o.side[a] = select(-1.0, 1.0, up); }
    }
  }
  return o;
}

// The block that holds global cell g (in the domain) at `level`: rel 0 a leaf at that level,
// 1 the same-level parent of finer leaves, -1 a leaf one level coarser.
struct Holder { slot: i32, rel: i32 }

fn holder(level: u32, g: vec3<i32>) -> Holder {
  let shift = P.amr.z - level;
  var fb = (g << vec3<u32>(shift)) / BS;
  if (is2d()) { fb.z = 0; }
  var s = i32(textureLoad(IDX, fb, 0).x);
  let ls = u32(bt(s, 0u));
  if (ls == level) { return Holder(s, 0); }
  if (ls < level) { return Holder(s, -1); }
  for (var k = ls; k > level; k--) { s = bt(s, 5u); }
  return Holder(s, 1);
}

// Local index in block `slot` of global cell (or face) index g at the block's level.
fn local_of(slot: i32, g: vec3<i32>) -> vec3<i32> {
  var l = g - blk_coord(slot) * BS;
  if (is2d()) { l.z = 0; }
  return l;
}

// Conserved state of global cell g0 (anywhere, ghosts beyond the domain included) at `level`,
// as a block at that level sees it, with velocity mirrored or clamped as the boundary says.
fn cons_at(level: u32, g0: vec3<i32>) -> Cons {
  let m = map_cell(level, g0);
  let h = holder(level, m.g);
  var u: Cons;
  if (h.rel == 0) {
    u = u_leaf(h.slot, local_of(h.slot, m.g));
  } else if (h.rel == 1) {
    u = u_at(h.slot, local_of(h.slot, m.g));
  } else {
    u = prolong_at(m.g, h.slot);
  }
  // A plain copy keeps its bits: two blocks then see the same numbers for the same cell, and
  // compute the faces and edges they share identically.
  if (all(m.flip == vec3<f32>(1.0)) && all(m.side == vec3<f32>(0.0))) { return u; }
  let v = bc_velocity(u.a.yzw / max(u.a.x, P.rho_floor), Mapped(vec3<i32>(0), m.flip, m.side));
  return Cons(vec4<f32>(u.a.x, v * u.a.x), u.b);
}

// Fine cell gf (global, in the domain) interpolated from the coarser leaf `cs` that covers it:
// amr/assemble.wgsl's from_coarse, for any fine cell.
fn prolong_at(gf: vec3<i32>, cs: i32) -> Cons {
  let nc = i32(P.amr.y) * BS * (1 << u32(bt(cs, 0u)));
  var gc = gf >> vec3<u32>(1u);
  gc = ((gc % nc) + nc) % nc;
  var lc = gc - blk_coord(cs) * BS;
  if (is2d()) { lc.z = 0; }
  var v: array<MaybeCons, 7>;
  let n = select(7, 5, is2d());
  for (var k = 0; k < n; k++) {
    var at = lc;
    if (k > 0) { at[(k - 1) / 2] += select(-1, 1, (k & 1) == 0); }
    v[k] = u_near(cs, at);
  }
  return prolong(v[0].u, v[1], v[2], v[3], v[4], v[5], v[6], gf & vec3<i32>(1));
}

// The face texel of block `slot` holding face a of local cell l at the block's own level (l in
// the block's face range).
fn face_own(slot: i32, a: i32, l: vec3<i32>) -> f32 {
  return textureLoad(BF, atl(blk_origin(slot), l), 0)[a];
}

// Face a with face index g0 (along a: a face index, 0 to n; across: cell indices) at `level`,
// held at that level or restricted from a finer one: (value, 1), or (0, 0) where the region is
// coarser. Across a reflecting face of the domain the normal field changes sign; an outflow
// face repeats the boundary face.
fn face_lvl(level: u32, a: i32, g0: vec3<i32>) -> vec2<f32> {
  let n = n_at(level);
  // Across: as cells. (The field normal to a is even under a mirror across another axis.)
  var g = g0;
  for (var b = 0; b < 3; b++) {
    if (b != a) { g[b] = map_cell(level, g0).g[b]; }
  }
  var sgn = 1.0;
  let i = g0[a];
  if (i < 0 || i > n[a]) {
    let up = i > n[a];
    let bc = select(P.bc_lo[a], P.bc_hi[a], up);
    if (bc == 0u) {
      g[a] = ((i % n[a]) + n[a]) % n[a];
    } else if (bc == 2u) {
      g[a] = select(-i, 2 * n[a] - i, up);
      sgn = -1.0;
    } else {
      g[a] = select(0, n[a], up);
    }
  } else if (i == n[a] && P.bc_hi[a] == 0u) {
    g[a] = 0;
  }
  // The cell whose lower face it is, or the last cell for the domain's upper face.
  var c = g;
  c[a] = min(g[a], n[a] - 1);
  let h = holder(level, c);
  if (h.rel == 0) { return vec2<f32>(sgn * face_own(h.slot, a, local_of(h.slot, g)), 1.0); }
  // A face between two cells is also the upper face of the cell below it (texel BS of its
  // block). Where the cell above is at another level, the cell below's block holds the face at
  // this level as it evolves it: on a refinement boundary, the fine block's own face, which
  // its fluxes must see, not one prolonged from the coarse side.
  if (g[a] > 0 || P.bc_lo[a] == 0u) {
    var cb = g;
    cb[a] = select(g[a] - 1, n[a] - 1, g[a] == 0);
    let hb = holder(level, cb);
    if (hb.rel == 0) {
      var lb = local_of(hb.slot, cb);
      lb[a] += 1;
      return vec2<f32>(sgn * face_own(hb.slot, a, lb), 1.0);
    }
  }
  if (h.rel == 1) {
    // Restriction: the mean of the fine faces covering it (two in 2D, four in 3D).
    var sum = 0.0;
    var count = 0.0;
    let t1 = (a + 1) % 3;
    let t2 = (a + 2) % 3;
    let m1 = select(2, 1, is2d() && t1 == 2);
    let m2 = select(2, 1, is2d() && t2 == 2);
    for (var j = 0; j < m2; j++) {
      for (var k = 0; k < m1; k++) {
        var gf = 2 * g;
        gf[t1] += k;
        gf[t2] += j;
        if (is2d()) { gf.z = 0; }
        var cf = gf;
        cf[a] = min(gf[a], 2 * n[a] - 1);
        let f = holder(level + 1u, cf);
        sum += face_own(f.slot, a, local_of(f.slot, gf));
        count += 1.0;
      }
    }
    return vec2<f32>(sgn * sum / count, 1.0);
  }
  return vec2<f32>(0.0, 0.0);
}

// A coarse face (at level - 1) prolonged to the fine face on it at fine parity `par` across it:
// its value plus a limited slope along each axis across it, a quarter of a coarse cell either
// way, so the fine faces on it keep its mean. A slope whose neighbor is not held at that level
// (coarser still, or beyond a closed boundary handled as copies) is zero.
fn coarse_face_sloped(level: u32, a: i32, gc: vec3<i32>, par: vec3<i32>) -> f32 {
  let v0 = face_lvl(level - 1u, a, gc).x;
  var v = v0;
  for (var k = 1; k < 3; k++) {
    let t = (a + k) % 3;
    if (is2d() && t == 2) { continue; }
    var e = vec3<i32>(0);
    e[t] = 1;
    let m = face_lvl(level - 1u, a, gc - e);
    let p = face_lvl(level - 1u, a, gc + e);
    if (m.y > 0.0 && p.y > 0.0) {
      let s = limit4(vec4<f32>(v0 - m.x, 0.0, 0.0, 0.0), vec4<f32>(p.x - v0, 0.0, 0.0, 0.0)).x;
      v += s * (f32(par[t]) - 0.5) * 0.5;
    }
  }
  return v;
}

// Face a with face index g0 at `level`, anywhere: held there, restricted from a finer level, or
// prolonged from a coarser one. Prolongation: a fine face on a coarse face takes the coarse
// face with limited slopes across it; a fine face between two coarse faces takes the mean of the
// two prolonged that way. (Linear along the normal: inside a divergence-free coarse cell the
// fine cells' divergence is then of the order of the slopes' curvature, and these are ghosts;
// the regridding of phase 3 uses Toth & Roe's exactly divergence-free formulas.)
fn face_at(level: u32, a: i32, g0: vec3<i32>) -> f32 {
  let own = face_lvl(level, a, g0);
  if (own.y > 0.0) { return own.x; }
  // Map into the domain at this level first, as face_lvl does, then go to the coarse indices.
  let n = n_at(level);
  var g = g0;
  for (var b = 0; b < 3; b++) {
    if (b != a) { g[b] = map_cell(level, g0).g[b]; }
  }
  var sgn = 1.0;
  let i = g0[a];
  if (i < 0 || i > n[a]) {
    let up = i > n[a];
    let bc = select(P.bc_lo[a], P.bc_hi[a], up);
    if (bc == 0u) {
      g[a] = ((i % n[a]) + n[a]) % n[a];
    } else if (bc == 2u) {
      g[a] = select(-i, 2 * n[a] - i, up);
      sgn = -1.0;
    } else {
      g[a] = select(0, n[a], up);
    }
  }
  var gc = g >> vec3<u32>(1u);
  if (is2d()) { gc.z = 0; }
  let par = g & vec3<i32>(1);
  if ((g[a] & 1) == 0) { return sgn * coarse_face_sloped(level, a, gc, par); }
  var up = gc;
  up[a] += 1;
  return sgn * 0.5 * (coarse_face_sloped(level, a, gc, par) + coarse_face_sloped(level, a, up, par));
}

// Cell-centered field of global cell g at `level`: the mean of its two faces along each axis.
fn bcell_at(level: u32, g: vec3<i32>) -> vec3<f32> {
  var B: vec3<f32>;
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    if (a == 2 && is2d()) { e = vec3<i32>(0); }
    B[a] = 0.5 * (face_at(level, a, g) + face_at(level, a, g + e));
  }
  return B;
}
