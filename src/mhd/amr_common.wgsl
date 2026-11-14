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
// Requires P, amr/grid.wgsl, amr/cons.wgsl (Cons, MaybeCons, prolong), <bins>, and bindings U1,
// U2 (conserved state), BF (faces), H1, H2, HF (the half-step stores) and IDX (texture_3d<u32>).

// Time: which state a block's stores stand for, as a block in bin g_recv_bin sees them at stage
// ASM_STAGE of its step (docs/mhd-plan.md, section 5b). A neighbor's value is the blend
// w.x S + w.y H of its state texture S (U1, U2, BF) and its half-step stores H (H1, H2, HF):
//   stage 1 (the receiver's start): an active block's start (S), a mid-step one's midpoint (H)
//   stage 2 (the receiver's midpoint): an active block in the same bin at its midpoint (H); one
//     bin faster, at the end of its first substep, extrapolated (2 H - S); one bin slower and
//     active, a quarter through (S + H) / 2; mid-step (slower), three quarters through, between
//     its midpoint and the end its state texture already holds (S + H) / 2
// Without time bins every block is active and in step: stage 1 reads S and stage 2 H.
// Requires <bins> and bindings H1, H2, HF.
var<private> g_recv_bin: u32 = 0u;
override ASM_STAGE: u32 = 1u;

fn src_weights(slot: i32) -> vec2<f32> {
  if (!BINS_ON) { return select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), ASM_STAGE == 1u); }
  let act = blk_active(slot);
  if (ASM_STAGE == 1u) { return select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), act); }
  if (!act) { return vec2<f32>(0.5, 0.5); }
  let k = blk_bin(slot);
  if (k == g_recv_bin) { return vec2<f32>(0.0, 1.0); }
  if (k < g_recv_bin) { return vec2<f32>(-1.0, 2.0); }
  return vec2<f32>(0.5, 0.5);
}

// Conserved state of interior cell l of leaf `slot`, at the time the receiver wants.
fn u_leaf_m(slot: i32, l: vec3<i32>) -> Cons {
  let c = atl(blk_origin(slot), l);
  let w = src_weights(slot);
  if (w.y == 0.0) { return Cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0)); }
  let h = Cons(textureLoad(H1, c, 0), textureLoad(H2, c, 0));
  if (w.x == 0.0) { return h; }
  let s = Cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
  let b = Cons(w.x * s.a + w.y * h.a, vec4<f32>(w.x * s.b.xyz + w.y * h.b.xyz, h.b.w));
  // An extrapolation that leaves no density or internal energy falls back to the half step.
  if (w.x < 0.0 && !(b.a.x > 0.0 && b.b.y > 0.0)) { return h; }
  return b;
}

// Interior cell l of any block, restricted from the children of a parent (amr/cons.wgsl's u_at).
fn u_at_m(slot: i32, l: vec3<i32>) -> Cons {
  if (blk_leaf(slot)) { return u_leaf_m(slot, l); }
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let ch = bt(slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  if (ch < 0 || !blk_leaf(ch)) { return Cons(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 0.0, 0.0)); }
  let f = 2 * (l - (BS / 2) * h);
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  let nz = select(2, 1, is2d());
  for (var k = 0; k < nz; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let u = u_leaf_m(ch, f + vec3<i32>(i, j, k));
        a += u.a;
        b += u.b;
      }
    }
  }
  let w = select(0.125, 0.25, is2d());
  return Cons(a * w, b * w);
}

// A cell of block `slot` that may lie one cell outside it (amr/cons.wgsl's u_near).
fn u_near_m(slot: i32, l: vec3<i32>) -> MaybeCons {
  let s = ghost_src(slot, l);
  if (s.kind != 0) { return MaybeCons(Cons(vec4<f32>(0.0), vec4<f32>(0.0)), false); }
  var u = u_at_m(s.slot, s.l);
  u.a = vec4<f32>(u.a.x, u.a.yzw * s.flip);
  return MaybeCons(u, true);
}

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
    u = u_leaf_m(h.slot, local_of(h.slot, m.g));
  } else if (h.rel == 1) {
    u = u_at_m(h.slot, local_of(h.slot, m.g));
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
    v[k] = u_near_m(cs, at);
  }
  return prolong(v[0].u, v[1], v[2], v[3], v[4], v[5], v[6], gf & vec3<i32>(1));
}

// The face texel of block `slot` holding face a of local cell l at the block's own level (l in
// the block's face range).
fn face_own(slot: i32, a: i32, l: vec3<i32>) -> f32 {
  let c = atl(blk_origin(slot), l);
  let w = src_weights(slot);
  if (w.y == 0.0) { return textureLoad(BF, c, 0)[a]; }
  if (w.x == 0.0) { return textureLoad(HF, c, 0)[a]; }
  return w.x * textureLoad(BF, c, 0)[a] + w.y * textureLoad(HF, c, 0)[a];
}

// The texel of the last face face_lvl returned as a plain copy of a leaf's own face (copied
// true), for a pass that also wants that face's compensation (mhd/amr_regrid.wgsl).
var<private> g_copied: bool = false;
var<private> g_copied_at: vec3<i32>;

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
  g_copied = false;
  if (h.rel == 0) {
    g_copied = sgn > 0.0;
    g_copied_at = atl(blk_origin(h.slot), local_of(h.slot, g));
    return vec2<f32>(sgn * face_own(h.slot, a, local_of(h.slot, g)), 1.0);
  }
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
      g_copied = sgn > 0.0;
      g_copied_at = atl(blk_origin(hb.slot), lb);
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
//
// The five faces are read in one loop, so face_lvl is compiled into it once: every call site of
// these functions is inlined, and written out five times here (and again in face_at, and in each
// of the assembly's face reads) the block grid's passes took seconds to compile. The loops'
// bounds come from loop_count() (amr/grid.wgsl), so the compiler keeps them loops.
fn coarse_face_sloped(level: u32, a: i32, gc: vec3<i32>, par: vec3<i32>) -> f32 {
  // 0: the face; 1, 2: below and above it along a + 1; 3, 4: along a + 2 (not along z in 2D).
  var f: array<vec2<f32>, 5>;
  for (var i = 0; i < loop_count(5); i++) {
    var e = vec3<i32>(0);
    if (i > 0) {
      let t = (a + 1 + (i - 1) / 2) % 3;
      if (is2d() && t == 2) { continue; }
      e[t] = select(1, -1, (i & 1) == 1);
    }
    f[i] = face_lvl(level - 1u, a, gc + e);
  }
  let v0 = f[0].x;
  var v = v0;
  for (var k = 1; k < 3; k++) {
    let t = (a + k) % 3;
    if (is2d() && t == 2) { continue; }
    let m = f[2 * k - 1];
    let p = f[2 * k];
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
  // On a coarse face, that face; between two, the mean of both (in one loop: see loop_count).
  let between = (g[a] & 1) != 0;
  var sum = 0.0;
  for (var j = 0; j < loop_count(select(1, 2, between)); j++) {
    var c = gc;
    c[a] += j;
    sum += coarse_face_sloped(level, a, c, par);
  }
  return sgn * select(sum, 0.5 * sum, between);
}

// The faces of global cell g at `level`: its lower faces, and the cell-centered field, the mean
// of its two faces along each axis. The six faces are read in one loop (see loop_count).
struct CellFaces { lo: vec3<f32>, B: vec3<f32> }
fn faces_at(level: u32, g: vec3<i32>) -> CellFaces {
  var f: array<f32, 6>;
  for (var i = 0; i < loop_count(6); i++) {
    let a = i % 3;
    var e = vec3<i32>(0);
    if (i >= 3 && !(a == 2 && is2d())) { e[a] = 1; }
    f[i] = face_at(level, a, g + e);
  }
  let lo = vec3<f32>(f[0], f[1], f[2]);
  return CellFaces(lo, 0.5 * (lo + vec3<f32>(f[3], f[4], f[5])));
}
fn bcell_at(level: u32, g: vec3<i32>) -> vec3<f32> {
  return faces_at(level, g).B;
}
