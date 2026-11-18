// Face values of the implicit step for a gas with an energy on blocks (the uniform grid's are in
// pressure/correct_gas.wgsl). Requires W1, W2 (the advected state's primitives, ghosts filled),
// PBT (the last step's solved pressure, carried with its ghosts: amr/pressure/lag_fill.wgsl), prs() (the solved change, or zero
// for the source), FLO and FHI (the face velocities the update kept), and
// amr/pressure/common.wgsl.
//
// Pressures here are less the reference pressure (P.ohm.w), which the correction adds back
// where a pressure multiplies a velocity.
//
// A face between a lower and an upper cell: its pressure is the density-weighted mean
// (rho_hi p_lo + rho_lo p_hi) / (rho_lo + rho_hi), its velocity the density-weighted mean of the
// two advected ones less dt (p_hi - p_lo) / (rho_f dx), and it carries, beyond the advective
// flux, (that velocity less the one that carried the flux) times the upwind advected state.
// Where the far side is coarser, the far cell is the block's ghost (interpolated by the
// assembly; its solved change the ghost of common.wgsl). A cell whose neighbor is refined takes
// the mean of the fine faces' values, formed from the fine blocks' own cells and ghosts: the
// same numbers, so mass, momentum and energy are conserved across levels.
struct GCell { rho: f32, v: vec3<f32>, eint: f32, s: f32, p: f32 }
// (pressure, velocity) of the face, and what it carries: (mass, momentum), (energy, internal
// energy, scalar).
struct GFace { pv: vec2<f32>, c1: vec4<f32>, c2: vec3<f32> }

// The advected state of cell l (or ghost) of a block, with pressure p.
fn g_cell(slot: i32, l: vec3<i32>, p: f32) -> GCell {
  let c = atl(blk_origin(slot), l);
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  return GCell(abs(w1.x), w1.yzw, w2.w, w2.z, p);
}
fn g_base(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PBT, atl(blk_origin(slot), l), 0).x;
}

// One piece of code, whichever cell asks. `uf` is the velocity that carried the advective flux,
// dtdx dt over the cell size of the level the face belongs to.
fn g_face(lo: GCell, hi: GCell, a: i32, uf: f32, dtdx: f32) -> GFace {
  let sum = lo.rho + hi.rho;
  let pf = (hi.rho * lo.p + lo.rho * hi.p) / sum;
  let vh = (lo.rho * lo.v[a] + hi.rho * hi.v[a]) / sum - dtdx * (2.0 / sum) * (hi.p - lo.p);
  let du = vh - uf;
  var d = lo;
  if (du < 0.0) { d = hi; }
  let e = d.eint + 0.5 * d.rho * dot(d.v, d.v);
  return GFace(vec2<f32>(pf, vh), du * vec4<f32>(d.rho, d.rho * d.v), du * vec3<f32>(e, d.eint, d.rho * d.s));
}

// Face (a, up) of leaf cell l of block `slot`, whose solved change is dc. With `solved` false
// the changes are left out (the source: the face velocity and the base pressure's flux).
fn ap_gface(slot: i32, l: vec3<i32>, a: i32, up: bool, dc: f32, dtdx: f32, solved: bool) -> GFace {
  let n = ap_nb(slot, l, a, up);
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  let w = select(0.0, 1.0, solved);
  let own = g_cell(slot, l, g_base(slot, l) + w * dc);
  let none = GFace(vec2<f32>(own.p, 0.0), vec4<f32>(0.0), vec3<f32>(0.0));
  if (n.kind == 2) { return none; }
  // An inlet sends its own velocity through; an open face lets the gas leave at its own.
  if (n.kind == 4) { return GFace(vec2<f32>(own.p, textureLoad(W1, atl(blk_origin(slot), l + e), 0)[a + 1]), vec4<f32>(0.0), vec3<f32>(0.0)); }
  if (n.kind == 3) {
    var v = own.v[a];
    if (n.only) { v = select(min(v, 0.0), max(v, 0.0), up); }
    return GFace(vec2<f32>(own.p, v), vec4<f32>(0.0), vec3<f32>(0.0));
  }
  if (n.kind == 0 && n.fine) {
    var acc = GFace(vec2<f32>(0.0), vec4<f32>(0.0), vec3<f32>(0.0));
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      let fm = ap_mark(f.slot, f.l);
      if (fm > 0.5) { acc.pv += none.pv; continue; }
      let fo = atl(blk_origin(f.slot), f.l);
      if (fm < -0.5) { acc.pv += vec2<f32>(own.p, textureLoad(W1, fo, 0)[a + 1]); continue; }
      // The fine cell and its ghost toward this cell, whose change is the ghost's coarse side.
      let di = w * prs(f.slot, f.l);
      let ci = g_cell(f.slot, f.l, g_base(f.slot, f.l) + di);
      // (The ghost's pressure is on the line between the two cells' whole pressures, base and
      // change together: the base's stored ghost is only the predictor's.)
      let cg = g_cell(f.slot, f.l - e, ci.p + AP_TWO_THIRDS * (own.p - ci.p));
      var g: GFace;
      if (up) { g = g_face(cg, ci, a, textureLoad(FLO, fo, 0)[a], 2.0 * dtdx); }
      else { g = g_face(ci, cg, a, textureLoad(FHI, fo, 0)[a], 2.0 * dtdx); }
      acc.pv += g.pv;
      acc.c1 += g.c1;
      acc.c2 += g.c2;
    }
    let k = 1.0 / f32(nf);
    return GFace(acc.pv * k, acc.c1 * k, acc.c2 * k);
  }
  // The same level, or coarser: the far cell is this block's own cell or ghost.
  var pq = g_base(slot, l + e) + w * prs(n.slot, n.l);
  if (n.kind == 1) { pq = own.p + AP_TWO_THIRDS * (g_base(n.slot, n.l) + w * prs(n.slot, n.l) - own.p); }
  let far = g_cell(slot, l + e, pq);
  let co = atl(blk_origin(slot), l);
  if (up) { return g_face(own, far, a, textureLoad(FHI, co, 0)[a], dtdx); }
  return g_face(far, own, a, textureLoad(FLO, co, 0)[a], dtdx);
}
