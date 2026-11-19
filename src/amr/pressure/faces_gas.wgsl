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
struct GCell { rho: f32, v: vec3<f32>, eint: f32, s: f32, p: f32, phi: f32, x: vec3<f32> }
// The gravity field: (acceleration, potential), ghosts filled (amr/gravity).
@group(0) @binding(23) var G: texture_3d<f32>;
// The point masses, whose pull each pass adds itself (gravity/ext.wgsl).
#include "gravity/ext_bind.wgsl"

// The acceleration at a cell: the uniform field's and the gas's own.
fn g_accel(slot: i32, l: vec3<i32>) -> vec3<f32> {
  var g = P.gext.xyz;
  if (P.grav_on != 0u) { g += textureLoad(G, atl(blk_origin(slot), l), 0).xyz; }
  return g + ext_accel(blk_pos(slot, l));
}
// ap_lift (common.wgsl) with the gas's own gravity too: the field along the offset from fine
// cell l of block `slot` to the coarse cell's center, in fine cells.
fn g_lift(slot: i32, l: vec3<i32>, a: i32) -> f32 {
  var s = ap_lift(slot, l, a);
  if (P.grav_on != 0u || ext_on()) {
    let par = (blk_coord(slot) * BS + l) & vec3<i32>(1);
    let g = textureLoad(G, atl(blk_origin(slot), l), 0).xyz + ext_accel(blk_pos(slot, l));
    let axes = ap_axes();
    for (var b = 0; b < axes; b++) {
      if (b != a) { s += g[b] * (0.5 - f32(par[b])); }
    }
  }
  return s;
}
// (pressure, velocity) of the face, what it carries: (mass, momentum), (energy, internal
// energy, scalar), and the mass flux through it, on which a uniform field does its work.
//
// A uniform field kicks the face velocities, as on the uniform grid (pressure/kick.wgsl): a
// face is at rest where p_hi - p_lo = g h (rho_lo + rho_hi) / 2. A wall's face pressure is the
// cell's carried half a cell down the field, and a fine cell's ghost toward a coarser cell
// leaves out the field's part of the difference to that cell's offset center (common.wgsl
// ap_lift).
struct GFace { pv: vec2<f32>, c1: vec4<f32>, c2: vec3<f32>, m: f32, k: f32 }

// The advected state of cell l (or ghost) of a block, with pressure p.
fn g_cell(slot: i32, l: vec3<i32>, p: f32) -> GCell {
  let c = atl(blk_origin(slot), l);
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  var phi = 0.0;
  if (P.grav_on != 0u) { phi = textureLoad(G, c, 0).w; }
  return GCell(abs(w1.x), w1.yzw, w2.w, w2.z, p, phi, blk_pos(slot, l));
}
fn g_base(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PBT, atl(blk_origin(slot), l), 0).x;
}

// The density of a fine cell's ghost toward a coarser cell. The fine face must be at rest in a
// field under one condition for both cells, since they share its pressure: the fine cell's
// pressure carried half a fine cell along the field at its own density, and the coarse cell's
// carried half a coarse cell back at its own, must meet. With the ghost's pressure two thirds
// of the way to the coarse cell's and the face's weights as between cells of one level, that
// holds when the face's density (rho_i + rho_g) / 2 is the mean along the line between the two
// centers, (rho_i + 2 rho_c) / 3: so rho_g = (4 rho_c - rho_i) / 3, kept positive across a
// jump. (With the assembly's interpolated density a column in a field crept at Mach 1e-3
// across a refinement face: the coarse cell beside it felt 0.3% of the field.)
fn g_ghost_rho(rho_i: f32, rho_c: f32) -> f32 {
  return max((4.0 * rho_c - rho_i) / 3.0, 0.25 * min(rho_i, rho_c));
}

// One piece of code, whichever cell asks. `uf` is the velocity that carried the advective flux,
// dtdx dt over the cell size of the level the face belongs to.
fn g_face(lo: GCell, hi: GCell, a: i32, uf: f32, dtdx: f32) -> GFace {
  let sum = lo.rho + hi.rho;
  let pf = (hi.rho * lo.p + lo.rho * hi.p) / sum;
  // The kick: the uniform field's, and the potential's difference across the face
  // (pressure/kick.wgsl).
  // (and the point masses', half a step ahead along their motion)
  var kick = clock.dt * P.gext[a] - dtdx * (hi.phi - lo.phi);
  if (ext_on()) {
    // (The box: its root blocks, of BS level-0 cells.)
    let len = f32(P.amr.y) * f32(BS) * P.dx;
    if (lo.x[a] < P.org[a] || hi.x[a] > P.org[a] + len) {
      // The face that wraps around a periodic box takes the two cells' pulls where they are
      // (pressure/kick.wgsl has why).
      var xl = lo.x;
      var xh = hi.x;
      if (xl[a] < P.org[a]) { xl[a] += len; }
      if (xh[a] > P.org[a] + len) { xh[a] -= len; }
      kick += clock.dt * 0.5 * (ext_accel(xl)[a] + ext_accel(xh)[a]);
    } else {
      kick -= dtdx * ext_dphi(lo.x, hi.x - lo.x, 0.5 * clock.dt);
    }
  }
  let vh = (lo.rho * lo.v[a] + hi.rho * hi.v[a]) / sum + kick - dtdx * (2.0 / sum) * (hi.p - lo.p);
  let du = vh - uf;
  var d = lo;
  if (du < 0.0) { d = hi; }
  let e = d.eint + 0.5 * d.rho * dot(d.v, d.v);
  return GFace(vec2<f32>(pf, vh), du * vec4<f32>(d.rho, d.rho * d.v), du * vec3<f32>(e, d.eint, d.rho * d.s), 0.5 * sum * vh, kick);
}

// Face (a, up) of leaf cell l of block `slot`, whose solved change is dc. With `solved` false
// the changes are left out (the source: the face velocity and the base pressure's flux).
fn ap_gface(slot: i32, l: vec3<i32>, a: i32, up: bool, dc: f32, dtdx: f32, solved: bool) -> GFace {
  let n = ap_nb(slot, l, a, up);
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  let w = select(0.0, 1.0, solved);
  let own = g_cell(slot, l, g_base(slot, l) + w * dc);
  let h = blk_dx(slot);
  let ga = g_accel(slot, l)[a];
  let kc = clock.dt * ga;
  let none = GFace(vec2<f32>(own.p + select(-0.5, 0.5, up) * h * ga * own.rho, 0.0), vec4<f32>(0.0), vec3<f32>(0.0), 0.0, kc);
  if (n.kind == 2) { return none; }
  // An inlet sends its own velocity through; an open face lets the gas leave at its own,
  // kicked by the field.
  if (n.kind == 4) { return GFace(vec2<f32>(none.pv.x, textureLoad(W1, atl(blk_origin(slot), l + e), 0)[a + 1]), vec4<f32>(0.0), vec3<f32>(0.0), 0.0, kc); }
  if (n.kind == 3) {
    var v = own.v[a] + kc;
    if (n.only) { v = select(min(v, 0.0), max(v, 0.0), up); }
    return GFace(vec2<f32>(own.p, v), vec4<f32>(0.0), vec3<f32>(0.0), own.rho * v, kc);
  }
  if (n.kind == 0 && n.fine) {
    var acc = GFace(vec2<f32>(0.0), vec4<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      let fm = ap_mark(f.slot, f.l);
      if (fm > 0.5) { acc.pv += none.pv; acc.k += kc; continue; }
      let fo = atl(blk_origin(f.slot), f.l);
      if (fm < -0.5) { acc.pv += vec2<f32>(none.pv.x, textureLoad(W1, fo, 0)[a + 1]); acc.k += kc; continue; }
      // The fine cell and its ghost toward this cell, whose change is the ghost's coarse side.
      let di = w * prs(f.slot, f.l);
      let ci = g_cell(f.slot, f.l, g_base(f.slot, f.l) + di);
      // (The ghost's pressure is on the line between the two cells' whole pressures, base and
      // change together: the base's stored ghost is only the predictor's.)
      var cg = g_cell(f.slot, f.l - e, ci.p + AP_TWO_THIRDS * (own.p - ci.p - own.rho * 0.5 * h * g_lift(f.slot, f.l, a)));
      cg.rho = g_ghost_rho(ci.rho, own.rho);
      var g: GFace;
      if (up) { g = g_face(cg, ci, a, textureLoad(FLO, fo, 0)[a], 2.0 * dtdx); }
      else { g = g_face(ci, cg, a, textureLoad(FHI, fo, 0)[a], 2.0 * dtdx); }
      acc.pv += g.pv;
      acc.c1 += g.c1;
      acc.c2 += g.c2;
      acc.m += g.m;
      acc.k += g.k;
    }
    let k = 1.0 / f32(nf);
    return GFace(acc.pv * k, acc.c1 * k, acc.c2 * k, acc.m * k, acc.k * k);
  }
  // The same level, or coarser: the far cell is this block's own cell or ghost.
  var pq = g_base(slot, l + e) + w * prs(n.slot, n.l);
  // (The field's part of the difference to an offset coarse cell, common.wgsl ap_lift, is
  // taken with the coarse cell's density here, not the fine cell's as for a liquid: the fine
  // cells' offsets then cancel in the mean of their face pressures, which is the coarse cell's
  // face pressure. With the fine cells' own densities a coarse cell beside a refinement face
  // along the field felt g h rho' / (8 rho) across it: Mach 1e-3 in a column that falls by e.)
  var rho_c = own.rho;
  if (n.kind == 1) {
    rho_c = abs(textureLoad(W1, atl(blk_origin(n.slot), n.l), 0).x);
    pq = own.p + AP_TWO_THIRDS * (g_base(n.slot, n.l) + w * prs(n.slot, n.l) - own.p - rho_c * h * g_lift(slot, l, a));
  }
  var far = g_cell(slot, l + e, pq);
  if (n.kind == 1) { far.rho = g_ghost_rho(own.rho, rho_c); }
  let co = atl(blk_origin(slot), l);
  if (up) { return g_face(own, far, a, textureLoad(FHI, co, 0)[a], dtdx); }
  return g_face(far, own, a, textureLoad(FLO, co, 0)[a], dtdx);
}
