// Face values for the source and the correction of the implicit step on blocks. Requires W1
// (the advected state's primitives, ghosts filled: amr/assemble.wgsl), prs(), and
// amr/pressure/common.wgsl.
//
// A face's momentum is the mean of its two cells' (density times velocity, from the same
// primitives whichever cell asks), its pressure the mean of their pressures, its mass flux
// the momentum less dt times the pressure difference across it. Where the far side is coarser,
// the far cell's momentum is the block's ghost there (interpolated by the assembly) and its
// pressure the ghost of common.wgsl. A cell whose neighbor is refined takes the mean of the
// fine faces' values, read from the fine blocks' own cells and ghosts: the same numbers.

// Momentum along axis a at cell l (which may be a ghost) of a block.
fn ap_mom(slot: i32, l: vec3<i32>, a: i32) -> f32 {
  let w = textureLoad(W1, atl(blk_origin(slot), l), 0);
  return w.x * w[a + 1];
}

// The mass flux through the face between a lower and an upper cell: one expression,
// whichever side asks (pressure/correct.wgsl).
fn ap_mass_flux(m_lo: f32, m_hi: f32, p_lo: f32, p_hi: f32, dtdx: f32) -> f32 {
  return 0.5 * (m_lo + m_hi) - dtdx * (p_hi - p_lo);
}
// A fine cell's ghost pressure toward a coarser cell.
fn ap_ghost(p_fine: f32, p_coarse: f32) -> f32 {
  return p_fine + AP_TWO_THIRDS * (p_coarse - p_fine);
}

// (pressure at the face, mass flux through it along +a) for face (a, up) of leaf cell l of
// block `slot`, whose pressure is pc. dtdx is dt over this block's cell size. With `solved`
// false the pressure terms are left out: the flux is the advected momentum's alone (the source).
fn ap_face(slot: i32, l: vec3<i32>, a: i32, up: bool, pc: f32, dtdx: f32, solved: bool) -> vec2<f32> {
  let n = ap_nb(slot, l, a, up);
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  let mc = ap_mom(slot, l, a);
  let w = select(0.0, 1.0, solved);
  if (n.kind == 2) { return vec2<f32>(pc, 0.0); }
  if (n.kind == 3) {
    var m = mc;
    if (n.only) { m = select(min(m, 0.0), max(m, 0.0), up); }
    return vec2<f32>(0.0, m + w * select(-2.0, 2.0, up) * dtdx * pc);
  }
  if (n.kind == 0 && n.fine) {
    // The fine faces, as the fine cells form them (their cell size is half this one's).
    var acc = vec2<f32>(0.0);
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      let pi = prs(f.slot, f.l);
      let pg = ap_ghost(pi, pc);
      let mi = ap_mom(f.slot, f.l, a);
      let mg = ap_mom(f.slot, f.l - e, a);
      if (up) { acc += vec2<f32>(0.5 * (pg + pi), ap_mass_flux(mg, mi, w * pg, w * pi, 2.0 * dtdx)); }
      else { acc += vec2<f32>(0.5 * (pi + pg), ap_mass_flux(mi, mg, w * pi, w * pg, 2.0 * dtdx)); }
    }
    return acc / f32(nf);
  }
  // The same level, or coarser: the far momentum is this block's own cell or ghost.
  var pq = prs(n.slot, n.l);
  if (n.kind == 1) { pq = ap_ghost(pc, pq); }
  let mq = ap_mom(slot, l + e, a);
  if (up) { return vec2<f32>(0.5 * (pc + pq), ap_mass_flux(mc, mq, w * pc, w * pq, dtdx)); }
  return vec2<f32>(0.5 * (pq + pc), ap_mass_flux(mq, mc, w * pq, w * pc, dtdx));
}
