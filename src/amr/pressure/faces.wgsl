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
  // (The assembly marks a solid cell by a negative density.)
  return abs(w.x) * w[a + 1];
}

// The mass flux through the face between a lower and an upper cell: one expression,
// whichever side asks (pressure/correct.wgsl). `kg` is dt times the uniform field along the
// normal and r_lo, r_hi the two cells' old densities: the field's kick on the face.
fn ap_mass_flux(m_lo: f32, m_hi: f32, p_lo: f32, p_hi: f32, dtdx: f32, kg: f32, r_lo: f32, r_hi: f32) -> f32 {
  return 0.5 * (m_lo + m_hi) + kg * 0.5 * (r_lo + r_hi) - dtdx * (p_hi - p_lo);
}
// A fine cell's ghost pressure toward a coarser cell.
fn ap_ghost(p_fine: f32, p_coarse: f32) -> f32 {
  return p_fine + AP_TWO_THIRDS * (p_coarse - p_fine);
}

// (pressure at the face, mass flux through it along +a, the density the field's kick takes
// there) for face (a, up) of leaf cell l of block `slot`, whose pressure is pc and old density
// rc. dtdx is dt over this block's cell size. With `solved` false the pressure terms are left
// out: the flux is the advected momentum's and the field's alone (the source).
//
// A uniform field (pressure/correct.wgsl): its kick is formed on the faces, dt g times the mean
// of the two cells' old densities (rho_old(), which the pass defines), and a wall's face
// pressure is the cell's carried half a cell down the field.
fn ap_face(slot: i32, l: vec3<i32>, a: i32, up: bool, pc: f32, rc: f32, dtdx: f32, solved: bool) -> vec3<f32> {
  let n = ap_nb(slot, l, a, up);
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  let mc = ap_mom(slot, l, a);
  let w = select(0.0, 1.0, solved);
  let kg = clock.dt * P.gext[a];
  let wall = vec3<f32>(pc + select(-0.5, 0.5, up) * blk_dx(slot) * P.gext[a] * rc, 0.0, rc);
  if (n.kind == 2) { return wall; }
  // An inlet: its own momentum goes through, and it pushes back as a wall does.
  if (n.kind == 4) { return vec3<f32>(wall.x, ap_mom(slot, l + e, a), rc); }
  if (n.kind == 3) {
    var m = mc + kg * rc;
    if (n.only) { m = select(min(m, 0.0), max(m, 0.0), up); }
    return vec3<f32>(0.0, m + w * select(-2.0, 2.0, up) * dtdx * pc, rc);
  }
  if (n.kind == 0 && n.fine) {
    // The fine faces, as the fine cells form them (their cell size is half this one's).
    var acc = vec3<f32>(0.0);
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      // A fine cell in an obstacle closes its face; one held at its state sends its momentum.
      let fm = ap_mark(f.slot, f.l);
      if (fm > 0.5) { acc += wall; continue; }
      if (fm < -0.5) { acc += vec3<f32>(wall.x, ap_mom(f.slot, f.l, a), rc); continue; }
      let pi = prs(f.slot, f.l);
      let ri = rho_old(f.slot, f.l);
      let pg = ap_ghost(pi, pc);
      let mi = ap_mom(f.slot, f.l, a);
      let mg = ap_mom(f.slot, f.l - e, a);
      // The field along the offset to this cell's center (common.wgsl ap_lift): out of the
      // face's pressure, and into its flux (the fine cell's ghost is on its side toward here).
      let lift = AP_TWO_THIRDS * ri * ap_lift(f.slot, f.l, a);
      let pt = pg - 0.5 * blk_dx(slot) * lift;
      let ft = select(1.0, -1.0, up) * clock.dt * lift;
      if (up) { acc += vec3<f32>(0.5 * (pt + pi), ft + ap_mass_flux(mg, mi, w * pg, w * pi, 2.0 * dtdx, kg, rc, ri), 0.5 * (rc + ri)); }
      else { acc += vec3<f32>(0.5 * (pi + pt), ft + ap_mass_flux(mi, mg, w * pi, w * pg, 2.0 * dtdx, kg, ri, rc), 0.5 * (ri + rc)); }
    }
    return acc / f32(nf);
  }
  // The same level, or coarser: the far momentum is this block's own cell or ghost.
  var pq = prs(n.slot, n.l);
  // Toward a coarser cell: the ghost, and the field along the offset to that cell's center.
  var pt = pq;
  var ft = 0.0;
  if (n.kind == 1) {
    pq = ap_ghost(pc, pq);
    let lift = AP_TWO_THIRDS * rc * ap_lift(slot, l, a);
    pt = pq - blk_dx(slot) * lift;
    ft = select(-1.0, 1.0, up) * clock.dt * lift;
  }
  let rq = rho_old(n.slot, n.l);
  let mq = ap_mom(slot, l + e, a);
  if (up) { return vec3<f32>(0.5 * (pc + pt), ft + ap_mass_flux(mc, mq, w * pc, w * pq, dtdx, kg, rc, rq), 0.5 * (rc + rq)); }
  return vec3<f32>(0.5 * (pt + pc), ft + ap_mass_flux(mq, mc, w * pq, w * pc, dtdx, kg, rq, rc), 0.5 * (rq + rc));
}
