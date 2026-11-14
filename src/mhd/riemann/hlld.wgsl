// HLLD flux for ideal MHD (Miyoshi & Kusano 2005), in the frame where x is the face normal.
// Five waves: the two fast waves, two rotational (Alfven) waves and the contact, so it resolves
// contacts and Alfven waves that HLLE smears. Next to flagged shocks (w_hll > 0) it blends
// toward HLLE, as HLLC does toward HLL. Requires mhd/riemann/common.wgsl.

// Star state (between a fast wave and the Alfven wave on its side).
struct Star { rho: f32, vt: vec2<f32>, bt: vec2<f32>, e: f32 }

fn hlld_star(s: MSide, bn: f32, sa: f32, sm: f32, pts: f32) -> Star {
  let rho = s.rho * (sa - s.v.x) / (sa - sm);
  let den = s.rho * (sa - s.v.x) * (sa - sm) - bn * bn;
  var vt = s.v.yz;
  var bt = s.bt;
  // Degenerate case (a fast wave meets the Alfven wave): the tangential state does not jump.
  if (abs(den) > 1.0e-6 * (s.rho * (sa - s.v.x) * (sa - s.v.x) + bn * bn)) {
    vt = s.v.yz - bn * s.bt * (sm - s.v.x) / den;
    bt = s.bt * (s.rho * (sa - s.v.x) * (sa - s.v.x) - bn * bn) / den;
  }
  let E = s.eint + 0.5 * s.rho * dot(s.v, s.v) + 0.5 * (bn * bn + dot(s.bt, s.bt));
  let vb = s.v.x * bn + dot(s.v.yz, s.bt);
  let vbs = sm * bn + dot(vt, bt);
  let e = ((sa - s.v.x) * E - side_pt(s, bn) * s.v.x + pts * sm + bn * (vb - vbs)) / (sa - sm);
  return Star(rho, vt, bt, e);
}

fn star_cons(t: Star, sm: f32) -> C7 {
  return C7(vec4<f32>(t.rho, t.rho * sm, t.rho * t.vt), vec4<f32>(t.e, t.bt, 0.0));
}

fn mhd_riemann(l: MSide, r: MSide, bn: f32, w_hll: f32) -> MFlux {
  if (w_hll >= 1.0) { return mhd_hlle(l, r, bn); }
  let cfl = side_cf(l, bn);
  let cfr = side_cf(r, bn);
  let sl = min(l.v.x - cfl, r.v.x - cfr);
  let sr = max(l.v.x + cfl, r.v.x + cfr);
  let fl = side_flux(l, bn);
  let fr = side_flux(r, bn);
  if (sl >= 0.0) { return mflux_of(fl, l.v.x, l, r); }
  if (sr <= 0.0) { return mflux_of(fr, r.v.x, l, r); }

  let ptl = side_pt(l, bn);
  let ptr = side_pt(r, bn);
  let ql = l.rho * (sl - l.v.x);
  let qr = r.rho * (sr - r.v.x);
  // Contact speed and the total pressure across the Riemann fan.
  let sm = (qr * r.v.x - ql * l.v.x - ptr + ptl) / (qr - ql);
  let pts = (qr * ptl - ql * ptr + ql * qr * (r.v.x - l.v.x)) / (qr - ql);
  let ls = hlld_star(l, bn, sl, sm, pts);
  let rs = hlld_star(r, bn, sr, sm, pts);
  let ul = side_cons(l, bn);
  let ur = side_cons(r, bn);
  let usl = star_cons(ls, sm);
  let usr = star_cons(rs, sm);
  let sql = sqrt(ls.rho);
  let sqr = sqrt(rs.rho);
  // Alfven waves.
  let sal = sm - abs(bn) / sql;
  let sar = sm + abs(bn) / sqr;

  var f: C7;
  if (sal >= 0.0) {
    f = c7_add(fl, c7_scale(c7_sub(usl, ul), sl));
  } else if (sar <= 0.0) {
    f = c7_add(fr, c7_scale(c7_sub(usr, ur), sr));
  } else {
    // Double-star states, between the Alfven waves.
    let sg = select(-1.0, 1.0, bn >= 0.0);
    let isum = 1.0 / (sql + sqr);
    let vt = (sql * ls.vt + sqr * rs.vt + (rs.bt - ls.bt) * sg) * isum;
    let bt = (sql * rs.bt + sqr * ls.bt + sql * sqr * (rs.vt - ls.vt) * sg) * isum;
    let vbss = sm * bn + dot(vt, bt);
    if (sm >= 0.0) {
      let e = ls.e - sql * (sm * bn + dot(ls.vt, ls.bt) - vbss) * sg;
      let uss = C7(vec4<f32>(ls.rho, ls.rho * sm, ls.rho * vt), vec4<f32>(e, bt, 0.0));
      let fs = c7_add(fl, c7_scale(c7_sub(usl, ul), sl));
      f = c7_add(fs, c7_scale(c7_sub(uss, usl), sal));
    } else {
      let e = rs.e + sqr * (sm * bn + dot(rs.vt, rs.bt) - vbss) * sg;
      let uss = C7(vec4<f32>(rs.rho, rs.rho * sm, rs.rho * vt), vec4<f32>(e, bt, 0.0));
      let fs = c7_add(fr, c7_scale(c7_sub(usr, ur), sr));
      f = c7_add(fs, c7_scale(c7_sub(uss, usr), sar));
    }
  }
  var out = mflux_of(f, sm, l, r);
  if (w_hll > 0.0) {
    let h = mhd_hlle(l, r, bn);
    out = MFlux(mix(out.m, h.m, w_hll), mix(out.mom, h.mom, w_hll), mix(out.e, h.e, w_hll), mix(out.bt, h.bt, w_hll),
                mix(out.ei, h.ei, w_hll), mix(out.s, h.s, w_hll), mix(out.vn, h.vn, w_hll));
  }
  return out;
}
