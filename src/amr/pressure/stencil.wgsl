// The operators on a level (amr/pressure/common.wgsl). Requires prs(slot, l), the pressure of
// a block's cell. Both return (sum, diag) with  h^2 (A p) = (diag + alpha h^2) p - sum.

// The plain operator: a refined neighbor counts as its parent's cell.
fn ap_plain(slot: i32, l: vec3<i32>) -> vec2<f32> {
  var s = vec2<f32>(0.0);
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    for (var u = 0; u < 2; u++) {
      let n = ap_nb(slot, l, a, u == 1);
      if (n.kind == 0) { s += vec2<f32>(prs(n.slot, n.l), 1.0); }
      if (n.kind == 1) { s += AP_TWO_THIRDS * vec2<f32>(prs(n.slot, n.l), 1.0); }
      if (n.kind == 3) { s.y += 2.0; }
    }
  }
  return s;
}

// The mean of the fine cells along a refined face.
fn ap_fine_mean(n: ANb, a: i32, up: bool) -> f32 {
  var m = 0.0;
  let nf = ap_nfine();
  for (var t = 0; t < nf; t++) {
    let f = ap_fine(n, a, up, t);
    m += prs(f.slot, f.l);
  }
  return m / f32(nf);
}

// (composite, plain) as two (sum, diag) pairs: they differ at refined faces only.
fn ap_both(slot: i32, l: vec3<i32>) -> vec4<f32> {
  var s = vec4<f32>(0.0);
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    for (var u = 0; u < 2; u++) {
      let n = ap_nb(slot, l, a, u == 1);
      if (n.kind == 0) {
        let own = vec2<f32>(prs(n.slot, n.l), 1.0);
        var comp = own;
        if (n.fine) { comp = 2.0 * AP_TWO_THIRDS * vec2<f32>(ap_fine_mean(n, a, u == 1), 1.0); }
        s += vec4<f32>(comp, own);
      }
      if (n.kind == 1) { s += AP_TWO_THIRDS * vec4<f32>(prs(n.slot, n.l), 1.0, prs(n.slot, n.l), 1.0); }
      if (n.kind == 3) { s += vec4<f32>(0.0, 2.0, 0.0, 2.0); }
    }
  }
  return s;
}
