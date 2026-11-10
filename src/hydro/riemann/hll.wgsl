// HLL flux in the frame where x is the face normal. w_hll is ignored.
fn riemann_flux(l: Side, r: Side, w_hll: f32) -> Flux {
  let sl = min(l.v.x - l.cs, r.v.x - r.cs);
  let sr = max(l.v.x + l.cs, r.v.x + r.cs);

  let el = l.eint + 0.5 * l.rho * dot(l.v, l.v);
  let er = r.eint + 0.5 * r.rho * dot(r.v, r.v);
  let ml = l.rho * l.v;
  let mr = r.rho * r.v;

  let fml = ml.x;
  let fmr = mr.x;
  let fpl = ml * l.v.x + vec3<f32>(l.p, 0.0, 0.0);
  let fpr = mr * r.v.x + vec3<f32>(r.p, 0.0, 0.0);
  let fel = (el + l.p) * l.v.x;
  let fer = (er + r.p) * r.v.x;

  var out: Flux;
  if (sl >= 0.0) {
    out.m = fml; out.mom = fpl; out.e = fel; out.vn = l.v.x;
  } else if (sr <= 0.0) {
    out.m = fmr; out.mom = fpr; out.e = fer; out.vn = r.v.x;
  } else {
    let ql = l.rho * (sl - l.v.x);
    let qr = r.rho * (sr - r.v.x);
    let inv = 1.0 / (sr - sl);
    out.m = (sr * fml - sl * fmr + sl * sr * (r.rho - l.rho)) * inv;
    out.mom = (sr * fpl - sl * fpr + sl * sr * (mr - ml)) * inv;
    out.e = (sr * fel - sl * fer + sl * sr * (er - el)) * inv;
    out.vn = (r.p - l.p + l.v.x * ql - r.v.x * qr) / (ql - qr);
  }
  let up = out.m >= 0.0;
  out.ei = out.m * select(r.eint / r.rho, l.eint / l.rho, up);
  out.s = out.m * select(r.s, l.s, up);
  return out;
}
