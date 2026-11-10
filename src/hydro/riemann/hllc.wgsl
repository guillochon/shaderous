// HLLC flux in the frame where x is the face normal, blended toward HLL by w_hll in [0, 1].
// Wave speeds are the Davis estimates, so nothing here depends on the equation of state
// beyond the pressure, sound speed and internal energy carried in each Side.
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
    let sm = (r.p - l.p + l.v.x * ql - r.v.x * qr) / (ql - qr);

    // HLL
    let inv = 1.0 / (sr - sl);
    let hm = (sr * fml - sl * fmr + sl * sr * (r.rho - l.rho)) * inv;
    let hp = (sr * fpl - sl * fpr + sl * sr * (mr - ml)) * inv;
    let he = (sr * fel - sl * fer + sl * sr * (er - el)) * inv;

    // HLLC
    var cm: f32;
    var cp: vec3<f32>;
    var ce: f32;
    if (sm >= 0.0) {
      let f = ql / (sl - sm);
      let ms = f * vec3<f32>(sm, l.v.y, l.v.z);
      let es = f * (el / l.rho + (sm - l.v.x) * (sm + l.p / ql));
      cm = fml + sl * (f - l.rho);
      cp = fpl + sl * (ms - ml);
      ce = fel + sl * (es - el);
    } else {
      let f = qr / (sr - sm);
      let ms = f * vec3<f32>(sm, r.v.y, r.v.z);
      let es = f * (er / r.rho + (sm - r.v.x) * (sm + r.p / qr));
      cm = fmr + sr * (f - r.rho);
      cp = fpr + sr * (ms - mr);
      ce = fer + sr * (es - er);
    }
    out.m = mix(cm, hm, w_hll);
    out.mom = mix(cp, hp, w_hll);
    out.e = mix(ce, he, w_hll);
    out.vn = sm;
  }
  // Internal energy and the passive scalar ride on the mass flux, upwinded.
  let up = out.m >= 0.0;
  out.ei = out.m * select(r.eint / r.rho, l.eint / l.rho, up);
  out.s = out.m * select(r.s, l.s, up);
  return out;
}
