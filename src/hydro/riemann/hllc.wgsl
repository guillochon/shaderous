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
  let fpl = ml * l.v.x + vec3<f32>(l.pd, 0.0, 0.0);
  let fpr = mr * r.v.x + vec3<f32>(r.pd, 0.0, 0.0);
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

    // HLL
    let inv = 1.0 / (sr - sl);
    let hm = (sr * fml - sl * fmr + sl * sr * (r.rho - l.rho)) * inv;
    let hp = (sr * fpl - sl * fpr + sl * sr * (mr - ml)) * inv;
    let he = (sr * fel - sl * fer + sl * sr * (er - el)) * inv;

    // Gas so cold that its sound speed is below the rounding of its velocity (c_s = 1e-7 at
    // v = 11: two cold streams pulled apart, a reflecting wall and its mirror image): sl - v.x
    // rounds to 0, ql or qr with it, and the contact speed below to 0 / 0. Otherwise ql < 0 < qr
    // always. The star region is then a vacuum, which HLL carries: take it.
    if (!(ql < 0.0 && qr > 0.0)) {
      out.m = hm; out.mom = hp; out.e = he; out.vn = 0.5 * (l.v.x + r.v.x);
      let up0 = out.m >= 0.0;
      out.ei = out.m * select(r.eint / r.rho, l.eint / l.rho, up0);
      out.s = out.m * select(r.s, l.s, up0);
      return out;
    }
    let sm = (r.pd - l.pd + l.v.x * ql - r.v.x * qr) / (ql - qr);

    // HLLC
    var cm: f32;
    var cp: vec3<f32>;
    var ce: f32;
    // The star state's differences from the outer state, U* - U. For a gas stored as its
    // deviation from a reference (hydro/ref_on.wgsl) they are written so that nothing large
    // cancels: with k = (S* - u) / (S - S*), rho* - rho = rho k, and so on. Formed as
    // rho (S - u) / (S - S*) - rho they round to 1e-7 of rho at a face at rest (S* = u = 0), a
    // mass flux of 1e-7 rho c every step that waves carry up a stratified column, where they
    // grow as rho^(-1/2) (docs/numerics.md section 7): a column against its reference held at
    // Mach 1.7 instead of 1e-8. Without a reference that rounding is one among others of its
    // size, and the form is the original (the two differ only in rounding, and the axisymmetric
    // column of the validation suite, at that floor, went from Mach 7.5e-5 to 1.7e-4).
    if (REF_DEV) {
      if (sm >= 0.0) {
        let k = (sm - l.v.x) / (sl - sm);
        let dr = l.rho * k;
        let f = l.rho + dr;
        cm = fml + sl * dr;
        cp = fpl + sl * vec3<f32>(sm * dr + l.rho * (sm - l.v.x), dr * l.v.y, dr * l.v.z);
        ce = fel + sl * (el * k + f * (sm - l.v.x) * (sm + l.p / ql));
      } else {
        let k = (sm - r.v.x) / (sr - sm);
        let dr = r.rho * k;
        let f = r.rho + dr;
        cm = fmr + sr * dr;
        cp = fpr + sr * vec3<f32>(sm * dr + r.rho * (sm - r.v.x), dr * r.v.y, dr * r.v.z);
        ce = fer + sr * (er * k + f * (sm - r.v.x) * (sm + r.p / qr));
      }
    } else if (sm >= 0.0) {
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
