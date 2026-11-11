// HLLC with reduced dissipation at low Mach number, in the frame where x is the face normal,
// blended toward HLL by w_hll in [0, 1].
//
// A Godunov flux damps a velocity jump across a face at a rate set by the sound speed,
// whatever the flow speed, so in slow flow the damping swamps the flow itself. Following
// Thornber et al. (2008), the velocity jump handed to the solver is scaled by the local Mach
// number,
//   v_L, v_R  ->  (v_L + v_R) / 2  +-  z (v_L - v_R) / 2,    z = min(1, max(Z_MIN, |v_L| / c_L, |v_R| / c_R))
// which makes that damping proportional to the flow speed and leaves the solver as it is at
// Mach 1 and above, shocks included.
//
// Some of it has to stay. This scheme's time integration (one predictor, one update) is not
// stable on its own with the third-order reconstruction, which is unlimited in smooth slow
// flow: it relies on the solver's damping. A Gresho vortex at Mach 0.01 on 64 x 64 cells
// gains kinetic energy without bound with Z_MIN = 0.02 (within one time unit) and 0.1 (within
// four); with 0.2, 0.3 and 0.5 it decays, keeping 92%, 90% and 85% after four time units
// where plain HLLC keeps 75%. Z_MIN is the middle one, for a margin. So the fix here removes
// up to 70% of the damping of slow flow, not all of it. The same is why the damping of
// pressure jumps is left alone: scaling the acoustic wave speeds down in both (HLLC-LM,
// Fleischmann et al. 2020) blows the vortex up at Mach 0.03.
const Z_MIN: f32 = 0.3;

fn riemann_flux(l0: Side, r0: Side, w_hll: f32) -> Flux {
  var l = l0;
  var r = r0;
  let z = min(1.0, max(Z_MIN, max(length(l0.v) / l0.cs, length(r0.v) / r0.cs)));
  let mean = 0.5 * (l0.v + r0.v);
  let half = 0.5 * z * (l0.v - r0.v);
  l.v = mean + half;
  r.v = mean - half;
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
