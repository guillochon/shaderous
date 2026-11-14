// What the MHD Riemann solvers share: the face states, the conserved variables and physical
// fluxes of one side, and HLLE. Everything is in the frame where x is the face normal:
// v = (v_n, v_t1, v_t2), bt = (B_t1, B_t2), and the normal field bn is the face's own, the same
// on both sides.

// One side of a face: e_int is the internal energy per unit volume.
struct MSide { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32, bt: vec2<f32> }
// Flux through a face, in the rotated frame until mhd_face_flux turns the momentum back:
//   (m, mom) mass and momentum; e total energy; bt the tangential fields' fluxes,
//   F_n(B_t1) and F_n(B_t2); ei and s the internal energy's and the scalar's advective parts;
//   vn the face velocity, for the internal energy's p div v.
struct MFlux { m: f32, mom: vec3<f32>, e: f32, bt: vec2<f32>, ei: f32, s: f32, vn: f32 }
// Seven conserved variables: a = (rho, momentum), b = (E, B_t1, B_t2, unused).
struct C7 { a: vec4<f32>, b: vec4<f32> }

fn c7_add(x: C7, y: C7) -> C7 { return C7(x.a + y.a, x.b + y.b); }
fn c7_sub(x: C7, y: C7) -> C7 { return C7(x.a - y.a, x.b - y.b); }
fn c7_scale(x: C7, s: f32) -> C7 { return C7(x.a * s, x.b * s); }

// Total pressure p + B^2 / 2.
fn side_pt(s: MSide, bn: f32) -> f32 {
  return s.p + 0.5 * (bn * bn + dot(s.bt, s.bt));
}

// Fast magnetosonic speed along the normal.
fn side_cf(s: MSide, bn: f32) -> f32 {
  let a2 = s.cs * s.cs;
  let b2 = (bn * bn + dot(s.bt, s.bt)) / s.rho;
  let sum = a2 + b2;
  return sqrt(0.5 * (sum + sqrt(max(sum * sum - 4.0 * a2 * bn * bn / s.rho, 0.0))));
}

fn side_cons(s: MSide, bn: f32) -> C7 {
  let E = s.eint + 0.5 * s.rho * dot(s.v, s.v) + 0.5 * (bn * bn + dot(s.bt, s.bt));
  return C7(vec4<f32>(s.rho, s.rho * s.v), vec4<f32>(E, s.bt, 0.0));
}

fn side_flux(s: MSide, bn: f32) -> C7 {
  let pt = side_pt(s, bn);
  let E = s.eint + 0.5 * s.rho * dot(s.v, s.v) + 0.5 * (bn * bn + dot(s.bt, s.bt));
  let vb = s.v.x * bn + dot(s.v.yz, s.bt);
  let m = s.rho * s.v.x;
  return C7(
    vec4<f32>(m, m * s.v.x + pt - bn * bn, m * s.v.y - bn * s.bt.x, m * s.v.z - bn * s.bt.y),
    vec4<f32>((E + pt) * s.v.x - bn * vb, s.bt.x * s.v.x - bn * s.v.y, s.bt.y * s.v.x - bn * s.v.z, 0.0),
  );
}

// The MFlux of a set of conserved fluxes, with the face velocity vn; the advected internal
// energy and scalar are taken from the upwind side.
fn mflux_of(f: C7, vn: f32, l: MSide, r: MSide) -> MFlux {
  let up = f.a.x >= 0.0;
  return MFlux(f.a.x, f.a.yzw, f.b.x, f.b.yz, f.a.x * select(r.eint / r.rho, l.eint / l.rho, up), f.a.x * select(r.s, l.s, up), vn);
}

fn mhd_hlle(l: MSide, r: MSide, bn: f32) -> MFlux {
  let cfl = side_cf(l, bn);
  let cfr = side_cf(r, bn);
  let sl = min(l.v.x - cfl, r.v.x - cfr);
  let sr = max(l.v.x + cfl, r.v.x + cfr);
  let fl = side_flux(l, bn);
  let fr = side_flux(r, bn);
  if (sl >= 0.0) { return mflux_of(fl, l.v.x, l, r); }
  if (sr <= 0.0) { return mflux_of(fr, r.v.x, l, r); }
  let ul = side_cons(l, bn);
  let ur = side_cons(r, bn);
  let inv = 1.0 / (sr - sl);
  let f = c7_scale(c7_add(c7_sub(c7_scale(fl, sr), c7_scale(fr, sl)), c7_scale(c7_sub(ur, ul), sl * sr)), inv);
  // Face velocity: the contact speed estimate with total pressures.
  let ql = l.rho * (sl - l.v.x);
  let qr = r.rho * (sr - r.v.x);
  let vn = (side_pt(r, bn) - side_pt(l, bn) + l.v.x * ql - r.v.x * qr) / (ql - qr);
  return mflux_of(f, vn, l, r);
}
