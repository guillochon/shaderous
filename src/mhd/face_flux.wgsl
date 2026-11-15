// The flux through one face for MHD, from the primitives of the two cells on each side
// (mhd/flux.wgsl on the uniform grid, mhd/amr_flux.wgsl on blocks): first order, or (second)
// a limited linear reconstruction of the primitives, then the Riemann solver, falling back to
// HLLE where w (the shock flags) asks. Also the upwind weight the edge fields use.
// Requires P, the equation of state, the limiter, hydro/pfloor.wgsl, mhd/common.wgsl (MQ),
// mhd/riemann/common.wgsl, the <mhd_riemann> slot and g_dx set.

fn rot(v: vec3<f32>, axis: i32) -> vec3<f32> {
  if (axis == 1) { return v.yzx; }
  if (axis == 2) { return v.zxy; }
  return v;
}
fn unrot(v: vec3<f32>, axis: i32) -> vec3<f32> {
  if (axis == 1) { return v.zxy; }
  if (axis == 2) { return v.yzx; }
  return v;
}

// Primitives to reconstruct: (rho, v) and (p, scalar, B_t1, B_t2), the field in the face's frame.
struct Rq { a: vec4<f32>, b: vec4<f32> }

fn rq_of(q: MQ, axis: i32) -> Rq {
  let B = rot(q.B, axis);
  return Rq(q.a, vec4<f32>(q.b.xy, B.yz));
}

// A side of the face from reconstructed primitives (in grid components) and the cell's sound
// speed as a fallback for a barotropic gas.
fn make_side(r: Rq, axis: i32, lo: vec2<f32>) -> MSide {
  var s: MSide;
  s.rho = max(r.a.x, max(lo.x, P.rho_floor));
  s.v = rot(r.a.yzw, axis);
  s.s = r.b.y;
  s.bt = r.b.zw;
  if (EOS_HAS_ENERGY) {
    s.p = max(r.b.x, max(lo.y, P.p_floor));
    let f = eos_face(s.rho, s.p);
    s.cs = f.x;
    s.eint = f.y;
  } else {
    let st = baro_state(s.rho);
    s.p = st.x;
    s.cs = st.y;
    s.eint = eos_eint_from_p(s.rho, s.p);
  }
  return s;
}

// Twice the increment from a cell's center to its face on the side of `near`, limited
// against `far`: the limiter's edge4 on both halves of the state.
fn rq_edge(far_a: vec4<f32>, near_a: vec4<f32>, far_b: vec4<f32>, near_b: vec4<f32>, c: MQ) -> Rq {
  let bs = sqrt(dot(c.B, c.B));
  return Rq(edge4(far_a, near_a, vec4<f32>(c.a.x, c.b.zzz)), edge4(far_b, near_b, vec4<f32>(c.b.x, 0.0, bs, bs)));
}

struct FaceOut { f: MFlux, w: f32 }

// First-order flux correction, decided face by face (after Athena++'s, which decides it cell by
// cell after the update). Stage 2 applies fluxes of the half-step state to the state at the
// start. Where stage 1 has filled a nearly empty cell (next to the edge of a star in
// near-vacuum: 350 times its density in half a step), the half-step cell pushes out through
// its other faces far more than the start held, and the update empties it while the energy that
// flowed in stays: floored pressures, speeds of 5 in the ambient gas that set the timestep, a
// passive scalar below zero. A cell whose density stage 1 changed by more than a factor of 4
// either way (more than a strong shock compresses gas) is suspect, and every face of a suspect
// cell takes the first-order HLLE flux of the state at the start instead: for that cell, the
// first-order step over dt, which keeps it positive. Both cells of a face (and both slabs or
// blocks that hold it) decide from the same two cells, so the flux stays shared: mass, momentum
// and energy are conserved, and the edge fields, from these fluxes, keep the field's divergence.
const FOFC_RATIO = 4.0;
fn fofc_suspect(start: f32, half: f32) -> bool {
  return half > FOFC_RATIO * start || start > FOFC_RATIO * half;
}

// The flux through the face between cells ql and qr along `axis` (qll and qrr: the next ones
// out), whose normal field is bn; kl and kr are the two cells' composition moments, dt the
// stage's step.
fn face_of(qll: MQ, ql: MQ, qr: MQ, qrr: MQ, bn: f32, axis: i32, second: bool, w: f32, kl: vec4<f32>, kr: vec4<f32>, dt: f32) -> FaceOut {
  var l = rq_of(ql, axis);
  var r = rq_of(qr, axis);
  if (second) {
    let ll = rq_of(qll, axis);
    let rr = rq_of(qrr, axis);
    let sl = rq_edge(l.a - ll.a, r.a - l.a, l.b - ll.b, r.b - l.b, ql);
    let sr = rq_edge(rr.a - r.a, r.a - l.a, rr.b - r.b, r.b - l.b, qr);
    let fl = Rq(l.a + 0.5 * sl.a, l.b + 0.5 * sl.b);
    let fr = Rq(r.a - 0.5 * sr.a, r.b - 0.5 * sr.b);
    // A face state with no density or pressure left: that side is first order.
    if (fl.a.x > 0.0 && fl.b.x > 0.0) { l = fl; }
    if (fr.a.x > 0.0 && fr.b.x > 0.0) { r = fr; }
  }
  // Neither side may fall below half the smaller of the two cells' own values.
  let lo = 0.5 * min(vec2<f32>(ql.a.x, ql.b.x), vec2<f32>(qr.a.x, qr.b.x));
  comp_use(kl);
  let sl = make_side(l, axis, lo);
  comp_use(kr);
  let sr = make_side(r, axis, lo);
  // Next to a shock that stage 1 flagged, HLLD falls back to HLLE, as HLLC does to HLL:
  // carbuncles and odd-even decoupling grow from too little dissipation along a shock front.
  var F = mhd_riemann(sl, sr, bn, w);
  F.mom = unrot(F.mom, axis);
  // Upwind weight for the edge fields (Athena++): the mass flux over a density, as a fraction of
  // a cell per step, saturated hard.
  let vc = 1024.0 * dt * F.m / (g_dx * (sl.rho + sr.rho));
  return FaceOut(F, 0.5 + clamp(vc, -0.5, 0.5));
}
