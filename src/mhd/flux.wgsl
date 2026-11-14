// MHD face fluxes on the uniform grid, for one stage of the van Leer integrator (Stone &
// Gardiner 2009; docs/mhd-plan.md). Each invocation computes the fluxes through the lower
// faces of one cell, so every face is computed once and the two cells it separates read the
// same bits.
//
//   STAGE 1: first order, from the cells' own states at the start of the step.
//   STAGE 2: limited linear reconstruction of the primitives from the half-step state.
//
// The faces stored are those of cells t in [-1, n] along each axis whose index along the
// face's own normal is in [0, n] (in 2D, the single layer along z, and no z faces): the edge
// electric fields on the grid's boundary need the faces of one row of ghost cells beyond it.
// Ghost cells take their state through the boundary mapping, as everywhere.
//
// Flux buffer of axis a, three vec4 per face (mhd/flux_index.wgsl):
//   (mass, momentum)   (total energy, internal energy, scalar, face velocity)
//   (F_a(B_t1), F_a(B_t2), upwind weight, 0)
// with t1 = a + 1 and t2 = a + 2 (mod 3), and the upwind weight of Athena++ for the edge
// fields: 1 where the gas flows toward +a, 0 toward -a, smooth in between.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> FX: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> FY: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> FZ: array<vec4<f32>>;

override STAGE: u32 = 1u;

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"
#include "mhd/riemann/common.wgsl"
#include <mhd_riemann>

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
    s.cs = eos_sound_speed(s.rho, s.p);
  } else {
    let st = baro_state(s.rho);
    s.p = st.x;
    s.cs = st.y;
  }
  s.eint = eos_eint_from_p(s.rho, s.p);
  return s;
}

// Twice the increment from a cell's center to its face on the side of `near`, limited
// against `far`: the limiter's edge4 on both halves of the state.
fn rq_edge(far_a: vec4<f32>, near_a: vec4<f32>, far_b: vec4<f32>, near_b: vec4<f32>, c: MQ) -> Rq {
  let bs = sqrt(dot(c.B, c.B));
  return Rq(edge4(far_a, near_a, vec4<f32>(c.a.x, c.b.zzz)), edge4(far_b, near_b, vec4<f32>(c.b.x, 0.0, bs, bs)));
}

struct FaceOut { f: MFlux, w: f32 }

// The flux through the lower face along `axis` of cell t.
fn face(t: vec3<i32>, axis: i32) -> FaceOut {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let bn = bface(axis, t);
  let ql = mq_load(t - e);
  let qr = mq_load(t);
  var l = rq_of(ql, axis);
  var r = rq_of(qr, axis);
  if (STAGE == 2u) {
    let qll = mq_load(t - 2 * e);
    let qrr = mq_load(t + e);
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
  comp_use(comp_tex(bc_map(t - e).c));
  let sl = make_side(l, axis, lo);
  comp_use(comp_tex(bc_map(t).c));
  let sr = make_side(r, axis, lo);
  var F = mhd_riemann(sl, sr, bn, 0.0);
  F.mom = unrot(F.mom, axis);
  // Upwind weight for the edge fields (Athena++): the mass flux over a density, as a fraction of
  // a cell per step, saturated hard.
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  let vc = 1024.0 * dt * F.m / (g_dx * (sl.rho + sr.rho));
  return FaceOut(F, 0.5 + clamp(vc, -0.5, 0.5));
}

fn put(axis: i32, i: u32, o: FaceOut) {
  let a = vec4<f32>(o.f.m, o.f.mom);
  let b = vec4<f32>(o.f.e, o.f.ei, o.f.s, o.f.vn);
  let c = vec4<f32>(o.f.bt, o.w, 0.0);
  if (axis == 0) { FX[3u * i] = a; FX[3u * i + 1u] = b; FX[3u * i + 2u] = c; }
  if (axis == 1) { FY[3u * i] = a; FY[3u * i + 1u] = b; FY[3u * i + 2u] = c; }
  if (axis == 2) { FZ[3u * i] = a; FZ[3u * i + 1u] = b; FZ[3u * i + 2u] = c; }
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fd = flux_dims();
  if (any(vec3<i32>(gid) >= fd)) { return; }
  g_dx = P.dx;
  g_dt = clock.dt;
  let t = vec3<i32>(gid) - flux_offset();
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);
  let i = flux_index(t);
  for (var a = 0; a < axes; a++) {
    if (t[a] >= 0 && t[a] <= n[a]) { put(a, i, face(t, a)); }
  }
}
