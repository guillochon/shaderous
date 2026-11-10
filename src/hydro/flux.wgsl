// Face fluxes from half-step states and slopes. Requires P, the equation of state, the
// limiter, hydro/q.wgsl, hydro/pfloor.wgsl, g_dx set, and two loaders supplied by the grid:
//   load_q(c)     primitives of a cell
//   load_pred(c)  its half-step state: a = (rho, v), b = (p, scalar, shock flag, first-order flag)
struct Side { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32 }
struct Flux { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32 }

#include <riemann>

// Cyclic rotation that puts the face normal in x, and its inverse.
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

// State on one side of a face: the half-step state plus half the edge increment `s` towards
// that face (sgn = +1 for the cell's upper face, -1 for its lower). First-order cells have none.
fn to_side(pr: Q, s: Q, sgn: f32, axis: i32) -> Side {
  let hs = 0.5 * sgn * (1.0 - pr.b.w);
  let a = pr.a + hs * s.a;
  let b = pr.b.xy + hs * s.b.xy;
  var o: Side;
  o.rho = max(a.x, P.rho_floor);
  o.v = rot(a.yzw, axis);
  o.s = b.y;
  if (EOS_HAS_ENERGY) {
    o.p = max(b.x, P.p_floor);
    o.cs = eos_sound_speed(o.rho, o.p);
  } else {
    // Barotropic: the face pressure is a function of the face density, not reconstructed.
    let st = baro_state(o.rho);
    o.p = st.x;
    o.cs = st.y;
  }
  o.eint = eos_eint_from_p(o.rho, o.p);
  return o;
}

// Flux through the face between cells l and r, along +axis, in grid-aligned components.
fn face_flux(l: Q, ls: Q, r: Q, rs: Q, axis: i32) -> Flux {
  var w = max(l.b.z, r.b.z);            // carbuncle fix: HLL next to flagged shocks
  if (!EOS_HAS_ENERGY) { w = 1.0; }   // no contact wave to resolve (HLLC was tried: no different)
  var F = riemann_flux(to_side(l, ls, 1.0, axis), to_side(r, rs, -1.0, axis), w);
  F.mom = unrot(F.mom, axis);
  return F;
}

struct Faces { lo: Flux, hi: Flux }

// Both faces of cell c along one axis. w0 and p0 are the cell's own primitives and half-step state.
fn axis_fluxes(c: vec3<i32>, e: vec3<i32>, axis: i32, w0: Q, p0: Q) -> Faces {
  let wm2 = load_q(c - 2 * e);
  let wm1 = load_q(c - e);
  let wp1 = load_q(c + e);
  let wp2 = load_q(c + 2 * e);
  let sm = edge_up(wm2, wm1, w0);
  let lo = edge_dn(wm1, w0, wp1);
  let hi = edge_up(wm1, w0, wp1);
  let sp = edge_dn(w0, wp1, wp2);
  return Faces(
    face_flux(load_pred(c - e), sm, p0, lo, axis),
    face_flux(p0, hi, load_pred(c + e), sp, axis),
  );
}

// Flux through the single face between cell a and cell a + e. The same arithmetic as
// axis_fluxes on the same inputs, so the two agree bit for bit.
fn flux_between(a: vec3<i32>, e: vec3<i32>, axis: i32) -> Flux {
  let wl = load_q(a);
  let wr = load_q(a + e);
  let sl = edge_up(load_q(a - e), wl, wr);
  let sr = edge_dn(wl, wr, load_q(a + 2 * e));
  return face_flux(load_pred(a), sl, load_pred(a + e), sr, axis);
}

fn flux_zero() -> Flux {
  return Flux(0.0, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0);
}

// Net inflow through a cell's faces, per unit dt/dx.
struct Net { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32 }

fn net_inflow(fx: Faces, fy: Faces, fz: Faces) -> Net {
  return Net(
    (fx.lo.m - fx.hi.m) + (fy.lo.m - fy.hi.m) + (fz.lo.m - fz.hi.m),
    (fx.lo.mom - fx.hi.mom) + (fy.lo.mom - fy.hi.mom) + (fz.lo.mom - fz.hi.mom),
    (fx.lo.e - fx.hi.e) + (fy.lo.e - fy.hi.e) + (fz.lo.e - fz.hi.e),
    (fx.lo.ei - fx.hi.ei) + (fy.lo.ei - fy.hi.ei) + (fz.lo.ei - fz.hi.ei),
    (fx.lo.s - fx.hi.s) + (fy.lo.s - fy.hi.s) + (fz.lo.s - fz.hi.s),
    (fx.lo.vn - fx.hi.vn) + (fy.lo.vn - fy.hi.vn) + (fz.lo.vn - fz.hi.vn),
  );
}
