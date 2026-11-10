// Unsplit MUSCL-Hancock update, gather form: each cell rebuilds the reconstruction of itself
// and its six neighbours (25-cell stencil) and evaluates its own six face fluxes.
//
// Performance note: everything here is straight-line code on locals. Caching the stencil in a
// shader-local array to avoid repeated loads was measured 3x slower (D3D12, RTX 3080); the
// dynamic indexing costs far more than the duplicate texture fetches.
//
// Conservation depends on the two cells sharing a face computing identical fluxes: a face is
// the upper face of one cell and the lower face of the next, which are different call sites
// of face_flux(). predict() and face_flux() are pure functions of the cell coordinates, so
// the results should match bit for bit, and the validation suite checks that they do
// (DEBUG_FLUX below).
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>

struct Side { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32 }
struct Flux { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32 }

#include <riemann>
#include "gravity/accel.wgsl"
#include "hydro/dual_energy.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var W1: texture_3d<f32>;
@group(0) @binding(5) var W2: texture_3d<f32>;
@group(0) @binding(6) var PHI: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var PHI_OLD: texture_3d<f32>;   // potential one step back

// Potential extrapolated to the middle of the step, for the time-centred energy source.
fn phi_half(q: vec3<i32>) -> f32 {
  return 1.5 * phi_ext(PHI, q) - 0.5 * phi_ext(PHI_OLD, q);
}

// Primitive state: a = (rho, vx, vy, vz), b = (p, scalar, c_s, 0).
struct Q { a: vec4<f32>, b: vec4<f32> }

// Half-step-predicted cell state, its limited slopes, and the shock flag.
struct Pred { q: Q, sx: Q, sy: Q, sz: Q, flag: f32 }

fn load_q(c: vec3<i32>) -> Q {
  let m = bc_map(c);
  let w1 = textureLoad(W1, m.c, 0);
  let w2 = textureLoad(W2, m.c, 0);
  return Q(vec4<f32>(w1.x, w1.yzw * m.flip), vec4<f32>(w2.x, w2.z, w2.y, 0.0));
}

fn slope(l: Q, c: Q, r: Q) -> Q {
  return Q(limit4(c.a - l.a, r.a - c.a), limit4(c.b - l.b, r.b - c.b));
}

fn predict(c: vec3<i32>) -> Pred {
  let q = load_q(c);
  let xm = load_q(c + vec3<i32>(-1, 0, 0));
  let xp = load_q(c + vec3<i32>(1, 0, 0));
  let ym = load_q(c + vec3<i32>(0, -1, 0));
  let yp = load_q(c + vec3<i32>(0, 1, 0));
  let zm = load_q(c + vec3<i32>(0, 0, -1));
  let zp = load_q(c + vec3<i32>(0, 0, 1));

  var out: Pred;
  out.sx = slope(xm, q, xp);
  out.sy = slope(ym, q, yp);
  out.sz = slope(zm, q, zp);

  // Shock detector: converging flow with a large relative pressure jump.
  let conv = (xp.a.y - xm.a.y) + (yp.a.z - ym.a.z) + (zp.a.w - zm.a.w);
  let jump = max(
    abs(xp.b.x - xm.b.x) / min(xp.b.x, xm.b.x),
    max(abs(yp.b.x - ym.b.x) / min(yp.b.x, ym.b.x), abs(zp.b.x - zm.b.x) / min(zp.b.x, zm.b.x)),
  );
  out.flag = select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);

  // Hancock predictor in primitive form, plus half a gravity kick.
  let h = 0.5 * clock.dt / P.dx;
  let rho = q.a.x;
  let v = q.a.yzw;
  let cs = q.b.z;
  let dv = out.sx.a.y + out.sy.a.z + out.sz.a.w;
  let da = v.x * out.sx.a + v.y * out.sy.a + v.z * out.sz.a;
  let db = v.x * out.sx.b + v.y * out.sy.b + v.z * out.sz.b;
  let gradp = vec3<f32>(out.sx.b.x, out.sy.b.x, out.sz.b.x);
  let m = bc_map(c);
  let g = gravity(PHI, m.c) * m.flip;

  out.q.a = vec4<f32>(
    rho - h * (da.x + rho * dv),
    v - h * (da.yzw + gradp / rho) + 0.5 * clock.dt * g,
  );
  out.q.b = vec4<f32>(q.b.x - h * (db.x + rho * cs * cs * dv), q.b.y - h * db.y, cs, 0.0);

  // Positivity: if any face state would be non-positive, this cell is first order this step.
  let rmin = out.q.a.x - 0.5 * max(abs(out.sx.a.x), max(abs(out.sy.a.x), abs(out.sz.a.x)));
  let pmin = out.q.b.x - 0.5 * max(abs(out.sx.b.x), max(abs(out.sy.b.x), abs(out.sz.b.x)));
  if (!(rmin > 0.0 && pmin > 0.0)) {
    let zero = Q(vec4<f32>(0.0), vec4<f32>(0.0));
    out.q = q;
    out.sx = zero;
    out.sy = zero;
    out.sz = zero;
  }
  return out;
}

fn face(pr: Pred, axis: i32, sgn: f32) -> Q {
  var s = pr.sx;
  if (axis == 1) { s = pr.sy; }
  if (axis == 2) { s = pr.sz; }
  let hs = 0.5 * sgn;
  return Q(pr.q.a + hs * s.a, pr.q.b + hs * s.b);
}

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

fn to_side(q: Q, axis: i32) -> Side {
  var s: Side;
  s.rho = max(q.a.x, P.rho_floor);
  s.v = rot(q.a.yzw, axis);
  s.p = max(q.b.x, P.p_floor);
  s.s = q.b.y;
  s.cs = eos_sound_speed(s.rho, s.p);
  s.eint = eos_eint_from_p(s.rho, s.p);
  return s;
}

// Flux through the face between two cells, along +axis, in grid-aligned components.
fn face_flux(l: Pred, r: Pred, axis: i32) -> Flux {
  var w = max(l.flag, r.flag);          // carbuncle fix: HLL next to flagged shocks
  if (!EOS_HAS_ENERGY) { w = 1.0; }
  var F = riemann_flux(to_side(face(l, axis, 1.0), axis), to_side(face(r, axis, -1.0), axis), w);
  F.mom = unrot(F.mom, axis);
  return F;
}

// Validation hook: when set, the pass writes the six face mass fluxes instead of the new state,
// so the test suite can check that both cells sharing a face computed the same bits.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= vec3<u32>(P.n))) { return; }
  let c = vec3<i32>(gid);
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  let p0 = predict(c);
  // Flux through each face, along +axis. "l" is the lower face, "h" the upper.
  let fxl = face_flux(predict(c - ex), p0, 0);
  let fxh = face_flux(p0, predict(c + ex), 0);
  let fyl = face_flux(predict(c - ey), p0, 1);
  let fyh = face_flux(p0, predict(c + ey), 1);
  let fzl = face_flux(predict(c - ez), p0, 2);
  let fzh = face_flux(p0, predict(c + ez), 2);

  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fxl.m, fxh.m, fyl.m, fyh.m));
    textureStore(O2, c, vec4<f32>(fzl.m, fzh.m, 0.0, 0.0));
    return;
  }

  // Net inflow per unit dt/dx.
  let dm = (fxl.m - fxh.m) + (fyl.m - fyh.m) + (fzl.m - fzh.m);
  let dmom = (fxl.mom - fxh.mom) + (fyl.mom - fyh.mom) + (fzl.mom - fzh.mom);
  let de = (fxl.e - fxh.e) + (fyl.e - fyh.e) + (fzl.e - fzh.e);
  let dei = (fxl.ei - fxh.ei) + (fyl.ei - fyh.ei) + (fzl.ei - fzh.ei);
  let ds = (fxl.s - fxh.s) + (fyl.s - fyh.s) + (fzl.s - fzh.s);
  let dvn = (fxl.vn - fxh.vn) + (fyl.vn - fyh.vn) + (fzl.vn - fzh.vn);

  // Flux-form gravitational work (P.flags bit 0): each face hands half of F_mass * dphi to the
  // two cells it separates, so the sum over cells telescopes to -d/dt of the potential energy.
  let flux_work = P.grav_on != 0u && (P.flags & 1u) != 0u;
  var dw = 0.0;
  if (flux_work) {
    let pc = phi_half(c);
    // (Giving each face's work wholly to the downstream cell was tried; it conserves just as
    // well in principle but heats low-density cells more and was worse on the Evrard test.)
    dw = -0.5 * (fxl.m * (pc - phi_half(c - ex)) + fxh.m * (phi_half(c + ex) - pc)
               + fyl.m * (pc - phi_half(c - ey)) + fyh.m * (phi_half(c + ey) - pc)
               + fzl.m * (pc - phi_half(c - ez)) + fzh.m * (phi_half(c + ez) - pc));
  }

  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let dtdx = clock.dt / P.dx;

  // Gravity is kick-drift-kick around the flux update: half a kick with the old potential here,
  // the other half in gravity/correct.wgsl once the new potential is known. (Applying the whole
  // kick after the fluxes would drain 1/2 rho g^2 dt^2 of internal energy per step from gas in
  // hydrostatic balance.) The energy source is either the flux-form work accumulated above, or
  // exactly the kinetic energy the kick adds, k . (m + k/2) / rho, which keeps gravity out of
  // the internal energy at the price of total-energy conservation.
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  if (P.grav_on != 0u) {
    kick = 0.5 * clock.dt * u1.x * gravity(PHI, c);
    if (!flux_work) { ekick = dot(kick, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor); }
  }

  let rho = max(u1.x + dtdx * dm, P.rho_floor);
  let mom = u1.yzw + kick + dtdx * dmom;
  let E = u2.x + ekick + dtdx * (de + dw);
  // Internal energy: advection plus -p div(v), with the half-step pressure and face velocities.
  let eint = u2.y + dtdx * (dei + max(p0.q.b.x, P.p_floor) * dvn);
  let sc = u2.z + dtdx * ds;

  let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);

  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(en.x, en.y, sc, 0.0));
}
