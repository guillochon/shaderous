// Unsplit MUSCL-Hancock update, gather form: each cell evaluates its own six face fluxes from
// the half-step states written by hydro/predict.wgsl and slopes recomputed from the primitives.
//
// The stencil is the cell and two neighbours each way along each axis (13 cells): face states
// of a neighbour need that neighbour's slope along the face normal only.
//
// Conservation depends on the two cells sharing a face computing identical fluxes: a face is
// the upper face of one cell and the lower face of the next, which are different call sites
// of face_flux(). Its inputs are pure functions of the cell coordinates, so the results should
// match bit for bit, and the validation suite checks that they do (DEBUG_FLUX below).
//
// Performance notes (D3D12, RTX 3080): keep this straight-line code on locals. Caching the
// stencil in a shader-local array was 3x slower than repeating the texture loads.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>

struct Side { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32 }
struct Flux { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32 }

#include <riemann>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var W1: texture_3d<f32>;
@group(0) @binding(5) var W2: texture_3d<f32>;
@group(0) @binding(6) var G: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var PA: texture_3d<f32>;
@group(0) @binding(10) var PB: texture_3d<f32>;

#include "hydro/recon.wgsl"
#include "gravity/gfield.wgsl"

// Half-step state of a cell: a = (rho, v), b = (p, scalar, shock flag, first-order flag).
// A cell beyond a boundary takes the state of the cell it maps to (mirrored if reflecting).
fn load_pred(c: vec3<i32>) -> Q {
  let m = bc_map(c);
  let a = textureLoad(PA, m.c, 0);
  return Q(vec4<f32>(a.x, a.yzw * m.flip), textureLoad(PB, m.c, 0));
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

// State on one side of a face: the half-step state moved half a cell along its slope
// (sgn = +1 for the cell's upper face, -1 for its lower). First-order cells have no slope.
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
  let sm = slope(wm2, wm1, w0);
  let s0 = slope(wm1, w0, wp1);
  let sp = slope(w0, wp1, wp2);
  return Faces(
    face_flux(load_pred(c - e), sm, p0, s0, axis),
    face_flux(p0, s0, load_pred(c + e), sp, axis),
  );
}

// Validation hook: when set, the pass writes the six face mass fluxes instead of the new state,
// so the test suite can check that both cells sharing a face computed the same bits.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  let w0 = load_q(c);
  let p0 = load_pred(c);
  let fx = axis_fluxes(c, ex, 0, w0, p0);
  let fy = axis_fluxes(c, ey, 1, w0, p0);
  let fz = axis_fluxes(c, ez, 2, w0, p0);

  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fx.lo.m, fx.hi.m, fy.lo.m, fy.hi.m));
    textureStore(O2, c, vec4<f32>(fz.lo.m, fz.hi.m, 0.0, 0.0));
    return;
  }

  // Net inflow per unit dt/dx.
  let dm = (fx.lo.m - fx.hi.m) + (fy.lo.m - fy.hi.m) + (fz.lo.m - fz.hi.m);
  let dmom = (fx.lo.mom - fx.hi.mom) + (fy.lo.mom - fy.hi.mom) + (fz.lo.mom - fz.hi.mom);
  let de = (fx.lo.e - fx.hi.e) + (fy.lo.e - fy.hi.e) + (fz.lo.e - fz.hi.e);
  let dei = (fx.lo.ei - fx.hi.ei) + (fy.lo.ei - fy.hi.ei) + (fz.lo.ei - fz.hi.ei);
  let ds = (fx.lo.s - fx.hi.s) + (fy.lo.s - fy.hi.s) + (fz.lo.s - fz.hi.s);
  let dvn = (fx.lo.vn - fx.hi.vn) + (fy.lo.vn - fy.hi.vn) + (fz.lo.vn - fz.hi.vn);

  // Flux-form gravitational work (P.flags bit 0): each face hands half of F_mass * dphi to the
  // two cells it separates, so the sum over cells telescopes to -d/dt of the potential energy.
  let flux_work = has_gravity() && (P.flags & 1u) != 0u;
  var dw = 0.0;
  if (flux_work) {
    // (Giving each face's work wholly to the downstream cell was tried; it conserves just as
    // well in principle but heats low-density cells more and was worse on the Evrard test.)
    dw = -0.5 * (fx.hi.m * dphi_half(c, ex) - fx.lo.m * dphi_half(c, -ex)
               + fy.hi.m * dphi_half(c, ey) - fy.lo.m * dphi_half(c, -ey)
               + fz.hi.m * dphi_half(c, ez) - fz.lo.m * dphi_half(c, -ez));
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
  if (has_gravity()) {
    kick = 0.5 * clock.dt * u1.x * g_load(c);
    if (!flux_work) { ekick = dot(kick, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor); }
  }

  let rho = max(u1.x + dtdx * dm, P.rho_floor);
  let mom = u1.yzw + kick + dtdx * dmom;
  let E = u2.x + ekick + dtdx * (de + dw);
  // Internal energy: advection plus -p div(v), with the half-step pressure and face velocities.
  let eint = u2.y + dtdx * (dei + max(p0.b.x, P.p_floor) * dvn);
  let sc = u2.z + dtdx * ds;

  let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);

  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(en.x, en.y, sc, 0.0));
}
