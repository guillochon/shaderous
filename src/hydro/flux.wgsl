// Face fluxes from half-step states and slopes. Requires P, the equation of state, the
// limiter, hydro/q.wgsl, hydro/pfloor.wgsl, g_dx set, and two loaders supplied by the grid:
//   load_q(c)     primitives of a cell
//   load_pred(c)  its half-step state: a = (rho, v), b = (p, scalar, shock flag, first-order flag)
// and, for the well-balanced pressure reconstruction (hydro/q.wgsl), the difference of the
// gravitational potential between two adjacent cells, phi(c + (k + 1) e) - phi(c + k e):
//   dphi_step(c, axis, k)  for the cell being updated, k from -2 to 1, as (seen, face): `seen`
//                          takes a cell beyond a boundary to have the potential of the cell
//                          its state is taken from, `face` is the true difference across a
//                          reflecting face (and equals `seen` everywhere else)
//   dphi_face(a, axis, k)  the plain difference, with no boundary involved, for flux_between
//   wall_step(a, axis, s)  for flux_between, the true step across a face whose plain step is s,
//                          and the sign each side takes it with (gravity/gfield.wgsl)
// and, for viscosity (hydro/diffusion.wgsl), load_vel(c) and face_rim(a, axis, t), and for a
// composition that varies (species/mixture.ts), load_comp(c): the moments of a cell. Each face
// state takes the composition of the cell it was reconstructed in.
// Both form the difference from the same two values whichever cell asks, so the two cells
// sharing a face still compute the same flux bit for bit.
struct Side { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32 }
// `d` is the viscous part of the flux on its own, (momentum, energy), which a cell needs to
// work out how much the viscosity heats it (hydro/diffusion.wgsl).
struct Flux { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32, d: vec4<f32> }

#include <riemann>
#include "hydro/diffusion.wgsl"
// The turbulence closure (closure/closure.ts), or nothing: <closure> is hydro/closure_on.wgsl
// or hydro/closure_off.wgsl. g_clos is the face's (a, b), which the functions that know the
// face's two cells (axis_fluxes, flux_between) set before they call face_flux.
#include <closure>
var<private> g_clos: vec3<f32> = vec3<f32>(0.0);
// With it, the passive scalar of the four cells along the face's normal, lowest first (the
// face lies between the second and the third), as they are at the start of the step.
var<private> g_clos_s: vec4<f32> = vec4<f32>(0.0);

// A face's scalar state moved to `want` by the closure, but no further from its own cell's
// value `own` than a limited slope could have put it: toward the other cell of the face (by
// `b`, the difference to it), and by no more than the cell's value differs from its neighbor
// on the far side (`a`); or as far as the reconstruction itself went (`base`), if that is
// further. So the closure never steepens the scalar past what a slope limiter allows, and no
// value leaves the range of its neighbors: without this the scalar rang at sharp fronts.
fn closure_bound(want: f32, base: f32, own: f32, a: f32, b: f32) -> f32 {
  let m = select(0.0, sign(b) * min(abs(a), abs(b)), a * b > 0.0);
  let d0 = base - own;
  return own + clamp(want - own, min(min(m, d0), 0.0), max(max(m, d0), 0.0));
}

// The scalar's compression: a face's state `have` moved on toward the other cell of the face
// by the share c of what is left to the limiter's bound (the same a and b as closure_bound),
// which keeps a front between two values of the dye from spreading over more cells each step.
// Not at an extremum, where there is no bound to move toward. The state's own cell value
// `own` is half a step on, so the result is held, as a value, between the face's two cells as
// they are at the start of the step (sc, so): it may reach the other cell's value, not pass it.
fn closure_compress(have: f32, own: f32, a: f32, b: f32, c: f32, sc: f32, so: f32) -> f32 {
  let m = select(0.0, sign(b) * min(abs(a), abs(b)), a * b > 0.0);
  let d1 = have - own;
  if (m == 0.0 || abs(m) <= abs(d1)) { return have; }
  let want = own + d1 + c * (m - d1);
  return max(min(want, max(max(sc, so), have)), min(min(sc, so), have));
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

// State on one side of a face: the half-step state plus half the edge increment `s` toward
// that face (sgn = +1 for the cell's upper face, -1 for its lower). First-order cells have none.
//
// `lo` = (density, pressure) is a lower bound from the two cells that share the face. The
// limiter lets a face value reach the neighboring cell's value, and the slopes come from the
// state at the start of the step while the value they are added to is half a step on. At a
// jump of several decades (the edge of a star in near-vacuum) the face density of the dense
// cell can therefore land at almost nothing, still positive, with the pressure not following:
// a state with an enormous sound speed, whose flux empties the light cell in one step.
fn to_side(pr: Q, s: Q, sgn: f32, axis: i32, lo: vec2<f32>) -> Side {
  let hs = 0.5 * sgn * (1.0 - pr.b.w);
  let a = pr.a + hs * s.a;
  let b = pr.b.xy + hs * s.b.xy;
  var o: Side;
  o.rho = max(a.x, max(lo.x, P.rho_floor));
  o.v = rot(a.yzw, axis);
  o.s = b.y;
  if (EOS_HAS_ENERGY) {
    o.p = max(b.x, max(lo.y, P.p_floor));
    let f = eos_face(o.rho, o.p);
    o.cs = f.x;
    o.eint = f.y;
  } else {
    // Barotropic: the face pressure is a function of the face density, not reconstructed.
    let st = baro_state(o.rho);
    o.p = st.x;
    o.cs = st.y;
    o.eint = eos_eint_from_p(o.rho, o.p);
  }
  return o;
}

// Flux through the face between cells l and r, along +axis, in grid-aligned components. kl
// and kr are the two cells' composition moments.
fn face_flux(l: Q, ls: Q, r: Q, rs: Q, axis: i32, kl: vec4<f32>, kr: vec4<f32>) -> Flux {
  var w = max(l.b.z, r.b.z);            // carbuncle fix: HLL next to flagged shocks
  // A barotropic gas takes HLL: it has no contact wave to resolve (HLLC was tried on its
  // collapse problems: no different). A liquid keeps HLLC, for the shear wave: HLL damps a jump
  // in the transverse velocity at the sound speed, and a Gresho vortex at Mach 0.1 on 64 x 64
  // cells then loses 8.4% of its kinetic energy in four turns where HLLC loses 2.2%
  // (plans/incompressible.md, 2.4 and 3.3).
  if (!EOS_HAS_ENERGY && !p_signed()) { w = 1.0; }
  // Neither face state may fall below half the smaller of the two cells' own values.
  let lo = 0.5 * min(vec2<f32>(l.a.x, l.b.x), vec2<f32>(r.a.x, r.b.x));
  comp_use(kl);
  var sl = to_side(l, ls, 1.0, axis, lo);
  comp_use(kr);
  var sr = to_side(r, rs, -1.0, axis, lo);
  if (CLOSURE) {
    // The closure: the Riemann solver is shown only the share 1 - a of the velocity's jump and
    // 1 - b of the passive scalar's, about the mean of the two face states. Its dissipation of
    // shear and its diffusion of the scalar are proportional to those jumps, so they fall by
    // the same shares; a is at most 0.8, so a fifth of the first always remains, and the
    // scalar's states are bounded as a limited slope bounds them (closure_bound). Where
    // they are zero (every face of a cell whose gate is shut) the states are left as they are,
    // to the bit.
    if (g_clos.x > 0.0) {
      let vm = 0.5 * (sl.v + sr.v);
      let vh = 0.5 * (1.0 - g_clos.x) * (sl.v - sr.v);
      sl.v = vm + vh;
      sr.v = vm - vh;
    }
    if (g_clos.y > 0.0) {
      let cm = 0.5 * (sl.s + sr.s);
      let ch = 0.5 * (1.0 - g_clos.y) * (sl.s - sr.s);
      let q = g_clos_s;
      sl.s = closure_bound(cm + ch, sl.s, l.b.y, q.y - q.x, q.z - q.y);
      sr.s = closure_bound(cm - ch, sr.s, r.b.y, q.z - q.w, q.y - q.z);
    }
    if (g_clos.z > 0.0) {
      let q = g_clos_s;
      sl.s = closure_compress(sl.s, l.b.y, q.y - q.x, q.z - q.y, g_clos.z, q.y, q.z);
      sr.s = closure_compress(sr.s, r.b.y, q.z - q.w, q.y - q.z, g_clos.z, q.z, q.y);
    }
  }
  var F = riemann_flux(sl, sr, w);
  F.mom = unrot(F.mom, axis);
  return F;
}

struct Faces { lo: Flux, hi: Flux }

// Both faces of cell c along one axis. w0 and p0 are the cell's own primitives and half-step state.
//
// Walls: where a neighbor is solid, the face between is a reflecting wall, and the cells
// beyond it are replaced by the images of the gas cells on this side (the cell itself for the
// first, its opposite neighbor for the second). The Riemann problem between a state and its
// image has no mass or energy flux, only pressure. A gas cell two away from a wall sees the
// image of its neighbor there, which is what that neighbor sees: both get the same flux.
fn axis_fluxes(c: vec3<i32>, e: vec3<i32>, axis: i32, w0: Q, p0: Q) -> Faces {
  var wm2 = load_q(c - 2 * e);
  var wm1 = load_q(c - e);
  var wp1 = load_q(c + e);
  var wp2 = load_q(c + 2 * e);
  var pm1 = load_pred(c - e);
  var pp1 = load_pred(c + e);
  // The neighbors as they are (solid or not), for the diffusive fluxes.
  let nm1 = wm1;
  let np1 = wp1;
  // Compositions: an image in a wall has that of the cell it mirrors, this one.
  let k0 = load_comp(c);
  var km1 = load_comp(c - e);
  var kp1 = load_comp(c + e);
  // Potential steps between the five cells, lowest pair first: as the deviations see them,
  // and the true ones across this cell's two faces.
  var d = vec4<f32>(0.0);
  var t = vec2<f32>(0.0);
  if (well_balanced()) {
    let below = dphi_step(c, axis, -1);
    let above = dphi_step(c, axis, 0);
    d = vec4<f32>(dphi_step(c, axis, -2).x, below.x, above.x, dphi_step(c, axis, 1).x);
    t = vec2<f32>(below.y, above.y);
  }
  if (has_solids()) {
    let below = q_solid(wm1);
    let above = q_solid(wp1);
    // An image has the potential of the cell it mirrors.
    if (below) { wm1 = q_mirror(w0, axis); pm1 = pred_mirror(p0, axis); d.y = 0.0; km1 = k0; }
    if (above) { wp1 = q_mirror(w0, axis); pp1 = pred_mirror(p0, axis); d.z = 0.0; kp1 = k0; }
    if (below) { wm2 = q_mirror(wp1, axis); d.x = -d.z; } else if (q_solid(wm2)) { wm2 = q_mirror(wm1, axis); d.x = 0.0; }
    if (above) { wp2 = q_mirror(wm1, axis); d.w = -d.y; } else if (q_solid(wp2)) { wp2 = q_mirror(wp1, axis); d.w = 0.0; }
  }
  // A neighbor that is this cell's image in a wall (the step the deviations see is not the
  // true one) reaches the wall from the other side: the negative of the true step.
  let sm = edge_up(wm2, wm1, w0, d.x, d.y, select(-t.x, t.x, d.y == t.x));
  let lo = edge_dn(wm1, w0, wp1, d.y, d.z, t.x);
  let hi = edge_up(wm1, w0, wp1, d.y, d.z, t.y);
  let sp = edge_dn(w0, wp1, wp2, d.z, d.w, select(-t.y, t.y, d.z == t.y));
  if (CLOSURE) { g_clos = closure_face(c - e, c); g_clos_s = vec4<f32>(wm2.b.y, wm1.b.y, w0.b.y, wp1.b.y); }
  let flo = add_diffusion(face_flux(pm1, sm, p0, lo, axis, km1, k0), c - e, e, axis, nm1, w0);
  if (CLOSURE) { g_clos = closure_face(c, c + e); g_clos_s = vec4<f32>(wm1.b.y, w0.b.y, wp1.b.y, wp2.b.y); }
  let fhi = add_diffusion(face_flux(p0, hi, pp1, sp, axis, k0, kp1), c, e, axis, w0, np1);
  return Faces(flo, fhi);
}

// Flux through the single face between cell a and cell a + e. The same arithmetic as
// axis_fluxes on the same inputs, so the two agree bit for bit.
fn flux_between(a: vec3<i32>, e: vec3<i32>, axis: i32) -> Flux {
  var wl = load_q(a);
  var wr = load_q(a + e);
  var wll = load_q(a - e);
  var wrr = load_q(a + 2 * e);
  var pl = load_pred(a);
  var pr = load_pred(a + e);
  let nl = wl;
  let nr = wr;
  var kl = load_comp(a);
  var kr = load_comp(a + e);
  // Potential steps between the four cells, lowest pair first.
  var d = vec3<f32>(0.0);
  if (well_balanced()) {
    d = vec3<f32>(dphi_face(a, axis, -1), dphi_face(a, axis, 0), dphi_face(a, axis, 1));
  }
  let across = d.y;
  // The true step across, and its sign for a side that is the image of the other in a wall of
  // the grid: what axis_fluxes uses there. Elsewhere (across, 1, 1).
  let ws = wall_step(a, axis, across);
  var sides = vec2<f32>(1.0);
  if (has_solids()) {
    let ls = q_solid(wl);
    let rs = q_solid(wr);
    if (ls && rs) { return flux_zero(); }
    if (rs) {
      // Wall on the right: as the left cell computes its upper face.
      if (q_solid(wll)) { wll = q_mirror(wl, axis); d.x = 0.0; }
      wr = q_mirror(wl, axis);
      pr = pred_mirror(pl, axis);
      kr = kl;
      wrr = q_mirror(wll, axis);
      d.y = 0.0;
      d.z = -d.x;
      sides.y = -1.0;
    } else if (ls) {
      if (q_solid(wrr)) { wrr = q_mirror(wr, axis); d.z = 0.0; }
      wl = q_mirror(wr, axis);
      pl = pred_mirror(pr, axis);
      kl = kr;
      wll = q_mirror(wrr, axis);
      d.y = 0.0;
      d.x = -d.z;
      sides.x = -1.0;
    } else {
      if (q_solid(wll)) { wll = q_mirror(wl, axis); d.x = 0.0; }
      if (q_solid(wrr)) { wrr = q_mirror(wr, axis); d.z = 0.0; }
    }
  }
  // ws.x is the true step across the face; a cell that is the other's image in a wall reaches
  // the wall from its own side.
  let sl = edge_up(wll, wl, wr, d.x, d.y, ws.x * (sides.x * ws.y));
  let sr = edge_dn(wl, wr, wrr, d.y, d.z, ws.x * (sides.y * ws.z));
  if (CLOSURE) { g_clos = closure_face(a, a + e); g_clos_s = vec4<f32>(wll.b.y, wl.b.y, wr.b.y, wrr.b.y); }
  return add_diffusion(face_flux(pl, sl, pr, sr, axis, kl, kr), a, e, axis, nl, nr);
}

fn flux_zero() -> Flux {
  return Flux(0.0, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0, vec4<f32>(0.0));
}

// Axisymmetric runs: a radial face's flux weighed by its area over the cell's (cyl_weights), and
// the angular momentum's by the square, so that R rho v_phi is conserved with no source.
fn cyl_face(f: Flux, w: f32) -> Flux {
  return Flux(f.m * w, vec3<f32>(f.mom.xy * w, f.mom.z * w * w), f.e * w, f.ei * w, f.s * w, f.vn * w, vec4<f32>(f.d.xy * w, f.d.z * w * w, f.d.w * w));
}
// Spherical runs: a face's flux weighed by its area over the cell's (sph_cell), the momentum's
// components each by `w` times their own lever arm (1 for v_r, so that r rho v_theta and
// r sin(theta) rho v_phi are carried in conservation form).
fn sph_face(f: Flux, w: f32, lever: vec3<f32>) -> Flux {
  let wm = w * lever;
  return Flux(f.m * w, f.mom * wm, f.e * w, f.ei * w, f.s * w, f.vn * w, vec4<f32>(f.d.xyz * wm, f.d.w * w));
}
// The pressure at a face, as its flux has it: the normal momentum flux less what the mass flux
// carries at the face's normal velocity.
fn face_pressure(f: Flux, axis: i32) -> f32 {
  return f.mom[axis] - f.m * f.vn;
}

// The flux through a plane of mirror symmetry (sym_lo): nothing crosses it, so only the normal
// momentum flux (the pressure, and viscosity's normal stress) is kept. The Riemann problem
// between a state and its image gives this up to rounding; it is set exactly, so that the
// mass on the grid is conserved to the bit.
fn mirror_flux(f: Flux, axis: i32) -> Flux {
  var o = flux_zero();
  o.mom[axis] = f.mom[axis];
  o.d[axis] = f.d[axis];
  return o;
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

// The same, with the heat that viscosity leaves in the cell (whose velocity is v) added to the
// internal energy's share.
fn net_inflow_heated(fx: Faces, fy: Faces, fz: Faces, v: vec3<f32>) -> Net {
  var net = net_inflow(fx, fy, fz);
  if (VISCOSITY) {
    net.ei += (visc_heat(fx.lo, v) - visc_heat(fx.hi, v)) + (visc_heat(fy.lo, v) - visc_heat(fy.hi, v)) + (visc_heat(fz.lo, v) - visc_heat(fz.hi, v));
  }
  return net;
}
