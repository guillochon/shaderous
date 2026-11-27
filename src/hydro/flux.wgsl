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
//   dphi_face(a, axis, k)  for flux_between, as (plain, true): the plain difference, with no
//                          boundary involved, and the true difference across a reflecting face
//   wall_step(a, axis, s, t)  for flux_between, the true step across a face whose plain step
//                          is s, and the sign each side takes it with: t is dphi_face's second
//                          part for k = 0 (gravity/gfield.wgsl), which a grid may take from its
//                          own tables instead (amr/update.wgsl)
// and, for viscosity (hydro/diffusion.wgsl), load_vel(c) and face_rim(a, axis, t), and for a
// composition that varies (species/mixture.ts), load_comp(c): the moments of a cell. Each face
// state takes the composition of the cell it was reconstructed in.
// Both form the difference from the same two values whichever cell asks, so the two cells
// sharing a face still compute the same flux bit for bit.
// `pd` is the pressure the momentum fluxes and the contact speed take: the pressure itself, or,
// for a gas stored as its deviation from a reference (hydro/ref_on.wgsl), its deviation p - p_ref,
// the reference's own pressure being balanced and left out. `p` is the whole pressure, for the
// sound speed and the energy flux.
struct Side { rho: f32, v: vec3<f32>, p: f32, s: f32, cs: f32, eint: f32, pd: f32 }
// `d` is the viscous part of the flux on its own, (momentum, energy), which a cell needs to
// work out how much the viscosity heats it (hydro/diffusion.wgsl).
struct Flux { m: f32, mom: vec3<f32>, e: f32, ei: f32, s: f32, vn: f32, d: vec4<f32> }

#include <refdev>
#include <riemann>
#include "hydro/diffusion.wgsl"
#include "hydro/moving_flags.wgsl"

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
// `rf` is the reference on this side (hydro/ref_on.wgsl ref_face) for a gas stored as its
// deviation from one, whose half-step state and increments are deviations: the face's whole
// density and pressure are the reference's at the face plus them. Zero otherwise.
fn to_side(pr: Q, s: Q, sgn: f32, axis: i32, lo: vec2<f32>, rf: vec4<f32>) -> Side {
  let hs = 0.5 * sgn * (1.0 - pr.b.w);
  let a = pr.a + hs * s.a;
  let b = pr.b.xy + hs * s.b.xy;
  var o: Side;
  o.rho = max(a.x + rf.z, max(lo.x, P.rho_floor));
  o.v = rot(a.yzw, axis);
  o.s = b.y;
  if (EOS_HAS_ENERGY) {
    o.p = max(b.x + rf.w, max(lo.y, P.p_floor));
    // (Raised by a floor or the bound, the deviation follows.)
    o.pd = select(b.x, o.p - rf.w, o.p != b.x + rf.w);
    if (!REF_DEV) { o.pd = o.p; }
    let f = eos_face(o.rho, o.p);
    o.cs = f.x;
    o.eint = f.y;
  } else {
    // Barotropic: the face pressure is a function of the face density, not reconstructed.
    let st = baro_state(o.rho);
    o.p = st.x;
    o.pd = o.p;
    o.cs = st.y;
    o.eint = eos_eint_from_p(o.rho, o.p);
  }
  return o;
}

// Flux through the face between cells l and r, along +axis, in grid-aligned components. kl
// and kr are the two cells' composition moments. dfl and dfr are the potential steps that the
// two sides' edge increments were given toward this face (edge_up's and edge_dn's `df`; zero
// without the balanced reconstruction). rl and rr are the two sides' references (ref_face) for
// a gas stored as its deviation from one, zero otherwise.
fn face_flux(l: Q, ls: Q, r: Q, rs: Q, axis: i32, kl: vec4<f32>, kr: vec4<f32>, dfl: f32, dfr: f32, rl: vec4<f32>, rr: vec4<f32>) -> Flux {
  var w = max(l.b.z, r.b.z);            // carbuncle fix: HLL next to flagged shocks
  // A barotropic gas takes HLL: it has no contact wave to resolve (HLLC was tried on its
  // collapse problems: no different). A liquid keeps HLLC, for the shear wave: HLL damps a jump
  // in the transverse velocity at the sound speed, and a Gresho vortex at Mach 0.1 on 64 x 64
  // cells then loses 8.4% of its kinetic energy in four turns where HLLC loses 2.2%
  // (plans/incompressible.md, 2.4 and 3.3).
  if (!EOS_HAS_ENERGY && !p_signed()) { w = 1.0; }
  // Neither face state may fall below half the smaller of the two cells' own values. For the
  // pressure, also below half of each cell's pressure carried to the face along hydrostatic
  // balance, p -/+ rho df / 2, where that is positive: in a column at rest that is the face
  // pressure itself. Between two gas cells it never binds (the upper cell's own pressure is
  // the lower), but at a wall the other side is this cell's image, with its pressure, and a
  // column with fewer than 2 cells per scale height has its balanced wall pressure,
  // p (1 - 1/2h), below p / 2: the bound raised it, and the top cell fell at Mach 0.002 (h =
  // 0.99) to 0.08 (h = 0.7) in a sound crossing (hydro-proofs' column maps).
  // (Of the whole densities and pressures, for a gas stored as its deviation from a reference.)
  let al = vec2<f32>(l.a.x + rl.x, l.b.x + rl.y);
  let ar = vec2<f32>(r.a.x + rr.x, r.b.x + rr.y);
  var pb = min(al.y, ar.y);
  if (!p_signed()) {
    let el = al.y - 0.5 * al.x * dfl;
    let er = ar.y + 0.5 * ar.x * dfr;
    pb = min(pb, min(select(al.y, el, el > 0.0), select(ar.y, er, er > 0.0)));
  }
  let lo = 0.5 * vec2<f32>(min(al.x, ar.x), pb);
  comp_use(kl);
  let sl = to_side(l, ls, 1.0, axis, lo, rl);
  comp_use(kr);
  let sr = to_side(r, rs, -1.0, axis, lo, rr);
  var F = riemann_flux(sl, sr, w);
  F.mom = unrot(F.mom, axis);
  return F;
}

struct Faces { lo: Flux, hi: Flux }

// Moving bodies (plans/wind-tunnel.md 6.2; hydro/moving.ts): a solid cell holds its body's
// velocity there in its momentum slot, so its primitives (load_q) carry it, and a wall moves.
// Its image is the gas cell's state with the normal velocity 2 v_s - v (reflected about the
// wall's), and the face carries no mass, only the pressure of the Riemann problem between them
// and its work, p v_s: a wall that pushes, whose cells change kind as it goes (6.3,
// hydro/moving_exchange.wgsl). Without moving bodies these are the plain mirror and flux.
fn mir(q: Q, axis: i32, vs: f32) -> Q {
  if (MOVING) {
    var a = q.a;
    a[axis + 1] = 2.0 * vs - a[axis + 1];
    return Q(a, vec4<f32>(q.b.xyz, 0.0));
  }
  return q_mirror(q, axis);
}
fn pmir(p: Q, axis: i32, vs: f32) -> Q {
  if (MOVING) {
    var a = p.a;
    a[axis + 1] = 2.0 * vs - a[axis + 1];
    return Q(a, p.b);
  }
  return pred_mirror(p, axis);
}
// A wall face's flux, where the wall may move at vs along the axis: the pressure and its work.
fn wall_face(F: Flux, wall: bool, axis: i32, vs: f32) -> Flux {
  if (!MOVING || !wall) { return F; }
  let p = F.mom[axis] - F.m * F.vn;
  var o = Flux(0.0, vec3<f32>(0.0), p * vs, 0.0, 0.0, vs, vec4<f32>(0.0));
  o.mom[axis] = p;
  return o;
}

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
  var below = false;
  var above = false;
  if (has_solids()) {
    below = q_solid(wm1);
    above = q_solid(wp1);
    // (The walls' velocities along the axis, for moving bodies: those of the solid cells.)
    let vb = nm1.a[axis + 1];
    let va = np1.a[axis + 1];
    // An image has the potential of the cell it mirrors.
    if (below) { wm1 = mir(w0, axis, vb); pm1 = pmir(p0, axis, vb); d.y = 0.0; km1 = k0; }
    if (above) { wp1 = mir(w0, axis, va); pp1 = pmir(p0, axis, va); d.z = 0.0; kp1 = k0; }
    if (below) { wm2 = mir(wp1, axis, vb); d.x = -d.z; } else if (q_solid(wm2)) { wm2 = mir(wm1, axis, wm2.a[axis + 1]); d.x = 0.0; }
    if (above) { wp2 = mir(wm1, axis, va); d.w = -d.y; } else if (q_solid(wp2)) { wp2 = mir(wp1, axis, wp2.a[axis + 1]); d.w = 0.0; }
  }
  // A neighbor that is this cell's image in a wall (the step the deviations see is not the
  // true one) reaches the wall from the other side: the negative of the true step.
  let fm = select(-t.x, t.x, d.y == t.x);
  let fp = select(-t.y, t.y, d.z == t.y);
  // A gas stored as its deviation from a reference (hydro/ref_on.wgsl): the five cells'
  // references, each increment's middle cell's for the limiter's scales, and each side's at
  // its face.
  var r0 = vec4<f32>(0.0);
  var r1 = vec4<f32>(0.0);
  var r2 = vec4<f32>(0.0);
  var r3 = vec4<f32>(0.0);
  var r4 = vec4<f32>(0.0);
  if (REF_DEV) {
    r0 = ref_at(c - 2 * e);
    r1 = ref_at(c - e);
    r2 = ref_at(c);
    r3 = ref_at(c + e);
    r4 = ref_at(c + 2 * e);
    g_qs = r1.xy;
  }
  let sm = edge_up(wm2, wm1, w0, d.x, d.y, fm);
  if (REF_DEV) { g_qs = r2.xy; }
  let lo = edge_dn(wm1, w0, wp1, d.y, d.z, t.x);
  let hi = edge_up(wm1, w0, wp1, d.y, d.z, t.y);
  if (REF_DEV) { g_qs = r3.xy; }
  let sp = edge_dn(w0, wp1, wp2, d.z, d.w, fp);
  if (REF_DEV) { g_qs = vec2<f32>(0.0); }
  return Faces(
    add_diffusion(wall_face(face_flux(pm1, sm, p0, lo, axis, km1, k0, fm, t.x, ref_face(r0, r1, r2, 1.0, fm), ref_face(r1, r2, r3, -1.0, t.x)), below, axis, nm1.a[axis + 1]), c - e, e, axis, nm1, w0),
    add_diffusion(wall_face(face_flux(p0, hi, pp1, sp, axis, k0, kp1, t.y, fp, ref_face(r1, r2, r3, 1.0, t.y), ref_face(r2, r3, r4, -1.0, fp)), above, axis, np1.a[axis + 1]), c, e, axis, w0, np1),
  );
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
  // Potential steps between the four cells, lowest pair first, and the true step across the face
  // for wall_step, from the same call as the middle step: each step maps two cells through the
  // boundaries and sums the point masses, and the compiler inlines every call, so a fourth call
  // for the wall's step cost the uniform update about 12 percent of its compile time (1.30 s
  // against 1.14 for a 2D Kelvin-Helmholtz run in Chrome on a cold cache). One call in a kept
  // loop compiled faster still (0.91 s), but serialized the loads: steps of a tidal disruption
  // took 2.6 percent longer.
  var d = vec3<f32>(0.0);
  var t = 0.0;
  if (well_balanced()) {
    let s0 = dphi_face(a, axis, 0);
    d = vec3<f32>(dphi_face(a, axis, -1).x, s0.x, dphi_face(a, axis, 1).x);
    t = s0.y;
  }
  let across = d.y;
  // The true step across, and its sign for a side that is the image of the other in a wall of
  // the grid: what axis_fluxes uses there. Elsewhere (across, 1, 1).
  let ws = wall_step(a, axis, across, t);
  var sides = vec2<f32>(1.0);
  var wall = false;
  var vw = 0.0;
  if (has_solids()) {
    let ls = q_solid(wl);
    let rs = q_solid(wr);
    if (ls && rs) { return flux_zero(); }
    wall = ls || rs;
    vw = select(nl.a[axis + 1], nr.a[axis + 1], rs);
    if (rs) {
      // Wall on the right: as the left cell computes its upper face.
      if (q_solid(wll)) { wll = mir(wl, axis, wll.a[axis + 1]); d.x = 0.0; }
      wr = mir(wl, axis, vw);
      pr = pmir(pl, axis, vw);
      kr = kl;
      wrr = mir(wll, axis, vw);
      d.y = 0.0;
      d.z = -d.x;
      sides.y = -1.0;
    } else if (ls) {
      if (q_solid(wrr)) { wrr = mir(wr, axis, wrr.a[axis + 1]); d.z = 0.0; }
      wl = mir(wr, axis, vw);
      pl = pmir(pr, axis, vw);
      kl = kr;
      wll = mir(wrr, axis, vw);
      d.y = 0.0;
      d.x = -d.z;
      sides.x = -1.0;
    } else {
      if (q_solid(wll)) { wll = mir(wl, axis, wll.a[axis + 1]); d.x = 0.0; }
      if (q_solid(wrr)) { wrr = mir(wr, axis, wrr.a[axis + 1]); d.z = 0.0; }
    }
  }
  // ws.x is the true step across the face; a cell that is the other's image in a wall reaches
  // the wall from its own side.
  let fl = ws.x * (sides.x * ws.y);
  let fr = ws.x * (sides.y * ws.z);
  // (A gas stored as its deviation from a reference: as axis_fluxes, hydro/ref_on.wgsl.)
  var r0 = vec4<f32>(0.0);
  var r1 = vec4<f32>(0.0);
  var r2 = vec4<f32>(0.0);
  var r3 = vec4<f32>(0.0);
  if (REF_DEV) {
    r0 = ref_at(a - e);
    r1 = ref_at(a);
    r2 = ref_at(a + e);
    r3 = ref_at(a + 2 * e);
    g_qs = r1.xy;
  }
  let sl = edge_up(wll, wl, wr, d.x, d.y, fl);
  if (REF_DEV) { g_qs = r2.xy; }
  let sr = edge_dn(wl, wr, wrr, d.y, d.z, fr);
  if (REF_DEV) { g_qs = vec2<f32>(0.0); }
  return add_diffusion(wall_face(face_flux(pl, sl, pr, sr, axis, kl, kr, fl, fr, ref_face(r0, r1, r2, 1.0, fl), ref_face(r1, r2, r3, -1.0, fr)), wall, axis, vw), a, e, axis, nl, nr);
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
