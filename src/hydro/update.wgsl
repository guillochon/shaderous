// Unsplit MUSCL-Hancock update on a uniform grid, gather form: each cell evaluates its own six
// face fluxes from the half-step states written by hydro/predict.wgsl and a reconstruction
// recomputed from the primitives, which are themselves converted from the conserved state on
// load (see hydro/recon.wgsl).
//
// The stencil is the cell and two neighbors each way along each axis (13 cells): face states
// of a neighbor need that neighbor's slope along the face normal only.
//
// Conservation depends on the two cells sharing a face computing identical fluxes: a face is
// the upper face of one cell and the lower face of the next, which are different call sites
// of face_flux(). Its inputs are pure functions of the cell coordinates, so the results should
// match bit for bit, and the validation suite checks that they do (DEBUG_FLUX below).
//
// Performance notes (D3D12, RTX 3080): keep this straight-line code on locals. Caching the
// stencil in a shader-local array was 3x slower than repeating the texture loads. The interior
// pipeline computes each face once per workgroup instead, when it can: see hydro/tile.wgsl.
#include "hydro/common.wgsl"
#include "hydro/drive.wgsl"
#include "gravity/ext_bind.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/fluff.wgsl"
#include <frame>
#include <psplit>
#include <psplit_store>
#include "hydro/moving_flags.wgsl"
#include "hydro/cut_flags.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(6) var G: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var PA: texture_3d<f32>;
@group(0) @binding(10) var PB: texture_3d<f32>;

#include "hydro/region.wgsl"
#include "hydro/recon.wgsl"
#include "gravity/gfield.wgsl"

// Half-step state of a cell. A cell beyond a boundary takes the state of the cell it maps to
// (mirrored if reflecting).
fn load_pred(c: vec3<i32>) -> Q {
  if (INTERIOR) { return Q(textureLoad(PA, c, 0), textureLoad(PB, c, 0)); }
  // (The half-step state: a shearing box's offset half a step on.)
  let g0 = g_shear;
  g_shear = g_shear_h;
  let m = bc_map(c);
  let a = textureLoad(PA, m.c, 0);
  var q = Q(vec4<f32>(a.x, bc_velocity(a.yzw, m)), textureLoad(PB, m.c, 0));
  if (shear_ghost(c)) {
    let c1 = shear_next(m.c);
    let a1 = textureLoad(PA, c1, 0);
    let f = shear_frac(c);
    q = Q(mix(q.a, vec4<f32>(a1.x, bc_velocity(a1.yzw, m)), f), mix(q.b, textureLoad(PB, c1, 0), f));
  }
  g_shear = g0;
  return q;
}

// For viscosity (hydro/diffusion.wgsl): the velocity alone, which costs one load and no
// equation of state. A solid cell is at rest, but in a moving body (hydro/moving.ts), whose
// velocity its momentum holds.
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  if (INTERIOR) {
    if (!MOVING && has_solids() && textureLoad(U2, c, 0).w > 0.5) { return vec3<f32>(0.0); }
    let u = textureLoad(U1, c, 0);
    return u.yzw / max(u.x, P.rho_floor);
  }
  let m = bc_map(c);
  if (!MOVING && has_solids() && textureLoad(U2, m.c, 0).w > 0.5) { return vec3<f32>(0.0); }
  let u = textureLoad(U1, m.c, 0);
  var v = u.yzw / max(u.x, P.rho_floor);
  if (shear_ghost(c)) {
    let u1 = textureLoad(U1, shear_next(m.c), 0);
    v = mix(v, u1.yzw / max(u1.x, P.rho_floor), shear_frac(c));
  }
  return bc_velocity(v, m);
}
// Also for viscosity: every cell has all its neighbors here (amr/update.wgsl is where not).
fn face_rim(a: vec3<i32>, axis: i32, t: i32) -> i32 {
  return 0;
}

#include "hydro/flux.wgsl"
#include "hydro/tile.wgsl"

// n, as a loop bound the shader compiler cannot see through (P.dims.x is never 0), so that it
// keeps the loop a loop: a large function called in a loop is compiled into it once, where
// unrolled, or called at several places, it is inlined at every one (as amr/grid.wgsl).
fn loop_count(n: i32) -> i32 {
  return select(n, 0, P.dims.x == 0u);
}
#include <walls>

// A shearing box: the flux through one of its x faces, averaged with what the cells across the
// boundary compute for the same face (sheared along y, their velocity boosted by the jump of
// the background flow). Each side's own flux is a different evaluation, from different ghosts;
// the average makes the two sides' totals over the face agree, so that mass, x and z momentum
// are conserved to rounding (the total, not each face: the weights differ).
fn shear_boost(f: Flux, u: f32) -> Flux {
  var o = f;
  o.mom.y = f.mom.y + u * f.m;
  o.e = f.e + u * f.mom.y + 0.5 * u * u * f.m;
  return o;
}
fn shear_face(c: vec3<i32>, own: Flux, upper: bool) -> Flux {
  let ny = i32(P.dims.y);
  // Where this face lies on the other side, in cells along y, half a step on.
  let J = f32(c.y) + select(-g_shear_h, g_shear_h, upper);
  let s = floor(J);
  let e = J - s;
  let j0 = ((i32(s) % ny) + ny) % ny;
  let j1 = (j0 + 1) % ny;
  // The other side's face: the upper face of its last cell, or the lower face of its first.
  let x = select(i32(P.dims.x) - 1, -1, upper);
  let ex = vec3<i32>(1, 0, 0);
  let u = select(1.0, -1.0, upper) * shear_jump();
  // (The two faces from one call site in a kept loop: flux_between is inlined whole at each.)
  var f0: Flux;
  var f1: Flux;
  for (var k = 0; k < loop_count(2); k++) {
    let f = shear_boost(flux_between(vec3<i32>(x, select(j0, j1, k == 1), c.z), ex, 0), u);
    if (k == 0) { f0 = f; } else { f1 = f; }
  }
  let w0 = 0.5 * (1.0 - e);
  let w1 = 0.5 * e;
  return Flux(
    0.5 * own.m + w0 * f0.m + w1 * f1.m,
    0.5 * own.mom + w0 * f0.mom + w1 * f1.mom,
    0.5 * own.e + w0 * f0.e + w1 * f1.e,
    0.5 * own.ei + w0 * f0.ei + w1 * f1.ei,
    0.5 * own.s + w0 * f0.s + w1 * f1.s,
    0.5 * own.vn + w0 * f0.vn + w1 * f1.vn,
    0.5 * own.d + w0 * f0.d + w1 * f1.d,
  );
}
#include <species>
#include "hydro/advance.wgsl"
#include "hydro/sources.wgsl"
#include "hydro/damping.wgsl"
#include <cooling>
#include "hydro/cooling.wgsl"

#include <debris>

// Debris energies (debris/debris.wgsl): the gas leaving through an open face of the box this
// step, binned by its orbital energy with the cell's velocity and potential at the face.
fn escape(c: vec3<i32>, v: vec3<f32>, fx: Faces, fy: Faces, fz: Faces, flat: bool) {
  let n = vec3<i32>(P.dims.xyz);
  let lo = vec3<f32>(fx.lo.m, fy.lo.m, fz.lo.m);
  let hi = vec3<f32>(fx.hi.m, fy.hi.m, fz.hi.m);
  // The passive scalar's flux, for problems that bin only the gas it marks.
  let los = vec3<f32>(fx.lo.s, fy.lo.s, fz.lo.s);
  let his = vec3<f32>(fx.hi.s, fy.hi.s, fz.hi.s);
  var xc = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  if (flat) { xc.z = 0.0; }
  // Face area times dt (P.geom.x is the cell volume, per unit length in z in 2D).
  let adt = P.geom.x / P.dx * clock.dt;
  let phi = textureLoad(G, c, 0).w;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    if (c[a] == 0 && P.bc_lo[a] != 0u) {
      var x = xc;
      x[a] -= 0.5 * P.dx;
      deb_tally(x, v, phi, 0.5 * clock.dt, -lo[a] * adt, -los[a] * adt);
    }
    if (c[a] == n[a] - 1 && P.bc_hi[a] != 0u) {
      var x = xc;
      x[a] += 0.5 * P.dx;
      deb_tally(x, v, phi, 0.5 * clock.dt, hi[a] * adt, his[a] * adt);
    }
  }
}

// A face's flux through a share s of its area.
fn flux_scale(f: Flux, s: f32) -> Flux {
  return Flux(f.m * s, f.mom * s, f.e * s, f.ei * s, f.s * s, f.vn * s, f.d * s);
}

// Validation hook: when set, the pass writes the six face mass fluxes instead of the new state,
// so the test suite can check that both cells sharing a face computed the same bits.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  set_proper_dx(P.dx, clock.ah);
  g_dt = clock.dt;
  // Comoving coordinates: cooling's rate at the proper density (hydro/cooling.wgsl).
  if (cosmo_on()) { g_a3 = clock.ah * clock.ah * clock.ah; }
  if (shear_on()) {
    g_shear = clock.shear / P.dx;
    g_shear_h = shear_half(clock.shear, clock.dt) / P.dx;
  }
  // The whole workgroup computes the faces of its box, before any invocation can leave.
  if (tiled()) { tile_fluxes(vec3<i32>(wid * vec3<u32>(WGX, WGY, WGZ) + REGION.origin), li); }
  let rc = region_cell(gid);
  if (!rc.mine) { return; }
  let c = rc.c;
  if (has_solids()) {
    // Solid and fixed cells keep their state.
    let held = textureLoad(U2, c, 0);
    if (abs(held.w) > 0.5) {
      textureStore(O1, c, textureLoad(U1, c, 0));
      textureStore(O2, c, held);
      store_wall_none(c);
      return;
    }
  }
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  // In 2D the single layer is its own neighbor along z: the two z fluxes are equal and cancel,
  // so they are not computed (and no load leaves the layer, which keeps g_inner valid).
  let flat = P.dims.z == 1u;
  g_inner = stencil_inside(c, 2);
  let w0 = load_q(c);
  let p0 = load_pred(c);
  var fx: Faces;
  var fy: Faces;
  var fz = Faces(flux_zero(), flux_zero());
  if (tiled()) {
    fx = tile_faces(0, c, ex);
    fy = tile_faces(1, c, ey);
    if (!flat) { fz = tile_faces(2, c, ez); }
  } else {
    // One call site in a loop the compiler keeps a loop: axis_fluxes holds two Riemann solves
    // and the stencil's equation-of-state calls, and each call site is inlined whole.
    for (var a = 0; a < loop_count(select(3, 2, flat)); a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      let f = axis_fluxes(c, e, a, w0, p0);
      if (a == 0) { fx = f; } else if (a == 1) { fy = f; } else { fz = f; }
    }
  }

  // A shearing box's x faces: the two sides' fluxes, averaged.
  if (shear_on()) {
    // (Both from one call site, as shear_face's own faces.)
    for (var k = 0; k < loop_count(2); k++) {
      let upper = k == 1;
      if (c.x == select(0, i32(P.dims.x) - 1, upper)) {
        var own = fx.lo;
        if (upper) { own = fx.hi; }
        let f = shear_face(c, own, upper);
        if (upper) { fx.hi = f; } else { fx.lo = f; }
      }
    }
  }
  // Partly filled cells (hydro/cut.ts): each face carries its flux through its open share alone,
  // nothing crosses the wall, and the cell's change is over its gas share (below).
  if (CUT) {
    let k = cut_at(c);
    fx.lo = flux_scale(fx.lo, cut_lo(c, 0));
    fx.hi = flux_scale(fx.hi, k.y);
    fy.lo = flux_scale(fy.lo, cut_lo(c, 1));
    fy.hi = flux_scale(fy.hi, k.z);
    if (!flat) {
      fz.lo = flux_scale(fz.lo, cut_lo(c, 2));
      fz.hi = flux_scale(fz.hi, k.w);
    }
  }
  // Planes of mirror symmetry: nothing crosses them.
  if (c.x == 0 && sym_lo(0)) { fx.lo = mirror_flux(fx.lo, 0); }
  if (c.y == 0 && sym_lo(1)) { fy.lo = mirror_flux(fy.lo, 1); }
  if (c.z == 0 && sym_lo(2) && !flat) { fz.lo = mirror_flux(fz.lo, 2); }

  // The species pass's records (hydro/species_on.wgsl): the bare mass fluxes, before (R, z) and
  // (r, theta) runs weigh them by their faces' areas over this cell's volume, which the species
  // pass does for each cell itself (a face's weight is not the same for the cells on its sides).
  store_mass_flux(c, fx, fy, fz, flat);

  // An axisymmetric run: the radial faces weighed by their areas, and the radial momentum's
  // geometric source (p + rho v_phi^2) / R, with p the mean of the two radial faces' pressures:
  // with it, uniform pressure is in balance to rounding, and a balanced state stays as the
  // Cartesian one does. (The velocities are the half step's.) Before anything reads the fluxes:
  // the flux-form gravitational work takes the weighted mass fluxes.
  var src_r = 0.0;
  if (cylindrical() && !DEBUG_FLUX) {
    let w = cyl_weights(c.x);
    let ps = 0.5 * (face_pressure(fx.lo, 0) + face_pressure(fx.hi, 0));
    src_r = (ps + p0.a.x * p0.a.w * p0.a.w) / (f32(c.x) + 0.5);
    fx.lo = cyl_face(fx.lo, w.x);
    fx.hi = cyl_face(fx.hi, w.y);
  }
  // A spherical run: the faces weighed by their areas, and the geometric sources of the r and
  // theta momenta. The pressure's share of each is (h - w_-) p_- + (w_+ - h) p_+ from the two
  // faces' own pressures: with uniform pressure it is (w_+ - w_-) p, the cell's exact average
  // of 2 p / r (or p cot(theta) / r), which the faces' difference cancels to rounding; and
  // with the faces' pressures apart it leaves the net force h (p_- - p_+), the Cartesian one,
  // so a hydrostatic atmosphere is held as the Cartesian reconstruction holds it. The angular
  // momenta need no source: the lever arms carry them (sph_face).
  var src_t = 0.0;
  if (spherical() && !DEBUG_FLUX) {
    let g = sph_cell(c.x, c.y);
    let v = p0.a.yzw;
    let rho = p0.a.x;
    let pr = vec2<f32>(face_pressure(fx.lo, 0), face_pressure(fx.hi, 0));
    let pt = vec2<f32>(face_pressure(fy.lo, 1), face_pressure(fy.hi, 1));
    src_r = (g.h.x - g.wr.x) * pr.x + (g.wr.y - g.h.x) * pr.y + rho * (v.y * v.y + v.z * v.z) * g.ir;
    src_t = (g.h.y - g.wt.x) * pt.x + (g.wt.y - g.h.y) * pt.y + rho * v.z * v.z * (g.wt.y - g.wt.x);
    fx.lo = sph_face(fx.lo, g.wr.x, vec3<f32>(1.0, g.lr.x, g.lr.x));
    fx.hi = sph_face(fx.hi, g.wr.y, vec3<f32>(1.0, g.lr.y, g.lr.y));
    fy.lo = sph_face(fy.lo, g.wt.x, vec3<f32>(1.0, 1.0, g.lt.x));
    fy.hi = sph_face(fy.hi, g.wt.y, vec3<f32>(1.0, 1.0, g.lt.y));
  }
  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fx.lo.m, fx.hi.m, fy.lo.m, fy.hi.m));
    textureStore(O2, c, vec4<f32>(fz.lo.m, fz.hi.m, 0.0, 0.0));
    return;
  }
  // Forces on bodies: their wall faces' fluxes (hydro/walls_on.wgsl).
  store_wall_forces(c, fx, fy, fz, flat);
  store_face_velocity(c, vec3<f32>(fx.hi.vn, fy.hi.vn, fz.hi.vn));
  if (DEBRIS) { escape(c, w0.a.yzw, fx, fy, fz, flat); }

  // Flux-form gravitational work (P.flags bit 0): each face hands half of F_mass * dphi to the
  // two cells it separates, so the sum over cells telescopes to -d/dt of the potential energy.
  // (Not with implicit pressure: gravity's kick and its work are then the pressure passes', formed
  // on the faces (pressure/correct_gas.wgsl), as the block grid's update has it. Added here too,
  // the work on gas that moves was counted twice: a star crossing the grid left its heat behind,
  // and collapsed within a dynamical time.)
  let flux_work = !PSPLIT && has_gravity() && (P.flags & 1u) != 0u;
  var dw = 0.0;
  if (flux_work) {
    // (Giving each face's work wholly to the downstream cell was tried; it conserves just as
    // well in principle but heats low-density cells more and was worse on the Evrard test.)
    dw = -0.5 * (fx.hi.m * dphi_half(c, ex) - fx.lo.m * dphi_half(c, -ex)
               + fy.hi.m * dphi_half(c, ey) - fy.lo.m * dphi_half(c, -ey)
               + fz.hi.m * dphi_half(c, ez) - fz.lo.m * dphi_half(c, -ez));
  }

  var u1 = textureLoad(U1, c, 0);
  var u2 = textureLoad(U2, c, 0);
  // A non-inertial frame's velocity-dependent forces, the first half step of them
  // (gravity/frame_off.wgsl); the second is applied after the second kick.
  if (FRAME_ON) {
    let fs = frame_open(u1, u2, 0.5 * clock.dt);
    u1 = fs.u1;
    u2 = fs.u2;
  }
  // From here on the equation of state is this cell's.
  comp_use(comp_tex(c));

  // Gravity is kick-drift-kick around the flux update: half a kick with the old potential here,
  // the other half in gravity/correct.wgsl once the new potential is known. (Applying the whole
  // kick after the fluxes would drain 1/2 rho g^2 dt^2 of internal energy per step from gas in
  // hydrostatic balance.) The energy source is either the flux-form work accumulated above, or
  // exactly the kinetic energy the kick adds, k . (m + k/2) / rho, which keeps gravity out of
  // the internal energy at the price of total-energy conservation. The flux form covers
  // self-gravity and the uniform field; the point masses' and the frame's part of the kick
  // always adds its energy in kick form (gravity/ext.wgsl).
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  // (With implicit pressure the uniform field's kick is the pressure passes': it has to be
  // formed on the faces, where the pressure difference that balances it is. pressure/correct.wgsl.)
  if (!PSPLIT && (has_gravity() || ext_on())) {
    let kx = 0.5 * clock.dt * u1.x * ext_accel(cell_pos(c));
    kick = 0.5 * clock.dt * u1.x * g_self(c) + kx;
    let ke = select(kx, kick, !flux_work);
    ekick = dot(ke, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor);
  }

  // Turbulence driving (hydro/drive.wgsl): a whole step's kick, in kick form.
  if (drive_on()) {
    let kd = drive_kick(kick, ekick, cell_pos(c), clock.dt, u1.x, u1.yzw);
    kick = kd.xyz;
    ekick = kd.w;
  }

  var net = net_inflow_heated(fx, fy, fz, w0.a.yzw);
  if (CUT) {
    let g = 1.0 / cut_at(c).x;
    net = Net(net.m * g, net.mom * g, net.e * g, net.ei * g, net.s * g, net.vn * g);
  }
  net.mom.x += src_r;
  if (spherical()) { net.mom.y += src_t; }
  var nw = advance(u1, u2, net, dw, p0.b.x, kick, ekick);
  if (has_sources()) { nw = apply_sources(nw, (vec3<f32>(c) + 0.5) / vec3<f32>(P.dims.xyz), clock.dt); }
  nw = apply_damping(nw, (vec3<f32>(c) + 0.5) / vec3<f32>(P.dims.xyz), clock.dt, clock.t);
  nw = apply_cooling(nw, clock.dt);
  // Implicit pressure in a rotating frame (the liquid; plans/wind-tunnel.md 2.1): the
  // centrifugal force's whole kick and the Coriolis force's second half-turn, here, before the
  // solve. (The explicit step has them in gravity/correct.wgsl, which this step does not run.)
  // For a liquid of uniform density the centrifugal force is a gradient, which the solve takes
  // into the pressure.
  if (PSPLIT && FRAME_ON) {
    var u1n = nw.u1;
    if (P.rot.w != 0.0) { u1n = vec4<f32>(u1n.x, u1n.yzw + clock.dt * u1n.x * rot_accel(cell_pos(c))); }
    let fs = frame_close(u1n, nw.u2, 0.5 * clock.dt);
    nw = NewState(fs.u1, fs.u2);
  }
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}
