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
// equation of state. A solid cell is at rest.
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  if (INTERIOR) {
    if (has_solids() && textureLoad(U2, c, 0).w > 0.5) { return vec3<f32>(0.0); }
    let u = textureLoad(U1, c, 0);
    return u.yzw / max(u.x, P.rho_floor);
  }
  let m = bc_map(c);
  if (has_solids() && textureLoad(U2, m.c, 0).w > 0.5) { return vec3<f32>(0.0); }
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
  let J = f32(c.y) + select(g_shear_h, -g_shear_h, upper);
  let s = floor(J);
  let e = J - s;
  let j0 = ((i32(s) % ny) + ny) % ny;
  let j1 = (j0 + 1) % ny;
  // The other side's face: the upper face of its last cell, or the lower face of its first.
  let x = select(i32(P.dims.x) - 1, -1, upper);
  let ex = vec3<i32>(1, 0, 0);
  let u = select(1.0, -1.0, upper) * shear_jump();
  let f0 = shear_boost(flux_between(vec3<i32>(x, j0, c.z), ex, 0), u);
  let f1 = shear_boost(flux_between(vec3<i32>(x, j1, c.z), ex, 0), u);
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

// Validation hook: when set, the pass writes the six face mass fluxes instead of the new state,
// so the test suite can check that both cells sharing a face computed the same bits.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  g_dx = cell_dx(P.dx, clock.ah);
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
    fx = axis_fluxes(c, ex, 0, w0, p0);
    fy = axis_fluxes(c, ey, 1, w0, p0);
    if (!flat) { fz = axis_fluxes(c, ez, 2, w0, p0); }
  }

  // A shearing box's x faces: the two sides' fluxes, averaged.
  if (shear_on()) {
    if (c.x == 0) { fx.lo = shear_face(c, fx.lo, false); }
    if (c.x == i32(P.dims.x) - 1) { fx.hi = shear_face(c, fx.hi, true); }
  }
  // Planes of mirror symmetry: nothing crosses them.
  if (c.x == 0 && sym_lo(0)) { fx.lo = mirror_flux(fx.lo, 0); }
  if (c.y == 0 && sym_lo(1)) { fy.lo = mirror_flux(fy.lo, 1); }
  if (c.z == 0 && sym_lo(2) && !flat) { fz.lo = mirror_flux(fz.lo, 2); }

  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fx.lo.m, fx.hi.m, fy.lo.m, fy.hi.m));
    textureStore(O2, c, vec4<f32>(fz.lo.m, fz.hi.m, 0.0, 0.0));
    return;
  }
  store_mass_flux(c, fx, fy, fz, flat);
  if (DEBRIS) { escape(c, w0.a.yzw, fx, fy, fz, flat); }

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
  if (has_gravity() || ext_on()) {
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

  // An axisymmetric run: the radial faces weighed by their areas, and the radial momentum's
  // geometric source (p + rho v_phi^2) / R, with p the mean of the two radial faces' pressures:
  // with it, uniform pressure is in balance to rounding, and a balanced state stays as the
  // Cartesian one does. (The velocities are the half step's.)
  var src_r = 0.0;
  if (cylindrical()) {
    let w = cyl_weights(c.x);
    let ps = 0.5 * (face_pressure(fx.lo, 0) + face_pressure(fx.hi, 0));
    src_r = (ps + p0.a.x * p0.a.w * p0.a.w) / (f32(c.x) + 0.5);
    fx.lo = cyl_face(fx.lo, w.x);
    fx.hi = cyl_face(fx.hi, w.y);
  }
  var net = net_inflow_heated(fx, fy, fz, w0.a.yzw);
  net.mom.x += src_r;
  var nw = advance(u1, u2, net, dw, p0.b.x, kick, ekick);
  if (has_sources()) { nw = apply_sources(nw, (vec3<f32>(c) + 0.5) / vec3<f32>(P.dims.xyz), clock.dt); }
  nw = apply_damping(nw, (vec3<f32>(c) + 0.5) / vec3<f32>(P.dims.xyz), clock.dt);
  nw = apply_cooling(nw, clock.dt);
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}
