// Viscosity and thermal conduction through one face, whatever the grid and whatever the flux
// they are added to (hydro/diffusion.wgsl for the gas alone, mhd/flux.wgsl and
// mhd/amr_flux.wgsl with a magnetic field; docs/numerics.md for the definitions):
//
//   viscosity    stress tau = rho nu (grad v + (grad v)^T - 2/3 (div v) I), with a constant
//                kinematic viscosity nu = P.diff.x. The momentum flux is -tau . n and the
//                energy flux -(tau . n) . v.
//   conduction   heat flux -kappa rho grad T, with kappa = P.diff.y constant and T as the
//                equation of state gives it (p / rho for an ideal gas). kappa has the units of
//                a diffusivity; the temperature of an ideal gas diffuses at (gamma - 1) kappa.
//
// (Both as Athena++ defines its nu_iso and kappa_iso; isotropic, also with a field: no
// Braginskii.) Which of the two are compiled in comes from the <diffusion> slot: the constants
// VISCOSITY and CONDUCTION.
//
// Walls. A face of an obstacle is a wall at rest that the gas sticks to (no slip) and that
// passes no heat. The faces of the domain are as they are for the gas: a reflecting one is a
// plane of symmetry, with no shear stress on it.
//
// Requires P, the equation of state, g_dx set, and from the grid (with load_comp(c), the
// composition of a cell, for conduction):
//   load_vel(c)             velocity of a cell (zero for a solid one)
//   face_rim(a, axis, t)    0 where both cells of the face between a and a + e_axis have their
//                           two neighbors along the transverse axis t; +1 / -1 where only the
//                           upper / lower ones can be read (the rim of a block's face)
#include <diffusion>

// What the diffusive fluxes need of a cell beside the face.
struct DSide { rho: f32, v: vec3<f32>, p: f32, solid: bool }

// dx times the derivative of the velocity along the transverse axis t, at the face between
// cell a and cell a + e: the mean of the centered differences in the two cells, or of one-sided
// ones at a rim. vl and vr are the velocities of the two cells.
fn visc_across(a: vec3<i32>, e: vec3<i32>, axis: i32, t: i32, vl: vec3<f32>, vr: vec3<f32>) -> vec3<f32> {
  var tv = vec3<i32>(0);
  tv[t] = 1;
  let b = a + e;
  let rim = face_rim(a, axis, t);
  if (rim > 0) { return 0.5 * ((load_vel(a + tv) - vl) + (load_vel(b + tv) - vr)); }
  if (rim < 0) { return 0.5 * ((vl - load_vel(a - tv)) + (vr - load_vel(b - tv))); }
  return 0.25 * ((load_vel(a + tv) - load_vel(a - tv)) + (load_vel(b + tv) - load_vel(b - tv)));
}

// Density on the face between l and r: the smaller of the two; at a wall, that of the gas.
// With the mean, a thin cell beside a dense one took the stress and heat of the dense gas: next
// to gas 1e7 times denser its velocity and temperature diffused 5e6 times faster than nu or
// kappa, which no timestep of hydro/diff_rate.wgsl allows for (found by tools/random-test, a
// dense blob in 1e-6 gas). With the smaller density neither cell diffuses faster than at its
// own density, the bound that timestep assumes. Where the density is smooth the two differ by
// half the difference between the cells.
fn diff_rho(l: DSide, r: DSide) -> f32 {
  if (l.solid) { return r.rho; }
  if (r.solid) { return l.rho; }
  return min(l.rho, r.rho);
}

// The viscous flux through the face between cell a (l) and cell a + e (r), along +axis:
// (momentum, energy), that is (-tau . n, -(tau . n) . v). Zero where both cells are solid.
fn visc_flux(a: vec3<i32>, e: vec3<i32>, axis: i32, l: DSide, r: DSide) -> vec4<f32> {
  if (l.solid && r.solid) { return vec4<f32>(0.0); }
  let wall = l.solid || r.solid;
  let rho = diff_rho(l, r);
  let t1 = (axis + 1) % 3;
  let t2 = (axis + 2) % 3;
  let vl = l.v;
  let vr = r.v;
  // dx times the derivatives of the velocity along the normal and the two transverse axes,
  // and the velocity on the face.
  var dn = vr - vl;
  var d1 = vec3<f32>(0.0);
  var d2 = vec3<f32>(0.0);
  var vf = 0.5 * (vl + vr);
  if (wall) {
    // The wall is half a cell from the gas cell's center, and nothing varies along it. It moves
    // with its solid cell's velocity: 0 for an obstacle at rest (whose gas is held at rest), a
    // moving body's (hydro/moving.ts) for one of those. (At rest, the same numbers as 2 v_r and
    // -2 v_l.)
    dn = 2.0 * (vr - vl);
    vf = select(vl, vr, r.solid);
  } else {
    let flat = P.dims.z == 1u;
    if (!(flat && t1 == 2)) { d1 = visc_across(a, e, axis, t1, vl, vr); }
    if (!(flat && t2 == 2)) { d2 = visc_across(a, e, axis, t2, vl, vr); }
  }
  // tau . n: component i is dv_i/dn + dv_n/dx_i, less 2/3 div v along the normal.
  var tau = dn;
  tau[axis] += dn[axis];
  tau[t1] += d1[axis];
  tau[t2] += d2[axis];
  tau[axis] -= (2.0 / 3.0) * (dn[axis] + d1[t1] + d2[t2]);
  tau *= P.diff.x * rho * (1.0 / g_dx);
  return vec4<f32>(-tau, -dot(tau, vf));
}

// The conductive heat flux through the same face; none through a wall.
fn cond_flux(a: vec3<i32>, e: vec3<i32>, l: DSide, r: DSide) -> f32 {
  if (l.solid || r.solid) { return 0.0; }
  // Each temperature with its own cell's composition.
  comp_use(load_comp(a + e));
  let tr = eos_temperature(r.rho, eos_eint_from_p(r.rho, r.p));
  comp_use(load_comp(a));
  let tl = eos_temperature(l.rho, eos_eint_from_p(l.rho, l.p));
  return -P.diff.y * diff_rho(l, r) * (tr - tl) * (1.0 / g_dx);
}

// What a face's viscous flux d = (momentum, energy) leaves as heat in a cell moving at v, per
// unit dt/dx and with the sign of an inflow (so: lower face minus upper face). The energy the
// stress carries in, less what goes into the cell's kinetic energy; summed over the faces it is
// tau : grad v. The internal energy that is advected on its own needs it
// (hydro/dual_energy.wgsl); the total energy has it already.
fn visc_heat_d(d: vec4<f32>, v: vec3<f32>) -> f32 {
  return d.w - dot(v, d.xyz);
}
