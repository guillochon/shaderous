// Viscosity and thermal conduction: what they add to the flux through a face.
//
//   viscosity    stress tau = rho nu (grad v + (grad v)^T - 2/3 (div v) I), with a constant
//                kinematic viscosity nu = P.diff.x. The momentum flux is -tau . n and the
//                energy flux -(tau . n) . v.
//   conduction   heat flux -kappa rho grad T, with kappa = P.diff.y constant and T as the
//                equation of state gives it (p / rho for an ideal gas). kappa has the units of
//                a diffusivity; the temperature of an ideal gas diffuses at (gamma - 1) kappa.
//
// (Both as Athena++ defines its nu_iso and kappa_iso.) Which of the two are compiled in comes
// from the <diffusion> slot: the constants VISCOSITY and CONDUCTION.
//
// The fluxes are taken from the primitives at the start of the step and added to the Riemann
// flux of the face, so everything that holds for that flux holds for them: the two cells
// sharing a face compute the same bits, and across a refinement boundary or between blocks
// with different timesteps both sides use one flux (amr/update.wgsl). hydro/diff_rate.wgsl
// limits the timestep.
//
// Walls. A face of an obstacle is a wall at rest that the gas sticks to (no slip) and that
// passes no heat. The faces of the domain are as they are for the gas: a reflecting one is a
// plane of symmetry, with no shear stress on it.
//
// Requires P, the equation of state, hydro/q.wgsl, Flux (hydro/flux.wgsl), g_dx set, and from
// the grid (with load_comp(c), the composition of a cell, for conduction):
//   load_vel(c)             velocity of a cell (zero for a solid one)
//   face_rim(a, axis, t)    0 where both cells of the face between a and a + e_axis have their
//                           two neighbors along the transverse axis t; +1 / -1 where only the
//                           upper / lower ones can be read (the rim of a block's face)
#include <diffusion>

fn q_temperature(q: Q) -> f32 {
  return eos_temperature(q.a.x, eos_eint_from_p(q.a.x, q.b.x));
}

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

// The flux F through the face between cell a (state l) and cell a + e (state r), along +axis,
// with the diffusive fluxes added. l and r are the cells as stored: either may be solid.
fn add_diffusion(Fin: Flux, a: vec3<i32>, e: vec3<i32>, axis: i32, l: Q, r: Q) -> Flux {
  var F = Fin;
  if (!VISCOSITY && !CONDUCTION) { return F; }
  let ls = q_solid(l);
  let rs = q_solid(r);
  if (ls && rs) { return F; }
  let wall = ls || rs;
  let inv = 1.0 / g_dx;
  // Density on the face; at a wall, that of the gas.
  var rho = 0.5 * (l.a.x + r.a.x);
  if (ls) { rho = r.a.x; }
  if (rs) { rho = l.a.x; }

  if (VISCOSITY) {
    let t1 = (axis + 1) % 3;
    let t2 = (axis + 2) % 3;
    let vl = l.a.yzw;
    let vr = r.a.yzw;
    // dx times the derivatives of the velocity along the normal and the two transverse axes,
    // and the velocity on the face.
    var dn = vr - vl;
    var d1 = vec3<f32>(0.0);
    var d2 = vec3<f32>(0.0);
    var vf = 0.5 * (vl + vr);
    if (wall) {
      // The wall is at rest half a cell from the gas cell's center, and nothing varies along it.
      dn = select(2.0 * vr, -2.0 * vl, rs);
      vf = vec3<f32>(0.0);
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
    tau *= P.diff.x * rho * inv;
    let work = -dot(tau, vf);
    F.mom -= tau;
    F.e += work;
    F.d = vec4<f32>(-tau, work);
  }
  if (CONDUCTION && !wall) {
    // Each temperature with its own cell's composition.
    comp_use(load_comp(a + e));
    let tr = q_temperature(r);
    comp_use(load_comp(a));
    let q = -P.diff.y * rho * (tr - q_temperature(l)) * inv;
    F.e += q;
    F.ei += q;
  }
  return F;
}

// What a face's viscous flux leaves as heat in a cell moving at v, per unit dt/dx and with the
// sign of an inflow (so: lower face minus upper face). The energy the stress carries in, less
// what goes into the cell's kinetic energy; summed over the faces it is tau : grad v. The
// internal energy that is advected on its own needs it (hydro/dual_energy.wgsl); the total
// energy has it already.
fn visc_heat(f: Flux, v: vec3<f32>) -> f32 {
  return f.d.w - dot(v, f.d.xyz);
}
