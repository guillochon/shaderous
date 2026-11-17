// Resistivity: a constant magnetic diffusivity eta = P.ohm.x, as Athena++'s eta_ohm. The
// electric field gains eta J, J = curl B, so that the induction equation gains eta lap B
// (mhd/corner.wgsl adds it to the edge fields of constrained transport, which keeps div B where
// it was). The total energy takes the matching Poynting flux eta J x B through every face (here,
// added to the face flux by mhd/diffusion.wgsl), so it is conserved to round-off, and the
// internal energy that is advected on its own takes the Ohmic heat eta J^2 (ohm_heat).
//
// J is taken from the cell-centered field throughout: at an edge, from the four cells around it
// (mhd/corner.wgsl); at a face, from its two cells and their transverse neighbors, as the
// viscous stress takes the velocity (hydro/diffusion_core.wgsl); at a cell, from its six
// neighbors. The same on both grids, and on blocks only the assembled cells are read.
//
// Requires P, g_dx set, the <diffusion> slot (RESISTIVITY), and load_b(c): the cell-centered
// field of cell c (a grid cell through the boundary mapping, or an atlas texel).

// dx times the derivative of B along the transverse axis t at the face between cells a and
// a + e: the mean of the centered differences in the two cells.
fn ohm_across(a: vec3<i32>, e: vec3<i32>, t: i32) -> vec3<f32> {
  var tv = vec3<i32>(0);
  tv[t] = 1;
  let b = a + e;
  return 0.25 * ((load_b(a + tv) - load_b(a - tv)) + (load_b(b + tv) - load_b(b - tv)));
}

// The energy flux eta (J x B) . n through the face between cell a (field bl) and a + e (br),
// normal to `axis`. With t1, t2 the transverse axes,
//   (J x B)_n = B_t1 (d_t1 B_n - d_n B_t1) + B_t2 (d_t2 B_n - d_n B_t2).
fn ohm_eflux(a: vec3<i32>, e: vec3<i32>, axis: i32, bl: vec3<f32>, br: vec3<f32>) -> f32 {
  let t1 = (axis + 1) % 3;
  let t2 = (axis + 2) % 3;
  let flat = P.dims.z == 1u;
  let bf = 0.5 * (bl + br);
  let dn = br - bl;
  var d1 = vec3<f32>(0.0);
  var d2 = vec3<f32>(0.0);
  if (!(flat && t1 == 2)) { d1 = ohm_across(a, e, t1); }
  if (!(flat && t2 == 2)) { d2 = ohm_across(a, e, t2); }
  let jb = bf[t1] * (d1[axis] - dn[t1]) + bf[t2] * (d2[axis] - dn[t2]);
  return P.ohm.x * jb / g_dx;
}

// The Ohmic heat of cell x, eta J^2, per unit dt/dx (as the update adds its fluxes), with J from
// centered differences of the neighbors' fields.
fn ohm_heat(x: vec3<i32>) -> f32 {
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let dX = 0.5 * (load_b(x + ex) - load_b(x - ex));
  let dY = 0.5 * (load_b(x + ey) - load_b(x - ey));
  var dZ = vec3<f32>(0.0);
  if (P.dims.z != 1u) {
    let ez = vec3<i32>(0, 0, 1);
    dZ = 0.5 * (load_b(x + ez) - load_b(x - ez));
  }
  // dx J = (d_y B_z - d_z B_y, d_z B_x - d_x B_z, d_x B_y - d_y B_x), each d times dx.
  let j = vec3<f32>(dY.z - dZ.y, dZ.x - dX.z, dX.y - dY.x);
  return P.ohm.x * dot(j, j) / g_dx;
}
