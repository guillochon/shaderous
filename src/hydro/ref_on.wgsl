// The gas stored as its deviation from a reference state at rest (SimConfig.reference;
// docs/numerics.md section 7, "Deviations from a reference"). The state textures hold
//   U1 = (rho - rho_ref, m)    U2 = (E - e_ref, rho e - e_ref, rho s, solid)
// with e_ref the reference's internal energy per volume, and the primitives the passes carry
// (hydro/q.wgsl Q) hold the density and pressure the same way: (rho - rho_ref, v) and
// (p - p_ref, scalar, c_s, solid). Single precision then holds a deviation to its own size: the
// small pressure differences of slow flow under a large background pressure, and those of a
// steep atmosphere at rest, which no longer round to the size of the pressure itself.
//
// The reference is in hydrostatic balance by construction (hydro/refstate.ts), with the
// trapezoid rule of the balanced reconstruction, so its own forces cancel exactly and are never
// computed: the pressure in the momentum fluxes is p - p_ref (Side.pd), the gravity kick is
// (rho - rho_ref) g, and the reconstruction's balanced part is -(rho - rho_ref) dphi. The
// reference enters elsewhere only as the coefficients it sets: densities in the mass fluxes,
// pressures in the sound speed and the energy fluxes.
//
// It varies along one axis, the uniform field's (P.refx.y), or not at all (3). Its values are
// in P.reft, cell by cell along that axis.
const REF_DEV = true;

// The reference at cell c, through the boundary mapping as the state's loads are:
// (density, pressure, internal energy per volume, 0).
fn ref_at(c: vec3<i32>) -> vec4<f32> {
  let a = i32(P.refx.y);
  if (a > 2) { return P.reft[0]; }
  let m = bc_map(c).c;
  return P.reft[clamp(m[a], 0, 2047)];
}

// Its density's limited difference across the middle of three cells along an axis (minmod: the
// reference may have a jump).
fn ref_slope(rm: vec4<f32>, rc: vec4<f32>, rp: vec4<f32>) -> f32 {
  let a = rp.x - rc.x;
  let b = rc.x - rm.x;
  if (a * b <= 0.0) { return 0.0; }
  return select(b, a, abs(a) < abs(b));
}

// The reference on one side of a face: (density and pressure of the cell, density and
// pressure at the face), for the cell rc between rm and rp along the face's axis, toward its
// upper face (sgn = 1) or lower one (-1); df is the potential step toward that face that the
// cell's edge increment was given (hydro/q.wgsl). The pressure is carried to the face along the
// balance, as the reconstruction carries it: p - sgn rho df / 2.
fn ref_face(rm: vec4<f32>, rc: vec4<f32>, rp: vec4<f32>, sgn: f32, df: f32) -> vec4<f32> {
  let rho = rc.x + 0.5 * sgn * ref_slope(rm, rc, rp);
  let p = rc.y - 0.5 * sgn * rc.x * df;
  return vec4<f32>(rc.x, rc.y, rho, p);
}
