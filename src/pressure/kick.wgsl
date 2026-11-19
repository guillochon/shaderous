// Gravity in the implicit step of a gas with an energy (plans/incompressible.md 4.7): it kicks
// the face velocities. Requires P and G, the gravity field (acceleration, and the potential at
// the half step that the explicit scheme's well-balanced reconstruction also differences).
//
// The kick of the face between a lower and an upper cell: dt times the uniform field, less dt
// times the potential's difference across the face over dx. A face is then at rest where
//   p_hi - p_lo = -(rho_lo + rho_hi) / 2 * (phi_hi - phi_lo - g dx),
// the trapezoid balance a settled star and a balanced column are in.
//
// Point masses (gravity/ext.wgsl; BH, which the pass binds) kick the faces the same way, by the
// difference of their potential across the face, taken half a step ahead along their motion.
// Their work on the gas is then in flux form like self-gravity's, not the kick form of the
// explicit scheme's passes: this step is for slow flows, where the two differ little.
fn face_kick(lo: vec3<i32>, hi: vec3<i32>, a: i32, dt: f32) -> f32 {
  var k = dt * P.gext[a];
  if (P.grav_on != 0u && P.bh.w == 0.0) { k -= dt * (textureLoad(G, hi, 0).w - textureLoad(G, lo, 0).w) / P.dx; }
  if (ext_on()) {
    if (hi[a] < lo[a]) {
      // The face that wraps around a periodic box: the masses' potential is not periodic, and
      // its difference to a point outside the box is not this face's. The two cells' pulls.
      k += dt * 0.5 * (ext_accel(cell_pos(lo))[a] + ext_accel(cell_pos(hi))[a]);
    } else {
      var d = vec3<f32>(0.0);
      d[a] = P.dx;
      k -= dt * ext_dphi(cell_pos(lo), d, 0.5 * dt) / P.dx;
    }
  }
  return k;
}
// The acceleration at a cell, for the faces with nothing beyond them (walls, open faces).
fn cell_accel(c: vec3<i32>) -> vec3<f32> {
  var g = P.gext.xyz;
  if (P.grav_on != 0u) { g += textureLoad(G, c, 0).xyz; }
  return g + ext_accel(cell_pos(c));
}
