// Neighbour sum of the 7-point Laplacian with boundary handling. Requires PHI (a storage
// texture with read access), P, L and phi_boundary().
//
// nb_sum returns (sum, diag) such that  h^2 Lap(phi) = sum - diag * phi_c.
// Periodic: neighbours wrap. Isolated: Dirichlet data at the face, ghost = 2 phi_face - phi_c,
// which moves one phi_c per boundary face onto the diagonal. Coarse levels solve for the
// error, whose boundary value is zero.

// Contribution of one neighbour: (value added to the sum, 1 if it added to the diagonal).
fn nb_one(c: vec3<i32>, o: vec3<i32>) -> vec2<f32> {
  let n = i32(L.n);
  let q = c + o;
  if (all(q >= vec3<i32>(0)) && all(q < vec3<i32>(n))) {
    return vec2<f32>(textureLoad(PHI, q).x, 0.0);
  }
  if (P.grav_bc == 0u) {
    return vec2<f32>(textureLoad(PHI, (q + n) % n).x, 0.0);
  }
  var s = 0.0;
  if (L.level == 0u) {
    let x = (vec3<f32>(c) + 0.5 + 0.5 * vec3<f32>(o)) * L.h - 0.5 * P.box;
    s = 2.0 * phi_boundary(x);
  }
  return vec2<f32>(s, 1.0);
}

fn nb_sum(c: vec3<i32>) -> vec2<f32> {
  let r = nb_one(c, vec3<i32>(-1, 0, 0)) + nb_one(c, vec3<i32>(1, 0, 0))
        + nb_one(c, vec3<i32>(0, -1, 0)) + nb_one(c, vec3<i32>(0, 1, 0))
        + nb_one(c, vec3<i32>(0, 0, -1)) + nb_one(c, vec3<i32>(0, 0, 1));
  return vec2<f32>(r.x, 6.0 + r.y);
}
