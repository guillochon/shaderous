// Neighbor sum of the 7-point Laplacian with boundary handling. Requires PHI (a storage
// texture with read access), P and L; binds BF, the isolated boundary's values
// (gravity/boundary.wgsl).
#include "gravity/boundary_index.wgsl"
@group(0) @binding(11) var<storage, read> BF: array<f32>;
//
// nb_sum returns (sum, diag) such that  h^2 Lap(phi) = sum - diag * phi_c.
// Periodic: neighbors wrap. Isolated: Dirichlet data at the face, ghost = 2 phi_face - phi_c,
// which moves one phi_c per boundary face onto the diagonal. Coarse levels solve for the
// error, whose boundary value is zero. A plane of mirror symmetry (sym_lo) is a zero-gradient
// face on every level: the ghost is the cell's own image, phi_c.

// Contribution of one neighbor: (value added to the sum, 1 if it added to the diagonal).
fn nb_one(c: vec3<i32>, o: vec3<i32>) -> vec2<f32> {
  let n = i32(L.n);
  let q = c + o;
  if (all(q >= vec3<i32>(0)) && all(q < vec3<i32>(n))) {
    return vec2<f32>(textureLoad(PHI, q).x, 0.0);
  }
  if (P.grav_bc == 0u) {
    return vec2<f32>(textureLoad(PHI, (q + n) % n).x, 0.0);
  }
  if (any(q < vec3<i32>(0))) {
    let a = select(select(2, 1, q.y < 0), 0, q.x < 0);
    if (sym_lo(a)) { return vec2<f32>(textureLoad(PHI, c).x, 0.0); }
  }
  var s = 0.0;
  if (L.level == 0u) { s = 2.0 * BF[bf_index(c, o, n)]; }
  return vec2<f32>(s, 1.0);
}

fn nb_sum(c: vec3<i32>) -> vec2<f32> {
  if (L.flat != 0u) {
    // An axisymmetric (R, z) run: (1/R) d/dR (R dphi/dR) + d2phi/dz2, the radial neighbors
    // weighed by R_-/R_c and R_+/R_c (cyl_weights; the axis face's weight is 0), and no z.
    let w = cyl_weights(c.x);
    let r = w.x * nb_one(c, vec3<i32>(-1, 0, 0)) + w.y * nb_one(c, vec3<i32>(1, 0, 0))
          + nb_one(c, vec3<i32>(0, -1, 0)) + nb_one(c, vec3<i32>(0, 1, 0));
    return vec2<f32>(r.x, w.x + w.y + 2.0 + r.y);
  }
  let r = nb_one(c, vec3<i32>(-1, 0, 0)) + nb_one(c, vec3<i32>(1, 0, 0))
        + nb_one(c, vec3<i32>(0, -1, 0)) + nb_one(c, vec3<i32>(0, 1, 0))
        + nb_one(c, vec3<i32>(0, 0, -1)) + nb_one(c, vec3<i32>(0, 0, 1));
  return vec2<f32>(r.x, 6.0 + r.y);
}
