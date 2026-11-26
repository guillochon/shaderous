// The 7-point Laplacian on one tree level of the block grid (5-point in 2D, and in an
// axisymmetric run the radial neighbors weighed by R_-/R_c and R_+/R_c, as the uniform grid's
// gravity/stencil.wgsl). Requires amr/gravity/common.wgsl and PHI: a storage texture with read
// access.
//
// g_nb returns (value, diag): the neighbor's contribution to the sum, and 1 if it added the
// center cell to the diagonal. h^2 Lap(phi) = sum - (6 + diag total) * phi_c.

fn g_nb(slot: i32, o: vec3<i32>, l: vec3<i32>, axis: i32, upper: bool) -> vec2<f32> {
  var q = l;
  q[axis] += select(-1, 1, upper);
  let B = GL.bsize;
  if (q[axis] >= 0 && q[axis] < B) {
    return vec2<f32>(textureLoad(PHI, o + q).x, 0.0);
  }
  let nb = bt(slot, 6u + u32(2 * axis) + u32(upper));
  if (nb >= 0) {
    q[axis] += select(B, -B, upper);
    return vec2<f32>(textureLoad(PHI, g_origin(nb) + q).x, 0.0);
  }
  if (nb == -1) {
    // Coarser region beyond: on the full grid the ghost cell holds the interpolated coarse
    // potential; on the error grids below it, the error there is zero.
    if (GL.k == 0u) { return vec2<f32>(textureLoad(PHI, o + q).x, 0.0); }
    return vec2<f32>(0.0, 0.0);
  }
  // A plane of mirror symmetry: the ghost is the cell's image, phi_c, on every sublevel.
  if (!upper && sym_lo(axis)) { return vec2<f32>(textureLoad(PHI, o + l).x, 0.0); }
  // Domain boundary (isolated): Dirichlet data at the face, ghost = 2 phi_face - phi_c, the only
  // neighbor with a diagonal term. On the full grid the ghost cell holds 2 phi_face, written as
  // the solve starts (amr/gravity/boundary.wgsl); on the error grids below it, 0.
  if (GL.k == 0u) { return vec2<f32>(textureLoad(PHI, o + q).x, 1.0); }
  return vec2<f32>(0.0, 1.0);
}

// The six neighbors' (value, diag), lower then upper along x, y, z (four in 2D).
struct GNb { n0: vec2<f32>, n1: vec2<f32>, n2: vec2<f32>, n3: vec2<f32>, n4: vec2<f32>, n5: vec2<f32> }

fn g_nbs(slot: i32, o: vec3<i32>, l: vec3<i32>, faces: i32) -> GNb {
  var n = GNb(g_nb(slot, o, l, 0, false), g_nb(slot, o, l, 0, true), g_nb(slot, o, l, 1, false), g_nb(slot, o, l, 1, true),
              vec2<f32>(0.0), vec2<f32>(0.0));
  if (faces == 6) {
    n.n4 = g_nb(slot, o, l, 2, false);
    n.n5 = g_nb(slot, o, l, 2, true);
  }
  return n;
}

fn g_sum(slot: i32, o: vec3<i32>, l: vec3<i32>) -> vec2<f32> {
  if (is2d()) {
    // (R, z): (1/R) d/dR (R dphi/dR) + d2phi/dz2 (cyl_weights; the axis face's weight is 0).
    var w = vec2<f32>(1.0);
    if (cylindrical()) { w = cyl_weights(blk_coord(slot).x * GL.bsize + l.x); }
    let n = g_nbs(slot, o, l, 4);
    let r = w.x * n.n0 + w.y * n.n1 + n.n2 + n.n3;
    return vec2<f32>(r.x, w.x + w.y + 2.0 + r.y);
  }
  let n = g_nbs(slot, o, l, 6);
  let r = n.n0 + n.n1 + n.n2 + n.n3 + n.n4 + n.n5;
  return vec2<f32>(r.x, 6.0 + r.y);
}
