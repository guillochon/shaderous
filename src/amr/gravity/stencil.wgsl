// The 7-point Laplacian on one tree level of the block grid. Requires amr/gravity/common.wgsl,
// gravity/multipole_eval.wgsl, and PHI: a storage texture with read access.
//
// g_nb returns (value, diag): the neighbour's contribution to the sum, and 1 if it added the
// centre cell to the diagonal. h^2 Lap(phi) = sum - (6 + diag total) * phi_c.

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
  // Domain boundary (isolated): Dirichlet data at the face, ghost = 2 phi_face - phi_c.
  var v = 0.0;
  if (GL.k == 0u) {
    var e = vec3<f32>(0.0);
    e[axis] = select(-0.5, 0.5, upper);
    let x = (vec3<f32>(blk_coord(slot) * B + l) + 0.5 + e) * g_h(slot) - 0.5 * P.box;
    v = 2.0 * phi_boundary(x);
  }
  return vec2<f32>(v, 1.0);
}

fn g_sum(slot: i32, o: vec3<i32>, l: vec3<i32>) -> vec2<f32> {
  let r = g_nb(slot, o, l, 0, false) + g_nb(slot, o, l, 0, true)
        + g_nb(slot, o, l, 1, false) + g_nb(slot, o, l, 1, true)
        + g_nb(slot, o, l, 2, false) + g_nb(slot, o, l, 2, true);
  return vec2<f32>(r.x, 6.0 + r.y);
}
