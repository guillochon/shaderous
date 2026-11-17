// The edge field of constrained transport from the four faces around an edge and the four
// cells around it, shared by both grids (mhd/emf.wgsl, mhd/amr_emf.wgsl). Requires fc(a, t), the
// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.

// What the edges need of a cell: its reference field -v x B, and its field B (for the
// resistive part, mhd/ohm.wgsl).
struct Ref { e: vec3<f32>, b: vec3<f32> }

// dx times the current J_c along axis c at the same edge, from the fields of the four cells
// around it (the arguments of corner()): J_c = d_a B_b - d_b B_a, each derivative the mean of
// the differences across the edge on its two sides.
fn edge_current(c: i32, r00: Ref, r10: Ref, r01: Ref, r11: Ref) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  let da = 0.5 * ((r10.b[b] + r11.b[b]) - (r00.b[b] + r01.b[b]));
  let db = 0.5 * ((r01.b[a] + r11.b[a]) - (r00.b[a] + r10.b[a]));
  return da - db;
}

// E along axis c at the edge on the lower a and b faces of cell t (a = c + 1, b = c + 2),
// given the four cells around it: r00 at t - ea - eb, r10 at t - eb, r01 at t - ea, r11 at t.
//
// Each of the four faces around the edge gives an estimate of the edge's field: its own value
// carried half a cell along the face to the edge, with the gradient taken between the face and
// the reference field of the cell on that side, from whichever cell is upwind across the face
// (the mass flux's direction). The edge's field is the mean of the four: Gardiner & Stone
// (2005), as Athena++ does it.
fn corner(c: i32, t: vec3<i32>, r00: Ref, r10: Ref, r01: Ref, r11: Ref) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  var ea = vec3<i32>(0);
  ea[a] = 1;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  let e00 = r00.e[c];
  let e10 = r10.e[c];
  let e01 = r01.e[c];
  let e11 = r11.e[c];
  // Faces normal to a, at b positions j (t) and j - 1 (t - eb): E_c = -F_a(B_b).
  let fa1 = fc(a, t);
  let fa0 = fc(a, t - eb);
  // Faces normal to b, at a positions i (t) and i - 1 (t - ea): E_c = F_b(B_a).
  let fb1 = fc(b, t);
  let fb0 = fc(b, t - ea);
  let ea1 = -fa1.x;
  let ea0 = -fa0.x;
  let eb1 = fb1.y;
  let eb0 = fb0.y;
  // The four estimates: from the a face below the edge (along b), the a face above it, the b
  // face left of it (along a) and the b face right of it.
  let below = ea0 + ((1.0 - fa0.z) * (eb1 - e10) + fa0.z * (eb0 - e00));
  let above = ea1 + ((1.0 - fa1.z) * (eb1 - e11) + fa1.z * (eb0 - e01));
  let left = eb0 + ((1.0 - fb0.z) * (ea1 - e01) + fb0.z * (ea0 - e00));
  let right = eb1 + ((1.0 - fb1.z) * (ea1 - e11) + fb1.z * (ea0 - e10));
  return 0.25 * ((left + right) + (below + above));
}

// The electric field along the walls an edge texel t lies on that move (hydro/boundary.wgsl):
// E = -v_wall x B_n, the field normal to the wall carried along it, so the footpoints of the
// field lines move with the wall. on, kinds and upper say, per axis, whether t lies on a
// line-tied or moving face of the domain normal to it, which kind, and which side. B_n is the
// mean of the two wall faces either side of the edge, so the flux through the wall is carried
// along it exactly (centered), whatever the field inside does. Requires wall_bn(a, t), the face
// field normal to a on the lower face along a of cell t, and hydro/boundary.wgsl.
fn moving_wall_e(t: vec3<i32>, on: vec3<bool>, kinds: vec3<u32>, upper: vec3<bool>) -> vec3<f32> {
  var E = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    if (!on[a] || kinds[a] != BC_MOVING_WALL) { continue; }
    let vw = bc_wall_velocity(kinds[a], a, upper[a]);
    for (var c = 0; c < 3; c++) {
      if (c == a) { continue; }
      var eb = vec3<i32>(0);
      eb[3 - a - c] = 1;
      var bn = vec3<f32>(0.0);
      bn[a] = 0.5 * (wall_bn(a, t) + wall_bn(a, t - eb));
      E[c] -= cross(vw, bn)[c];
    }
  }
  return E;
}
