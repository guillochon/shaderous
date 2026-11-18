// The kinds of box face, and what each does to a ghost: the one place that says so for every
// grid (hydro/common.wgsl's bc_axis on the uniform grid, amr/grid.wgsl's block ghosts,
// mhd/amr_common.wgsl's cells and faces at any level). problems/boundaries.ts holds the same
// table for the interface; the codes are the values of P.bc_lo and P.bc_hi.
//
//   0 periodic      the opposite face's cells
//   1 outflow       the edge cell, copied (zero gradient)
//   2 reflecting    the mirror image, its normal velocity and normal field reversed: a wall the
//                   gas slips along, a plane of symmetry for the field
//   3 outflow only  the edge cell, with its normal velocity clamped to point outward
//   4 line-tied     the mirror image, its whole velocity reversed and its field kept: a wall the
//                   gas sticks to (no slip) and the field lines stay anchored in. With a field,
//                   the electric field along it is zero (mhd/emf.wgsl, mhd/amr_emf.wgsl), so the
//                   flux through it never changes.
//   5 moving wall   a line-tied wall moving along itself at the face's velocity (P.wallv): the
//                   ghost's velocity is 2 v_wall - v, and the electric field along it is
//                   -v_wall x B_n (mhd/corner.wgsl), so the field lines' footpoints move with it.
const BC_PERIODIC = 0u;
const BC_OUTFLOW = 1u;
const BC_REFLECT = 2u;
const BC_OUTFLOW_ONLY = 3u;
const BC_LINE_TIED = 4u;
const BC_MOVING_WALL = 5u;

// A wall: its ghosts are mirror images of the cells inside.
fn bc_wall(k: u32) -> bool {
  return k == BC_REFLECT || bc_tied(k);
}

// A wall the gas sticks to and the field lines are anchored in, at rest or moving.
fn bc_tied(k: u32) -> bool {
  return k == BC_LINE_TIED || k == BC_MOVING_WALL;
}

// The velocity of the face of kind k normal to `axis` (its lower or upper one): a moving wall's,
// along the wall (its normal part dropped), and zero for every other kind.
fn bc_wall_velocity(k: u32, axis: i32, up: bool) -> vec3<f32> {
  if (k != BC_MOVING_WALL) { return vec3<f32>(0.0); }
  var v = P.wallv[2 * axis + select(0, 1, up)].xyz;
  v[axis] = 0.0;
  return v;
}

// Cell index i (outside [0, n), at most n beyond it) along an axis whose face on that side is
// of kind k: the cell whose state stands for it. Periodic is clamped for a single-layer axis
// (n = 1), where the stencil reaches two cells out.
fn bc_ghost_cell(i: i32, n: i32, k: u32) -> i32 {
  let up = i >= n;
  if (k == BC_PERIODIC) { return clamp(select(i + n, i - n, up), 0, n - 1); }
  if (bc_wall(k)) { return select(-1 - i, 2 * n - 1 - i, up); }
  return select(0, n - 1, up);
}

// Signs of the ghost's velocity components beyond a face of kind k normal to `axis`.
fn bc_vsign(k: u32, axis: i32) -> vec3<f32> {
  if (bc_tied(k)) { return vec3<f32>(-1.0); }
  var s = vec3<f32>(1.0);
  if (k == BC_REFLECT) { s[axis] = -1.0; }
  return s;
}

// Signs of the ghost's field components beyond the same face.
fn bc_bsign(k: u32, axis: i32) -> vec3<f32> {
  var s = vec3<f32>(1.0);
  if (k == BC_REFLECT) { s[axis] = -1.0; }
  return s;
}

// The same where the face may be a plane of mirror symmetry (sym_lo): there B, a pseudovector,
// keeps its normal component and changes the sign of the tangential ones, the opposite of a
// conducting wall.
fn bc_bsign_at(k: u32, axis: i32, up: bool) -> vec3<f32> {
  let s = bc_bsign(k, axis);
  return select(s, -s, !up && P.sym[axis] == 1u);
}

// Beyond an outflow-only face: -1 below, +1 above (the side the ghost's normal velocity is
// clamped toward), else 0.
fn bc_side(k: u32, up: bool) -> f32 {
  if (k != BC_OUTFLOW_ONLY) { return 0.0; }
  return select(-1.0, 1.0, up);
}

// Face index i along its own axis (outside [0, n], the faces of an axis of n cells) beyond a face
// of kind k: (the face whose value stands for it, the sign of that value). A wall's ghost faces
// are mirror images about the wall face, the normal field reversed at a reflecting one; an open
// face repeats the boundary face.
fn bc_ghost_face(i: i32, n: i32, k: u32) -> vec2<i32> {
  let up = i > n;
  if (k == BC_PERIODIC) { return vec2<i32>(((i % n) + n) % n, 1); }
  if (bc_wall(k)) { return vec2<i32>(select(-i, 2 * n - i, up), select(1, -1, k == BC_REFLECT)); }
  return vec2<i32>(select(0, n, up), 1);
}

// A line-tied face holds the field lines that cross it: the electric field has no component
// along it. Given which axes' faces an edge texel lies on that are line-tied, the factors that
// keep E's components: E_c survives unless the edge lies in a line-tied face normal to another
// axis (the component normal to that face is not along it).
fn bc_tied_keep(on: vec3<bool>) -> vec3<f32> {
  return vec3<f32>(
    select(1.0, 0.0, on.y || on.z),
    select(1.0, 0.0, on.z || on.x),
    select(1.0, 0.0, on.x || on.y),
  );
}
