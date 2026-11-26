// The potential's Dirichlet data at isolated faces of the box, once per solve: into the ghost cell
// next to each such face on the full-resolution grid (sublevel 0), 2 phi_face, which the stencil
// reads as that neighbor's contribution (amr/gravity/stencil.wgsl, g_nb). The multipole
// expansion is then evaluated once per face cell and solve, not by every cell on the box's edge
// in every sweep, residual and restriction: kept in those passes, it made them several times
// slower even where no cell reached it (2026-11-26: a sweep of the finest sublevel of a tidal
// disruption, 93 us against 25 us with the expansion compiled in but never run, and 5 us without it).
//
// The ghost (rest) pass that ends every solve (amr/gravity/ghost.wgsl, MODE 1) writes these cells
// over with the potential extended past the face, for the field pass.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, write>;

// As ghost.wgsl's: per block 6 faces x 2 layers x BS x BS cells (in 2D 4 x 2 x BS), in rows of
// 64, one block of the list per workgroup row. Only the layer next to the block is written.
const FACE_GHOSTS = 6 * 2 * BS * BS;
const FACE_GHOSTS_2D = 4 * 2 * BS;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let i = i32(wid.x * 64u + lix);
  let row = flat_row(wid);
  let flat = is2d();
  if (row >= LIST[0] || i >= select(FACE_GHOSTS, FACE_GHOSTS_2D, flat)) { return; }
  let slot = i32(LIST[1u + row]);
  var face = i / (2 * BS * BS);          // -x, +x, -y, +y, -z, +z
  var r = i % (2 * BS * BS);
  var layer = r / (BS * BS);
  if (flat) {
    face = i / (2 * BS);
    r = i % (2 * BS);
    layer = r / BS;
  }
  if (layer != 0) { return; }
  let axis = face / 2;
  let upper = (face & 1) == 1;
  // Beyond this face: a block of the level (>= 0), a coarser one (-1), or the box's edge.
  let nb = bt(slot, 6u + u32(2 * axis) + u32(upper));
  if (nb >= -1) { return; }
  // A plane of mirror symmetry: the stencil takes the cell's image there instead.
  if (!upper && sym_lo(axis)) { return; }
  // The cell inside the face, as the stencil's g_nb sees it, and the face's center, computed as
  // the stencil did when it evaluated the expansion itself (h at sublevel 0 is the block's dx).
  var l = vec3<i32>(0);
  if (flat) {
    l[1 - axis] = r % BS;
  } else {
    l[(axis + 1) % 3] = r % BS;
    l[(axis + 2) % 3] = (r / BS) % BS;
  }
  l[axis] = select(0, BS - 1, upper);
  var e = vec3<f32>(0.0);
  e[axis] = select(-0.5, 0.5, upper);
  var x = (vec3<f32>(blk_coord(slot) * BS + l) + 0.5 + e) * blk_dx(slot) + P.org.xyz;
  // (A 2D run lies in the plane z = 0, as its moments are taken.)
  if (flat) { x.z = 0.0; }
  var q = l;
  q[axis] += select(-1, 1, upper);
  textureStore(PHI, blk_origin(slot) + q, vec4<f32>(2.0 * phi_boundary(x), 0.0, 0.0, 0.0));
}
