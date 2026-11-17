// MHD on the block grid: where a cell of any level lives, ghosts beyond the domain included.
// Shared by mhd/amr_common.wgsl and mhd/amr_update.wgsl. Requires P, amr/grid.wgsl,
// hydro/boundary.wgsl (through hydro/common.wgsl) and a binding IDX (texture_3d<u32>: the leaf
// slot per finest-level block).

// Cells per side of the domain at a level (1 along z in 2D).
fn n_at(level: u32) -> vec3<i32> {
  let n = i32(P.amr.y) * BS * (1 << level);
  return select(vec3<i32>(n), vec3<i32>(n, n, 1), is2d());
}

// A cell mapped into the domain: its index, the signs of its velocity and field components
// (reversed by walls as hydro/boundary.wgsl says), and for outflow-only faces the side
// (-1 / +1) to clamp the normal velocity toward.
struct GCell { g: vec3<i32>, flip: vec3<f32>, side: vec3<f32>, bflip: vec3<f32> }

fn map_cell(level: u32, g0: vec3<i32>) -> GCell {
  let n = n_at(level);
  var o = GCell(g0, vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
  for (var a = 0; a < 3; a++) {
    let i = g0[a];
    if (i >= 0 && i < n[a]) { continue; }
    let up = i >= n[a];
    let bc = select(P.bc_lo[a], P.bc_hi[a], up);
    if (bc == BC_PERIODIC) {
      // Exactly, however far out (the modulo only here: it is slow).
      o.g[a] = ((i % n[a]) + n[a]) % n[a];
      continue;
    }
    o.g[a] = bc_ghost_cell(i, n[a], bc);
    if (bc_wall(bc)) {
      o.flip *= bc_vsign(bc, a);
      o.bflip *= bc_bsign(bc, a);
    }
    o.side[a] = bc_side(bc, up);
  }
  return o;
}

// The block that holds global cell g (in the domain) at `level`: rel 0 a leaf at that level,
// 1 the same-level parent of finer leaves, -1 a leaf one level coarser.
struct Holder { slot: i32, rel: i32 }

fn holder(level: u32, g: vec3<i32>) -> Holder {
  let shift = P.amr.z - level;
  var fb = (g << vec3<u32>(shift)) / BS;
  if (is2d()) { fb.z = 0; }
  var s = i32(textureLoad(IDX, fb, 0).x);
  let ls = u32(bt(s, 0u));
  if (ls == level) { return Holder(s, 0); }
  if (ls < level) { return Holder(s, -1); }
  for (var k = ls; k > level; k--) { s = bt(s, 5u); }
  return Holder(s, 1);
}

// Local index in block `slot` of global cell (or face) index g at the block's level.
fn local_of(slot: i32, g: vec3<i32>) -> vec3<i32> {
  var l = g - blk_coord(slot) * BS;
  if (is2d()) { l.z = 0; }
  return l;
}

// Face a with face index g0 (along a: a face index, 0 to n; across: cell indices) at `level`,
// mapped into the domain: the face whose value stands for it, and that value's sign. Across
// the other axes it maps as cells do (the field normal to a is even under a mirror across
// another axis); beyond a face of the domain along a, as hydro/boundary.wgsl's bc_ghost_face.
struct FaceMap { g: vec3<i32>, sgn: f32 }

fn face_map(level: u32, a: i32, g0: vec3<i32>) -> FaceMap {
  let n = n_at(level);
  var g = g0;
  // map_cell inside the loop, as written: called once in the compiled code, not inlined at
  // every call site (it doubled the MHD block passes' step time when hoisted out).
  for (var b = 0; b < 3; b++) {
    if (b != a) { g[b] = map_cell(level, g0).g[b]; }
  }
  var sgn = 1.0;
  let i = g0[a];
  if (i < 0 || i > n[a]) {
    let f = bc_ghost_face(i, n[a], select(P.bc_lo[a], P.bc_hi[a], i > n[a]));
    g[a] = f.x;
    sgn = f32(f.y);
  }
  return FaceMap(g, sgn);
}
