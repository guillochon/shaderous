// Block grid: the block table and the mapping from (block, cell) to atlas texels.
// Requires P. Layouts are documented in src/amr/tree.ts.
//
// Each block stores PADW^3 texels in the atlas: BS^3 cells plus NGH ghost layers on each face
// (2D grids have a single layer in z and no z ghosts). Only face ghosts are ever filled or
// read; the passes are written so that edges and corners are not needed.
const BS = 16;
const NGH = 2;
const PADW = 20;
const BT_STRIDE = 32u;

@group(0) @binding(30) var<storage, read> BT: array<i32>;

fn bt(slot: i32, k: u32) -> i32 {
  return BT[u32(slot) * BT_STRIDE + k];
}
fn is2d() -> bool {
  return P.dims.z == 1u;
}
fn blk_leaf(slot: i32) -> bool {
  return (bt(slot, 4u) & 3) == 3;
}
fn blk_coord(slot: i32) -> vec3<i32> {
  return vec3<i32>(bt(slot, 1u), bt(slot, 2u), bt(slot, 3u));
}
fn blk_dx(slot: i32) -> f32 {
  return P.dx / f32(1u << u32(bt(slot, 0u)));
}

// Atlas texel of a block's cell (0, 0, 0).
fn blk_origin(slot: i32) -> vec3<i32> {
  let a = i32(P.amr.x);
  if (is2d()) {
    return vec3<i32>((slot % a) * PADW + NGH, (slot / a) * PADW + NGH, 0);
  }
  return vec3<i32>(slot % a, (slot / a) % a, slot / (a * a)) * PADW + NGH;
}

// Atlas texel of local cell l of the block whose origin is o. l may be a face ghost.
fn atl(o: vec3<i32>, l: vec3<i32>) -> vec3<i32> {
  if (is2d()) { return vec3<i32>(o.xy + l.xy, 0); }
  return o + l;
}

// Centre of local cell l, relative to the box centre.
fn blk_pos(slot: i32, l: vec3<i32>) -> vec3<f32> {
  var x = (vec3<f32>(blk_coord(slot) * BS + l) + 0.5) * blk_dx(slot) - 0.5 * P.box;
  if (is2d()) { x.z = 0.0; }
  return x;
}

// Where the value of a (possibly ghost) cell comes from.
//   kind 0: cell `l` of block `slot` at the same level: the cell itself, a neighbour's cell, or
//           its image in a physical boundary (then flip / side say how to treat the velocity)
//   kind 1: beyond a face whose neighbour is one level coarser; `slot` is that coarser leaf
//   kind 2: an edge or corner ghost, which nothing uses
struct Src { kind: i32, slot: i32, l: vec3<i32>, flip: vec3<f32>, side: vec3<f32> }

fn ghost_src(slot: i32, l: vec3<i32>) -> Src {
  var s = Src(0, slot, l, vec3<f32>(1.0), vec3<f32>(0.0));
  let lo = l < vec3<i32>(0);
  let hi = l >= vec3<i32>(BS);
  let out = vec3<i32>(lo) + vec3<i32>(hi);
  let count = out.x + out.y + out.z;
  if (count == 0) { return s; }
  if (count > 1) { s.kind = 2; return s; }
  var axis = 0;
  if (out.y != 0) { axis = 1; }
  if (out.z != 0) { axis = 2; }
  let upper = l[axis] >= BS;
  let nb = bt(slot, 6u + u32(2 * axis) + u32(upper));
  if (nb >= 0) {
    // Same-level neighbour (periodic wrap included).
    s.slot = nb;
    s.l[axis] = l[axis] + select(BS, -BS, upper);
  } else if (nb == -1) {
    s.kind = 1;
    s.slot = bt(slot, 20u + u32(2 * axis) + u32(upper));
  } else if (nb == -3) {
    // Reflecting: mirror image, normal velocity reversed.
    s.l[axis] = select(-1 - l[axis], 2 * BS - 1 - l[axis], upper);
    s.flip[axis] = -1.0;
  } else {
    // Outflow: copy the boundary cell. Outflow-only also clamps the ghost's normal velocity.
    s.l[axis] = select(0, BS - 1, upper);
    if (nb == -4) { s.side[axis] = select(-1.0, 1.0, upper); }
  }
  return s;
}

// Local index along an axis of the cell whose state stands for cell i of a block's row, where
// i may lie up to two cells outside the block: the mirror image beyond a reflecting face of
// the domain, the edge cell beyond an outflow face, and i itself otherwise (then it is a
// stored ghost of a neighbouring block).
fn row_src(slot: i32, axis: i32, i: i32) -> i32 {
  if (i >= 0 && i < BS) { return i; }
  let upper = i >= BS;
  let nb = bt(slot, 6u + u32(2 * axis) + u32(upper));
  if (nb == -3) { return select(-1 - i, 2 * BS - 1 - i, upper); }
  if (nb == -2 || nb == -4) { return select(0, BS - 1, upper); }
  return i;
}

fn src_velocity(v: vec3<f32>, s: Src) -> vec3<f32> {
  return bc_velocity(v, Mapped(vec3<i32>(0), s.flip, s.side));
}
