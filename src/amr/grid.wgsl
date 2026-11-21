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

// A rectangular grid (src/domain.ts gridDims): the mirrored axes halved, whose root blocks stop
// half way along the tree's cube of coordinates (P.sym.w, a mask). Fractions of the computed box
// (what the display and the probe hand over) become fractions of that cube by cube_scale().
fn cube_scale() -> vec3<f32> {
  let m = P.sym.w;
  return select(vec3<f32>(1.0), vec3<f32>(0.5), vec3<bool>((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u));
}

fn bt(slot: i32, k: u32) -> i32 {
  return BT[u32(slot) * BT_STRIDE + k];
}
fn is2d() -> bool {
  return P.dims.z == 1u;
}
// n, as a loop bound the shader compiler cannot see through (P.dims.x is never 0): it then keeps
// the loop a loop. A large function called in a loop is compiled into it once; unrolled, or
// called at several places, it is inlined at every one (the MHD block passes took seconds to
// compile that way).
fn loop_count(n: i32) -> i32 {
  return select(n, 0, P.dims.x == 0u);
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

// A spherical run on blocks (hydro/common.wgsl): the geometry of a block's cells, from their
// global indices on its level. P.sph is the finest level's; a block of level L is
// P.amr.z - L levels coarser.
fn blk_sph_grid(slot: i32) -> SphGrid {
  return sph_level(i32(P.amr.z) - bt(slot, 0u));
}
fn blk_sph(slot: i32, l: vec3<i32>) -> SphCell {
  let g = blk_coord(slot) * BS + l;
  return sph_cell_g(g.x, g.y, blk_sph_grid(slot));
}
fn blk_sph_volume(slot: i32, l: vec3<i32>) -> f32 {
  let g = blk_coord(slot) * BS + l;
  return sph_volume_g(g.x, g.y, blk_sph_grid(slot), blk_dx(slot));
}
// The four children of the cell at global index pg of level pl: (the radial pair's volumes,
// the polar pair's dcos), for volume-weighted restriction and prolongation.
fn sph_children(pl: i32, pg: vec2<i32>) -> vec4<f32> {
  let q = sph_level(i32(P.amr.z) - pl - 1);
  let c = 2 * pg;
  return vec4<f32>(sph_v3(c.x, q), sph_v3(c.x + 1, q), sph_dcos(c.y, q), sph_dcos(c.y + 1, q));
}
// Child (i, j) of cell l of block `slot`: its weight in the parent's volume-weighted mean against
// the plain one (the four sum to 4): its volume over the mean of theirs. An axisymmetric run's is
// its radius's (cyl_child_weight); 1 otherwise.
fn child_weight(slot: i32, l: vec3<i32>, i: i32, j: i32) -> f32 {
  let pg = blk_coord(slot) * BS + l;
  if (spherical()) {
    let v = sph_children(bt(slot, 0u), pg.xy);
    return 4.0 * select(v.x, v.y, i == 1) * select(v.z, v.w, j == 1) / ((v.x + v.y) * (v.z + v.w));
  }
  return cyl_child_weight(pg.x, 2 * pg.x + i);
}
// A spherical run: a refined neighbor's fine face (cell `a` of block `src` below or above it)
// against the mean of the two that cover cell l's face along `axis`: its area over theirs, so
// that the coarse face's flux is their area-weighted mean. A radial face's area goes as its
// dcos(theta), a polar face's as the integral of r dr over it (r_c dr, or r_c^2 sinh D).
fn sph_fine_face(slot: i32, l: vec3<i32>, axis: i32, src: i32, a: vec3<i32>) -> f32 {
  let q = blk_sph_grid(src);
  let fg = blk_coord(src) * BS + a;
  let cg = 2 * (blk_coord(slot) * BS + l);
  var w0 = 0.0;
  var w1 = 0.0;
  var hi = false;
  if (axis == 0) {
    w0 = sph_dcos(cg.y, q);
    w1 = sph_dcos(cg.y + 1, q);
    hi = fg.y == cg.y + 1;
  } else {
    let r0 = sph_cell_g(cg.x, 0, q).r;
    let r1 = sph_cell_g(cg.x + 1, 0, q).r;
    w0 = select(r0, r0 * r0, q.d != 0.0);
    w1 = select(r1, r1 * r1, q.d != 0.0);
    hi = fg.x == cg.x + 1;
  }
  return 2.0 * select(w0, w1, hi) / (w0 + w1);
}

// For prolongation of the cell at global index pg of level pl: the shift of its children's
// offsets along r and theta that centers them on its centroid, so that their volume-weighted
// mean is the parent (in quarters of the parent: (v_+ - v_-) / (v_+ + v_-) / 4).
fn sph_centroid(pl: i32, pg: vec2<i32>) -> vec2<f32> {
  let v = sph_children(pl, pg);
  return 0.25 * vec2<f32>((v.y - v.x) / (v.y + v.x), (v.w - v.z) / (v.w + v.z));
}

// Atlas texel of a block's cell (0, 0, 0).
fn blk_origin(slot: i32) -> vec3<i32> {
  let a = i32(P.amr.x);
  if (is2d()) {
    // Layers of a x a slots, one cell deep, stacked along z.
    return vec3<i32>((slot % a) * PADW + NGH, ((slot / a) % a) * PADW + NGH, slot / (a * a));
  }
  return vec3<i32>(slot % a, (slot / a) % a, slot / (a * a)) * PADW + NGH;
}

// View only, one field held (amr/field.wgsl BAKED): texel of local cell l of block `slot` in the
// baked field's atlas, slots of BS cells laid out as the atlas's, without ghost cells.
fn bk_texel(slot: i32, l: vec3<i32>) -> vec3<i32> {
  let a = i32(P.amr.x);
  let s = vec3<i32>(slot % a, (slot / a) % a, slot / (a * a));
  if (is2d()) { return vec3<i32>(s.xy * BS + l.xy, s.z); }
  return s * BS + l;
}

// Atlas texel of local cell l of the block whose origin is o. l may be a face ghost.
fn atl(o: vec3<i32>, l: vec3<i32>) -> vec3<i32> {
  if (is2d()) { return vec3<i32>(o.xy + l.xy, o.z); }
  return o + l;
}

// Center of local cell l, relative to the box center.
fn blk_pos(slot: i32, l: vec3<i32>) -> vec3<f32> {
  var x = (vec3<f32>(blk_coord(slot) * BS + l) + 0.5) * blk_dx(slot) + P.org.xyz;
  if (is2d()) { x.z = 0.0; }
  return x;
}

// Where the value of a (possibly ghost) cell comes from.
//   kind 0: cell `l` of block `slot` at the same level: the cell itself, a neighbor's cell, or
//           its image in a physical boundary (then flip / side say how to treat the velocity)
//   kind 1: beyond a face whose neighbor is one level coarser; `slot` is that coarser leaf
//   kind 2: an edge or corner ghost, which nothing uses
struct Src { kind: i32, slot: i32, l: vec3<i32>, flip: vec3<f32>, side: vec3<f32>, bflip: vec3<f32>, vadd: vec3<f32> }

fn ghost_src(slot: i32, l: vec3<i32>) -> Src {
  var s = Src(0, slot, l, vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(0.0));
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
    // Same-level neighbor (periodic wrap included).
    s.slot = nb;
    s.l[axis] = l[axis] + select(BS, -BS, upper);
  } else if (nb == -1) {
    s.kind = 1;
    s.slot = bt(slot, 20u + u32(2 * axis) + u32(upper));
  } else {
    // A face of the domain, of kind -1 - nb (hydro/boundary.wgsl): mirror image or edge cell.
    let k = u32(-1 - nb);
    s.l[axis] = bc_ghost_cell(l[axis], BS, k);
    s.flip = bc_vsign(k, axis);
    s.bflip = bc_bsign_at(k, axis, upper);
    // Across the axis of an axisymmetric run v_phi and B_phi change sign, as in hydro/common.wgsl
    // bc_map. (Both poles of a spherical run are such axes.)
    if (bc_wall(k) && ((!upper && P.sym[axis] == 2u) || (spherical() && axis == 1))) {
      s.flip.z = -s.flip.z;
      s.bflip.z = -s.bflip.z;
    }
    s.side[axis] = bc_side(k, upper);
    s.vadd = 2.0 * bc_wall_velocity(k, axis, upper);
  }
  return s;
}

// Local index along an axis of the cell whose state stands for cell i of a block's row, where
// i may lie up to two cells outside the block: the mirror image beyond a reflecting face of
// the domain, the edge cell beyond an outflow face, and i itself otherwise (then it is a
// stored ghost of a neighboring block).
fn row_src(slot: i32, axis: i32, i: i32) -> i32 {
  if (i >= 0 && i < BS) { return i; }
  let upper = i >= BS;
  let nb = bt(slot, 6u + u32(2 * axis) + u32(upper));
  if (nb <= -2) { return bc_ghost_cell(i, BS, u32(-1 - nb)); }
  return i;
}

fn src_velocity(v: vec3<f32>, s: Src) -> vec3<f32> {
  return bc_velocity(v, Mapped(vec3<i32>(0), s.flip, s.side, s.bflip, s.vadd));
}
