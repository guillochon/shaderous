// Self-gravity on the block grid: shared definitions for the per-level multigrid.
//
// The potential lives on every block, parents included. Each tree level is solved in turn,
// coarsest first, on the set of blocks at that level:
//   - between same-level blocks the stencil reads straight across
//   - where the region beyond a face is coarser, the ghost cell holds a value interpolated
//     from the coarser level's finished solution (a Dirichlet condition; "one-way" coupling)
//   - at the domain boundary: periodic neighbors, or the multipole expansion (isolated)
// Parents see the restricted density of their children, so each level is a complete solve.
//
// Each level is accelerated by multigrid within the blocks: sublevel k has BS >> k cells per
// block side (16, 8, 4, 2, 1), with the same block connectivity. Sublevel 0 uses the hydro
// atlas layout (two ghost layers); the others have one.
//
// A 2D grid (an axisymmetric (R, z) run) has blocks one cell deep: each sublevel's atlas is
// laid out as the hydro one is in 2D (amr/grid.wgsl, blk_origin), layers of slots one texel
// deep, and a dispatch's four invocations along z take four blocks of the list, as
// amr/list.wgsl's 2D passes do. The stencil then has four neighbors, the radial ones weighed by
// R_-/R_c and R_+/R_c (amr/gravity/stencil.wgsl), and transfers between sublevels are by ring
// volume (amr/gravity/restrict.wgsl).
//
// Requires P and amr/grid.wgsl.
struct GLvl {
  bsize: i32,     // cells per block side at this sublevel
  pad: i32,       // ghost layers stored around each block
  stride: i32,    // texels per block side in this sublevel's atlas
  k: u32,         // sublevel: 0 is the full-resolution grid of the tree level
  color: u32,     // red-black parity handled by this dispatch
  obsize: i32,    // the same three for the other sublevel a transfer pass touches
  opad: i32,
  ostride: i32,
}
@group(0) @binding(1) var<uniform> GL: GLvl;

// Atlas texel of a block's cell (0, 0, 0) at this sublevel, and at the other one.
fn g_origin(slot: i32) -> vec3<i32> {
  let a = i32(P.amr.x);
  if (is2d()) { return vec3<i32>((slot % a) * GL.stride + GL.pad, ((slot / a) % a) * GL.stride + GL.pad, slot / (a * a)); }
  return vec3<i32>(slot % a, (slot / a) % a, slot / (a * a)) * GL.stride + GL.pad;
}
fn g_origin_other(slot: i32) -> vec3<i32> {
  let a = i32(P.amr.x);
  if (is2d()) { return vec3<i32>((slot % a) * GL.ostride + GL.opad, ((slot / a) % a) * GL.ostride + GL.opad, slot / (a * a)); }
  return vec3<i32>(slot % a, (slot / a) % a, slot / (a * a)) * GL.ostride + GL.opad;
}

// Cell size at this sublevel.
fn g_h(slot: i32) -> f32 {
  return blk_dx(slot) * f32(BS / GL.bsize);
}

// Decode an invocation into a block of the dispatched list and a cell of it.
struct GInv { ok: bool, slot: i32, l: vec3<i32> }

// Invocations a block takes along z at a sublevel of `bsize` cells: whole workgroups of four.
// A list longer than SLAB_BLOCKS is dispatched in slabs of that many blocks stacked along z
// (amr/list.wgsl, AmrGravity.cells() and color()).
fn g_zspan(bsize: i32) -> u32 {
  return 4u * u32((bsize + 3) / 4);
}

// 2D: a block of `bsize` cells per side at invocation gid, four blocks to a quad along z, the
// quads laid end to end along x in slabs of SLAB_QUADS (amr/dispatch.ts, gravityCells2d).
fn g_inv2d(gid: vec3<u32>, bsize: i32) -> GInv {
  let span = u32(max(bsize, 8));
  let b = 4u * (gid.x / span + (gid.z / 4u) * SLAB_QUADS) + gid.z % 4u;
  let l = vec3<i32>(i32(gid.x % span), i32(gid.y), 0);
  if (b >= LIST[0] || l.x >= bsize || l.y >= bsize) { return GInv(false, 0, l); }
  return GInv(true, i32(LIST[1u + b]), l);
}

fn g_inv(gid: vec3<u32>) -> GInv {
  if (is2d()) { return g_inv2d(gid, GL.bsize); }
  let span = u32(max(GL.bsize, 8));
  let zs = g_zspan(GL.bsize);
  let b = gid.x / span + (gid.z / zs) * SLAB_BLOCKS;
  let l = vec3<i32>(i32(gid.x % span), i32(gid.y), i32(gid.z % zs));
  if (b >= LIST[0] || any(l >= vec3<i32>(GL.bsize))) { return GInv(false, 0, l); }
  return GInv(true, i32(LIST[1u + b]), l);
}

// Follow same-level neighbors to the block that holds cell l, which may lie outside `slot`
// along any axes. ok is false if that walks off the level (a coarser region or the domain edge).
// A cell beyond a plane of mirror symmetry is its image, the potential being even across it.
struct GRes { ok: bool, slot: i32, l: vec3<i32> }

fn g_resolve(slot0: i32, l0: vec3<i32>, bsize: i32) -> GRes {
  var slot = slot0;
  var l = l0;
  let bc = blk_coord(slot0);
  for (var axis = 0; axis < 3; axis++) {
    if (l[axis] < 0 && bc[axis] == 0 && sym_lo(axis)) { l[axis] = -1 - l[axis]; }
  }
  for (var axis = 0; axis < 3; axis++) {
    if (l[axis] < 0) {
      let nb = bt(slot, 6u + u32(2 * axis));
      if (nb < 0) { return GRes(false, slot, l); }
      slot = nb;
      l[axis] += bsize;
    } else if (l[axis] >= bsize) {
      let nb = bt(slot, 7u + u32(2 * axis));
      if (nb < 0) { return GRes(false, slot, l); }
      slot = nb;
      l[axis] -= bsize;
    }
  }
  return GRes(true, slot, l);
}
