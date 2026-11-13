// The list of slots a pass is dispatched over: LIST[0] is the count, the slots follow.
//
// Dispatch layout: blocks are laid end to end along x. A pass over cells with `span` texels
// per block side (BS for interior passes, 24 for padded ones, a multiple of the workgroup
// size) decodes the invocation into a block and a local cell. A 2D block has one layer of
// cells, so there the four layers of a workgroup in z take four consecutive blocks of the list
// (the dispatch has a quarter as many workgroups along x and one along z).
//
// A GPU runs at most 65535 workgroups along each axis of a dispatch, fewer than a fine 2D mesh
// has blocks (4096^2 cells is 65536 root blocks). So a 2D list longer than a slab is cut into
// slabs stacked along z: SLAB_QUADS groups of four blocks per slab in an interior or padded
// pass, FLAT_ROWS blocks per slab in a flat one. A 3D mesh holds few enough blocks not to need
// it (AmrSim caps it). The host (amr_sim.ts) and amr/bins_args.wgsl dispatch to match.
@group(0) @binding(31) var<storage, read> LIST: array<u32>;

const SLAB_QUADS = 16384u;
const FLAT_ROWS = 32768u;

struct Inv { ok: bool, slot: i32, l: vec3<i32> }

// Interior cells: l in [0, BS).
fn inv_interior(gid: vec3<u32>) -> Inv {
  var b = gid.x / u32(BS);
  var l = vec3<i32>(i32(gid.x % u32(BS)), i32(gid.y), i32(gid.z));
  if (is2d()) {
    b = 4u * (b + (gid.z / 4u) * SLAB_QUADS) + gid.z % 4u;
    l.z = 0;
  }
  if (b >= LIST[0] || l.y >= BS) { return Inv(false, 0, l); }
  return Inv(true, i32(LIST[1u + b]), l);
}

// Flat dispatch over cells and ghosts out to `reach` layers, for passes declared with
// @workgroup_size(FLAT_WG): workgroup (x, y) covers cells x * FLAT_WG onward, in x-fastest
// order, of block y of the list. The padded region is not a multiple of 8 cells across, so
// this leaves far fewer idle invocations than the 8 x 8 x 4 tiling of inv_padded().
const FLAT_WG = 64u;

// The block of the list a flat pass's workgroup works on: its row, in slabs along z.
fn flat_row(wid: vec3<u32>) -> u32 {
  return wid.y + wid.z * FLAT_ROWS;
}

fn inv_flat(wid: vec3<u32>, lix: u32, reach: i32) -> Inv {
  let w = BS + 2 * reach;
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  let l = vec3<i32>(i % w, (i / w) % w, i / (w * w)) - reach;
  let row = flat_row(wid);
  if (row >= LIST[0] || i >= n) { return Inv(false, 0, l); }
  return Inv(true, i32(LIST[1u + row]), select(l, vec3<i32>(l.xy, 0), is2d()));
}

// Cells and ghosts out to `reach` layers: l in [-reach, BS + reach).
fn inv_padded(gid: vec3<u32>, reach: i32) -> Inv {
  var b = gid.x / 24u;
  var l = vec3<i32>(i32(gid.x % 24u), i32(gid.y), i32(gid.z)) - reach;
  if (is2d()) {
    b = 4u * (b + (gid.z / 4u) * SLAB_QUADS) + gid.z % 4u;
    l.z = 0;
  }
  let top = BS + reach;
  if (b >= LIST[0] || l.x >= top || l.y >= top || l.z >= top) { return Inv(false, 0, l); }
  return Inv(true, i32(LIST[1u + b]), l);
}
