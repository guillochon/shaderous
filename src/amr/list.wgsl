// The list of slots a pass is dispatched over: LIST[0] is the count, the slots follow.
//
// Dispatch layout: blocks are laid end to end along x. A pass over cells with `span` texels
// per block side (BS for interior passes, 24 for padded ones, a multiple of the workgroup
// size) decodes the invocation into a block and a local cell.
@group(0) @binding(31) var<storage, read> LIST: array<u32>;

struct Inv { ok: bool, slot: i32, l: vec3<i32> }

// Interior cells: l in [0, BS).
fn inv_interior(gid: vec3<u32>) -> Inv {
  let b = gid.x / u32(BS);
  let l = vec3<i32>(i32(gid.x % u32(BS)), i32(gid.y), i32(gid.z));
  let depth = select(BS, 1, is2d());
  if (b >= LIST[0] || l.y >= BS || l.z >= depth) { return Inv(false, 0, l); }
  return Inv(true, i32(LIST[1u + b]), l);
}

// Cells and ghosts out to `reach` layers: l in [-reach, BS + reach).
fn inv_padded(gid: vec3<u32>, reach: i32) -> Inv {
  let b = gid.x / 24u;
  var l = vec3<i32>(i32(gid.x % 24u), i32(gid.y), i32(gid.z)) - reach;
  if (is2d()) { l.z = i32(gid.z); }
  let top = BS + reach;
  let ztop = select(top, 1, is2d());
  if (b >= LIST[0] || l.x >= top || l.y >= top || l.z >= ztop) { return Inv(false, 0, l); }
  return Inv(true, i32(LIST[1u + b]), l);
}
