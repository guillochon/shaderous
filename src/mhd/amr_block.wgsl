// MHD on the block grid: what the flux, edge and update passes share. Each works on a chunk of
// the leaf list at a time (CHUNK: the first entry and how many), one workgroup row per block,
// over a box of texels of the block; the assembled state (amr_assemble.wgsl) is read from the
// block's own padded storage.
//
// The flux scratch holds, per block of the chunk and per axis, an entry for every cell t of
// [-1, BS] along each axis (the single layer along z in 2D), three vec4 per entry (as on the
// uniform grid). The flux register keeps, per slot, the fluxes through the block's six faces,
// two vec4 per face (mass and momentum; energy, internal energy, scalar, face velocity): what
// a coarser neighbor needs for its flux matching after every block's flux pass has run.
// Requires P, amr/grid.wgsl, amr/list.wgsl and a binding CHUNK.

struct Chunk { first: u32, count: u32, unused0: u32, unused1: u32 }

// Side of the flux box, and its cells per block (18 or 18^2 x 1 in 2D).
const FBOX = BS + 2;
fn fbox_cells() -> u32 {
  return select(u32(FBOX * FBOX * FBOX), u32(FBOX * FBOX), is2d());
}
fn naxes() -> i32 {
  return select(3, 2, is2d());
}
// Scratch entry (in vec4) of axis a's face below cell t of the chunk's block `row`.
fn flux_entry(row: u32, a: i32, t: vec3<i32>) -> u32 {
  let s = t + 1;
  let z = select(s.z, 0, is2d());
  let i = u32((z * FBOX + s.y) * FBOX + s.x);
  return 3u * ((row * u32(naxes()) + u32(a)) * fbox_cells() + i);
}
// Register entry (in vec4) of the face along a at the block's lower (hi false) or upper side,
// at cell (t1, t2) across it.
fn reg_entry(slot: i32, a: i32, hi: bool, t1: i32, t2: i32) -> u32 {
  let plane = u32(2 * a) + select(0u, 1u, hi);
  return 2u * ((u32(slot) * 6u + plane) * u32(BS * BS) + u32(t1 + BS * t2));
}

// A chunk's block for workgroup row `row`, and the local texel for invocation i of a box
// starting at `lo`, `w` texels per side (one layer along z in 2D).
struct BoxInv { ok: bool, slot: i32, row: u32, l: vec3<i32> }

fn inv_box(wid: vec3<u32>, lix: u32, lo: i32, w: i32) -> BoxInv {
  let row = flat_row(wid);
  let n = select(w * w * w, w * w, is2d());
  let i = i32(wid.x * FLAT_WG + lix);
  var l = vec3<i32>(i % w, (i / w) % w, i / (w * w)) + lo;
  if (is2d()) { l.z = 0; }
  if (row >= CHUNK.count || i >= n) { return BoxInv(false, 0, row, l); }
  return BoxInv(true, i32(LIST[1u + CHUNK.first + row]), row, l);
}

// The assembled state of texel l of a block whose origin is o.
fn mq_block(o: vec3<i32>, l: vec3<i32>) -> MQ {
  let c = atl(o, l);
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  return MQ(w1, vec4<f32>(w2.x, w2.z, w2.y, w2.w), textureLoad(W3, c, 0).xyz);
}
