// The first-order flux correction after the update on the block grid (mhd/amr_fofc.wgsl): adding a
// leaf to the list of the step, once (SEEN holds the step's stamp per slot), and sizing the
// dispatches over the list chunk by chunk as it grows (ARGS, as mhd/amr_chunk_args.wgsl lays them
// out: rows here, the rest set when the buffer is made); and adding a leaf's face neighbors that
// start a step (with time bins, a neighbor at its midpoint takes the corrected register in its
// deposit). Requires mhd/amr_fofc_core.wgsl, <bins>, an override PER_CHUNK and bindings RL
// (count, then slots), ARGS and SEEN, all array<atomic<u32>>.

fn list_leaf(slot: i32) {
  let stamp = u32(fofc_stamp());
  if (atomicExchange(&SEEN[u32(slot)], stamp) == stamp) { return; }
  let at = atomicAdd(&RL[0], 1u);
  atomicStore(&RL[1u + at], u32(slot));
  list_rows(at);
}
// The chunk of list entry `at` dispatches at least the rows up to it.
fn list_rows(at: u32) {
  let c = at / PER_CHUNK;
  let rows = at % PER_CHUNK + 1u;
  atomicMax(&ARGS[6u * c + 1u], rows);
  atomicMax(&ARGS[6u * c + 4u], rows);
}
// The leaves at points one cell beyond each face of `slot`, at the middles of its quarters (halves
// in 2D), so that a neighbor one level finer is found at each of its blocks.
fn list_neighbors(slot: i32) {
  let level = u32(bt(slot, 0u));
  let base = blk_coord(slot) * BS;
  for (var k = 0; k < loop_count(6 * 4); k++) {
    let a = k / 8;
    if (a >= select(3, 2, is2d())) { break; }
    let up = ((k / 4) & 1) == 1;
    let q = k & 3;
    let b = (a + 1) % 3;
    let c = (a + 2) % 3;
    var l = vec3<i32>(0);
    l[a] = select(-1, BS, up);
    l[b] = select(BS / 4, 3 * BS / 4, (q & 1) == 1);
    l[c] = select(BS / 4, 3 * BS / 4, (q & 2) == 2);
    if (is2d()) {
      if (c == 2 && q >= 2) { continue; }
      if (b == 2 && (q & 1) == 1) { continue; }
      l.z = 0;
    }
    let g = base + l;
    if (fofc_beyond_wall(level, g)) { continue; }
    let s = fofc_leaf_at(level, map_cell(level, g).g);
    if (s >= 0 && s != slot && blk_active(s)) { list_leaf(s); }
  }
}
