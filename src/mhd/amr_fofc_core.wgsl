// The first-order flux correction after the update on the block grid (mhd/amr_fofc.wgsl): what
// its passes share that binds nothing of its own. Requires P, clock, amr/grid.wgsl and
// mhd/amr_lookup.wgsl (IDX).

const FOFC_KEEP = 0.125;
fn fofc_stamp() -> f32 {
  return clock.steps + 1.0;
}
// The atlas texel of a block that holds its mark: a corner of its ghost padding.
fn fofc_corner(slot: i32) -> vec3<i32> {
  return atl(blk_origin(slot), vec3<i32>(-NGH, -NGH, select(-NGH, 0, is2d())));
}
// Beyond a wall or an open face of the domain there is no cell of another block.
fn fofc_beyond_wall(level: u32, g: vec3<i32>) -> bool {
  let n = n_at(level);
  for (var a = 0; a < select(3, 2, is2d()); a++) {
    if (g[a] < 0 && P.bc_lo[a] != BC_PERIODIC) { return true; }
    if (g[a] >= n[a] && P.bc_hi[a] != BC_PERIODIC) { return true; }
  }
  return false;
}
// The leaf at global cell g (in the domain) of `level`: that level's, the coarser one around it,
// or the finer one at its lower corner; -1 if none.
fn fofc_leaf_at(level: u32, g: vec3<i32>) -> i32 {
  let h = holder(level, g);
  if (h.rel <= 0) { return h.slot; }
  var gf = 2 * g;
  if (is2d()) { gf.z = 0; }
  let f = holder(level + 1u, gf);
  return select(-1, f.slot, f.rel == 0);
}
