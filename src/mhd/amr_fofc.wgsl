// First-order flux correction after the update on the block grid: mhd/fofc.wgsl's (the uniform
// grid's) for blocks. A cell that stage 2's update leaves with less than FOFC_KEEP of its density,
// or with no internal energy, is marked; then (mhd/amr_mhd.ts encode, encodeSubstep):
//
//   scan   list the step's leaves that have a marked cell or a face neighbor with one
//          (mhd/amr_fofc_scan.wgsl);
//   fluxes stage 2's flux pass again over that list (mhd/amr_flux.wgsl, REDO): a face of a marked
//          cell takes the first-order HLLE flux of the state at the start in its mass, momentum
//          and energy, into the scratch and, on the block's own faces, the register; the block's
//          other faces on its surface take back what the register holds of them, the very flux
//          the neighbor beside it used; inner faces are computed again;
//   update stage 2's update again over the list (mhd/amr_update.wgsl, REDO): the cells alone, from
//          those fluxes (a coarse block beside a refined one takes the fine register as always);
//          a cell beside a corrected face keeps its advected internal energy.
//
// The faces of the field, their constrained transport and the edge fields are not touched, so
// div B is as it was. A block and its neighbor both list each other, and both correct their
// shared face from the same inputs, so mass, momentum and energy stay conserved. With time bins
// only the leaves that start a step are listed; a slower neighbor at its midpoint takes the
// corrected register in its deposit, which comes after.
//
// A mark is the step's stamp in the fourth channel, otherwise zero, of the new face texture: at
// the cell's own texel, and at the block's corner padding texel, which nothing else uses, for the
// block. The step counter moves on every substep, so a mark never outlives its substep.
//
// Requires P, clock, amr/grid.wgsl, mhd/amr_lookup.wgsl (IDX) and a binding NF (the step's new
// faces, sampled).

const FOFC_KEEP = 0.125;

fn fofc_stamp() -> f32 {
  return clock.steps + 1.0;
}
// The atlas texel of a block that holds its mark: a corner of its ghost padding.
fn fofc_corner(slot: i32) -> vec3<i32> {
  return atl(blk_origin(slot), vec3<i32>(-NGH, -NGH, select(-NGH, 0, is2d())));
}
fn fofc_block_marked(slot: i32) -> bool {
  return textureLoad(NF, fofc_corner(slot), 0).w == fofc_stamp();
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
// Global cell g (in the domain) of `level` is marked: in the leaf of that level, or the coarser
// one around it (a finer neighbor's cells mark its own faces, not this level's).
fn fofc_cell_marked(level: u32, g: vec3<i32>) -> bool {
  let h = holder(level, g);
  if (h.rel > 0) { return false; }
  var gc = g;
  if (h.rel < 0) { gc = g >> vec3<u32>(1u); }
  if (is2d()) { gc.z = 0; }
  return textureLoad(NF, atl(blk_origin(h.slot), local_of(h.slot, gc)), 0).w == fofc_stamp();
}
// Local cell l (inside the block or a face ghost) of block `slot` is marked.
fn fofc_local_marked(slot: i32, l: vec3<i32>) -> bool {
  let top = vec3<i32>(BS, BS, select(BS, 1, is2d()));
  if (all(l >= vec3<i32>(0)) && all(l < top)) {
    return textureLoad(NF, atl(blk_origin(slot), l), 0).w == fofc_stamp();
  }
  let level = u32(bt(slot, 0u));
  let g = blk_coord(slot) * BS + l;
  // A ghost beyond a wall stands for this block's own cell (mirrored), whose mark is its own.
  return fofc_cell_marked(level, map_cell(level, g).g);
}
