// First-order flux correction after the update, on the uniform grid (after Athena++'s). Stage 2 of
// the van Leer step can empty a cell: next to near-vacuum gas (a thin corona over a magnetized
// disk) the second-order fluxes of the half-step state carry out more mass than the cell held,
// though no cell's density moved enough over the half step for the test before the fluxes
// (mhd/face_flux.wgsl, fofc_suspect) to catch it. So stage 2 also checks every cell after its
// update: one left with less than FOFC_KEEP of its density, or with no internal energy, is listed.
// After each slab's update (mhd/mhd.ts encode):
//
//   fix   over the list (mhd/flux.wgsl, fix_main): each face of a listed cell
//         that this slab's flux buffers hold takes the first-order HLLE flux of the state at the
//         start of the step, in its mass, momentum and energy entries. The third entry, the
//         field's fluxes and the upwind weight the edge fields were made from, is left alone: the
//         field, its constrained transport and div B are those of the second-order step.
//   redo  the update again (mhd/update.wgsl, redo_main), of each listed cell and its face
//         neighbors, from the fluxes as they now stand, where the cell has been updated and its
//         faces and field are at hand: in this slab but its top layer (whose upper faces the next
//         slab writes), or the layer below it.
//
// The list runs over the whole step, from slab to slab: a later slab fixes the faces it shares
// with cells listed before it, and redoes the cells of both sides. Every cell is checked on the
// fluxes of the second-order step alone, and every redo sees all the corrections made around it,
// so the result does not depend on the slabs. Both cells of a corrected face read the same
// corrected flux, so mass, momentum and energy are conserved as before.
//
// The fix and the redo run FOFC_GROUPS workgroups that step through the list, so in a step that
// lists nothing they end at once. (An indirect dispatch sized by the list would need the counts in
// a buffer of their own, and the update pass has no storage buffer to spare under WebGPU's default
// limit of 8.)
//
// Requires P, mhd/common.wgsl, and a binding FOFC (Fofc).

const FOFC_KEEP = 0.125;
const FOFC_GROUPS = 16u;
const FOFC_THREADS = 4096u; // FOFC_GROUPS workgroups of 256

// The listed cells, as indices into the grid; the flux pass of stage 2 resets the count at the
// first slab.
struct Fofc {
  count: atomic<u32>,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  cells: array<u32>,
}

fn fofc_cell(c: vec3<i32>) -> u32 {
  let n = mhd_dims();
  return u32((c.z * n.y + c.y) * n.x + c.x);
}
// The listed cells, at most what the list holds.
fn fofc_count() -> u32 {
  return min(atomicLoad(&FOFC.count), arrayLength(&FOFC.cells));
}
fn fofc_listed(i: u32) -> vec3<i32> {
  let n = vec3<u32>(mhd_dims());
  let c = FOFC.cells[i];
  return vec3<i32>(vec3<u32>(c % n.x, (c / n.x) % n.y, c / (n.x * n.y)));
}
// List cell t (the update, stage 2), if the list has room; past that it stays as the step left it.
fn fofc_add(t: vec3<i32>) {
  let i = atomicAdd(&FOFC.count, 1u);
  if (i < arrayLength(&FOFC.cells)) { FOFC.cells[i] = fofc_cell(t); }
}
