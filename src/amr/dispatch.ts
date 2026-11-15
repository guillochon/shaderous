/**
 * Workgroup counts of the 3D passes over a list of blocks (amr/list.wgsl, inv_padded and
 * inv_interior). A pass runs three (padded) or two (interior) workgroups per block along x, and
 * a dispatch may have at most 65535 along an axis, so a list longer than SLAB_BLOCKS is cut into
 * slabs stacked along z, each a block's depth of workgroups: five (20 cells) in a padded pass,
 * four (16) in an interior one. amr/bins_args.wgsl sizes its passes the same way.
 */
export const SLAB_BLOCKS = 16384; // must match amr/list.wgsl and amr/bins_args.wgsl

const slabs = (count: number) => Math.ceil(count / SLAB_BLOCKS);

/** Workgroups of a padded pass over `count` blocks of a 3D mesh: 24 x 24 x 20 cells each. */
export const padded3d = (count: number): [number, number, number] => [Math.min(count, SLAB_BLOCKS) * 3, 3, 5 * slabs(count)];

/** Workgroups of an interior pass over `count` blocks of a 3D mesh: 16 x 16 x 16 cells each. */
export const interior3d = (count: number): [number, number, number] => [Math.min(count, SLAB_BLOCKS) * 2, 2, 4 * slabs(count)];

/**
 * Self-gravity's multigrid (amr/gravity.ts) at a sublevel of `bsize` cells per block side: every
 * cell (span max(bsize, 8) along x per block), or one color of a red-black sweep (half the cells
 * along x, several small blocks to a workgroup). Slabs as above, a block taking 4 ceil(bsize / 4)
 * invocations along z (amr/gravity/common.wgsl, g_zspan). amr/gravity/gate.wgsl and
 * amr/bins_args.wgsl size the same dispatches on the GPU.
 */
export const gravityCells = (count: number, bsize: number): [number, number, number] =>
  [Math.min(count, SLAB_BLOCKS) * (Math.max(bsize, 8) / 8), Math.ceil(bsize / 8), Math.ceil(bsize / 4) * slabs(count)];
export const gravityColor = (count: number, bsize: number): [number, number, number] =>
  [Math.ceil((Math.min(count, SLAB_BLOCKS) * Math.max(bsize / 2, 1)) / 8), Math.ceil(bsize / 8), Math.ceil(bsize / 4) * slabs(count)];

/** Rows of a pass with a workgroup row per block (amr/list.wgsl, flat_row): FLAT_ROWS rows to a slab along z. */
export const FLAT_ROWS = 32768; // must match amr/list.wgsl
export const rows = (x: number, count: number): [number, number, number] => [x, Math.min(count, FLAT_ROWS), Math.ceil(count / FLAT_ROWS)];
