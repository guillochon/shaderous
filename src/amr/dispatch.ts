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
