import { describe, expect, it } from 'vitest';
import { burnInterior, FLAT_ROWS, gravityCells, gravityColor, interior3d, padded3d, rows, SLAB_BLOCKS, SLAB_QUADS } from '../../src/amr/dispatch';
import { shaderFiles } from '../../src/gpu/shaders';

describe('3D dispatches over blocks in slabs (amr/dispatch.ts)', () => {
  it('dispatches a list within one slab as before slabs existed', () => {
    expect(padded3d(1000)).toEqual([3000, 3, 5]);
    expect(interior3d(1000)).toEqual([2000, 2, 4]);
    expect(padded3d(SLAB_BLOCKS)).toEqual([SLAB_BLOCKS * 3, 3, 5]);
    // The multigrid's sizes before slabs: cells count * max(b, 8) / 8 x ceil(b / 8) x ceil(b / 4),
    // one color ceil(count * max(b / 2, 1) / 8) x the same; face ghosts 48 x count.
    for (const b of [16, 8, 4, 2, 1]) {
      expect(gravityCells(1000, b)).toEqual([1000 * (Math.max(b, 8) / 8), Math.ceil(b / 8), Math.ceil(b / 4)]);
      expect(gravityColor(1000, b)).toEqual([Math.ceil((1000 * Math.max(b / 2, 1)) / 8), Math.ceil(b / 8), Math.ceil(b / 4)]);
    }
    expect(rows(48, 1000)).toEqual([48, 1000, 1]);
  });

  it('stacks longer lists in slabs along z, within the 65535 workgroups an axis may have', () => {
    expect(padded3d(SLAB_BLOCKS + 1)).toEqual([SLAB_BLOCKS * 3, 3, 10]);
    expect(interior3d(3 * SLAB_BLOCKS)).toEqual([SLAB_BLOCKS * 2, 2, 12]);
    expect(gravityCells(SLAB_BLOCKS + 1, 16)).toEqual([SLAB_BLOCKS * 2, 2, 8]);
    expect(gravityColor(2 * SLAB_BLOCKS, 8)).toEqual([SLAB_BLOCKS / 2, 1, 4]);
    expect(rows(48, FLAT_ROWS + 1)).toEqual([48, FLAT_ROWS, 2]);
    for (const n of [35 ** 3, 60 ** 3, 102 ** 3]) {
      const all = [padded3d(n), interior3d(n), rows(48, n), ...[16, 8, 4, 2, 1].flatMap((b) => [gravityCells(n, b), gravityColor(n, b)])];
      for (const g of all) expect(Math.max(...g)).toBeLessThanOrEqual(65535);
    }
  });

  it("covers the burn's interior cells as inv_interior decodes them, within the limit", () => {
    // A full slab at four workgroups a block along x was 65536: past 16383 leaves every burn pass
    // of a run with a network was invalid, and the run hung waiting for it.
    for (const f of ['hydro/burn_net.wgsl', 'hydro/burn_watch.wgsl']) {
      const wg = shaderFiles[f]?.match(/@workgroup_size\((\d+), (\d+), (\d+)\)\s*fn main/)?.slice(1).map(Number);
      expect(wg, f).toBeDefined();
      const [wx, wy, wz] = wg!;
      for (const n of [1, 1000, SLAB_BLOCKS - 1, SLAB_BLOCKS, SLAB_BLOCKS + 1, 102 ** 3]) {
        const g = burnInterior(3, n);
        expect(Math.max(...g)).toBeLessThanOrEqual(65535);
        const slabs = Math.ceil(n / SLAB_BLOCKS);
        expect([g[0] * wx, g[1] * wy, g[2] * wz]).toEqual([Math.min(n, SLAB_BLOCKS) * 16, 16, 16 * slabs]);
      }
      for (const n of [1, 4 * SLAB_QUADS, 4 * SLAB_QUADS + 1, 1 << 20]) {
        const g = burnInterior(2, n);
        expect(Math.max(...g)).toBeLessThanOrEqual(65535);
        const q = Math.ceil(n / 4);
        // four blocks to a quad, one to each of the workgroup's z layers in a slab
        expect([g[0] * wx, g[1] * wy, g[2] * wz]).toEqual([Math.min(q, SLAB_QUADS) * 16, 16, 4 * Math.ceil(q / SLAB_QUADS)]);
      }
    }
  });

  it('uses the slab sizes the shaders decode with', () => {
    for (const f of ['amr/list.wgsl', 'amr/bins_args.wgsl']) {
      expect(Number(shaderFiles[f]?.match(/const SLAB_BLOCKS = (\d+)u;/)?.[1]), f).toBe(SLAB_BLOCKS);
      expect(Number(shaderFiles[f]?.match(/const FLAT_ROWS = (\d+)u;/)?.[1]), f).toBe(FLAT_ROWS);
    }
  });
});
