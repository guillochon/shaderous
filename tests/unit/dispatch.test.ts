import { describe, expect, it } from 'vitest';
import { interior3d, padded3d, SLAB_BLOCKS } from '../../src/amr/dispatch';
import { shaderFiles } from '../../src/gpu/shaders';

describe('3D dispatches over blocks in slabs (amr/dispatch.ts)', () => {
  it('dispatches a list within one slab as before slabs existed', () => {
    expect(padded3d(1000)).toEqual([3000, 3, 5]);
    expect(interior3d(1000)).toEqual([2000, 2, 4]);
    expect(padded3d(SLAB_BLOCKS)).toEqual([SLAB_BLOCKS * 3, 3, 5]);
  });

  it('stacks longer lists in slabs along z, within the 65535 workgroups an axis may have', () => {
    expect(padded3d(SLAB_BLOCKS + 1)).toEqual([SLAB_BLOCKS * 3, 3, 10]);
    expect(interior3d(3 * SLAB_BLOCKS)).toEqual([SLAB_BLOCKS * 2, 2, 12]);
    for (const n of [35 ** 3, 102 ** 3]) for (const g of [padded3d(n), interior3d(n)]) expect(Math.max(...g)).toBeLessThanOrEqual(65535);
  });

  it('uses the slab size the shaders decode with', () => {
    for (const f of ['amr/list.wgsl', 'amr/bins_args.wgsl']) {
      const m = shaderFiles[f]?.match(/const SLAB_BLOCKS = (\d+)u;/);
      expect(Number(m?.[1]), f).toBe(SLAB_BLOCKS);
    }
  });
});
