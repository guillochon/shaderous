import { describe, expect, it } from 'vitest';
import { speciesAtlasSize, speciesGroupOffset } from '../../src/species/species';

describe('species atlas layout', () => {
  it('stacks groups along z while they fit, as it always did', () => {
    for (const stride of [1, 64, 256, 512, 680]) {
      for (let g = 0; (g + 1) * stride <= 2048 && g < 8; g++) expect(speciesGroupOffset(g, stride)).toEqual([0, 0, g * stride]);
    }
    expect(speciesAtlasSize(512, 512, 512, 4)).toEqual([512, 512, 2048]);
    expect(speciesAtlasSize(256, 256, 1, 3)).toEqual([256, 256, 3]);
  });

  it('starts further stacks along y past 2048 texels', () => {
    // 51 slots of 20 texels a side, four groups: two stacks of two.
    const s = 1020;
    expect([0, 1, 2, 3].map((g) => speciesGroupOffset(g, s))).toEqual([
      [0, 0, 0],
      [0, 0, s],
      [0, s, 0],
      [0, s, s],
    ]);
    expect(speciesAtlasSize(s, s, s, 4)).toEqual([s, 2 * s, 2 * s]);
    // Past 1024 texels, one group per stack.
    expect(speciesAtlasSize(1100, 1100, 1100, 3)).toEqual([1100, 3300, 1100]);
  });

  it('places every group of every stack apart, inside the atlas', () => {
    for (const [stride, groups] of [
      [700, 4],
      [1020, 4],
      [1300, 2],
      [400, 9],
    ]) {
      const size = speciesAtlasSize(stride, stride, stride, groups);
      const seen = new Set<string>();
      for (let g = 0; g < groups; g++) {
        const o = speciesGroupOffset(g, stride);
        seen.add(o.join());
        for (let a = 0; a < 3; a++) expect(o[a] + stride).toBeLessThanOrEqual(size[a]);
      }
      expect(seen.size).toBe(groups);
    }
  });
});
