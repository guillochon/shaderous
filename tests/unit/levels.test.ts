import { describe, expect, it } from 'vitest';
import { gridBytes, leafIndexBytes, maxAmrLevels } from '../../src/create';
import { AmrSim } from '../../src/amr/amr_sim';
import { MAX_AMR_LEVELS } from '../../src/amr/tree';
import { OPTIONS } from '../../src/setup/options';
import { decodeLink } from '../../src/ui/link';

describe('levels of refinement', () => {
  it('allows up to seven, with a whole block per side at the root', () => {
    expect(maxAmrLevels(2048)).toBe(7);
    expect(maxAmrLevels(16384)).toBe(7);
    expect(maxAmrLevels(1024)).toBe(6);
    expect(maxAmrLevels(512)).toBe(5);
    expect(maxAmrLevels(32)).toBe(1);
    expect(maxAmrLevels(16)).toBe(0);
    expect(maxAmrLevels(4096)).toBe(7);
  });

  it('counts the leaf index in the memory of an adaptive grid (64 MB at 4096^3)', () => {
    expect(leafIndexBytes(4096, 3)).toBe(64 * 2 ** 20);
    // 4096^3 on 7 levels: 2 x 2 x 2 root blocks, as many again to refine into, and the index.
    expect(gridBytes(4096, 3, 7)).toBe(16 * AmrSim.blockBytes(3) + 64 * 2 ** 20);
  });

  it('offers 0 to 7 in setup documents', () => {
    const c = OPTIONS.find((o) => o.path === 'domain.amr')!.control as { min?: number; max?: number };
    expect(c.min).toBe(0);
    expect(c.max).toBe(MAX_AMR_LEVELS);
  });

  it('keeps seven levels in a link, and clamps past them', () => {
    expect(decodeLink(new URLSearchParams('problem=sedov&dim=2&n=16384&amr=7')).amr).toBe(7);
    expect(decodeLink(new URLSearchParams('problem=sedov&dim=2&n=16384&amr=9')).amr).toBe(MAX_AMR_LEVELS);
  });

  it('takes per-block timesteps up to cycles of 32 substeps', () => {
    expect(decodeLink(new URLSearchParams('problem=sedov&dim=2&n=2048&amr=7&tbins=5')).cfg.timeBins).toBe(5);
    const c = OPTIONS.find((o) => o.path === 'mesh.timeBins')!.control as { choices?: { value: unknown }[] };
    expect(c.choices?.map((x) => x.value)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
