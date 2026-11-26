import { describe, expect, it } from 'vitest';
import { maxAmrLevels } from '../../src/create';
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
});
