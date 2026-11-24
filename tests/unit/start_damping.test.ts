// Start-up damping (hydro/damping.wgsl): the setup's physics.startDamping reaches the problem,
// the config and the Params block (P.calm), and is off unless both its rate and time are positive.
import { describe, expect, it } from 'vitest';
import { checkSetup } from '../../src/setup/check';
import { compileSetup } from '../../src/setup/compile';
import type { SetupSpec } from '../../src/setup/spec';
import { configFor, packParams } from '../../src/sim';

const box = (startDamping?: { rate?: number; time?: number }): SetupSpec => ({
  format: 'shaderous-setup',
  version: 1,
  domain: { ndim: 3, n: 32, box: 1 },
  physics: { ...(startDamping ? { startDamping } : {}) },
  background: { rho: 1, p: 1 },
});

const calm = (spec: SetupSpec) => {
  const problem = compileSetup(spec).problem;
  const cfg = configFor(problem, 32);
  const f = new Float32Array(packParams(cfg, { dims: [32, 32, 32], dx: 1 / 32, rhoMean: 1, selfGravity: false, masses: 0 }));
  return [cfg.startDamping, Array.from(f.slice(136, 140))] as const;
};

describe('start-up damping', () => {
  it('goes from the setup into P.calm', () => {
    expect(checkSetup(box({ rate: 1, time: 10 })).filter((i) => i.level === 'error')).toEqual([]);
    const [cfg, p] = calm(box({ rate: 1, time: 10 }));
    expect(cfg).toEqual({ rate: 1, time: 10 });
    expect(p).toEqual([1, 10, 0, 0]);
  });

  it('is off without it, or with a zero rate or time', () => {
    expect(calm(box())[1]).toEqual([0, 0, 0, 0]);
    expect(calm(box({ rate: 1 }))[1]).toEqual([0, 0, 0, 0]);
    expect(calm(box({ time: 5 }))[1]).toEqual([0, 0, 0, 0]);
  });
});
