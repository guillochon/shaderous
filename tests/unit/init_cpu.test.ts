import { describe, expect, it } from 'vitest';
import { indicator, samplesPerCell, surfaceOf } from '../../src/amr/init_cpu';
import { BlockTree, BS } from '../../src/amr/tree';
import { makeEos } from '../../src/eos/eos';
import { problemById } from '../../src/problems/index';
import { configFor } from '../../src/sim';

/** The starting mesh's levels, as AmrSim.reset builds it from the setup (without the GPU). */
function startingMesh(id: string, n: number, levels: number, cap: number) {
  const problem = problemById(id);
  const cfg = configFor(problem, n, { amrLevels: levels, ndim: 3 });
  const n0 = n >> levels;
  const dx0 = problem.box / n0;
  const eos = makeEos(cfg.eos, cfg.eosParams);
  const surface = surfaceOf(problem.init(n, eos, 3, cfg.params), eos);
  const tree = new BlockTree(3, n0 / BS, levels, cap, cfg.bcLo, cfg.bcHi, false);
  tree.levelLimit = levels;
  const seen = new Map<unknown, { sub: number; ind: number }>();
  for (let pass = 0; pass < levels; pass++) {
    const leaves = tree.leaves().length;
    const changed = tree.adapt((b) => {
      const sub = samplesPerCell(cfg, levels, b.level, leaves);
      const s = seen.get(b);
      if (s && s.sub >= sub) return s.ind;
      const ind = indicator(cfg, dx0, b, surface, sub);
      seen.set(b, { sub, ind });
      return ind;
    }, cfg.refineHi, cfg.refineLo);
    if (!changed) break;
  }
  return tree;
}

describe('starting mesh on grids finer than 512^3 (amr/init_cpu.ts)', () => {
  it('keeps the samples per cell of grids up to 512^3', () => {
    const cfg = (n: number) => configFor(problemById('sedov'), n, { amrLevels: 4, ndim: 3 });
    // 8 root blocks at 512^3 with 4 levels: 4 samples per root cell side, as before the budget scaled.
    expect(samplesPerCell(cfg(512), 4, 0, 8)).toBe(4);
    expect(samplesPerCell(cfg(256), 4, 0, 1)).toBe(8);
  });

  it('refines a blast 3.5 finest cells across to the finest level at 1024^3 with 5 levels', () => {
    const tree = startingMesh('sedov', 1024, 5, 4000);
    const deepest = Math.max(...tree.leaves().map((b) => b.level));
    expect(deepest).toBe(5);
  });
});
