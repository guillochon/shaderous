// Picks the grid for a configuration: uniform, or block-adaptive when refinement levels are set.
import type { Gpu } from './gpu/device';
import { Sim, type SimConfig, type Simulation } from './sim';
import { AmrSim } from './amr/amr_sim';
import { BS } from './amr/tree';
import { memoryBudget } from './gpu/memory';
import type { Problem } from './problems/types';

/** Whether a preset can run on the adaptive grid: all can in 3D; in 2D, those without self-gravity. */
export const supportsAmr = (_p: Problem) => true;

/** The deepest refinement that leaves at least one whole block per side at the coarsest level. */
export function maxAmrLevels(n: number): number {
  let l = 0;
  while (l < 5 && (n >> (l + 1)) % BS === 0 && n >> (l + 1) >= BS) l++;
  return l;
}

/**
 * Whether a grid of n cells per side at the finest of `levels` fits the GPU memory budget: its
 * root blocks, with as many again to refine into.
 */
export function amrFits(n: number, ndim: 2 | 3, levels: number): boolean {
  const roots = ((n >> levels) / BS) ** ndim;
  return 2 * roots * AmrSim.blockBytes(ndim) <= memoryBudget();
}

/** The fewest levels of refinement (at least 1) whose root blocks fit, or 0 if none do. */
export function minAmrLevels(n: number, ndim: 2 | 3): number {
  for (let l = 1; l <= maxAmrLevels(n); l++) if (amrFits(n, ndim, l)) return l;
  return 0;
}

/** The shader that displays a configuration's run (Simulation.fieldShader), before it exists. */
export const fieldShaderFor = (cfg: SimConfig) => (cfg.amrLevels > 0 ? AmrSim.fieldShader : Sim.fieldShader);

/**
 * @param opts.deferUpdate on the adaptive mesh, return before the slowest pass has compiled: the
 *   run can be drawn meanwhile, and is `preparing` (takes no steps) until it has (AmrSim.create).
 */
export async function createSim(gpu: Gpu, cfg: SimConfig, opts: { deferUpdate?: boolean } = {}): Promise<Simulation> {
  return cfg.amrLevels > 0 ? AmrSim.create(gpu, cfg, opts) : Sim.create(gpu, cfg);
}
