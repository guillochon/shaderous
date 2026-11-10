// Picks the grid for a configuration: uniform, or block-adaptive when refinement levels are set.
import type { Gpu } from './gpu/device';
import { Sim, type SimConfig, type Simulation } from './sim';
import { AmrSim } from './amr/amr_sim';
import { BS } from './amr/tree';
import type { Problem } from './problems/types';

/** Whether a preset can run on the adaptive grid: all can in 3D; in 2D, those without self-gravity. */
export const supportsAmr = (_p: Problem) => true;

/** The deepest refinement that leaves at least one whole block per side at the coarsest level. */
export function maxAmrLevels(n: number): number {
  let l = 0;
  while (l < 4 && (n >> (l + 1)) % BS === 0 && n >> (l + 1) >= BS) l++;
  return l;
}

export async function createSim(gpu: Gpu, cfg: SimConfig): Promise<Simulation> {
  return cfg.amrLevels > 0 ? AmrSim.create(gpu, cfg) : Sim.create(gpu, cfg);
}
