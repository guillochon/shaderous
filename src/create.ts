// Picks the grid for a configuration: uniform, or block-adaptive when refinement levels are set.
import type { Gpu } from './gpu/device';
import { onBlocks, Sim, type SimConfig, type Simulation } from './sim';
import { AmrSim } from './amr/amr_sim';
import { BS, MAX_AMR_LEVELS } from './amr/tree';
import { memoryBudget } from './gpu/memory';
import type { Problem } from './problems/types';

/**
 * Whether a preset can run on the adaptive grid: all but a shearing box. (With a magnetic field the
 * mesh stays as the initial conditions refined it, until regridding carries the field:
 * docs/numerics.md section 21.)
 */
export const supportsAmr = (p: Problem) => !p.shear;

/** The deepest refinement that leaves at least one whole block per side at the coarsest level. */
export function maxAmrLevels(n: number): number {
  let l = 0;
  while (l < MAX_AMR_LEVELS && (n >> (l + 1)) % BS === 0 && n >> (l + 1) >= BS) l++;
  return l;
}

/** The leaf index of an adaptive grid (AmrSim's 'leaf index' texture): one r32uint per block of the finest level. */
export const leafIndexBytes = (n: number, ndim: 1 | 2 | 3) => 4 * (n / BS) ** ndim;

/**
 * Whether a grid of n cells per side at the finest of `levels` fits the GPU memory budget: its
 * root blocks, with as many again to refine into, and its leaf index.
 */
export function amrFits(n: number, ndim: 1 | 2 | 3, levels: number): boolean {
  return gridBytes(n, ndim, levels) <= memoryBudget();
}

/**
 * GPU memory a grid takes at the least: a uniform grid's cells, or an adaptive one's root
 * blocks with as many again to refine into, and its leaf index.
 */
export function gridBytes(n: number, ndim: 1 | 2 | 3, levels: number): number {
  if (levels > 0) return 2 * ((n >> levels) / BS) ** ndim * AmrSim.blockBytes(ndim) + leafIndexBytes(n, ndim);
  // A uniform grid past a texture's side is blocks with nothing to refine into (onBlocks).
  if (onBlocks({ n, ndim, amrLevels: 0 })) return (n / BS) ** ndim * AmrSim.blockBytes(ndim) + leafIndexBytes(n, ndim);
  return n ** ndim * Sim.cellBytes(ndim);
}

/** The fewest levels of refinement (at least 1) whose root blocks fit, or 0 if none do. */
export function minAmrLevels(n: number, ndim: 1 | 2 | 3): number {
  for (let l = 1; l <= maxAmrLevels(n); l++) if (amrFits(n, ndim, l)) return l;
  return 0;
}

/** The shader that displays a configuration's run (Simulation.fieldShader), before it exists. */
export const fieldShaderFor = (cfg: SimConfig) => (onBlocks(cfg) ? AmrSim.fieldShader : Sim.fieldShader);

/**
 * @param opts.deferUpdate on the adaptive mesh, return before the slowest pass has compiled: the
 *   run can be drawn meanwhile, and is `preparing` (takes no steps) until it has (AmrSim.create).
 */
export async function createSim(gpu: Gpu, cfg: SimConfig, opts: { deferUpdate?: boolean } = {}): Promise<Simulation> {
  // (A uniform 1D or 2D grid past a 3D texture's side runs on blocks with no levels: onBlocks.)
  return onBlocks(cfg) ? AmrSim.create(gpu, cfg, opts) : Sim.create(gpu, cfg);
}
