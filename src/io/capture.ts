// Snapshots of a running simulation, and runs started from snapshots.
import { AmrSim } from '../amr/amr_sim';
import { BS } from '../amr/tree';
import type { Problem } from '../problems';
import { Sim, type SimConfig, type Simulation } from '../sim';
import { primitivesFrom, SnapshotSampler, type Restart, type Snapshot, type SnapshotMeta } from './snapshot';

/** Read the state back from the GPU as blocks of primitive variables. */
export async function capture(sim: Simulation, meta?: Omit<SnapshotMeta, 'perturber'>): Promise<Snapshot> {
  const cfg = sim.cfg;
  const half = cfg.problem.box / 2;
  const flat = cfg.ndim === 2;
  let rootSize: number[];
  let blockSize: number[];
  let levels: Int32Array;
  let locations: Int32Array;
  let u1: Float32Array;
  let u2: Float32Array;
  let restart: Awaited<ReturnType<Sim['readRestart']>>;

  if (sim instanceof AmrSim) {
    // Leaves come back block by block, each as [z][y][x]: the layout a snapshot uses.
    const cells = await sim.readLeaves();
    const leaves = sim.tree.leaves();
    const n0 = sim.n >> cfg.amrLevels;
    rootSize = [n0, n0, flat ? 1 : n0];
    blockSize = [BS, BS, flat ? 1 : BS];
    levels = Int32Array.from(leaves.map((b) => b.level));
    locations = Int32Array.from(leaves.flatMap((b) => [b.c[0], b.c[1], flat ? 0 : b.c[2]]));
    ({ u1, u2 } = cells);
    restart = await sim.readRestart();
  } else if (sim instanceof Sim) {
    // A uniform grid is a single block.
    rootSize = [sim.n, sim.n, sim.nz];
    blockSize = rootSize.slice();
    levels = new Int32Array(1);
    locations = new Int32Array(3);
    ({ u1, u2 } = await sim.readState());
    restart = await sim.readRestart();
  } else {
    throw new Error('this kind of simulation cannot be exported');
  }

  return {
    ndim: cfg.ndim,
    rootSize,
    blockSize,
    lo: [-half, -half, -half],
    hi: [half, half, half],
    time: restart.time,
    steps: Math.round(restart.steps),
    levels,
    locations,
    data: primitivesFrom(cfg, u1, u2),
    meta: meta ? { ...meta, perturber: restart.perturber } : undefined,
  };
}

/** What to change in a configuration so the run starts from a snapshot. */
export function restartFrom(snap: Snapshot, problem: Problem, adaptive: boolean, rootCells: number): Pick<SimConfig, 'problem' | 'restart'> {
  const sampler = new SnapshotSampler(snap);
  const box = problem.box;
  const frac = (x: number) => x / box + 0.5;
  const restart: Restart = { time: snap.time, steps: snap.steps, perturber: snap.meta?.perturber };
  if (adaptive) {
    // Levels count from the root grid, and the file's root may not be the run's.
    const shift = Math.round(Math.log2(snap.rootSize[0] / rootCells));
    restart.levelAt = (x, y, z) => Math.max(0, sampler.level(frac(x), frac(y), frac(z)) + shift);
  }
  return {
    // The same preset, except that its initial conditions are read out of the snapshot.
    // (Its obstacles and inflow are not in the file: they come from the preset again.)
    problem: {
      ...problem,
      init: (n, eos, ndim, params) => {
        const preset = problem.init(n, eos, ndim, params);
        return (x, y, z) => {
          const { solid, fixed } = preset(x, y, z);
          return { ...sampler.prim(frac(x), frac(y), frac(z)), solid, fixed };
        };
      },
    },
    restart,
  };
}
