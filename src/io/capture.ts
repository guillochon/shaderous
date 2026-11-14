// Snapshots of a running simulation, and runs started from snapshots.
import { AmrSim } from '../amr/amr_sim';
import { BS } from '../amr/tree';
import type { Problem } from '../problems';
import { Sim, type SimConfig, type Simulation } from '../sim';
import { primitivesFrom, SNAPSHOT_FORMAT, SnapshotSampler, type Restart, type Snapshot, type SnapshotMeta } from './snapshot';

/** Read the state back from the GPU as blocks of primitive variables. */
export async function capture(sim: Simulation, meta?: Omit<SnapshotMeta, 'masses' | 'perturber'>): Promise<Snapshot> {
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
  let X: Float32Array[];

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
    X = await sim.readLeafSpecies();
    restart = await sim.readRestart();
  } else if (sim instanceof Sim) {
    // A uniform grid is a single block.
    rootSize = [sim.n, sim.n, sim.nz];
    blockSize = rootSize.slice();
    levels = new Int32Array(1);
    locations = new Int32Array(3);
    ({ u1, u2 } = await sim.readState());
    X = await sim.readSpecies();
    restart = await sim.readRestart();
  } else {
    throw new Error('this kind of simulation cannot be exported');
  }

  // A magnetic field: cell-centered for analysis (Bcc1-3, as Athena++ names them), and each
  // cell's lower faces (Bf1-3) with the upper boundary faces in the metadata, for a restart.
  const mag: Record<string, Float32Array> = {};
  let mhdMeta: SnapshotMeta['mhd'];
  if (sim instanceof Sim && sim.mhd) {
    const f = await sim.mhd.readFaces(sim.cur);
    const n = sim.n;
    const nz = sim.nz;
    const fx = n + 1;
    const texel = (i: number, j: number, k: number) => 4 * ((k * fx + j) * fx + i);
    const up = (a: number, m: number, len: number) => (m + 1 === len && cfg.bcLo[a] === 0 ? 0 : m + 1);
    for (const k of ['Bf1', 'Bf2', 'Bf3', 'Bcc1', 'Bcc2', 'Bcc3']) mag[k] = new Float32Array(n * n * nz);
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const o = (k * n + j) * n + i;
          const hi = [texel(up(0, i, n), j, k), texel(i, up(1, j, n), k), texel(i, j, nz === 1 ? 0 : up(2, k, nz))];
          for (let a = 0; a < 3; a++) {
            const lo = f[texel(i, j, k) + a];
            mag[`Bf${a + 1}`][o] = lo;
            mag[`Bcc${a + 1}`][o] = 0.5 * (lo + f[hi[a] + a]);
          }
        }
    const upper = [0, 1, 2].map((a) => {
      if (cfg.bcLo[a] === 0 || (a === 2 && nz === 1)) return [];
      const plane: number[] = [];
      const [l1, l2] = a === 2 ? [n, n] : [n, nz];
      for (let q = 0; q < l2; q++)
        for (let p = 0; p < l1; p++) {
          const t = a === 0 ? texel(n, p, q) : a === 1 ? texel(p, n, q) : texel(p, q, nz);
          plane.push(f[t + a]);
        }
      return plane;
    });
    mhdMeta = { n, nz, upper };
  }
  const debris = await sim.debrisLedger();
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
    data: { ...primitivesFrom(cfg, u1, u2), ...Object.fromEntries((cfg.problem.species ?? []).map((k, i) => [k, X[i]])), ...mag },
    meta: meta ? { format: SNAPSHOT_FORMAT, ...meta, masses: restart.masses, ...(debris ? { debris } : {}), ...(mhdMeta ? { mhd: mhdMeta } : {}) } : undefined,
  };
}

/** What to change in a configuration so the run starts from a snapshot. */
export function restartFrom(snap: Snapshot, problem: Problem, adaptive: boolean, rootCells: number): Pick<SimConfig, 'problem' | 'restart'> {
  const sampler = new SnapshotSampler(snap);
  const box = problem.box;
  const frac = (x: number) => x / box + 0.5;
  const restart: Restart = { time: snap.time, steps: snap.steps, masses: snap.meta?.masses ?? (snap.meta?.perturber ? [snap.meta.perturber] : undefined), debris: snap.meta?.debris };
  if (adaptive) {
    // Levels count from the root grid, and the file's root may not be the run's.
    const shift = Math.round(Math.log2(snap.rootSize[0] / rootCells));
    restart.levelAt = (x, y, z) => Math.max(0, sampler.level(frac(x), frac(y), frac(z)) + shift);
  }
  // A magnetic field comes back face by face, which needs the grid it was saved on.
  let mhd = problem.mhd;
  if (mhd) {
    const m = snap.meta?.mhd;
    const d = snap.data;
    if (adaptive || !m || !d.Bf1 || !d.Bf2 || !d.Bf3 || m.n !== rootCells || snap.rootSize[0] !== rootCells) {
      throw new Error('a magnetized run restarts only from a file it wrote itself, on the same uniform grid');
    }
    mhd = {
      faces: (n, nz) => {
        if (n !== m.n || nz !== m.nz) throw new Error('a magnetized run restarts only on the grid it was saved on');
        const fx = n + 1;
        const fz = nz === 1 ? 1 : nz + 1;
        const f = new Float32Array(fx * fx * fz * 4);
        const texel = (i: number, j: number, k: number) => 4 * ((k * fx + j) * fx + i);
        const B = [d.Bf1!, d.Bf2!, d.Bf3!];
        for (let k = 0; k < nz; k++)
          for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) f[texel(i, j, k) + a] = B[a][(k * n + j) * n + i];
        m.upper.forEach((plane, a) => {
          if (!plane.length) return;
          const [l1, l2] = a === 2 ? [n, n] : [n, nz];
          for (let q = 0; q < l2; q++)
            for (let p = 0; p < l1; p++) {
              const t = a === 0 ? texel(n, p, q) : a === 1 ? texel(p, n, q) : texel(p, q, nz);
              f[t + a] = plane[q * l1 + p];
            }
        });
        return f;
      },
    };
  }
  return {
    // The same preset, except that its initial conditions are read out of the snapshot.
    // (Its obstacles and inflow are not in the file: they come from the preset again.)
    problem: {
      ...problem,
      ...(mhd ? { mhd } : {}),
      init: (n, eos, ndim, params) => {
        const preset = problem.init(n, eos, ndim, params);
        return (x, y, z) => {
          const { solid, fixed, X } = preset(x, y, z);
          // Species the file lacks start as the preset has them.
          const w = sampler.prim(frac(x), frac(y), frac(z), problem.species);
          return { ...w, X: w.X ?? X, solid, fixed };
        };
      },
    },
    restart,
  };
}
