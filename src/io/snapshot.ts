// The state of the gas as a set of blocks of primitive variables, independent of how a run
// stores it: what gets written to a file and read back, and how a run is started from one.
import { makeEos } from '../eos/eos';
import type { PointMass, Prim } from '../problems';
import type { SimConfig } from '../sim';

export const VARIABLES = ['rho', 'press', 'vel1', 'vel2', 'vel3', 'r0'] as const;
export type Variable = (typeof VARIABLES)[number];

export interface Snapshot {
  ndim: 2 | 3;
  /** cells along each axis of the root grid, and of every block */
  rootSize: number[];
  blockSize: number[];
  /** lower and upper edge of the domain along each axis */
  lo: number[];
  hi: number[];
  time: number;
  steps: number;
  /** refinement level and position (in blocks, at that level) of each block; leaves only */
  levels: Int32Array;
  locations: Int32Array; // 3 per block
  /** one array per variable: blocks in order, each block as [z][y][x] */
  data: Partial<Record<Variable, Float32Array>>;
  /** written by this program: enough to set the run up again exactly */
  meta?: SnapshotMeta;
}

export interface SnapshotMeta {
  program: 'shaderous';
  build: string;
  /** the settings, as the query string of a shareable link */
  query: string;
  /** the point masses, if the run has any */
  masses?: PointMass[];
  /** files written before there could be several: position and velocity of the one mass */
  perturber?: { pos: number[]; vel: number[] };
}

export const blockCells = (s: Snapshot) => s.blockSize[0] * s.blockSize[1] * s.blockSize[2];
export const blockCount = (s: Snapshot) => s.levels.length;
export const maxLevel = (s: Snapshot) => s.levels.reduce((m, l) => Math.max(m, l), 0);

/** Primitive variables from interleaved conserved state (rho, m) and (E, e_int, rho s, 0). */
export function primitivesFrom(cfg: SimConfig, u1: Float32Array, u2: Float32Array): Record<Variable, Float32Array> {
  const eos = makeEos(cfg.eos, cfg.eosParams);
  const count = u1.length / 4;
  const out = Object.fromEntries(VARIABLES.map((v) => [v, new Float32Array(count)])) as Record<Variable, Float32Array>;
  for (let i = 0; i < count; i++) {
    const rho = u1[4 * i];
    out.rho[i] = rho;
    out.vel1[i] = u1[4 * i + 1] / rho;
    out.vel2[i] = u1[4 * i + 2] / rho;
    out.vel3[i] = u1[4 * i + 3] / rho;
    out.press[i] = eos.hasEnergy ? eos.pressure(rho, u2[4 * i + 1]) : eos.pressure(rho, 0);
    out.r0[i] = u2[4 * i + 2] / rho;
  }
  return out;
}

/**
 * Point lookup into a snapshot: the cell that contains a position, given as a fraction of the
 * domain along each axis. Leaves do not overlap, so the block that held the previous point is
 * tried first; most lookups come in runs through one block.
 */
export class SnapshotSampler {
  private index = new Map<number, number>();
  private last = -1;
  private readonly top: number;
  private readonly rootBlocks: number[];
  private readonly cells: number;

  constructor(readonly snap: Snapshot) {
    this.top = maxLevel(snap);
    this.rootBlocks = snap.rootSize.map((n, a) => n / snap.blockSize[a]);
    this.cells = blockCells(snap);
    for (let b = 0; b < snap.levels.length; b++) {
      this.index.set(this.key(snap.levels[b], snap.locations[3 * b], snap.locations[3 * b + 1], snap.locations[3 * b + 2]), b);
    }
    for (const v of ['rho', 'press', 'vel1', 'vel2', 'vel3'] as const) {
      if (!snap.data[v]) throw new Error(`the file has no "${v}" variable`);
    }
  }

  private key(level: number, i: number, j: number, k: number): number {
    // Block positions fit 12 bits per axis up to level ~8 of a 16-block root; levels fit 4.
    return ((level * 4096 + k) * 4096 + j) * 4096 + i;
  }

  /** Index of the block holding the point and of the cell inside it, or -1 if none does. */
  private find(f: number[]): { block: number; cell: number } | null {
    const s = this.snap;
    const tryBlock = (b: number) => {
      const l = s.levels[b];
      let cell = 0;
      let stride = 1;
      for (let a = 0; a < 3; a++) {
        const g = Math.min(Math.floor(f[a] * (s.rootSize[a] << l)), (s.rootSize[a] << l) - 1);
        const local = g - s.locations[3 * b + a] * s.blockSize[a];
        if (local < 0 || local >= s.blockSize[a]) return -1;
        cell += local * stride;
        stride *= s.blockSize[a];
      }
      return cell;
    };
    if (this.last >= 0) {
      const cell = tryBlock(this.last);
      if (cell >= 0) return { block: this.last, cell };
    }
    for (let l = this.top; l >= 0; l--) {
      const at = [0, 1, 2].map((a) => Math.min(Math.floor(f[a] * (this.rootBlocks[a] << l)), (this.rootBlocks[a] << l) - 1));
      const b = this.index.get(this.key(l, at[0], at[1], at[2]));
      if (b === undefined) continue;
      this.last = b;
      return { block: b, cell: tryBlock(b) };
    }
    return null;
  }

  private where(fx: number, fy: number, fz: number) {
    const clamp = (v: number) => Math.min(Math.max(v, 0), 1 - 1e-7);
    const hit = this.find([clamp(fx), clamp(fy), this.snap.rootSize[2] === 1 ? 0 : clamp(fz)]);
    if (!hit) throw new Error('the file does not cover the whole domain');
    return hit;
  }

  /** Primitive state at a point. */
  prim(fx: number, fy: number, fz: number): Prim {
    const { block, cell } = this.where(fx, fy, fz);
    const o = block * this.cells + cell;
    const d = this.snap.data;
    return { rho: d.rho![o], vx: d.vel1![o], vy: d.vel2![o], vz: d.vel3![o], p: d.press![o], s: d.r0 ? d.r0[o] : 0 };
  }

  /** Refinement level of the block at a point. */
  level(fx: number, fy: number, fz: number): number {
    return this.snap.levels[this.where(fx, fy, fz).block];
  }
}

/** How a run is started from a snapshot instead of from its preset's initial conditions. */
export interface Restart {
  time: number;
  steps: number;
  /** the point masses' state, laid over the preset's */
  masses?: Partial<PointMass>[];
  /** refinement level the mesh should have at a position (adaptive runs) */
  levelAt?: (x: number, y: number, z: number) => number;
}
