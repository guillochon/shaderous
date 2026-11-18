/**
 * The CPU half of building an adaptive run's initial state (AmrSim.reset): the refinement
 * indicator that shapes the starting mesh, and each leaf's conserved state and magnetic faces
 * from the problem's setup. Pure functions of the configuration, so the page and the workers of
 * amr/init_pool.ts run the very same code and get bit-identical results.
 */
import { domainLo } from '../domain';
import { BS } from './tree';
import type { Eos } from '../eos/eos';
import { makeEos } from '../eos/eos';
import { burnIndicator, burnIndices, isNetworkBurn } from '../burn/burn';
import { cellMark, type Prim } from '../problems/types';
import { burnSpec, type SimConfig } from '../sim';
import { packSpecies, SPECIES_JUMP_SCALE } from '../species/species';
import { hydrogenParticles, mixMoments, type Moments } from '../species/mixture';
import { blockFaces } from '../mhd/amr_mhd';

/** Samples of the setup the indicator may take per pass over all leaves, in 2D and 3D. */
export const INDICATOR_SAMPLES_2D = 1 << 25;
export const INDICATOR_SAMPLES_3D = 1 << 24;

/** A block's place: its level and its coordinates in blocks of that level. */
export interface BlockAt {
  level: number;
  c: readonly number[];
}

/** One sample of the setup, as the indicator reads it. */
export type Sample = { rho: number; p: number; X?: readonly number[] };
export type Setup = (x: number, y: number, z: number) => Prim;

/**
 * The pressure of a state from the setup: its own, or the equation of state's from the density
 * alone where the gas has no energy of its own (barotropic).
 */
export const pressureOf = (eos: Eos, w: { rho: number; p: number }): number => (eos.hasEnergy ? w.p : eos.pressure(w.rho, 0));

/**
 * The setup as the indicator samples it. An obstacle counts as a jump in density, so its surface
 * starts refined. The indicator reads each sample at once and keeps none, so one object is reused
 * for them all: the mesh takes tens of millions of samples for a large 3D start.
 */
export function surfaceOf(fn: Setup, eos: Eos): (x: number, y: number, z: number) => Sample {
  const sample = { rho: 0, p: 0, X: undefined as readonly number[] | undefined };
  return (x, y, z) => {
    const w = fn(x, y, z);
    sample.rho = w.solid ? 4 * w.rho : w.rho;
    sample.p = pressureOf(eos, w);
    sample.X = w.solid ? undefined : w.X;
    return sample;
  };
}

/**
 * Samples per cell side when building the starting mesh: down to the finest level, within a
 * budget. A few blocks share a total budget per pass, so a deep mesh still sees a feature a few
 * finest cells across (a blast) that coarse sampling of a root block would step over; many
 * blocks get the floor per block.
 */
export function samplesPerCell(cfg: SimConfig, levels: number, level: number, leaves: number): number {
  const three = cfg.ndim === 3;
  const floor = three ? 2 : 4;
  // The 3D budget was sized for grids up to 512^3. A finer grid gets more in proportion to its
  // cells, so the samples per finest cell stay as they were: a blast 3.5 finest cells across
  // at 1024^3 with 5 levels is otherwise stepped over at the root level, and never refined.
  const budget = three ? INDICATOR_SAMPLES_3D * Math.max(1, (cfg.n / 512) ** 3) : INDICATOR_SAMPLES_2D;
  const share = Math.floor(Math.pow(budget / Math.max(leaves, 1), 1 / cfg.ndim) / (BS + 2));
  return Math.min(1 << (levels - level), Math.max(floor, share > 0 ? 1 << Math.floor(Math.log2(share)) : 0));
}

/**
 * Refinement indicator of a block from the analytic setup, for building the initial mesh: the
 * same normalized jump across two cell widths that the GPU pass measures, but sampled more finely
 * than the block's own cells. A feature smaller than a coarse cell (a point explosion, say) would
 * otherwise fall between cell centers and never trigger refinement.
 */
export function indicator(cfg: SimConfig, dx0: number, b: BlockAt, fn: (x: number, y: number, z: number) => Sample, sub: number): number {
  const three = cfg.ndim === 3;
  const dx = dx0 / (1 << b.level);
  const lo = domainLo(cfg);
  const h = dx / sub;
  const m = (BS + 2) * sub; // one cell of margin on each side
  const mz = three ? m : 1;
  const at = (i: number, a: number) => b.c[a] * BS * dx + (i - sub + 0.5) * h + lo[a];
  // Gas dense enough that the preset wants it at the finest level (stars): as amr/flag.wgsl.
  // Any sample above it decides the block, so with several samples per cell one per cell is
  // looked at first (the cell-centered ones of the full sampling below): a block inside a star
  // or disk is then settled at a small fraction of the cost.
  const dense = cfg.refineDense;
  if (dense > 0 && sub > 1) {
    const first = sub >> 1;
    for (let k = three ? first : 0; k < mz; k += sub)
      for (let j = first; j < m; j += sub)
        for (let i = first; i < m; i += sub) if (fn(at(i, 0), at(j, 1), three ? at(k, 2) : 0).rho > dense) return Infinity;
  }
  const rho = new Float64Array(m * m * mz);
  const pr = new Float64Array(m * m * mz);
  // The species' mass fractions, normalized as packSpecies stores them.
  const ns = cfg.problem.species?.length ?? 0;
  const xs = Array.from({ length: ns }, () => new Float64Array(m * m * mz));
  for (let k = 0; k < mz; k++)
    for (let j = 0; j < m; j++)
      for (let i = 0; i < m; i++) {
        const w = fn(at(i, 0), at(j, 1), three ? at(k, 2) : 0);
        if (dense > 0 && w.rho > dense) return Infinity;
        rho[(k * m + j) * m + i] = w.rho;
        pr[(k * m + j) * m + i] = w.p;
        if (ns) {
          let sum = 0;
          for (let s = 0; s < ns; s++) sum += Math.max(w.X?.[s] ?? 0, 0);
          for (let s = 0; s < ns; s++) xs[s][(k * m + j) * m + i] = sum > 0 ? Math.max(w.X?.[s] ?? 0, 0) / sum : s === 0 ? 1 : 0;
        }
      }
  // The normalized second derivative of amr/flag.wgsl, at the finest spacing sampled.
  let ind = 0;
  const strides = three ? [sub, sub * m, sub * m * m] : [sub, sub * m];
  const ns2 = strides.length;
  const lohner = (q: Float64Array, a: number) => {
    let num = 0;
    let den = 0;
    const q0 = q[a];
    for (let t = 0; t < ns2; t++) {
      const s = strides[t];
      const qp = q[a + s];
      const qm = q[a - s];
      num += (qp - 2 * q0 + qm) ** 2;
      den += (Math.abs(qp - q0) + Math.abs(q0 - qm) + 0.01 * (Math.abs(qp) + 2 * Math.abs(q0) + Math.abs(qm))) ** 2;
    }
    return den > 0 ? Math.sqrt(num / den) : 0;
  };
  // The mass fractions' jumps between neighbors, as species_indicator (species/species.ts).
  const jump = (q: Float64Array, a: number) => {
    let j = 0;
    for (const s of strides) j = Math.max(j, Math.abs(q[a + s] - q[a]), Math.abs(q[a] - q[a - s]));
    return SPECIES_JUMP_SCALE * j;
  };
  // How fast it burns, against the block's cells (burnIndicator, as amr/flag.wgsl).
  const spec = burnSpec(cfg);
  const burn = spec && !isNetworkBurn(spec) ? spec : null;
  const fuel = burn ? xs[burnIndices(burn, cfg.problem.species)[0]] : null;
  const eos = burn ? makeEos(cfg.eos, cfg.eosParams) : null;
  for (let k = three ? sub : 0; k < (three ? m - sub : 1); k++)
    for (let j = sub; j < m - sub; j++)
      for (let i = sub; i < m - sub; i++) {
        const a = (k * m + j) * m + i;
        ind = Math.max(ind, lohner(rho, a), lohner(pr, a));
        for (const x of xs) ind = Math.max(ind, jump(x, a));
        if (fuel) ind = Math.max(ind, burnIndicator(burn!, eos!, rho[a], pr[a], fuel[a], dx));
      }
  return ind;
}

/** A leaf's initial state, as AmrSim.reset writes it into the atlas. */
export interface LeafState {
  /** (rho, rho v) and (E, e, rho s, mark) of each cell, BS^3 (2D: BS^2) texels of four */
  u1: Float32Array<ArrayBuffer>;
  u2: Float32Array<ArrayBuffer>;
  /** the magnetic faces (amr_mhd.ts blockFaces' layout), with a field */
  faces: Float32Array<ArrayBuffer> | null;
  /** each cell's composition moments, where the species set the equation of state */
  km: Float32Array<ArrayBuffer> | null;
  /** the species' partial densities, groups of four */
  sp: Float32Array<ArrayBuffer> | null;
  /** the leaf's mass */
  mass: number;
  /** a cell of the leaf is inside an obstacle */
  solids: boolean;
}

/** What the fill needs beyond the configuration, made once per reset. */
export interface FillContext {
  cfg: SimConfig;
  dx0: number;
  levels: number;
  fn: Setup;
  eos: Eos;
  /** the composition coefficients, where the species set the equation of state */
  kc: Moments[] | null;
  /** species groups of four */
  groups: number;
  mhd: boolean;
}

/** A leaf's conserved state (and magnetic faces) from the setup, at its cell centers. */
export function leafState(x: FillContext, b: BlockAt): LeafState {
  const { cfg: c, fn, eos, kc, groups: G } = x;
  const three = c.ndim === 3;
  const depth = three ? BS : 1;
  const lo = domainLo(c);
  const blockCells = BS * BS * depth;
  const u1 = new Float32Array(blockCells * 4);
  const u2 = new Float32Array(blockCells * 4);
  const sp = G ? new Float32Array(blockCells * 4 * G) : null;
  const km = kc ? new Float32Array(blockCells * 4) : null;
  const dx = x.dx0 / (1 << b.level);
  const vol = dx ** c.ndim;
  // A magnetic field: the leaf's faces, and their energy in each cell's total.
  const faces = x.mhd ? (blockFaces(c, b as never, x.dx0, x.levels) as Float32Array<ArrayBuffer>) : null;
  const W = BS + 1;
  const f = (a: number, ii: number, jj: number, kk: number) => faces![4 * ((kk * W + jj) * W + ii) + a];
  let mass = 0;
  let solids = false;
  let o = 0;
  for (let k = 0; k < depth; k++)
    for (let j = 0; j < BS; j++)
      for (let i = 0; i < BS; i++, o += 4) {
        const w = fn((b.c[0] * BS + i + 0.5) * dx + lo[0], (b.c[1] * BS + j + 0.5) * dx + lo[1], three ? (b.c[2] * BS + k + 0.5) * dx + lo[2] : 0);
        const p = pressureOf(eos, w);
        // Where the species set the equation of state, the cell's own.
        let ce = eos;
        if (km) {
          const mo = mixMoments(kc!, w.X);
          km.set(mo, o);
          ce = makeEos(c.eos, c.eosParams, mo, hydrogenParticles(c.problem.composition));
        }
        const eint = ce.eintFromP(w.rho, p);
        const mark = cellMark(w);
        if (mark) solids = true;
        const go = mark === 1 ? 0 : 1; // the gas kept inside an obstacle is at rest
        mass += w.rho * vol;
        u1[o] = w.rho;
        u1[o + 1] = w.rho * w.vx * go;
        u1[o + 2] = w.rho * w.vy * go;
        u1[o + 3] = w.rho * w.vz * go;
        u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz) * go;
        if (faces) {
          const bx = 0.5 * (f(0, i, j, k) + f(0, i + 1, j, k));
          const by = 0.5 * (f(1, i, j, k) + f(1, i, j + 1, k));
          const bz = three ? 0.5 * (f(2, i, j, k) + f(2, i, j, k + 1)) : f(2, i, j, k);
          u2[o] += 0.5 * (bx * bx + by * by + bz * bz);
        }
        // With implicit pressure on blocks a liquid's unused internal-energy slot carries the
        // solved pressure from step to step (amr/pressure/correct.wgsl): it starts as p(rho).
        u2[o + 1] = c.pressure === 'implicit' && !eos.hasEnergy ? p : eint;
        u2[o + 2] = w.rho * w.s;
        u2[o + 3] = mark;
        if (sp) packSpecies(sp, o, blockCells * 4, G, w.rho, w.X);
      }
  return { u1, u2, faces, km, sp, mass, solids };
}
