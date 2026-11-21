// View-only loading (plans/view-only.md): a checkpoint read in parts, with its blocks finer than a
// chosen level merged into their ancestors at that level, so a run too large to continue on this
// device can still be looked at. The result is an ordinary snapshot of the coarser mesh, which
// the adaptive mesh loads as it would any restart (AmrSim in view mode holds only what the
// display reads).
import type { AthdfReader } from './athdf';
import { maxLevel, type Snapshot } from './snapshot';
import { BS, PAD } from '../amr/tree';

/**
 * GPU bytes per atlas block in view only (AmrSim with SimConfig.viewOnly): the state at one
 * parity and the primitives the display derives from it (three rgba32float), the display's own
 * atlas (render/renderer.ts, rgba16float), a field's faces at one parity and its flux register
 * (still allocated), and the species' groups at one parity with the composition's moments.
 */
export function viewBlockBytes(ndim: 2 | 3, mhd: boolean, speciesGroups: number, composition: boolean): number {
  const texel = 3 * 16 + 8 + (mhd ? 16 : 0) + speciesGroups * 16 + (composition ? 16 : 0);
  // The flux register: mhd/amr_mhd.ts registerBytes (not imported: it brings every shader into the reader's worker).
  return PAD ** ndim * texel + (mhd ? 6 * (ndim === 3 ? BS * BS : BS) * 32 : 0);
}

/** Variables of a file that are not species: the primitives, the passive scalar, the field. */
const NOT_SPECIES = /^(rho|press|vel\d|r0|B(cc|f)\d)$/;

/** Species groups of four among a file's variables (what speciesGroups would give its problem). */
export const fileSpeciesGroups = (variables: string[]): number => Math.ceil(variables.filter((v) => !NOT_SPECIES.test(v)).length / 4);

/** How a file will be viewed: the level its blocks are merged to, and what that holds. */
export interface ViewPlan {
  /** the finest level kept (the file's own finest when nothing is merged) */
  level: number;
  /** the file's finest level */
  fileLevel: number;
  /** leaf blocks after merging, and blocks of the tree with their ancestors (atlas slots) */
  leaves: number;
  nodes: number;
  /** leaf blocks in the file */
  fileLeaves: number;
}

/** Axes along which blocks split (a 2D file's z, a cylindrical file's phi, do not). */
const activeAxes = (s: Pick<Snapshot, 'rootSize' | 'blockSize'>) => [0, 1, 2].filter((a) => s.blockSize[a] > 1);

const key = (l: number, i: number, j: number, k: number) => ((l * 65536 + k) * 65536 + j) * 65536 + i;

/** Leaves and tree blocks (leaves and all their ancestors) once the file's blocks are merged to `cap`. */
export function countAtCap(s: Pick<Snapshot, 'levels' | 'locations' | 'rootSize' | 'blockSize'>, cap: number): { leaves: number; nodes: number } {
  const act = activeAxes(s);
  const leaves = new Set<number>();
  const nodes = new Set<number>();
  for (let b = 0; b < s.levels.length; b++) {
    const lv = s.levels[b];
    const l = Math.min(lv, cap);
    const sh = lv - l;
    const c = [0, 1, 2].map((a) => (act.includes(a) ? s.locations[3 * b + a] >> sh : 0));
    leaves.add(key(l, c[0], c[1], c[2]));
    for (let m = l; m >= 0; m--) {
      const k = key(m, c[0] >> (l - m), c[1] >> (l - m), c[2] >> (l - m));
      if (nodes.has(k)) break;
      nodes.add(k);
    }
  }
  return { leaves: leaves.size, nodes: nodes.size };
}

/**
 * The finest level whose tree fits `maxNodes` atlas slots (all of the file's when it fits), or
 * null when not even the root grid does.
 */
export function planView(s: Pick<Snapshot, 'levels' | 'locations' | 'rootSize' | 'blockSize'>, maxNodes: number, maxLevelWanted = Infinity): ViewPlan | null {
  const fileLevel = s.levels.reduce((m, l) => Math.max(m, l), 0);
  for (let level = Math.min(fileLevel, maxLevelWanted); level >= 0; level--) {
    const { leaves, nodes } = countAtCap(s, level);
    if (nodes <= maxNodes) return { level, fileLevel, leaves, nodes, fileLeaves: s.levels.length };
  }
  return null;
}

/** Variables averaged by volume; the rest (velocities, the passive scalar, species) by mass. */
const byVolume = (name: string) => name === 'rho' || name === 'press' || /^B(cc|f)?\d$/.test(name);

/**
 * Read the file's blocks merged to level `cap`, `chunk` blocks at a time, so no copy of the
 * whole file is held: cells finer than `cap` are averaged into the cell of their ancestor that
 * holds them (density, pressure and magnetic field by volume; velocities, the passive scalar
 * and species by mass), and a magnetized run's faces by area, which keeps the field
 * divergence-free. Returns a snapshot of the merged mesh in the program's axis order.
 * `wanted` picks the variables (all by default; the five primitives are always read).
 */
export function readForView(
  r: AthdfReader,
  cap: number,
  opts: { chunk?: number; wanted?: (name: string) => boolean; progress?: (fraction: number) => void } = {},
): Snapshot {
  const head = r.header();
  const act = activeAxes(head);
  const d = act.length;
  const bs = head.blockSize;
  const nb = r.blocks;
  const cells = r.cells;
  const fileTop = maxLevel(head);
  const required = ['rho', 'press', 'vel1', 'vel2', 'vel3'];
  for (const v of required) if (!r.variables.includes(v)) throw new Error(`the file has no "${v}" variable`);
  const vars = r.variables.filter((v) => required.includes(v) || (opts.wanted?.(v) ?? true));
  // A uniform grid's single block is split into blocks of 16 by the caller; here a file's blocks must be the program's.
  if (nb > 1 && act.some((a) => bs[a] !== 16)) throw new Error(`blocks of ${bs.join(' x ')} cells cannot be viewed: the program's are 16 a side`);

  // The merged mesh: one block per distinct (level, position) at or above cap, in file order.
  const out = new Map<number, number>();
  const outLevels: number[] = [];
  const outLoc: number[] = [];
  const target = new Int32Array(nb);
  for (let b = 0; b < nb; b++) {
    const lv = head.levels[b];
    const l = Math.min(lv, cap);
    const c = [0, 1, 2].map((a) => (act.includes(a) ? head.locations[3 * b + a] >> (lv - l) : 0));
    const k = key(l, c[0], c[1], c[2]);
    let o = out.get(k);
    if (o === undefined) {
      o = outLevels.length;
      out.set(k, o);
      outLevels.push(l);
      outLoc.push(...c);
    }
    target[b] = o;
  }
  const no = outLevels.length;
  const data: Record<string, Float32Array> = {};
  for (const v of vars) data[v] = new Float32Array(no * cells);
  const merging = cap < fileTop;
  const W = [0, 1, 2].map((a) => bs[a] + (act.includes(a) ? 1 : 0));
  const per = r.hasFaces ? W[0] * W[1] * W[2] : 0;
  const faces = r.hasFaces ? new Float32Array(no * per * 3) : undefined;
  if (r.hasFaces && per !== r.facesPerBlock) throw new Error('the face field does not have the shape the header describes');

  const chunk = Math.max(1, opts.chunk ?? 1024);
  for (let b0 = 0; b0 < nb; b0 += chunk) {
    const b1 = Math.min(nb, b0 + chunk);
    const rho = r.read('rho', b0, b1);
    for (const v of vars) {
      const src = v === 'rho' ? rho : r.read(v, b0, b1);
      const dst = data[v];
      const mass = !byVolume(v);
      for (let b = b0; b < b1; b++) {
        const o = target[b];
        const lv = head.levels[b];
        const s = lv - Math.min(lv, cap);
        const from = (b - b0) * cells;
        if (s === 0) {
          // A block kept as it is: copied (a velocity's sum of rho v is formed below, then divided back).
          if (mass && merging) for (let c = 0; c < cells; c++) dst[o * cells + c] = src[from + c] * rho[from + c];
          else dst.set(src.subarray(from, from + cells), o * cells);
          continue;
        }
        const w = 1 / 2 ** (d * s);
        // Offset of this block's cells within its ancestor, in its own cells.
        const off = [0, 1, 2].map((a) => (act.includes(a) ? (head.locations[3 * b + a] - (outLoc[3 * o + a] << s)) * bs[a] : 0));
        let c = 0;
        for (let k = 0; k < bs[2]; k++) {
          const K = (off[2] + k) >> s;
          for (let j = 0; j < bs[1]; j++) {
            const J = (off[1] + j) >> s;
            const row = o * cells + (K * bs[1] + J) * bs[0];
            for (let i = 0; i < bs[0]; i++, c++) {
              const I = (off[0] + i) >> s;
              dst[row + I] += w * (mass ? src[from + c] * rho[from + c] : src[from + c]);
            }
          }
        }
      }
    }
    if (faces) {
      const f = r.readFaces(b0, b1);
      for (let b = b0; b < b1; b++) {
        const o = target[b];
        const lv = head.levels[b];
        const s = lv - Math.min(lv, cap);
        const from = (b - b0) * per * 3;
        if (s === 0) {
          faces.set(f.subarray(from, from + per * 3), o * per * 3);
          continue;
        }
        // Each fine face on one of the ancestor's planes, by area. A face between two of its
        // fine blocks is taken from the block above it (its lower face), and the ancestor's own
        // upper faces from the blocks below them, so each is counted once.
        const w = 1 / 2 ** ((d - 1) * s);
        const m = (1 << s) - 1;
        const off = [0, 1, 2].map((a) => (act.includes(a) ? (head.locations[3 * b + a] - (outLoc[3 * o + a] << s)) * bs[a] : 0));
        for (const a of act) {
          for (let k = 0; k < W[2]; k++)
            for (let j = 0; j < W[1]; j++)
              for (let i = 0; i < W[0]; i++) {
                const t = [i, j, k];
                // Faces across axis a run to bs along a; the others' upper texels are not faces of a.
                if (act.some((x) => x !== a && t[x] === bs[x])) continue;
                const g = off[a] + t[a];
                if (g & m) continue; // inside a coarse cell
                const G = g >> s;
                if (t[a] === bs[a] && G !== bs[a]) continue; // a fine block's upper face inside the ancestor
                const T = [0, 1, 2].map((x) => (x === a ? G : (off[x] + t[x]) >> s));
                const ft = (k * W[1] + j) * W[0] + i;
                const ct = (T[2] * W[1] + T[1]) * W[0] + T[0];
                faces[(o * per + ct) * 3 + a] += w * f[from + ft * 3 + a];
              }
        }
      }
    }
    opts.progress?.(b1 / nb);
  }
  if (merging) {
    const rho = data.rho;
    for (const v of vars) {
      if (byVolume(v)) continue;
      const a = data[v];
      for (let c = 0; c < a.length; c++) a[c] /= rho[c] || 1;
    }
  }
  return r.finish({ ...head, levels: Int32Array.from(outLevels), locations: Int32Array.from(outLoc), data, faces });
}
