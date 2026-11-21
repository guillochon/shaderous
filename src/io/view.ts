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
 * atlas (render/renderer.ts, rgba16float), a field's faces at one parity, and the species'
 * groups at one parity with the composition's moments.
 */
export function viewBlockBytes(ndim: 2 | 3, mhd: boolean, speciesGroups: number, composition: boolean): number {
  const texel = 3 * 16 + 8 + (mhd ? 16 : 0) + speciesGroups * 16 + (composition ? 16 : 0);
  return PAD ** ndim * texel;
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

/** Blocks of a merged mesh, a batch of them at a time (streamForView), in the program's axis order. */
export interface ViewBatch {
  levels: Int32Array;
  locations: Int32Array;
  /** each variable, blocks in order, each block as [z][y][x] */
  data: Record<string, Float32Array>;
  /** a magnetized run's faces (Snapshot.faces), when asked for and the file has them */
  faces?: Float32Array;
}

/**
 * Read the file's blocks merged to level `cap`, `chunk` blocks at a time, so no copy of the
 * whole file is held: cells finer than `cap` are averaged into the cell of their ancestor that
 * holds them (density, pressure and magnetic field by volume; velocities, the passive scalar
 * and species by mass), and a magnetized run's faces by area, which keeps the field
 * divergence-free. A merged block is handed to `emit` as soon as the last of the file's blocks
 * inside it has been read, `batch` blocks at a time, so only the blocks still being filled are
 * held. `vars` picks the variables (all by default; the density is always read, as the mass
 * weight); `faces: false` leaves the faces out.
 */
export function streamForView(
  r: AthdfReader,
  cap: number,
  opts: { vars?: string[]; faces?: boolean; chunk?: number; batch?: number; progress?: (fraction: number) => void; emit: (b: ViewBatch) => void },
): void {
  const head = r.header();
  const act = activeAxes(head);
  const d = act.length;
  const bs = head.blockSize;
  const nb = r.blocks;
  const cells = r.cells;
  if (!r.variables.includes('rho')) throw new Error('the file has no "rho" variable');
  const vars = ['rho', ...(opts.vars ?? r.variables).filter((v) => v !== 'rho' && r.variables.includes(v))];
  if (act.some((a) => bs[a] !== 16)) throw new Error(`blocks of ${bs.join(' x ')} cells cannot be viewed: the program's are 16 a side`);

  // The merged mesh: one block per distinct (level, position) at or above cap, in file order.
  const out = new Map<number, number>();
  const outLevels: number[] = [];
  const outLoc: number[] = [];
  const remaining: number[] = [];
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
      remaining.push(0);
    }
    remaining[o]++;
    target[b] = o;
  }
  const W = [0, 1, 2].map((a) => bs[a] + (act.includes(a) ? 1 : 0));
  const per = W[0] * W[1] * W[2];
  const withFaces = r.hasFaces && opts.faces !== false;
  if (withFaces && per !== r.facesPerBlock) throw new Error('the face field does not have the shape the header describes');

  // The merged blocks being filled: a block kept as it is (raw) holds the file's values; a
  // merged one sums of value (or of rho value) times its share of the volume.
  const open = new Map<number, { data: Record<string, Float32Array>; faces?: Float32Array; raw: boolean }>();
  const ready: number[] = [];
  const batchSize = Math.max(1, opts.batch ?? 1024);
  const flush = () => {
    if (!ready.length) return;
    const n = ready.length;
    const data: Record<string, Float32Array> = {};
    for (const v of vars) data[v] = new Float32Array(n * cells);
    const faces = withFaces ? new Float32Array(n * per * 3) : undefined;
    const levels = new Int32Array(n);
    const locations = new Int32Array(3 * n);
    ready.forEach((o, i) => {
      const blk = open.get(o)!;
      open.delete(o);
      if (!blk.raw) {
        const rho = blk.data.rho;
        for (const v of vars) {
          if (byVolume(v)) continue;
          const a = blk.data[v];
          for (let c = 0; c < cells; c++) a[c] /= rho[c] || 1;
        }
      }
      for (const v of vars) data[v].set(blk.data[v], i * cells);
      if (faces) faces.set(blk.faces!, i * per * 3);
      levels[i] = outLevels[o];
      locations.set(outLoc.slice(3 * o, 3 * o + 3), 3 * i);
    });
    ready.length = 0;
    const s = r.finish({ ...head, levels, locations, data, faces });
    opts.emit({ levels: s.levels, locations: s.locations, data: s.data as Record<string, Float32Array>, faces: s.faces });
  };

  const chunk = Math.max(1, opts.chunk ?? 1024);
  for (let b0 = 0; b0 < nb; b0 += chunk) {
    const b1 = Math.min(nb, b0 + chunk);
    const src: Record<string, Float32Array> = {};
    for (const v of vars) src[v] = r.read(v, b0, b1);
    const rho = src.rho;
    const f = withFaces ? r.readFaces(b0, b1) : null;
    for (let b = b0; b < b1; b++) {
      const o = target[b];
      const lv = head.levels[b];
      const s = lv - Math.min(lv, cap);
      const from = (b - b0) * cells;
      let blk = open.get(o);
      if (!blk) {
        blk = { data: {}, raw: s === 0, faces: withFaces ? new Float32Array(per * 3) : undefined };
        for (const v of vars) blk.data[v] = s === 0 ? src[v].slice(from, from + cells) : new Float32Array(cells);
        if (s === 0 && f) blk.faces = f.slice((b - b0) * per * 3, (b - b0 + 1) * per * 3);
        open.set(o, blk);
      }
      if (s > 0) {
        const w = 1 / 2 ** (d * s);
        // Offset of this block's cells within its ancestor, in its own cells.
        const off = [0, 1, 2].map((a) => (act.includes(a) ? (head.locations[3 * b + a] - (outLoc[3 * o + a] << s)) * bs[a] : 0));
        for (const v of vars) {
          const dst = blk.data[v];
          const sv = src[v];
          const mass = !byVolume(v);
          let c = 0;
          for (let k = 0; k < bs[2]; k++) {
            const K = (off[2] + k) >> s;
            for (let j = 0; j < bs[1]; j++) {
              const row = (K * bs[1] + ((off[1] + j) >> s)) * bs[0];
              for (let i = 0; i < bs[0]; i++, c++) dst[row + ((off[0] + i) >> s)] += w * (mass ? sv[from + c] * rho[from + c] : sv[from + c]);
            }
          }
        }
        if (f) {
          // Each fine face on one of the ancestor's planes, by area. A face between two of its
          // fine blocks is taken from the block above it (its lower face), and the ancestor's
          // own upper faces from the blocks below them, so each is counted once.
          const wf = 1 / 2 ** ((d - 1) * s);
          const m = (1 << s) - 1;
          const ff = (b - b0) * per * 3;
          const faces = blk.faces!;
          for (const a of act) {
            // Faces across axis a run to bs along a (W texels); along the others, to bs - 1.
            const n = [0, 1, 2].map((x) => (x === a ? W[x] : bs[x]));
            for (let k = 0; k < n[2]; k++) {
              const gk = off[2] + k;
              if (a === 2 && (gk & m || (k === bs[2] && gk >> s !== bs[2]))) continue;
              const K = gk >> s;
              for (let j = 0; j < n[1]; j++) {
                const gj = off[1] + j;
                if (a === 1 && (gj & m || (j === bs[1] && gj >> s !== bs[1]))) continue;
                const J = gj >> s;
                const fine = ff + (k * W[1] + j) * W[0] * 3 + a;
                const coarse = (K * W[1] + J) * W[0] * 3 + a;
                for (let i = 0; i < n[0]; i++) {
                  const gi = off[0] + i;
                  // (Inside a coarse cell, or a fine block's upper face inside the ancestor: not one of its faces.)
                  if (a === 0 && (gi & m || (i === bs[0] && gi >> s !== bs[0]))) continue;
                  faces[coarse + (gi >> s) * 3] += wf * f[fine + i * 3];
                }
              }
            }
          }
        }
      }
      if (--remaining[o] === 0) {
        ready.push(o);
        if (ready.length >= batchSize) flush();
      }
    }
    opts.progress?.(b1 / nb);
  }
  flush();
}

/**
 * The file's header with the block tables of its mesh merged to level `cap` (streamForView's
 * blocks, in the program's axis order), and no data: what a view that holds one field is built
 * from (Snapshot.view baked), its cells baked from the file afterwards.
 */
export function tablesAtCap(r: AthdfReader, cap: number): Snapshot {
  const head = r.header();
  const act = activeAxes(head);
  const seen = new Set<number>();
  const levels: number[] = [];
  const locations: number[] = [];
  for (let b = 0; b < head.levels.length; b++) {
    const lv = head.levels[b];
    const l = Math.min(lv, cap);
    const c = [0, 1, 2].map((a) => (act.includes(a) ? head.locations[3 * b + a] >> (lv - l) : 0));
    const k = key(l, c[0], c[1], c[2]);
    if (seen.has(k)) continue;
    seen.add(k);
    levels.push(l);
    locations.push(...c);
  }
  return r.finish({ ...head, levels: Int32Array.from(levels), locations: Int32Array.from(locations), data: {} });
}

/**
 * The whole merged mesh as one snapshot (streamForView's batches together), in the program's
 * axis order. `wanted` picks the variables (all by default; the five primitives are always read).
 */
export function readForView(
  r: AthdfReader,
  cap: number,
  opts: { chunk?: number; wanted?: (name: string) => boolean; progress?: (fraction: number) => void } = {},
): Snapshot {
  const required = ['rho', 'press', 'vel1', 'vel2', 'vel3'];
  for (const v of required) if (!r.variables.includes(v)) throw new Error(`the file has no "${v}" variable`);
  const vars = r.variables.filter((v) => required.includes(v) || (opts.wanted?.(v) ?? true));
  const batches: ViewBatch[] = [];
  streamForView(r, cap, { vars, chunk: opts.chunk, progress: opts.progress, emit: (b) => batches.push(b) });
  const cat = <T extends Float32Array | Int32Array>(parts: T[], make: (n: number) => T): T => {
    const outArr = make(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
      outArr.set(p as never, o);
      o += p.length;
    }
    return outArr;
  };
  const head = r.finish(r.header());
  const data: Record<string, Float32Array> = {};
  for (const v of Object.keys(batches[0]?.data ?? {})) data[v] = cat(batches.map((b) => b.data[v]), (n) => new Float32Array(n));
  return {
    ...head,
    levels: cat(batches.map((b) => b.levels), (n) => new Int32Array(n)),
    locations: cat(batches.map((b) => b.locations), (n) => new Int32Array(n)),
    data,
    faces: r.hasFaces ? cat(batches.map((b) => b.faces!), (n) => new Float32Array(n)) : undefined,
  };
}

/**
 * A uniform grid's file (one block) read as blocks of 16 cells, all at one level `m` below a
 * root grid of N / 2^m cells (the most levels that keep the root a multiple of 16), so the same
 * merging that coarsens an adaptive mesh's levels downsamples it by powers of two. Blocks run
 * along x fastest, then y, then z, so a range of them is read as slabs of 16 rows. A magnetized
 * run's faces (Snapshot.faces) are put together from its lower faces Bf1-3 and the upper
 * boundary planes in its metadata (or, along a periodic axis, the lower faces opposite).
 */
export function blocksOfUniform(r: AthdfReader): AthdfReader {
  const file = r.header();
  if (file.levels.length !== 1) return r;
  const N = file.blockSize;
  const act = [0, 1, 2].filter((a) => N[a] > 1);
  if (act.some((a) => N[a] % 16 !== 0)) throw new Error(`a uniform grid of ${N.join(' x ')} cells cannot be viewed: each side must be a multiple of 16`);
  let m = 0;
  while (act.every((a) => (N[a] >> (m + 1)) % 16 === 0 && N[a] % (16 << (m + 1)) === 0)) m++;
  const nbA = [0, 1, 2].map((a) => (act.includes(a) ? N[a] / 16 : 1));
  const blocks = nbA[0] * nbA[1] * nbA[2];
  const bsz = [0, 1, 2].map((a) => (act.includes(a) ? 16 : 1));
  const cells = bsz[0] * bsz[1] * bsz[2];
  const locations = new Int32Array(3 * blocks);
  for (let b = 0; b < blocks; b++) locations.set([b % nbA[0], Math.floor(b / nbA[0]) % nbA[1], Math.floor(b / (nbA[0] * nbA[1]))], 3 * b);
  const head: Snapshot = { ...file, rootSize: [0, 1, 2].map((a) => (act.includes(a) ? N[a] >> m : 1)), blockSize: bsz, levels: new Int32Array(blocks).fill(m), locations, data: {} };
  const bf = ['Bf1', 'Bf2', 'Bf3'];
  const hasFaces = act.every((a) => r.variables.includes(bf[a]));
  const W = bsz.map((s, a) => s + (act.includes(a) ? 1 : 0));
  const per = W[0] * W[1] * W[2];
  const upper = file.meta?.mhd?.upper;
  // The rows (y, z) of blocks a range covers, each read as one slab of the file.
  const rows = (b0: number, b1: number) => {
    const out: { y: number; z: number; x0: number; x1: number; first: number }[] = [];
    for (let b = b0; b < b1; ) {
      const x = b % nbA[0];
      const y = Math.floor(b / nbA[0]) % nbA[1];
      const z = Math.floor(b / (nbA[0] * nbA[1]));
      const x1 = Math.min(nbA[0], x + (b1 - b));
      out.push({ y, z, x0: x, x1, first: b });
      b += x1 - x;
    }
    return out;
  };
  // A slab [z0, z1) x [y0, y1) x all x of a variable, as [z][y][x].
  const slab = (name: string, z0: number, z1: number, y0: number, y1: number) => r.readSlab(name, 0, [z0, z1], [y0, y1], [0, N[0]]);
  return {
    blocks,
    cells,
    variables: r.variables.filter((v) => !/^Bf\d$/.test(v)),
    hasFaces,
    facesPerBlock: hasFaces ? per : 0,
    header: () => ({ ...head, levels: head.levels.slice(), locations: head.locations.slice(), data: {} }),
    read(name, b0, b1) {
      const outArr = new Float32Array((b1 - b0) * cells);
      for (const row of rows(b0, b1)) {
        const z0 = row.z * bsz[2];
        const y0 = row.y * bsz[1];
        const s = slab(name, z0, z0 + bsz[2], y0, y0 + bsz[1]);
        for (let bx = row.x0; bx < row.x1; bx++) {
          let o = (row.first + bx - row.x0 - b0) * cells;
          for (let k = 0; k < bsz[2]; k++)
            for (let j = 0; j < bsz[1]; j++) {
              const at = (k * bsz[1] + j) * N[0] + bx * 16;
              outArr.set(s.subarray(at, at + bsz[0]), o);
              o += bsz[0];
            }
        }
      }
      return outArr;
    },
    readSlab: (name, block, z, y, x) => r.readSlab(name, block, z, y, x),
    readFaces(b0, b1) {
      const outArr = new Float32Array((b1 - b0) * per * 3);
      for (const row of rows(b0, b1)) {
        const z0 = row.z * bsz[2];
        const y0 = row.y * bsz[1];
        // One cell more along each axis in use (the upper faces), within the grid.
        const z1 = Math.min(N[2], z0 + W[2]);
        const y1 = Math.min(N[1], y0 + W[1]);
        const comps = act.map((a) => ({ a, s: slab(bf[a], z0, z1, y0, y1) }));
        // Along a periodic axis the grid's upper faces are its lower ones: the first layer.
        const wrap = act.map((a) => (upper?.[a]?.length ? null : a === 2 ? slab(bf[a], 0, 1, y0, y1) : a === 1 ? slab(bf[a], z0, z1, 0, 1) : null));
        const ny = y1 - y0;
        for (let bx = row.x0; bx < row.x1; bx++) {
          const o = (row.first + bx - row.x0 - b0) * per * 3;
          for (let k = 0; k < W[2]; k++)
            for (let j = 0; j < W[1]; j++)
              for (let i = 0; i < W[0]; i++) {
                const g = [bx * 16 + i, y0 + j, z0 + k];
                const t = [i, j, k];
                for (const { a, s } of comps) {
                  // Faces across a reach index bs along a; the other axes' upper texels are not faces of a.
                  if (act.some((x) => x !== a && t[x] === bsz[x])) continue;
                  let v: number;
                  if (g[a] < N[a]) v = s[((g[2] - z0) * ny + (g[1] - y0)) * N[0] + g[0]];
                  else if (upper?.[a]?.length) {
                    const [p, q, l1] = a === 0 ? [g[1], g[2], N[1]] : a === 1 ? [g[0], g[2], N[0]] : [g[0], g[1], N[0]];
                    v = upper[a][q * l1 + p];
                  } else if (a === 0) v = s[((g[2] - z0) * ny + (g[1] - y0)) * N[0]];
                  else if (a === 1) v = wrap[act.indexOf(a)]![(g[2] - z0) * N[0] + g[0]];
                  else v = wrap[act.indexOf(a)]![(g[1] - y0) * N[0] + g[0]];
                  outArr[o + ((k * W[1] + j) * W[0] + i) * 3 + a] = v;
                }
              }
        }
      }
      return outArr;
    },
    finish: (s) => r.finish(s),
    close: () => r.close(),
  };
}

/**
 * The cells of the file under `points` (fractions of the box, in the program's axis order), at
 * the file's own resolution: each point's block read from the file. For probes of a view that
 * holds one field (baked) or a merged mesh.
 */
export function probeFile(r: AthdfReader, points: number[][]): { at: number[]; level: number; values: Record<string, number> }[] {
  const raw = r.header();
  const head = r.finish(r.header());
  const flat = head.ndim === 2;
  const top = maxLevel(head);
  const rootBlocks = head.rootSize.map((n, a) => n / head.blockSize[a]);
  const index = new Map<number, number>();
  for (let b = 0; b < head.levels.length; b++) index.set(key(head.levels[b], head.locations[3 * b], head.locations[3 * b + 1], flat ? 0 : head.locations[3 * b + 2]), b);
  // (A cylindrical file's own order is (R, phi, z), io/athdf.ts swapAxes23: its cells are the same in memory.)
  return points.map((p) => {
    const f = [0, 1, 2].map((a) => (flat && a === 2 ? 0 : Math.min(Math.max(p[a] ?? 0, 0), 1 - 1e-7)));
    for (let l = top; l >= 0; l--) {
      const at = [0, 1, 2].map((a) => Math.min(Math.floor(f[a] * (rootBlocks[a] << l)), (rootBlocks[a] << l) - 1));
      const b = index.get(key(l, at[0], at[1], flat ? 0 : at[2]));
      if (b === undefined) continue;
      let cell = 0;
      let stride = 1;
      for (let a = 0; a < 3; a++) {
        const n = head.blockSize[a];
        const g = Math.min(Math.floor(f[a] * (head.rootSize[a] << l)), (head.rootSize[a] << l) - 1);
        cell += Math.min(Math.max(g - head.locations[3 * b + a] * n, 0), n - 1) * stride;
        stride *= n;
      }
      const data: Record<string, Float32Array> = {};
      for (const v of r.variables) data[v] = r.read(v, b, b + 1);
      // (finish swaps a cylindrical file's second and third velocities into the program's order.)
      const s = r.finish({ ...raw, levels: Int32Array.of(raw.levels[b]), locations: raw.locations.slice(3 * b, 3 * b + 3), data });
      const values: Record<string, number> = {};
      for (const v of r.variables) values[v] = s.data[v]![cell];
      return { at: p, level: l, values };
    }
    return { at: p, level: -1, values: {} };
  });
}

/** The file's variables a field baked from it reads (AmrSim.bake): the primitives, and what the field needs besides. */
export function bakeVariables(field: number, species: string | null): string[] {
  const vars = ['rho', 'press', 'vel1', 'vel2', 'vel3'];
  if (field === 2) vars.push('r0');
  if (field >= 22 && field < 30) vars.push('Bcc1', 'Bcc2', 'Bcc3');
  if (field === 19 && species) vars.push(species);
  return vars;
}

/**
 * GPU bytes per atlas block of a view that holds one field (AmrSim with viewBaked): the field and
 * the density at each cell, no ghost cells (rg32float), and the display's atlas the ray-marcher
 * reads in 3D (render/renderer.ts, rgba16float, with ghost cells).
 */
export const viewBakedBlockBytes = (ndim: 2 | 3): number => BS ** ndim * 8 + (ndim === 3 ? PAD ** 3 * 8 : 0);

/** GPU bytes per atlas block of a magnetized run's face field (one rgba32float atlas), which field lines trace. */
export const viewFacesBlockBytes = (ndim: 2 | 3): number => PAD ** ndim * 16;
