// Snapshots as HDF5 files in the layout Athena++ writes (.athdf), which yt, VisIt and others
// read: https://github.com/PrincetonUniversity/athena/wiki/HDF5-Format
//
//   root attributes  NumCycles, Time, Coordinates, RootGridX1..3 (min, max, ratio),
//                    RootGridSize, NumMeshBlocks, MeshBlockSize, MaxLevel,
//                    NumVariables, DatasetNames, VariableNames
//   Levels           int32  [blocks]
//   LogicalLocations int64  [blocks, 3]     position at the block's own level
//   x1f, x2f, x3f    float32 [blocks, n+1]  cell faces;   x1v, x2v, x3v  [blocks, n]  centers
//   prim             float32 [variables, blocks, nx3, nx2, nx1]
//
// A uniform grid is one block. The passive scalar is the variable "r0" of "prim", and each
// species' mass fraction the variable named after the species ("he4", "c12"). Files this
// program writes also carry a "Shaderous" attribute (JSON) with the settings of the run.
//
// HDF5 itself comes from h5wasm (the HDF5 library compiled to WebAssembly), loaded on first
// use: it is several megabytes and most visits never need it.
import { blockCells, blockCount, maxLevel, VARIABLES, type Snapshot, type SnapshotMeta } from './snapshot';

// The part of h5wasm used here, so a caller (the unit tests, under Node) can supply its build.
export interface H5 {
  /** resolves to the module, whose in-memory file system holds the files being read or written */
  ready: Promise<{ FS: H5Files }>;
  File: new (name: string, mode: string) => H5File;
}
interface H5Files {
  writeFile(path: string, data: Uint8Array): void;
  readFile(path: string): Uint8Array;
  unlink(path: string): void;
  // Emscripten's: the web build mounts a File without copying it, in a worker (io/view_worker.ts).
  mkdir?(path: string): void;
  mount?(type: unknown, opts: unknown, path: string): void;
  unmount?(path: string): void;
  filesystems?: Record<string, unknown>;
}
export interface Library {
  File: H5['File'];
  FS: H5Files;
}
interface H5Attr {
  value: unknown;
  json_value?: unknown;
}
interface H5Dataset {
  shape: number[] | null;
  value: unknown;
  slice(ranges: (number | null)[][]): unknown;
  write_slice(ranges: (number | null)[][], data: unknown): void;
  resize(shape: number[]): number;
}
interface H5File {
  attrs: Record<string, H5Attr>;
  get(name: string): unknown;
  keys(): string[];
  create_attribute(name: string, data: unknown, shape?: number[] | null, dtype?: string | null): void;
  create_dataset(args: Record<string, unknown>): H5Dataset;
  flush(): void;
  close(): unknown;
}

let loaded: Promise<Library> | null = null;
export async function library(h5?: H5): Promise<Library> {
  if (h5) return { File: h5.File, FS: (await h5.ready).FS };
  loaded ??= import('h5wasm').then(async (m) => {
    const lib = (m.default ?? m) as unknown as H5;
    return { File: lib.File, FS: (await lib.ready).FS };
  });
  return loaded;
}

/**
 * Load the HDF5 library ahead of the first file. The page does this soon after it starts: a
 * page left open across a new release can no longer fetch its own build's files, and a state
 * that could not be saved for that reason is a run lost.
 */
export const warm = (): Promise<unknown> => library();

let counter = 0;

/**
 * An axisymmetric snapshot (x = R, y = z, v_z = v_phi) relabeled into Athena++'s cylindrical
 * order (R, phi, z), one cell in phi spanning 2 pi, and back: the cells keep their memory order
 * (a block's [1][nz][nR] is its [nz][1][nR]), and only the axes' names, the second and third
 * velocity and field components and the block tables move.
 */
function swapAxes23(s: Snapshot, toFile: boolean): Snapshot {
  const at = (v: number[], phi: number) => (toFile ? [v[0], phi, v[1]] : [v[0], v[2], phi]);
  const loc = new Int32Array(s.locations.length);
  for (let b = 0; b < s.locations.length / 3; b++) {
    const l = [s.locations[3 * b], s.locations[3 * b + 1], s.locations[3 * b + 2]];
    loc.set(toFile ? [l[0], 0, l[1]] : [l[0], l[2], 0], 3 * b);
  }
  const data = { ...s.data };
  [data.vel2, data.vel3] = [s.data.vel3, s.data.vel2];
  return {
    ...s,
    rootSize: at(s.rootSize, 1),
    blockSize: at(s.blockSize, 1),
    lo: at(s.lo, 0),
    hi: at(s.hi, toFile ? 2 * Math.PI : 0),
    locations: loc,
    data,
  };
}

/**
 * @param gzip compression level for the cell data, 1 (fastest) to 9; 0 writes it uncompressed.
 *             gzip is HDF5's standard filter, so every reader handles it.
 */
export async function writeAthdf(s0: Snapshot, h5?: H5, gzip = 0): Promise<Uint8Array> {
  const lib = await library(h5);
  const name = `write-${counter++}.athdf`;
  writeInto(lib, name, s0, gzip);
  const bytes = lib.FS.readFile(name);
  lib.FS.unlink(name);
  return bytes;
}

/**
 * The same, written straight to `file` by a build of HDF5 whose files are the computer's own
 * (h5wasm/node), so no copy of the file is ever held in memory: the runner writes a checkpoint
 * of gigabytes so.
 */
export async function writeAthdfFile(file: string, s: Snapshot, h5: H5, gzip = 0): Promise<void> {
  writeInto(await library(h5), file, s, gzip);
}

/** Blocks of the face field written or read at a time, so HDF5's memory holds a part of it only. */
const FACE_SLAB = 2048;

function writeInto(lib: Library, name: string, s0: Snapshot, gzip: number): void {
  const cyl = s0.geometry === 'cylindrical';
  const s = cyl ? swapAxes23(s0, true) : s0;
  const f = new lib.File(name, 'w');
  const nb = blockCount(s);
  const [nx, ny, nz] = s.blockSize;
  // The primitives first, in their order, then the species.
  const present = [...VARIABLES.filter((v) => s.data[v]), ...Object.keys(s.data).filter((k) => !(VARIABLES as readonly string[]).includes(k) && s.data[k])];

  f.create_attribute('NumCycles', s.steps, null, '<i4');
  f.create_attribute('Time', s.time, null, '<f4');
  // Strings are fixed-length byte strings, as Athena++ writes them: yt decodes them as bytes.
  // (A spherical run's axes are Athena++'s spherical_polar order (r, theta, phi) as they are.)
  const sph = s0.geometry === 'spherical';
  const coords = sph ? 'spherical_polar' : cyl ? 'cylindrical' : 'cartesian';
  f.create_attribute('Coordinates', coords, null, `S${coords.length}`);
  for (let a = 0; a < 3; a++) f.create_attribute(`RootGridX${a + 1}`, new Float32Array([s.lo[a], s.hi[a], a === 0 && sph ? (s.x1rat ?? 1) : 1]), [3], '<f4');
  f.create_attribute('RootGridSize', new Int32Array(s.rootSize), [3], '<i4');
  f.create_attribute('NumMeshBlocks', nb, null, '<i4');
  f.create_attribute('MeshBlockSize', new Int32Array(s.blockSize), [3], '<i4');
  f.create_attribute('MaxLevel', maxLevel(s), null, '<i4');
  f.create_attribute('NumVariables', new Int32Array([present.length]), [1], '<i4');
  f.create_attribute('DatasetNames', ['prim'], [1], 'S21');
  f.create_attribute('VariableNames', present as unknown as string[], [present.length], 'S21');
  if (s.meta) f.create_attribute('Shaderous', JSON.stringify(s.meta));

  f.create_dataset({ name: 'Levels', data: s.levels, shape: [nb], dtype: '<i4' });
  f.create_dataset({ name: 'LogicalLocations', data: BigInt64Array.from(s.locations, (v) => BigInt(v)), shape: [nb, 3], dtype: '<q' });
  for (let a = 0; a < 3; a++) {
    const n = s.blockSize[a];
    const faces = new Float32Array(nb * (n + 1));
    const centers = new Float32Array(nb * n);
    for (let b = 0; b < nb; b++) {
      const dx = (s.hi[a] - s.lo[a]) / (s.rootSize[a] << s.levels[b]);
      const x0 = s.lo[a] + s.locations[3 * b + a] * n * dx;
      for (let i = 0; i <= n; i++) faces[b * (n + 1) + i] = x0 + i * dx;
      for (let i = 0; i < n; i++) centers[b * n + i] = x0 + (i + 0.5) * dx;
      if (sph && a < 2) {
        // Spherical: r's faces from x1rat (a uniform grid: every block at level 0), and the volume-weighted
        // centers Athena++ writes for r and theta.
        const rat = s.x1rat ?? 1;
        const ff = (i: number) => (a === 1 || rat === 1 ? x0 + i * dx : s.lo[0] * rat ** (s.locations[3 * b] * n + i));
        for (let i = 0; i <= n; i++) faces[b * (n + 1) + i] = ff(i);
        for (let i = 0; i < n; i++) {
          const [l, h] = [ff(i), ff(i + 1)];
          centers[b * n + i] = a === 0 ? (0.75 * (h ** 4 - l ** 4)) / (h ** 3 - l ** 3) : (Math.sin(h) - h * Math.cos(h) - Math.sin(l) + l * Math.cos(l)) / (Math.cos(l) - Math.cos(h));
        }
      }
    }
    f.create_dataset({ name: `x${a + 1}f`, data: faces, shape: [nb, n + 1], dtype: '<f4' });
    f.create_dataset({ name: `x${a + 1}v`, data: centers, shape: [nb, n], dtype: '<f4' });
  }

  // The cell data is written one variable at a time, so no second copy of all of it is needed:
  // the dataset starts with the first variable and grows along its first axis.
  const perVar = [nb, nz, ny, nx];
  const chunkBlocks = Math.max(1, Math.min(nb, Math.floor((1 << 22) / blockCells(s))));
  const prim = f.create_dataset({
    name: 'prim',
    data: s.data[present[0]]!,
    shape: [1, ...perVar],
    maxshape: [present.length, ...perVar],
    chunks: [1, chunkBlocks, nz, ny, nx],
    dtype: '<f4',
    ...(gzip > 0 ? { compression: 'gzip', compression_opts: gzip } : {}),
  });
  if (present.length > 1) prim.resize([present.length, ...perVar]);
  for (let v = 1; v < present.length; v++) prim.write_slice([[v, v + 1], [], [], [], []], s.data[present[v]]!);
  if (s.faces) {
    // A magnetized run on blocks: the face field (Snapshot.faces), for a restart.
    // Written a slab of blocks at a time: at tens of thousands of blocks the whole of it is
    // gigabytes, and HDF5's memory (WebAssembly's) holds at most 4 GB.
    const shape = [nb, nz === 1 ? 1 : nz + 1, ny + 1, nx + 1, 3];
    const per = shape[1] * shape[2] * shape[3] * 3;
    const first = Math.min(nb, FACE_SLAB);
    const set = f.create_dataset({
      name: 'ShaderousFaces',
      data: s.faces.subarray(0, first * per),
      shape: [first, ...shape.slice(1)],
      maxshape: shape,
      chunks: [1, ...shape.slice(1)],
      dtype: '<f4',
      ...(gzip > 0 ? { compression: 'gzip', compression_opts: gzip } : {}),
    });
    if (nb > first) set.resize(shape);
    for (let b = first; b < nb; b += FACE_SLAB) {
      const e = Math.min(nb, b + FACE_SLAB);
      set.write_slice([[b, e], [], [], [], []], s.faces.subarray(b * per, e * per));
    }
  }

  f.flush();
  f.close();
}

export async function readAthdf(bytes: Uint8Array, h5?: H5): Promise<Snapshot> {
  // Every HDF5 file starts with this signature; the library's own complaint is less clear.
  const signature = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 8 || signature.some((b, i) => bytes[i] !== b)) throw new Error('this is not an HDF5 file');
  const lib = await library(h5);
  const name = `read-${counter++}.athdf`;
  lib.FS.writeFile(name, bytes);
  try {
    return readFrom(lib, name);
  } finally {
    lib.FS.unlink(name);
  }
}

/** The same, read straight from `file` by a build of HDF5 whose files are the computer's own (h5wasm/node). */
export async function readAthdfFile(file: string, h5: H5): Promise<Snapshot> {
  return readFrom(await library(h5), file);
}

function readFrom(lib: Library, name: string): Snapshot {
  const r = openAthdf(lib, name);
  try {
    const snap = r.header();
    for (const v of r.variables) snap.data[v] = r.read(v, 0, r.blocks);
    if (r.hasFaces) snap.faces = r.readFaces(0, r.blocks);
    return r.finish(snap);
  } finally {
    r.close();
  }
}

/**
 * An .athdf file open for reading in parts (io/view.ts reads one too large to hold): its header
 * as a snapshot without data, and any variable or the face field over a range of blocks.
 */
export interface AthdfReader {
  /** blocks in the file */
  readonly blocks: number;
  /** cells per block */
  readonly cells: number;
  /** every variable of the file, the primitives and the species, by name */
  readonly variables: string[];
  /** the file has a magnetized run's faces (Snapshot.faces) */
  readonly hasFaces: boolean;
  /** texels of faces per block, when it has them */
  readonly facesPerBlock: number;
  /** a new snapshot of the header (no data, in the file's own axis order) */
  header(): Snapshot;
  /** variable `name` for blocks b0 to b1 (exclusive), each block as [z][y][x] */
  read(name: string, b0: number, b1: number): Float32Array;
  /** the faces of blocks b0 to b1, three per texel */
  readFaces(b0: number, b1: number): Float32Array;
  /** turn a snapshot read from this file into the program's axis order (a cylindrical file's) */
  finish(s: Snapshot): Snapshot;
  close(): void;
}

/** Open an .athdf file in the library's file system for reading in parts (readFrom reads it whole). */
export function openAthdf(lib: Library, name: string): AthdfReader {
  let f: H5File;
  try {
    f = new lib.File(name, 'r');
  } catch {
    throw new Error('this is not an HDF5 file');
  }
  try {
    const attr = (key: string): unknown => {
      const a = f.attrs[key];
      if (!a) throw new Error(`not an Athena++ HDF5 file: it has no "${key}" attribute`);
      return a.value;
    };
    const numbers = (key: string): number[] => {
      const v = attr(key);
      return typeof v === 'number' || typeof v === 'bigint' ? [Number(v)] : Array.from(v as ArrayLike<number | bigint>, Number);
    };
    const strings = (key: string): string[] => {
      const v = attr(key);
      return (Array.isArray(v) ? v : [v]).map((x) => String(x).replace(/\0+$/, '').trim());
    };
    const dataset = (key: string): H5Dataset => {
      const d = f.get(key) as H5Dataset | null;
      if (!d || !d.shape) throw new Error(`not an Athena++ HDF5 file: it has no "${key}" dataset`);
      return d;
    };

    const coords = strings('Coordinates')[0].toLowerCase();
    // Cylindrical files are read when they are axisymmetric: one cell in phi.
    const cyl = coords === 'cylindrical' && numbers('RootGridSize')[1] === 1;
    if (coords !== 'cartesian' && !cyl) throw new Error(`only Cartesian grids, and axisymmetric cylindrical ones, can be read (this file is ${strings('Coordinates')[0]})`);
    const rootSize = numbers('RootGridSize');
    const blockSize = numbers('MeshBlockSize');
    const nb = numbers('NumMeshBlocks')[0];
    const grid = [1, 2, 3].map((a) => numbers(`RootGridX${a}`));
    if (grid.some((g) => Math.abs(g[2] - 1) > 1e-6)) throw new Error('only uniformly spaced grids can be read');

    const head: Snapshot = {
      ndim: rootSize[2] === 1 ? 2 : 3,
      rootSize,
      blockSize,
      lo: grid.map((g) => g[0]),
      hi: grid.map((g) => g[1]),
      time: numbers('Time')[0],
      steps: f.attrs.NumCycles ? numbers('NumCycles')[0] : 0,
      levels: Int32Array.from(dataset('Levels').value as ArrayLike<number>),
      locations: Int32Array.from(dataset('LogicalLocations').value as ArrayLike<number | bigint>, Number),
      data: {},
    };
    if (head.levels.length !== nb || head.locations.length !== 3 * nb) throw new Error('the block tables do not match NumMeshBlocks');
    const meta = f.attrs.Shaderous?.value;
    if (typeof meta === 'string') {
      try {
        const parsed = JSON.parse(meta) as SnapshotMeta;
        if (parsed.program === 'shaderous' && typeof parsed.query === 'string') head.meta = parsed;
      } catch {
        // Not ours after all; read it as any other file.
      }
    }

    // Variables are listed dataset by dataset; find each one we use (the first of a name).
    const names = strings('VariableNames');
    const counts = numbers('NumVariables');
    const sets = strings('DatasetNames');
    const cells = blockCells(head);
    const where = new Map<string, { set: H5Dataset; index: number }>();
    let first = 0;
    for (let d = 0; d < sets.length; d++) {
      for (let v = 0; v < counts[d]; v++) {
        const variable = names[first + v];
        if (!variable || where.has(variable)) continue;
        const set = dataset(sets[d]);
        if (set.shape!.length !== 5 || set.shape![1] !== nb || set.shape![2] * set.shape![3] * set.shape![4] !== cells) throw new Error(`"${sets[d]}" does not have the shape the header describes`);
        where.set(variable, { set, index: v });
      }
      first += counts[d];
    }
    const faces = f.get('ShaderousFaces') as H5Dataset | null;
    const facesPerBlock = faces?.shape ? faces.shape.slice(1, 4).reduce((a, b) => a * b, 1) : 0;
    const asF32 = (got: unknown) => (got instanceof Float32Array ? got : Float32Array.from(got as ArrayLike<number>));
    return {
      blocks: nb,
      cells,
      variables: [...where.keys()],
      hasFaces: facesPerBlock > 0,
      facesPerBlock,
      header: () => ({ ...head, levels: head.levels.slice(), locations: head.locations.slice(), data: {} }),
      read(name, b0, b1) {
        const w = where.get(name);
        if (!w) throw new Error(`the file has no variable "${name}"`);
        const got = asF32(w.set.slice([[w.index, w.index + 1], [b0, b1], [], [], []]));
        if (got.length !== (b1 - b0) * cells) throw new Error(`"${name}" does not have the shape the header describes`);
        return got;
      },
      readFaces(b0, b1) {
        // A slab of blocks at a time, as it is written.
        const out = new Float32Array((b1 - b0) * facesPerBlock * 3);
        for (let b = b0; b < b1; b += FACE_SLAB) {
          const e = Math.min(b1, b + FACE_SLAB);
          out.set(faces!.slice([[b, e], [], [], [], []]) as ArrayLike<number>, (b - b0) * facesPerBlock * 3);
        }
        return out;
      },
      finish: (s) => (cyl ? { ...swapAxes23(s, false), ndim: 2, geometry: 'cylindrical' } : s),
      close: () => void f.close(),
    };
  } catch (e) {
    f.close();
    throw e;
  }
}
