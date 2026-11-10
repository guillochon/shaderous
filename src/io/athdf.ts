// Snapshots as HDF5 files in the layout Athena++ writes (.athdf), which yt, VisIt and others
// read: https://github.com/PrincetonUniversity/athena/wiki/HDF5-Format
//
//   root attributes  NumCycles, Time, Coordinates, RootGridX1..3 (min, max, ratio),
//                    RootGridSize, NumMeshBlocks, MeshBlockSize, MaxLevel,
//                    NumVariables, DatasetNames, VariableNames
//   Levels           int32  [blocks]
//   LogicalLocations int64  [blocks, 3]     position at the block's own level
//   x1f, x2f, x3f    float32 [blocks, n+1]  cell faces;   x1v, x2v, x3v  [blocks, n]  centres
//   prim             float32 [variables, blocks, nx3, nx2, nx1]
//
// A uniform grid is one block. The passive scalar is the variable "r0" of "prim". Files this
// program writes also carry a "Shaderous" attribute (JSON) with the settings of the run.
//
// HDF5 itself comes from h5wasm (the HDF5 library compiled to WebAssembly), loaded on first
// use: it is several megabytes and most visits never need it.
import { blockCells, blockCount, maxLevel, VARIABLES, type Snapshot, type SnapshotMeta, type Variable } from './snapshot';

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
}
interface Library {
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
async function library(h5?: H5): Promise<Library> {
  if (h5) return { File: h5.File, FS: (await h5.ready).FS };
  loaded ??= import('h5wasm').then(async (m) => {
    const lib = (m.default ?? m) as unknown as H5;
    return { File: lib.File, FS: (await lib.ready).FS };
  });
  return loaded;
}

let counter = 0;

/**
 * @param gzip compression level for the cell data, 1 (fastest) to 9; 0 writes it uncompressed.
 *             gzip is HDF5's standard filter, so every reader handles it.
 */
export async function writeAthdf(s: Snapshot, h5?: H5, gzip = 0): Promise<Uint8Array> {
  const lib = await library(h5);
  const name = `write-${counter++}.athdf`;
  const f = new lib.File(name, 'w');
  const nb = blockCount(s);
  const [nx, ny, nz] = s.blockSize;
  const present = VARIABLES.filter((v) => s.data[v]);

  f.create_attribute('NumCycles', s.steps, null, '<i4');
  f.create_attribute('Time', s.time, null, '<f4');
  // Strings are fixed-length byte strings, as Athena++ writes them: yt decodes them as bytes.
  f.create_attribute('Coordinates', 'cartesian', null, 'S9');
  for (let a = 0; a < 3; a++) f.create_attribute(`RootGridX${a + 1}`, new Float32Array([s.lo[a], s.hi[a], 1]), [3], '<f4');
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
    const centres = new Float32Array(nb * n);
    for (let b = 0; b < nb; b++) {
      const dx = (s.hi[a] - s.lo[a]) / (s.rootSize[a] << s.levels[b]);
      const x0 = s.lo[a] + s.locations[3 * b + a] * n * dx;
      for (let i = 0; i <= n; i++) faces[b * (n + 1) + i] = x0 + i * dx;
      for (let i = 0; i < n; i++) centres[b * n + i] = x0 + (i + 0.5) * dx;
    }
    f.create_dataset({ name: `x${a + 1}f`, data: faces, shape: [nb, n + 1], dtype: '<f4' });
    f.create_dataset({ name: `x${a + 1}v`, data: centres, shape: [nb, n], dtype: '<f4' });
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

  f.flush();
  f.close();
  const bytes = lib.FS.readFile(name);
  lib.FS.unlink(name);
  return bytes;
}

export async function readAthdf(bytes: Uint8Array, h5?: H5): Promise<Snapshot> {
  // Every HDF5 file starts with this signature; the library's own complaint is less clear.
  const signature = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 8 || signature.some((b, i) => bytes[i] !== b)) throw new Error('this is not an HDF5 file');
  const lib = await library(h5);
  const name = `read-${counter++}.athdf`;
  lib.FS.writeFile(name, bytes);
  let f: H5File;
  try {
    f = new lib.File(name, 'r');
  } catch {
    lib.FS.unlink(name);
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

    if (strings('Coordinates')[0].toLowerCase() !== 'cartesian') throw new Error(`only Cartesian grids can be read (this file is ${strings('Coordinates')[0]})`);
    const rootSize = numbers('RootGridSize');
    const blockSize = numbers('MeshBlockSize');
    const nb = numbers('NumMeshBlocks')[0];
    const grid = [1, 2, 3].map((a) => numbers(`RootGridX${a}`));
    if (grid.some((g) => Math.abs(g[2] - 1) > 1e-6)) throw new Error('only uniformly spaced grids can be read');

    const snap: Snapshot = {
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
    if (snap.levels.length !== nb || snap.locations.length !== 3 * nb) throw new Error('the block tables do not match NumMeshBlocks');

    // Variables are listed dataset by dataset; find each one we use.
    const names = strings('VariableNames');
    const counts = numbers('NumVariables');
    const sets = strings('DatasetNames');
    const cells = blockCells(snap);
    let first = 0;
    for (let d = 0; d < sets.length; d++) {
      for (let v = 0; v < counts[d]; v++) {
        const variable = names[first + v] as Variable;
        if (!VARIABLES.includes(variable) || snap.data[variable]) continue;
        const set = dataset(sets[d]);
        const got = set.slice([[v, v + 1], [], [], [], []]) as ArrayLike<number>;
        if (got.length !== nb * cells) throw new Error(`"${sets[d]}" does not have the shape the header describes`);
        snap.data[variable] = got instanceof Float32Array ? got : Float32Array.from(got);
      }
      first += counts[d];
    }

    const meta = f.attrs.Shaderous?.value;
    if (typeof meta === 'string') {
      try {
        const parsed = JSON.parse(meta) as SnapshotMeta;
        if (parsed.program === 'shaderous' && typeof parsed.query === 'string') snap.meta = parsed;
      } catch {
        // Not ours after all; read it as any other file.
      }
    }
    return snap;
  } finally {
    f.close();
    lib.FS.unlink(name);
  }
}
