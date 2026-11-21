// Reads a checkpoint for view-only loading (io/view.ts) off the page's thread. The file is
// mounted with Emscripten's WORKERFS, which reads it with slice() as HDF5 asks: a File in place,
// or a URL through a stand-in whose slice() makes range requests (urlFile). Either way no copy of
// the file is made, so files larger than WebAssembly's memory, or than the browser would hold as
// a download, open. A uniform grid's file is read as
// blocks of 16 (blocksOfUniform). Messages (io/view_open.ts):
//   in  { open: File | url }                    out { header, variables, hasFaces }
//   in  { read: level }                         out { progress } ..., then { snapshot }
//   in  { stream: level, vars, faces, batch }   out { progress } and { batch } ..., then { done }
//   in  { probe: points }                       out { probes }
//   in  { tables: level }                       out { snapshot } (block tables only: tablesAtCap)
//   errors come back as { error }.
import { library, openAthdf, type AthdfReader, type Library } from './athdf';
import { blocksOfUniform, probeFile, readForView, streamForView, tablesAtCap, type ViewBatch } from './view';
import type { Snapshot } from './snapshot';

let lib: Library | null = null;
let reader: AthdfReader | null = null;
const MOUNT = '/view';

function release() {
  reader?.close();
  reader = null;
  try {
    lib?.FS.unmount?.(MOUNT);
  } catch {
    // not mounted
  }
}

/** A read of a URL's file (urlFile): what WORKERFS's reader is handed in place of a Blob. */
type Bytes = { size: number; bytes: ArrayBuffer };

/**
 * WORKERFS reads a File's slice() with a FileReaderSync: in its place, one that also takes the
 * bytes urlFile's slice() gives.
 */
function bytesReader(fs: Library['FS']) {
  const w = fs.filesystems!.WORKERFS as { reader?: unknown };
  // (A worker's own, which the page's typings do not declare.)
  const real = new (globalThis as unknown as { FileReaderSync: new () => { readAsArrayBuffer(b: Blob): ArrayBuffer } }).FileReaderSync();
  w.reader = { readAsArrayBuffer: (c: Blob | Bytes) => (c instanceof Blob ? real.readAsArrayBuffer(c) : c.bytes) };
}

/**
 * A file at a URL as WORKERFS takes a File: its size, and slice() answered by synchronous range
 * requests (a worker may make them), a megabyte at a time at least and the last 64 MB kept, so
 * HDF5's many small reads of its metadata do not each go to the server.
 */
function urlFile(url: string, name: string) {
  const BLOCK = 1 << 20;
  const KEEP = 64;
  const head = new XMLHttpRequest();
  head.open('HEAD', url, false);
  head.send();
  if (head.status < 200 || head.status >= 300) throw new Error(`${url}: ${head.status} ${head.statusText}`);
  const size = Number(head.getResponseHeader('content-length'));
  if (!(size > 0)) throw new Error(`${url}: the server gives no length for it`);
  const cache = new Map<number, Uint8Array>();
  // Blocks b0 to b1 (exclusive) in one request.
  const fetchBlocks = (b0: number, b1: number): Uint8Array[] => {
    const start = b0 * BLOCK;
    const end = Math.min(size, b1 * BLOCK) - 1;
    const x = new XMLHttpRequest();
    x.open('GET', url, false);
    x.responseType = 'arraybuffer';
    x.setRequestHeader('Range', `bytes=${start}-${end}`);
    try {
      x.send();
    } catch (e) {
      throw new Error(`${url}: bytes ${start} to ${end} could not be read (${e instanceof Error ? e.message : String(e)})`);
    }
    if (x.status !== 206 && !(x.status === 200 && start === 0 && end === size - 1)) throw new Error(`${url}: the server does not answer range requests (${x.status})`);
    const got = new Uint8Array(x.response as ArrayBuffer);
    const out: Uint8Array[] = [];
    for (let b = b0; b < b1; b++) out.push(got.subarray((b - b0) * BLOCK, Math.min(got.length, (b - b0 + 1) * BLOCK)));
    return out;
  };
  return {
    name,
    size,
    lastModifiedDate: new Date(),
    // (Not a Blob: the bytes themselves, which the reader WORKERFS is given takes as they are.
    // Making a Blob of every read filled a fresh browser profile's store of them.)
    slice(s: number, e: number): Bytes {
      e = Math.min(e, size);
      if (e <= s) return { size: 0, bytes: new ArrayBuffer(0) };
      const b0 = Math.floor(s / BLOCK);
      const b1 = Math.ceil(e / BLOCK);
      // This read's blocks, gathered before the cache changes: the cached ones, then each run of
      // missing ones in one request.
      const blocks: (Uint8Array | undefined)[] = [];
      for (let b = b0; b < b1; b++) blocks.push(cache.get(b));
      for (let i = 0; i < blocks.length; ) {
        if (blocks[i]) {
          i++;
          continue;
        }
        let j = i;
        while (j < blocks.length && !blocks[j]) j++;
        fetchBlocks(b0 + i, b0 + j).forEach((blk, k) => (blocks[i + k] = blk));
        i = j;
      }
      const out = new Uint8Array(e - s);
      let o = 0;
      blocks.forEach((blk, i) => {
        const b = b0 + i;
        cache.delete(b);
        cache.set(b, blk!); // most recently used last
        const lo = b === b0 ? s - b * BLOCK : 0;
        const hi = b === b1 - 1 ? e - b * BLOCK : blk!.length;
        out.set(blk!.subarray(lo, hi), o);
        o += hi - lo;
      });
      while (cache.size > KEEP) cache.delete(cache.keys().next().value!);
      return { size: out.length, bytes: out.buffer };
    },
  };
}

const buffers = (s: Pick<Snapshot, 'levels' | 'locations' | 'data' | 'faces'>) =>
  [s.levels.buffer, s.locations.buffer, ...Object.values(s.data).map((a) => a!.buffer), ...(s.faces ? [s.faces.buffer] : [])] as ArrayBuffer[];

type Msg = { open?: File | string; read?: number; stream?: number; vars?: string[]; faces?: boolean; batch?: number; probe?: number[][]; tables?: number };

self.onmessage = async (e: MessageEvent<Msg>) => {
  try {
    let last = 0;
    const progress = (f: number) => {
      if (f - last >= 0.01 || f === 1) self.postMessage({ progress: (last = f) });
    };
    const m = e.data;
    if (m.open) {
      lib ??= await library();
      release();
      const fs = lib.FS;
      if (!fs.mount || !fs.filesystems?.WORKERFS) throw new Error('this build of HDF5 cannot read a file in place');
      try {
        fs.mkdir?.(MOUNT);
      } catch {
        // already there
      }
      bytesReader(fs);
      const file = typeof m.open === 'string' ? urlFile(new URL(m.open, self.location.href).href, 'checkpoint.athdf') : m.open;
      fs.mount(fs.filesystems.WORKERFS, { files: [file] }, MOUNT);
      reader = blocksOfUniform(openAthdf(lib, `${MOUNT}/${file.name}`));
      self.postMessage({ header: reader.header(), variables: reader.variables, hasFaces: reader.hasFaces });
    } else if (m.read !== undefined) {
      if (!reader) throw new Error('no file is open');
      const snap = readForView(reader, m.read, { progress });
      self.postMessage({ snapshot: snap }, { transfer: buffers(snap) });
    } else if (m.stream !== undefined) {
      if (!reader) throw new Error('no file is open');
      streamForView(reader, m.stream, {
        vars: m.vars,
        faces: !!m.faces,
        batch: m.batch,
        progress,
        emit: (b: ViewBatch) => self.postMessage({ batch: b }, { transfer: buffers(b) }),
      });
      self.postMessage({ done: true });
    } else if (m.tables !== undefined) {
      if (!reader) throw new Error('no file is open');
      self.postMessage({ snapshot: tablesAtCap(reader, m.tables) });
    } else if (m.probe) {
      if (!reader) throw new Error('no file is open');
      self.postMessage({ probes: probeFile(reader, m.probe) });
    } else {
      release();
    }
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
