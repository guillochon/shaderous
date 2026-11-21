// Reads a checkpoint for view-only loading (io/view.ts) off the page's thread. The file is
// mounted with Emscripten's WORKERFS, which reads it with File.slice as HDF5 asks: no copy of it
// is made, so files larger than WebAssembly's memory open. Messages (io/view_open.ts):
//   in  { open: File }                  out { header, variables, hasFaces }
//   in  { read: level, wanted?: names } out { progress } ..., then { snapshot }
//   errors come back as { error }.
import { library, openAthdf, type AthdfReader, type Library } from './athdf';
import { readForView } from './view';
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

self.onmessage = async (e: MessageEvent<{ open?: File; read?: number; wanted?: string[] }>) => {
  try {
    if (e.data.open) {
      lib ??= await library();
      release();
      const fs = lib.FS;
      if (!fs.mount || !fs.filesystems?.WORKERFS) throw new Error('this build of HDF5 cannot read a file in place');
      try {
        fs.mkdir?.(MOUNT);
      } catch {
        // already there
      }
      fs.mount(fs.filesystems.WORKERFS, { files: [e.data.open] }, MOUNT);
      reader = openAthdf(lib, `${MOUNT}/${e.data.open.name}`);
      const header = reader.header();
      self.postMessage({ header, variables: reader.variables, hasFaces: reader.hasFaces });
    } else if (e.data.read !== undefined) {
      if (!reader) throw new Error('no file is open');
      const wanted = e.data.wanted ? new Set(e.data.wanted) : null;
      let last = 0;
      const snap: Snapshot = readForView(reader, e.data.read, {
        wanted: wanted ? (v) => wanted.has(v) : undefined,
        progress: (f) => {
          if (f - last >= 0.01 || f === 1) self.postMessage({ progress: (last = f) });
        },
      });
      const transfer = [snap.levels.buffer, snap.locations.buffer, ...Object.values(snap.data).map((a) => a!.buffer), ...(snap.faces ? [snap.faces.buffer] : [])];
      self.postMessage({ snapshot: snap }, { transfer: transfer as ArrayBuffer[] });
    } else {
      release();
    }
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
