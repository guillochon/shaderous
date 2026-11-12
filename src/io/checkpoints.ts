// Where checkpoints go: a folder the user picks once (the File System Access API, in Chrome and
// Edge), or, in a browser without it, one download per file.

/** The part of the File System Access API used here (not yet in TypeScript's DOM types). */
interface DirectoryHandle {
  name: string;
  getFileHandle(name: string, opts: { create: boolean }): Promise<{
    createWritable(): Promise<{ write(data: BufferSource | Blob): Promise<void>; close(): Promise<void> }>;
  }>;
  queryPermission?(opts: { mode: 'readwrite' }): Promise<PermissionState>;
  requestPermission?(opts: { mode: 'readwrite' }): Promise<PermissionState>;
}
type PickerWindow = { showDirectoryPicker?(opts?: { id?: string; mode?: 'readwrite' }): Promise<DirectoryHandle> };

/** The browser can write files into a folder the user picks. */
export const canPickFolder = (): boolean => typeof (window as PickerWindow).showDirectoryPicker === 'function';

export interface CheckpointSink {
  /** Where the files go, for the page to say. */
  readonly where: string;
  write(name: string, bytes: Uint8Array): Promise<void>;
}

/** Hand a file to the browser as a download. */
export function download(name: string, bytes: Uint8Array, type = 'application/x-hdf5') {
  const url = URL.createObjectURL(new Blob([bytes as Uint8Array<ArrayBuffer>], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/** Each checkpoint as a download (the browser may ask once to allow several). */
export const downloadSink: CheckpointSink = {
  where: 'your downloads',
  write: async (name, bytes) => download(name, bytes),
};

/**
 * Ask the user for a folder to write checkpoints into. Must be called from a click or other
 * user gesture. Null if they cancel; throws if the browser has no folder picker.
 */
export async function pickFolder(): Promise<CheckpointSink | null> {
  let dir: DirectoryHandle;
  try {
    dir = await (window as PickerWindow).showDirectoryPicker!({ id: 'shaderous-checkpoints', mode: 'readwrite' });
  } catch (e) {
    if ((e as DOMException)?.name === 'AbortError') return null;
    throw e;
  }
  return {
    where: `the folder “${dir.name}”`,
    async write(name, bytes) {
      // Permission granted at the pick lasts for the page; ask again if the browser dropped it.
      if (dir.queryPermission && (await dir.queryPermission({ mode: 'readwrite' })) !== 'granted') {
        if ((await dir.requestPermission?.({ mode: 'readwrite' })) !== 'granted') throw new Error(`no permission to write to ${dir.name}`);
      }
      const file = await dir.getFileHandle(name, { create: true });
      const out = await file.createWritable();
      try {
        await out.write(bytes as Uint8Array<ArrayBuffer>);
      } finally {
        await out.close();
      }
    },
  };
}

/** Checkpoint times: multiples of `every`. The first one after time `t` (t itself excluded). */
export function nextCheckpoint(t: number, every: number): number {
  // Times are f32 on the GPU: a clock stopped at a checkpoint reads a hair either side of it.
  const k = Math.floor(t / every + 1e-4) + 1;
  return k * every;
}

/** The checkpoint's number: which multiple of `every` time `t` is. */
export const checkpointIndex = (t: number, every: number): number => Math.round(t / every);
