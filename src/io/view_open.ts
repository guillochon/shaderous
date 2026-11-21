// The page's side of view-only loading: a checkpoint opened in a worker (io/view_worker.ts),
// its header read, its blocks read merged to a level (io/view.ts), whole or a batch at a time,
// and probed at the file's own resolution.
import type { Snapshot } from './snapshot';
import type { ViewBatch } from './view';

export interface ViewFile {
  /** the file's header: a snapshot with its block tables and settings, without data (a uniform grid's as blocks of 16) */
  header: Snapshot;
  variables: string[];
  hasFaces: boolean;
  /** The file's blocks merged to `level`, as a snapshot; `progress` gets the fraction read. */
  read(level: number, progress?: (fraction: number) => void): Promise<Snapshot>;
  /** The same, variables `vars` only (and the faces with `faces`), handed to `batch` some blocks at a time as they are complete. */
  stream(level: number, vars: string[], batch: (b: ViewBatch) => void, progress?: (fraction: number) => void, faces?: boolean): Promise<void>;
  /** The header with the block tables merged to `level` and no data (io/view.ts tablesAtCap). */
  tables(level: number): Promise<Snapshot>;
  /** The file's cells under these points (fractions of the box): io/view.ts probeFile. */
  probe(points: number[][]): Promise<{ at: number[]; level: number; values: Record<string, number> }[]>;
  close(): void;
}

type Reply = {
  error?: string;
  header?: Snapshot;
  variables?: string[];
  hasFaces?: boolean;
  progress?: number;
  snapshot?: Snapshot;
  batch?: ViewBatch;
  done?: boolean;
  probes?: Awaited<ReturnType<ViewFile['probe']>>;
};

/** Open a checkpoint for reading in parts: a File (read in place), or a URL the page can fetch from (read with range requests). */
export async function openViewFile(file: File | string): Promise<ViewFile> {
  const worker = new Worker(new URL('./view_worker.ts', import.meta.url), { type: 'module' });
  let progress: ((f: number) => void) | undefined;
  let onBatch: ((b: ViewBatch) => void) | undefined;
  let waiting: { resolve: (r: Reply) => void; reject: (e: Error) => void } | null = null;
  // One request at a time: the next waits for the last.
  let queue: Promise<unknown> = Promise.resolve();
  worker.onmessage = (e: MessageEvent<Reply>) => {
    if (e.data.progress !== undefined) return progress?.(e.data.progress);
    if (e.data.batch) return onBatch?.(e.data.batch);
    const w = waiting;
    waiting = null;
    if (e.data.error) w?.reject(new Error(e.data.error));
    else w?.resolve(e.data);
  };
  worker.onerror = (e) => {
    const w = waiting;
    waiting = null;
    w?.reject(new Error(e.message || 'the file reader stopped'));
  };
  const ask = (msg: unknown, p?: (f: number) => void, b?: (x: ViewBatch) => void): Promise<Reply> => {
    const run = queue.then(
      () =>
        new Promise<Reply>((resolve, reject) => {
          progress = p;
          onBatch = b;
          waiting = { resolve, reject };
          worker.postMessage(msg);
        }),
    );
    queue = run.catch(() => {});
    return run;
  };
  try {
    const head = await ask({ open: file });
    return {
      header: head.header!,
      variables: head.variables!,
      hasFaces: head.hasFaces!,
      read: async (level, p) => (await ask({ read: level }, p)).snapshot!,
      stream: async (level, vars, batch, p, faces = false) => void (await ask({ stream: level, vars, faces, batch: 512 }, p, batch)),
      probe: async (points) => (await ask({ probe: points })).probes!,
      tables: async (level) => (await ask({ tables: level })).snapshot!,
      close: () => worker.terminate(),
    };
  } catch (e) {
    worker.terminate();
    throw e;
  }
}
