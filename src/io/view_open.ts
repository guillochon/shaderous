// The page's side of view-only loading: a checkpoint opened in a worker (io/view_worker.ts),
// its header read, and its blocks read merged to a level (io/view.ts).
import type { Snapshot } from './snapshot';

export interface ViewFile {
  /** the file's header: a snapshot with its block tables and settings, without data */
  header: Snapshot;
  variables: string[];
  hasFaces: boolean;
  /** The file's blocks merged to `level`, as a snapshot; `progress` gets the fraction read. */
  read(level: number, progress?: (fraction: number) => void): Promise<Snapshot>;
  close(): void;
}

type Reply = { error?: string; header?: Snapshot; variables?: string[]; hasFaces?: boolean; progress?: number; snapshot?: Snapshot };

export async function openViewFile(file: File): Promise<ViewFile> {
  const worker = new Worker(new URL('./view_worker.ts', import.meta.url), { type: 'module' });
  let progress: ((f: number) => void) | undefined;
  let waiting: { resolve: (r: Reply) => void; reject: (e: Error) => void } | null = null;
  worker.onmessage = (e: MessageEvent<Reply>) => {
    if (e.data.progress !== undefined) return progress?.(e.data.progress);
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
  const ask = (msg: unknown) =>
    new Promise<Reply>((resolve, reject) => {
      waiting = { resolve, reject };
      worker.postMessage(msg);
    });
  try {
    const head = await ask({ open: file });
    return {
      header: head.header!,
      variables: head.variables!,
      hasFaces: head.hasFaces!,
      async read(level, p) {
        progress = p;
        try {
          return (await ask({ read: level })).snapshot!;
        } finally {
          progress = undefined;
        }
      },
      close: () => worker.terminate(),
    };
  } catch (e) {
    worker.terminate();
    throw e;
  }
}
