/**
 * Workers for the CPU half of an adaptive run's initial state (AmrSim.reset): the refinement
 * indicators of the starting mesh and the leaves' states, spread over the CPU's cores. A large
 * 3D start (the 512^3 MRI disk) samples its setup some hundred million times, which takes about
 * a minute on one thread.
 *
 * The workers run amr/init_cpu.ts, as the page does without them, so the results are
 * bit-identical either way. Functions do not cross to a worker: it gets the setup document the
 * run's problem was compiled from (setup/compile.ts) and compiles it again. A problem that is
 * not its document compiled (one carried in a snapshot, or altered by hand), or a configuration
 * that cannot be cloned, is built on the page instead. So is everything when the pool fails in
 * any way.
 */
import { compiledFrom } from '../setup/compile';
import { meshFilesFor } from '../geometry/store';
import type { SetupSpec } from '../setup/spec';
import type { SimConfig } from '../sim';
import type { BlockAt, LeafState } from './init_cpu';

export type InitRequest =
  | { kind: 'setup'; key: string; job: number; setup: SetupSpec; box: number; cfg: Omit<SimConfig, 'problem'>; n: number; dx0: number; levels: number; meshes: Record<string, Uint8Array> }
  | { kind: 'indicators'; key: string; job: number; blocks: (BlockAt & { sub: number })[] }
  | { kind: 'leaves'; key: string; job: number; blocks: (BlockAt & { index: number })[] };

export type InitReply =
  | { kind: 'ready'; key: string; job: number }
  | { kind: 'indicators'; key: string; job: number; values: Float64Array }
  | { kind: 'leaf'; key: string; job: number; index: number; state: LeafState }
  | { kind: 'done'; key: string; job: number }
  | { kind: 'error'; key: string; job: number; message: string };

/** Most workers: beyond this the page's own uploads and the memory of each worker's copy of the code outweigh more cores. */
const MAX_WORKERS = 12;
/** Blocks per message: small enough to balance blocks of very different cost, large enough that messages are few. */
const INDICATOR_CHUNK = 8;
const LEAF_CHUNK = 16;
/** A worker that has not loaded and answered within this long is taken as broken. */
const READY_MS = 30_000;

let pool: Worker[] | null | undefined;
let runs = 0;

/** The workers, made on first use; null where there are none to be had or too few cores to gain. */
function workers(): Worker[] | null {
  if (pool !== undefined) return pool;
  pool = null;
  try {
    // A switch for checking the workers against the page: localStorage shaderous.initWorkers = '0'.
    if (globalThis.localStorage?.getItem('shaderous.initWorkers') === '0') return null;
    const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency ?? 1 : 1;
    const count = Math.min(MAX_WORKERS, cores - 1);
    if (typeof Worker === 'undefined' || count < 2) return null;
    pool = Array.from({ length: count }, () => new Worker(new URL('./init_worker.ts', import.meta.url), { type: 'module' }));
  } catch (e) {
    console.warn('initial state: no workers, building on the page', e);
    pool = null;
  }
  return pool;
}

/** Drop the pool after a failure, so the next run builds on the page. */
function disable(why: string) {
  console.warn(`initial state: workers failed (${why}); building on the page`);
  for (const w of pool ?? []) w.terminate();
  pool = null;
}

/** The configuration without what cannot cross to a worker, or null if a worker could not rebuild it. */
function shareable(cfg: SimConfig): { plain: Omit<SimConfig, 'problem'>; setup: SetupSpec } | null {
  try {
    const setup = cfg.problem.setup;
    if (cfg.restart || !setup) return null;
    // The problem must be its document compiled, but for the side of the box (a parameter may set it).
    const base = compiledFrom(setup).problem as unknown as Record<string, unknown>;
    const mine = cfg.problem as unknown as Record<string, unknown>;
    const keys = new Set([...Object.keys(base), ...Object.keys(mine)]);
    for (const k of keys) if (k !== 'box' && base[k] !== mine[k]) return null;
    const plain = Object.fromEntries(Object.entries(cfg).filter(([k, v]) => k !== 'problem' && typeof v !== 'function'));
    structuredClone(plain);
    structuredClone(setup);
    return { plain: plain as Omit<SimConfig, 'problem'>, setup };
  } catch {
    return null;
  }
}

/** One reset's use of the pool: every worker holds the run's setup. */
export class InitRun {
  private job = 0;
  private constructor(
    private readonly ws: Worker[],
    private readonly key: string,
  ) {}

  /** Set the workers up for this run, or null to build on the page. */
  static async start(cfg: SimConfig, n: number, dx0: number, levels: number): Promise<InitRun | null> {
    const ws = workers();
    const share = ws && shareable(cfg);
    if (!ws || !share) return null;
    const key = `run${++runs}`;
    const run = new InitRun(ws, key);
    try {
      await run.all((w, job) => {
        // (The mesh files the setup refers to: a worker compiles the setup itself. Each builds its own distance field.)
        w.postMessage({ kind: 'setup', key, job, setup: share.setup, box: cfg.problem.box, cfg: share.plain, n, dx0, levels, meshes: meshFilesFor(share.setup) } satisfies InitRequest);
        return (m) => m.kind === 'ready';
      }, READY_MS);
      return run;
    } catch (e) {
      disable(String(e));
      return null;
    }
  }

  /** Each worker sends one message and waits for the reply `done` accepts. */
  private all(send: (w: Worker, job: number) => (m: InitReply) => boolean, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      let left = this.ws.length;
      const timer = setTimeout(() => finish(new Error('workers did not answer')), timeoutMs);
      const off: (() => void)[] = [];
      const finish = (err?: Error) => {
        clearTimeout(timer);
        for (const f of off) f();
        if (err) reject(err);
        else resolve();
      };
      for (const w of this.ws) {
        const job = ++this.job;
        const accept = send(w, job);
        const onMsg = (e: MessageEvent<InitReply>) => {
          const m = e.data;
          if (m.key !== this.key || m.job !== job) return;
          if (m.kind === 'error') return finish(new Error(m.message));
          if (accept(m) && --left === 0) finish();
        };
        const onErr = (e: ErrorEvent) => finish(new Error(e.message || 'worker error'));
        w.addEventListener('message', onMsg);
        w.addEventListener('error', onErr);
        off.push(() => {
          w.removeEventListener('message', onMsg);
          w.removeEventListener('error', onErr);
        });
      }
    });
  }

  /**
   * Chunks of `items` handed to whichever worker is free, `handle` called with each reply, until
   * every chunk's `done` reply has come.
   */
  private queue<T>(items: T[], size: number, request: (job: number, chunk: T[]) => InitRequest, handle: (m: InitReply, chunk: T[]) => boolean): Promise<void> {
    return new Promise((resolve, reject) => {
      let next = 0;
      let open = 0;
      let failed = false;
      const off: (() => void)[] = [];
      const finish = (err?: Error) => {
        for (const f of off) f();
        if (err) reject(err);
        else resolve();
      };
      const jobs = new Map<number, T[]>();
      const feed = (w: Worker) => {
        if (next >= items.length) {
          if (open === 0 && !failed) finish();
          return;
        }
        const chunk = items.slice(next, next + size);
        next += size;
        const job = ++this.job;
        jobs.set(job, chunk);
        open++;
        w.postMessage(request(job, chunk));
      };
      for (const w of this.ws) {
        const onMsg = (e: MessageEvent<InitReply>) => {
          const m = e.data;
          if (failed || m.key !== this.key || !jobs.has(m.job)) return;
          if (m.kind === 'error') {
            failed = true;
            return finish(new Error(m.message));
          }
          if (handle(m, jobs.get(m.job)!)) {
            jobs.delete(m.job);
            open--;
            feed(w);
          }
        };
        const onErr = (e: ErrorEvent) => {
          failed = true;
          finish(new Error(e.message || 'worker error'));
        };
        w.addEventListener('message', onMsg);
        w.addEventListener('error', onErr);
        off.push(() => {
          w.removeEventListener('message', onMsg);
          w.removeEventListener('error', onErr);
        });
      }
      if (items.length === 0) return finish();
      for (const w of this.ws) feed(w);
    });
  }

  /** The indicator of each block, in order. */
  async indicators(blocks: (BlockAt & { sub: number })[]): Promise<Float64Array> {
    const out = new Float64Array(blocks.length);
    const items = blocks.map((b, i) => ({ level: b.level, c: [...b.c], sub: b.sub, i }));
    try {
      await this.queue(
        items,
        INDICATOR_CHUNK,
        (job, chunk) => ({ kind: 'indicators', key: this.key, job, blocks: chunk }),
        (m, chunk) => {
          if (m.kind !== 'indicators') return false;
          chunk.forEach((b, k) => (out[b.i] = m.values[k]));
          return true;
        },
      );
    } catch (e) {
      disable(String(e));
      throw e;
    }
    return out;
  }

  /** Each leaf's state, handed to `onLeaf` with its index as it arrives (in no particular order). */
  async leaves(blocks: BlockAt[], onLeaf: (index: number, s: LeafState) => void): Promise<void> {
    const items = blocks.map((b, index) => ({ level: b.level, c: [...b.c], index }));
    try {
      await this.queue(
        items,
        LEAF_CHUNK,
        (job, chunk) => ({ kind: 'leaves', key: this.key, job, blocks: chunk }),
        (m) => {
          if (m.kind === 'leaf') {
            onLeaf(m.index, m.state);
            return false;
          }
          return m.kind === 'done';
        },
      );
    } catch (e) {
      disable(String(e));
      throw e;
    }
  }
}
