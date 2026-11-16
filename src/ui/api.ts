// A programmatic handle on the simulator page, `window.shaderous`, for scripts and agents that
// drive it without a person at the controls (Playwright, the Chrome DevTools Protocol, the
// browser console). See docs/api.md.
//
// Every call returns a promise, and everything it resolves to is plain JSON (numbers, strings,
// arrays, objects; binary files as base64), so it crosses page.evaluate() and similar bridges
// intact. Calls that change the run are served one at a time, in the order they were made.
import type { Gpu } from '../gpu/device';
import type { LinkOverrides } from './link';
import { decodeLink } from './link';
import { capture } from '../io/capture';
import { AmrSim } from '../amr/amr_sim';
import { SnapshotSampler, VARIABLES, type Snapshot } from '../io/snapshot';
import type { Diagnostics, SimConfig, Simulation } from '../sim';
import { EOS_LABELS } from '../eos/eos';
import { COOLING_LABELS } from '../cooling/cooling';
import { FAMILIES, FIELDS, GRAVITY_MODES, LIMITER_KINDS, RIEMANN_KINDS } from '../problems';
import { VIEW_EXTRAS, VIEW_MODES, VIEW_OPTIONS } from '../render/view_options';
import { defaultFrameDt, problemUnits } from '../problems/types';
import type { ViewState } from '../render/renderer';
import { COLORMAPS } from '../render/transfer';
import { overlayGutters, paintOverlays, type OverlayInfo, type OverlayOptions } from '../render/overlays';
import { download } from '../io/checkpoints';
import { BUILD, errorLog } from './errors';
import { PRESET_FILES, presetSpec } from '../presets/index';
import { checkSetup } from '../setup/check';
import { readSetup } from '../setup/codec';
import { factsOf, factsWithout, flatten, OPTIONS, optionAt, PART_OPTIONS, partFields, unavailable, valueAt, type Option } from '../setup/options';
import { FLOWS } from '../setup/parts/flow';
import { cloneSetup, SETUP_FORMAT, setPath, type SetupSpec } from '../setup/spec';

/** What the page lends the API: its state, and the operations its own controls use. */
export interface AppHooks {
  gpu(): Gpu;
  /** The GPU adapter, as the page names it. */
  adapter(): string;
  sim(): Simulation | undefined;
  cfg(): SimConfig;
  view: ViewState;
  canvas: HTMLCanvasElement;
  /** Grid sizes on offer; above uniformMax a grid must be adaptive (`amr` of 1 or more). */
  resolutions(): { 2: number[]; 3: number[]; uniformMax: { 2: number; 3: number } };
  /** Resolves once every queued rebuild has finished. */
  idle(): Promise<void>;
  build(link: LinkOverrides): Promise<void>;
  /** resolves to what an older file of ours could not carry over as written */
  importAthdf(bytes: Uint8Array): Promise<string[]>;
  restart(): Promise<void>;
  running(): boolean;
  finished(): boolean;
  setRunning(on: boolean): void;
  setFinished(on: boolean): void;
  /** While set, the page's own frame loop takes no steps: the API does. */
  setDriving(on: boolean): void;
  /** Set the clock's stop time from the checkpoint schedule again (after a run to a time). */
  rearm(t: number): void;
  query(): string;
  /** Take up new readouts: the displayed range and the density cut follow them. */
  refresh(d: Diagnostics): void;
  /**
   * Encode and submit one frame of the current view: at `size` pixels, or with null at the
   * canvas's size on the page. A size holds until the next draw.
   */
  draw(size: [number, number] | null, asPage?: boolean, capture?: boolean): void;
  /** The frame last drawn with `capture` set, read back from the GPU (Renderer.captured). */
  captured(): Promise<HTMLCanvasElement>;
  /** Resolves once the range read back from the last draw, if any, has been taken up. */
  rangeSettled(): Promise<void>;
  /** Whether the displayed range follows from what a draw finds (not from `d`, nor typed in). */
  rangedByDrawing(d: Diagnostics): boolean;
  /** The view area's size as the page draws it, in pixels (0 x 0 without a layout). */
  viewSize(): [number, number];
  /**
   * The simulation box as the page draws it: its left, top, right and bottom as fractions of the
   * view area, outside 0 to 1 where it runs off. Null when it cannot be told (no layout, or the
   * camera is inside the box).
   */
  viewBounds(): [number, number, number, number] | null;
  /** The view's background, as a CSS color. */
  background(): string;
  athdf(): Promise<{ time: number; bytes: Uint8Array; stem: string }>;
  setColormap(name: string): void;
  setAutoRange(on: boolean): void;
  sync(): void;
  /** What the color bar and the triad show, for the view as it stands (render/overlays.ts). */
  overlayInfo(): OverlayInfo;
  /**
   * A movie is being recorded at `size` pixels, or with null no longer: while one is, the page's
   * frame loop draws nothing, and the canvas shows the movie's frames letterboxed.
   */
  setRecording(size: [number, number] | null): void;
  /** A frame of the recording under way has been added. */
  recordProgress(status: RecordingStatus): void;
  /** The setup document the run stands for now: its preset or file, with every setting changed since. */
  runningSpec(): SetupSpec;
  /** Take an edited document: the page does the least that gets the run there. */
  applySpec(spec: SetupSpec): Promise<void>;
  /** Make a document runnable: the id it runs under (its preset's if it is that preset, else the custom setup's). */
  adopt(spec: SetupSpec): string;
  /** GPU memory the grid of a document needs, and what the page may take, in bytes. */
  gridMemory(spec: SetupSpec): { need: number; budget: number };
  /** Checkpoints are on: the clock is held at each (cfg.tStop) until it is written. */
  checkpointsOn(): boolean;
  /** Write the checkpoint the clock is held at, and arm the next (as the page does when it plays). */
  writeCheckpoint(): Promise<void>;
}

/** Options of run(). */
export interface RunOptions {
  /** Take this many steps (at most). */
  steps?: number;
  /** Run until the clock reads this time: the last step is shortened to land on it exactly. */
  until?: number;
  /** Give up after this much wall time, returning where the run got to. */
  timeoutMs?: number;
}

export interface RunResult {
  t: number;
  dt: number;
  steps: number;
  /** steps taken by this call */
  taken: number;
  seconds: number;
  /** why the call returned: `frame` only at a movie's soft stop (record() with `dt`) */
  stopped: 'steps' | 'time' | 'end' | 'timeout' | 'frame';
  /** the timestep of the last step taken, which a held clock's dt (0) no longer shows */
  lastDt: number;
}

/** run()'s options inside the API: a soft stop as well. */
interface RunInternal extends RunOptions {
  /**
   * A movie's frame time (SimConfig.tFrame): the clock holds at the step boundary nearest to it,
   * and the call returns with `frame`.
   */
  frameAt?: number;
}

/** The clock as the stepping reads it back. */
type Clock = Pick<Diagnostics, 't' | 'dt' | 'steps'>;

/** Options of record(). */
export interface RecordOptions {
  /** A frame every this many steps... */
  every?: number;
  /**
   * ...or a frame near every multiple of this much code time (the default cadence, at a 600th
   * of the problem's dynamical time): at the step boundary nearest to it, so that no step is
   * shortened and the run is the one it would have been unrecorded.
   */
  dt?: number;
  /** Stop after this many frames (the first, at the start, included). */
  frames?: number;
  /** Stop when the clock reaches this time (landed on exactly, as run() does). */
  until?: number;
  /** Give up after this much wall time, finishing the file with what was recorded. */
  timeoutMs?: number;
  /** Playback rate, frames per second (default 60). */
  fps?: number;
  /**
   * The movie's size in pixels, rounded down to even numbers. By default the view area's own,
   * framed as the page frames it (1920 x 1080 when the page has no layout).
   */
  width?: number;
  height?: number;
  /** Bits per second (default about 0.2 bits per pixel per frame). */
  bitrate?: number;
  /** The color bar, the triad and the status line, all on by default. */
  overlays?: OverlayOptions;
  /** Hand the file to the browser as a download (default true). */
  download?: boolean;
  /** Return the file as base64 too (default false). */
  base64?: boolean;
}

export interface RecordResult {
  name: string;
  frames: number;
  /** Length of the movie, in seconds. */
  seconds: number;
  width: number;
  height: number;
  /** The clock at the last frame, and the steps taken while recording. */
  t: number;
  steps: number;
  bytes: number;
  /** why the recording ended */
  stopped: 'frames' | 'time' | 'end' | 'stopped' | 'replaced' | 'timeout';
  base64?: string;
}

/** A recording under way, as the page's Record button shows it. */
export interface RecordingStatus {
  frames: number;
  seconds: number;
  t: number;
}

/** Fractions of the box, 0 to 1 along each axis (z is ignored in 2D). */
export type Point = [number, number, number];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Bytes as base64, without building one giant string of char codes. */
async function toBase64(bytes: Uint8Array): Promise<string> {
  const url = await new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(new Blob([bytes as Uint8Array<ArrayBuffer>]));
  });
  return url.slice(url.indexOf(',') + 1);
}

function fromBase64(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

/** A setup as a query string: one already, or an object of the same keys. */
/**
 * Names the settings went by when they were controls of the side bar: each still sets its
 * option. Add to these, never remove.
 */
export const CONTROL_ALIASES: Record<string, string> = {
  dim: 'domain.ndim', res: 'domain.n', levels: 'domain.amr', refhi: 'mesh.refineHi', refrho: 'mesh.refineRho', reflo: 'mesh.refineLo', tbins: 'mesh.timeBins',
  fluffrho: 'physics.fluff.rho', fluffu: 'physics.fluff.u', fluffv: 'physics.fluff.v', fluffva: 'physics.fluff.va', cooling: 'physics.cooling',
  visc: 'physics.viscosity', cond: 'physics.conductivity', jeansfloor: 'physics.jeansFloor', jeanscells: 'physics.jeansCells',
};

/** An option as the scripting interface lists it. */
function optionInfo(o: Option, path: string) {
  const c = o.control;
  return {
    id: o.linkKey ?? path,
    path,
    label: o.label,
    title: o.tip,
    page: o.page,
    kind: c.type === 'check' ? 'checkbox' : c.type,
    apply: o.apply,
    ...(o.default !== undefined ? { default: o.default } : {}),
    ...(c.type === 'number' ? { ...(c.min !== undefined ? { min: c.min } : {}), ...(c.max !== undefined ? { max: c.max } : {}), ...(c.integer ? { integer: true } : {}) } : {}),
    ...(c.type === 'select' ? { options: c.choices.map((k) => ({ value: k.value, label: k.label })) } : {}),
  };
}

function setupQuery(setup: string | Record<string, string | number | boolean>): URLSearchParams {
  if (typeof setup === 'string') return new URLSearchParams(setup.replace(/^.*\?/, ''));
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(setup)) q.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  return q;
}

/** Diagnostics as JSON: field ranges keyed by field name. */
function diagJson(d: Diagnostics) {
  const { ranges, ...rest } = d;
  return { ...rest, ranges: Object.fromEntries(ranges.map((r, i) => [FIELDS[i], r])) };
}

/** The variables a snapshot can be sampled for: the primitives, each species, and `level`. */
function variableOf(snap: Snapshot, sampler: SnapshotSampler, species: readonly string[], name: string): (p: Point) => number {
  const prim = (p: Point) => sampler.prim(p[0], p[1], p[2], species);
  switch (name) {
    case 'rho': return (p) => prim(p).rho;
    case 'press': return (p) => prim(p).p;
    case 'vel1': return (p) => prim(p).vx;
    case 'vel2': return (p) => prim(p).vy;
    case 'vel3': return (p) => prim(p).vz;
    case 'r0': return (p) => prim(p).s;
    case 'level': return (p) => sampler.level(p[0], p[1], p[2]);
  }
  const k = species.indexOf(name);
  if (k >= 0 && snap.data[name]) return (p) => prim(p).X![k];
  throw new Error(`unknown variable "${name}": use one of ${[...VARIABLES, 'level', ...species].join(', ')}`);
}

export function installApi(app: AppHooks, ready: Promise<void>) {
  // Calls that change the run, one at a time.
  let chain: Promise<unknown> = Promise.resolve();
  const serial = <T>(job: () => Promise<T>): Promise<T> => {
    const next = chain.then(async () => {
      await ready;
      await app.idle();
      return job();
    });
    chain = next.catch(() => {});
    return next;
  };
  const current = (): Simulation => {
    const s = app.sim();
    if (!s) throw new Error(lastError() ?? 'no simulation is loaded');
    return s;
  };
  const lastError = () => {
    const e = errorLog();
    return e.length ? e[e.length - 1] : undefined;
  };
  /** A build that failed reports its error and leaves no simulation: say why. */
  const built = async (job: () => Promise<void>) => {
    const before = errorLog().length;
    await job();
    await app.idle();
    if (!app.sim()) throw new Error(errorLog().length > before ? lastError() : 'the simulation could not be built');
    app.setRunning(false);
  };

  async function state() {
    const s = current();
    const cfg = app.cfg();
    const d = s.preparing ? undefined : await s.diagnostics();
    return {
      problem: cfg.problem.id,
      label: cfg.problem.label,
      ndim: cfg.ndim,
      n: cfg.n,
      amr: cfg.amrLevels,
      t: d?.t ?? cfg.restart?.time ?? 0,
      dt: d?.dt ?? 0,
      steps: d?.steps ?? 0,
      tEnd: Number.isFinite(cfg.tEnd) ? cfg.tEnd : null,
      running: app.running(),
      finished: app.finished(),
      preparing: s.preparing,
      query: app.query(),
      setup: app.runningSpec(),
      adapter: app.adapter(),
    };
  }

  // The stepping reads only the clock back, not the full diagnostics: each batch copies it into a
  // small buffer of its own, so the next batch can be submitted while one is being read.
  const clockPool: GPUBuffer[] = [];
  let poolDevice: GPUDevice | null = null;
  /** Copy the clock out at the end of `enc`; once `enc` is submitted, the returned call reads it. */
  function clockCopy(enc: GPUCommandEncoder, s: Simulation): () => Promise<Clock> {
    const device = app.gpu().device;
    if (poolDevice !== device) {
      clockPool.length = 0;
      poolDevice = device;
    }
    const buf = clockPool.pop() ?? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(s.clockBuffer, 0, buf, 0, 16);
    return () => {
      const read = (async () => {
        await buf.mapAsync(GPUMapMode.READ);
        const [dt, t, , steps] = new Float32Array(buf.getMappedRange().slice(0));
        buf.unmap();
        if (poolDevice === device) clockPool.push(buf);
        return { t, dt, steps };
      })();
      // A read left behind by a run that was replaced fails quietly.
      read.catch(() => {});
      return read;
    };
  }
  async function readClock(s: Simulation): Promise<Clock> {
    const enc = app.gpu().device.createCommandEncoder();
    const read = clockCopy(enc, s);
    app.gpu().device.queue.submit([enc.finish()]);
    return read();
  }
  // How fast each run steps, carried from one run() to the next: a movie calls run() for every
  // frame, and should not start again from batches of one step each time.
  const paces = new WeakMap<Simulation, { msPerStep: number; dt: number; grow: number }>();

  async function run(opts: RunInternal = {}): Promise<RunResult> {
    const s = current();
    const cfg = app.cfg();
    const { steps, until, timeoutMs, frameAt } = opts;
    if (steps === undefined && until === undefined && frameAt === undefined) throw new Error('run() needs `steps`, `until`, or both');
    if (steps !== undefined && !(steps >= 0)) throw new Error('`steps` must be a number of steps, 0 or more');
    app.setRunning(false);
    app.setDriving(true);
    const t0 = performance.now();
    const stopWas = cfg.tStop;
    // The checkpoint the clock is armed to stop at, if any (cfg.tStop is `until` when sooner).
    let checkpoint = stopWas;
    let pace = paces.get(s);
    if (!pace) paces.set(s, (pace = { msPerStep: 0, dt: 0, grow: 1 }));
    let lastDt = pace.dt;
    // A uniform grid keeps two batches in flight, about 50 ms of GPU time each, so the GPU steps
    // on while the last batch's clock is read back. An adaptive mesh steps one batch at a time:
    // its refinement checks are read back after a batch and applied at the start of the next, so
    // a batch encoded before that read would move the re-mesh, and the run would depend on timing.
    const adaptive = s instanceof AmrSim;
    const IN_FLIGHT = adaptive ? 1 : 2;
    const BATCH_MS = adaptive ? 100 : 50;
    const result = (d: Clock, start: number, stopped: RunResult['stopped']): RunResult => ({
      t: d.t, dt: d.dt, steps: d.steps, taken: d.steps - start, seconds: (performance.now() - t0) / 1000, stopped, lastDt: d.dt > 0 ? d.dt : lastDt,
    });
    // Batches submitted and not yet read back, oldest first.
    const inFlight: { read: Promise<Clock>; taken: number; at: number }[] = [];
    let inFlightSteps = 0;
    let lastDone = 0;
    let d: Clock = { t: 0, dt: 0, steps: 0 };
    let rounds = 0;
    const settle = async () => {
      const b = inFlight.shift()!;
      const was = d;
      d = await b.read;
      const now = performance.now();
      inFlightSteps -= b.taken;
      // The batch's GPU time: from when it could start (submitted, and the one before it done).
      const m = (now - Math.max(b.at, lastDone)) / b.taken;
      lastDone = now;
      pace.msPerStep = pace.msPerStep ? 0.7 * pace.msPerStep + 0.3 * m : m;
      pace.grow = Math.min(pace.grow * 2, 256);
      rounds++;
      // The step about to be taken; held at a stop it reads 0, and the mean step of the batch
      // stands in, so a frame-by-frame run does not judge every batch by a step long past.
      if (d.dt > 0) lastDt = pace.dt = d.dt;
      else if (d.steps > was.steps && d.t > was.t) pace.dt = (d.t - was.t) / (d.steps - was.steps);
      if (!(d.t === d.t) || !(d.dt === d.dt)) throw new Error(`the clock is NaN after ${d.steps} steps`);
    };
    const drain = async () => {
      while (inFlight.length) await settle();
    };
    /**
     * Steps until the clock next stops, beyond those in flight, as far as the last timestep tells:
     * at a frame time the step boundary nearest to it (the step that reaches it already reads the hold), at
     * `until` the step that lands on it, at a checkpoint or the end that step and a held one.
     * Steps past a stop are no-ops that still cost a full pass, so batches go no further.
     */
    const ahead = (): number => {
      const dt = d.dt > 0 ? d.dt : pace.dt;
      if (!(dt > 0)) return Infinity;
      const tp = d.t + inFlightSteps * dt;
      let n = Infinity;
      // A held clock's next step is taken at dt = 0 (it only sets the step anew): one more.
      if (frameAt !== undefined) n = Math.min(n, Math.round((frameAt - tp) / dt) + (d.dt > 0 ? 0 : 1));
      if (until !== undefined) n = Math.min(n, Math.ceil((until - tp) / dt - 1e-6));
      const stop = Math.min(checkpoint ?? Infinity, Number.isFinite(cfg.tEnd) ? cfg.tEnd : Infinity);
      if (stop < Infinity) n = Math.min(n, Math.ceil((stop - tp) / dt - 1e-6) + 1);
      return n;
    };
    try {
      await untilPrepared(s, t0, timeoutMs);
      d = await readClock(s);
      const start = d.steps;
      if (until !== undefined && !(until > d.t)) return result(d, start, 'time');
      if (until !== undefined || frameAt !== undefined) {
        // Hold the clock at `until`: the step that reaches it is shortened to land on it. A frame
        // time only holds it at the step boundary nearest to it (hydro/clock.wgsl).
        if (until !== undefined) cfg.tStop = Math.min(until, stopWas ?? Infinity);
        cfg.tFrame = frameAt;
        s.cfg = cfg;
        s.writeParams();
      }
      for (;;) {
        if (app.sim() !== s) throw new Error('the run was replaced while stepping');
        let left = steps === undefined ? Infinity : steps - (d.steps - start);
        // About to stop: what is still in flight is read first, so the stop is judged on it.
        if (inFlight.length && (left <= 0 || (d.dt === 0 && rounds > 0) || (until !== undefined && d.t >= until * (1 - 1e-6)))) {
          await drain();
          continue;
        }
        if (left <= 0) return result(d, start, 'steps');
        if (until !== undefined && d.t >= until * (1 - 1e-6)) return result(d, start, 'time');
        // Held at the end time (or at `until`, a hair short of it in single precision). The
        // clock's dt is that of the last step, so only a step taken here tells.
        if (d.dt === 0 && rounds > 0) {
          if (until !== undefined && d.t >= until * (1 - 1e-4)) return result(d, start, 'time');
          // Held at a checkpoint short of the end time: write it, as the page's own loop does
          // when it plays, and go on to the next.
          if (checkpoint !== undefined && app.checkpointsOn() && d.t >= checkpoint * (1 - 1e-5) && !(d.t >= cfg.tEnd * (1 - 1e-5))) {
            await app.writeCheckpoint();
            if (app.sim() !== s) throw new Error('the run was replaced while stepping');
            // The page has armed the next one (none if checkpoints stopped on an error).
            checkpoint = cfg.tStop;
            if (checkpoint !== undefined && !(checkpoint > d.t * (1 + 1e-6))) throw new Error(`the checkpoint at t = ${d.t} could not be written, and holds the clock`);
            if (until !== undefined) cfg.tStop = Math.min(until, checkpoint ?? Infinity);
            s.cfg = cfg;
            s.writeParams();
            rounds = 0;
            continue;
          }
          // Short of every hard stop: held at the frame time.
          const hard = Math.min(Number.isFinite(cfg.tEnd) ? cfg.tEnd : Infinity, cfg.tStop ?? Infinity);
          if (frameAt !== undefined && d.t < hard * (1 - 1e-4)) return result(d, start, 'frame');
          app.setFinished(true);
          return result(d, start, 'end');
        }
        if (timeoutMs !== undefined && performance.now() - t0 > timeoutMs) {
          await drain();
          return result(d, start, 'timeout');
        }
        // Keep IN_FLIGHT batches submitted, none past the next stop. Batch sizes double from one
        // step, carried over between calls, and are held to about BATCH_MS of GPU time: where
        // batches end is the same from run to run unless steps are slow.
        let resume: Promise<void> | null | undefined;
        left -= inFlightSteps;
        while (inFlight.length < IN_FLIGHT) {
          const size = Math.min(pace.grow, pace.msPerStep ? Math.max(1, Math.round(BATCH_MS / pace.msPerStep)) : Infinity);
          let n = Math.min(size, left, ahead());
          if (n <= 0) {
            if (inFlight.length) break;
            // The estimate is short (the step grew): one step at a time from here.
            n = 1;
          }
          const gpu = app.gpu();
          const enc = gpu.device.createCommandEncoder();
          const taken = s.encodeSteps(enc, n);
          const copy = taken ? clockCopy(enc, s) : null;
          gpu.device.queue.submit([enc.finish()]);
          s.submitted();
          if (!copy) {
            // An adaptive run waiting on a refinement check, or a star still settling.
            resume = s.stepsResume();
            break;
          }
          inFlight.push({ read: copy(), taken, at: performance.now() });
          inFlightSteps += taken;
          left -= taken;
        }
        if (inFlight.length) await settle();
        else if (resume !== undefined) await (resume ?? sleep(5));
      }
    } finally {
      if ((until !== undefined || frameAt !== undefined) && app.sim() === s) {
        cfg.tStop = stopWas;
        cfg.tFrame = undefined;
        s.cfg = cfg;
        await drain().catch(() => {});
        app.rearm(d.t);
      }
      app.setDriving(false);
    }
  }

  async function untilPrepared(s: Simulation, t0: number, timeoutMs?: number): Promise<void> {
    while (s.preparing) {
      if (timeoutMs !== undefined && performance.now() - t0 > timeoutMs) throw new Error('timed out while preparing the run (a star settling, or shaders compiling)');
      await sleep(20);
    }
  }

  async function waitPrepared(s: Simulation, t0: number, timeoutMs?: number): Promise<Diagnostics> {
    await untilPrepared(s, t0, timeoutMs);
    return s.diagnostics();
  }

  /** The state of the gas, read back now, and a sampler over it. */
  async function sampled() {
    const s = current();
    const snap = await capture(s);
    const species = app.cfg().problem.species ?? [];
    return { snap, sampler: new SnapshotSampler(snap), species };
  }

  // Screenshots and movie frames read back from the GPU rather than from the canvas: set by the
  // headless runner when the browser composites without the GPU (SHADEROUS_NO_GPU_COMPOSITING,
  // tools/headless/session.mjs), where a WebGPU canvas cannot be read.
  const readBack = (globalThis as { shaderousReadBack?: boolean }).shaderousReadBack === true;

  /**
   * Draw the view with fresh readouts, at `size` (or the canvas's own with null), and hand the
   * canvas to `read` in the same task as the draw: the canvas holds the frame only until the
   * task ends. With readBack, the frame is drawn off the canvas and handed over as a 2D canvas.
   */
  async function drawFresh<T>(s: Simulation, size: [number, number] | null, read: (c: HTMLCanvasElement) => T, asPage = false, known?: Diagnostics): Promise<T> {
    const d = s.preparing ? null : (known ?? (await s.diagnostics()));
    // A range readback still under way would leave this draw's range unread.
    await app.rangeSettled();
    if (d) app.refresh(d);
    app.draw(size, asPage, readBack);
    // Fields ranged by the display pass need one frame drawn before their range is known: draw
    // again with it. The state is the same, so the readouts are too.
    if (d && app.rangedByDrawing(d)) {
      await app.rangeSettled();
      app.refresh(d);
      app.draw(size, asPage, readBack);
    }
    return read(readBack ? await app.captured() : app.canvas);
  }

  async function screenshot(opts: { width?: number; height?: number; overlays?: OverlayOptions } = {}): Promise<string> {
    const s = current();
    const c = app.canvas;
    // The canvas's own size, unless asked for another, or the page is not laid out (a hidden tab).
    const max = app.gpu().device.limits.maxTextureDimension2D;
    const px = (v: number | undefined, fallback: number) => Math.max(16, Math.min(max, Math.round(v ?? fallback)));
    const laidOut = c.clientWidth > 0 && c.clientHeight > 0;
    const size: [number, number] | null =
      opts.width !== undefined || opts.height !== undefined || !laidOut ? [px(opts.width, 1024), px(opts.height, opts.width ?? 1024)] : null;
    const ov = opts.overlays;
    try {
      return await drawFresh(s, size, (canvas) => {
        if (!ov?.colorbar && !ov?.triad && !ov?.status) return canvas.toDataURL('image/png');
        const out = document.createElement('canvas');
        out.width = canvas.width;
        out.height = canvas.height;
        const g = out.getContext('2d')!;
        g.drawImage(canvas, 0, 0);
        paintOverlays(g, out.width, out.height, app.overlayInfo(), ov);
        return out.toDataURL('image/png');
      });
    } finally {
      if (size) app.draw(null);
    }
  }

  // The recording under way, if any.
  let recording: { status: RecordingStatus; stop: boolean } | null = null;

  async function record(opts: RecordOptions = {}): Promise<RecordResult> {
    const s = current();
    const cfg = app.cfg();
    if (opts.every !== undefined && opts.dt !== undefined) throw new Error('record() takes `every` or `dt`, not both');
    if (opts.every !== undefined && !(opts.every >= 1)) throw new Error('`every` is a number of steps, 1 or more');
    if (opts.dt !== undefined && !(opts.dt > 0)) throw new Error('`dt` is a time interval, more than 0');
    if (opts.frames !== undefined && !(opts.frames >= 1)) throw new Error('`frames` is a number of frames, 1 or more');
    // The encoder and muxer load with the first recording.
    const { MovieWriter, movieSize, frameLayout } = await import('../io/movie');
    const fps = Math.max(1, Math.min(240, opts.fps ?? 60));
    const max = Math.min(4096, app.gpu().device.limits.maxTextureDimension2D);
    const overlays: OverlayOptions = { colorbar: opts.overlays?.colorbar ?? true, triad: opts.overlays?.triad ?? true, status: opts.overlays?.status ?? true };
    // No size given: the view area as the page draws it, framed as on the page.
    const [vw, vh] = app.viewSize();
    const asPage = opts.width === undefined && opts.height === undefined && vw >= 16 && vh >= 16;
    // The size each frame is drawn at.
    const size: [number, number] = asPage
      ? movieSize(vw, vh, max)
      : movieSize(opts.width ?? 1920, opts.height ?? (opts.width !== undefined ? Math.round((opts.width * 9) / 16) : 1080), max);
    // Of the view area the movie takes the box as it is framed when the recording starts, with a
    // small margin, and room beside it for the axes and the color bar: not the empty sides of a
    // wide window. At a size asked for, the frame is the movie.
    let layout = { width: size[0], height: size[1], sx: 0, sy: 0, sw: size[0], sh: size[1], dx: 0, dy: 0, dw: size[0], dh: size[1] };
    if (asPage) {
      const info = app.overlayInfo();
      const ruler = document.createElement('canvas').getContext('2d');
      const measure = ruler
        ? (text: string, px: number) => {
            ruler.font = `${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
            return ruler.measureText(text).width;
          }
        : undefined;
      layout = frameLayout(size[0], size[1], app.viewBounds(), (h) => overlayGutters(h, info, overlays, measure), max);
    }
    const { width, height } = layout;
    const background = app.background();
    // By default a frame every 600th of the problem's dynamical time: one dynamical time plays
    // in 10 s at 60 fps.
    const frameDt = opts.every === undefined ? (opts.dt ?? defaultFrameDt(cfg.problem)) : undefined;
    const every = frameDt !== undefined ? 0 : Math.round(opts.every!);
    // Ten minutes of movie at most: the file is held in memory.
    const maxFrames = Math.min(opts.frames ?? Infinity, fps * 600);
    const writer = await MovieWriter.open({ width, height, fps, bitrate: opts.bitrate });
    const t0 = performance.now();
    const state = { status: { frames: 0, seconds: 0, t: 0 }, stop: false };
    recording = state;
    app.setRunning(false);
    app.setRecording(size);
    let stopped: RecordResult['stopped'] = 'stopped';
    let d = await waitPrepared(s, t0, opts.timeoutMs);
    const startSteps = d.steps;
    let frameSteps = -1;
    const frame = async () => {
      d = await s.diagnostics();
      await drawFresh(s, size, (canvas) => {
        const g = writer.ctx;
        g.fillStyle = background;
        g.fillRect(0, 0, width, height);
        // The drawn frame may be larger or smaller than asked for (a read-back): by its own size.
        const kx = canvas.width / size[0];
        const ky = canvas.height / size[1];
        g.drawImage(canvas, layout.sx * kx, layout.sy * ky, layout.sw * kx, layout.sh * ky, layout.dx, layout.dy, layout.dw, layout.dh);
        paintOverlays(g, width, height, app.overlayInfo(), overlays);
      }, asPage, d);
      await writer.add();
      frameSteps = d.steps;
      state.status = { frames: writer.frames, seconds: writer.seconds, t: d.t };
      app.recordProgress(state.status);
    };
    try {
      await frame();
      // The time cadence's next frame time: the first multiple of dt past the clock.
      let k = frameDt !== undefined ? Math.floor(d.t / frameDt + 1e-6) + 1 : 0;
      for (;;) {
        if (writer.frames >= maxFrames) { stopped = 'frames'; break; }
        if (state.stop) { stopped = 'stopped'; break; }
        if (opts.until !== undefined && d.t >= opts.until * (1 - 1e-6)) { stopped = 'time'; break; }
        const left = opts.timeoutMs === undefined ? undefined : opts.timeoutMs - (performance.now() - t0);
        if (left !== undefined && left <= 0) { stopped = 'timeout'; break; }
        if (app.sim() !== s) { stopped = 'replaced'; break; }
        let r: RunResult;
        try {
          r = frameDt !== undefined ? await run({ frameAt: k * frameDt, until: opts.until, timeoutMs: left }) : await run({ steps: every, until: opts.until, timeoutMs: left });
        } catch (e) {
          if (app.sim() !== s) { stopped = 'replaced'; break; }
          throw e;
        }
        if (r.stopped === 'timeout') { stopped = 'timeout'; break; }
        // A held clock that took no step since the last frame is at that frame's boundary
        // again: this frame time is nearest to it too, and is skipped, not recorded twice.
        if (r.steps !== frameSteps) await frame();
        if (r.stopped === 'end') { stopped = 'end'; break; }
        if (r.stopped === 'time') { stopped = 'time'; break; }
        if (frameDt !== undefined) {
          // The next frame time that lies more than half a step past the clock (as near as the
          // last step tells; one that does not is skipped above).
          const dt = frameDt;
          k = Math.max(k + 1, Math.floor((r.t + 0.5 * r.lastDt) / dt + 1e-6) + 1);
        }
      }
    } catch (e) {
      await writer.cancel();
      throw e;
    } finally {
      recording = null;
      if (app.sim() === s && cfg.tFrame !== undefined) {
        cfg.tFrame = undefined;
        s.cfg = cfg;
        s.writeParams();
      }
      app.setRecording(null);
      app.draw(null);
    }
    const bytes = await writer.finish();
    const name = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-t${+d.t.toPrecision(4)}.mp4`;
    if (opts.download ?? true) download(name, bytes, 'video/mp4');
    return {
      name, frames: writer.frames, seconds: writer.seconds, width, height, t: d.t, steps: d.steps - startSteps, bytes: bytes.length, stopped,
      ...(opts.base64 ? { base64: await toBase64(bytes) } : {}),
    };
  }

  const VIEW_KEYS: string[] = VIEW_OPTIONS.map((o) => o.key);

  function setView(changes: Partial<ViewState> & { colormap?: string; autoRange?: boolean }) {
    const { colormap, autoRange, ...v } = changes;
    for (const k of Object.keys(v)) if (!VIEW_KEYS.includes(k)) throw new Error(`unknown view setting "${k}": use one of ${[...VIEW_KEYS, ...VIEW_EXTRAS.map((o) => o.key)].join(', ')}`);
    if (v.field !== undefined && !(v.field >= 0 && v.field < FIELDS.length)) throw new Error(`field is an index into FIELDS, 0 to ${FIELDS.length - 1}`);
    if (v.mode !== undefined && !VIEW_MODES.includes(v.mode)) throw new Error(`mode is one of ${VIEW_MODES.join(', ')}`);
    if (colormap !== undefined) {
      if (!(colormap in COLORMAPS)) throw new Error(`unknown colormap "${colormap}": use one of ${Object.keys(COLORMAPS).join(', ')}`);
      app.setColormap(colormap);
    }
    Object.assign(app.view, v);
    if (v.lo !== undefined || v.hi !== undefined) app.setAutoRange(false);
    if (autoRange !== undefined) app.setAutoRange(autoRange);
    app.sync();
    return { ...app.view, colormap: colormap };
  }

  /** The panel's controls that are showing, with what each holds. */
  function controls() {
    const side = document.getElementById('side')!;
    const els = side.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>('input[id], select[id], button[id]');
    const out = [];
    for (const el of els) {
      if (el.closest('[hidden]') || el.disabled) continue;
      if (el.id.startsWith('sheet-') || el.id.endsWith('-slide')) continue;
      const box = el.closest('label, .f');
      const caption = box?.querySelector('span')?.textContent ?? box?.textContent ?? '';
      const label = (el.getAttribute('aria-label') ?? (caption.trim() || el.textContent?.trim()) ?? '').trim();
      const title = el.title || box?.getAttribute('title') || '';
      if (el instanceof HTMLButtonElement) out.push({ id: el.id, kind: 'button', label, title });
      else if (el instanceof HTMLSelectElement) {
        out.push({ id: el.id, kind: 'select', label, title, value: el.value, options: [...el.options].map((o) => ({ value: o.value, label: o.label })) });
      } else if (el.type === 'checkbox') out.push({ id: el.id, kind: 'checkbox', label, title, value: el.checked });
      else if (el.type !== 'file') {
        out.push({ id: el.id, kind: el.type, label, title, value: el.value, ...(el.min ? { min: +el.min } : {}), ...(el.max ? { max: +el.max } : {}) });
      }
    }
    return [...out, ...settings()];
  }

  /** The settings of the run's setup document (setup/options.ts), each with its value and whether it is available. */
  function settings() {
    const spec = app.runningSpec();
    const facts = factsOf(spec);
    const out = [];
    const listed = (o: Option) => ['number', 'select', 'check', 'text', 'vec3', 'faces'].includes(o.control.type);
    for (const o of flatten(OPTIONS)) {
      if (!listed(o)) continue;
      const why = unavailable(o, facts);
      out.push({ ...optionInfo(o, o.path), value: valueAt(spec, o.path) ?? null, ...(why.length ? { unavailable: why } : {}) });
    }
    (spec.parts ?? []).forEach((part, i) => {
      const own = factsWithout(spec, i);
      for (const o of flatten(partFields(part))) {
        if (!listed(o)) continue;
        const path = `parts.${i}.${o.path}`;
        const why = unavailable(o, own);
        out.push({ ...optionInfo({ ...o, linkKey: undefined }, path), part: part.id ?? part.kind, value: valueAt(spec, path) ?? null, ...(why.length ? { unavailable: why } : {}) });
      }
    });
    return out;
  }

  /** Set a setting of the setup document, by its link key, its path, or the name it had as a control of the bar. */
  async function setOption(id: string, value: unknown) {
    const spec = cloneSetup(app.runningSpec());
    const face = /^bc-(lo|hi)-([012])$/.exec(id);
    if (face) {
      // One face of the box; a periodic face takes the opposite one with it.
      const bc = cloneSetup((valueAt(spec, 'domain.bc') ?? { lo: [1, 1, 1], hi: [1, 1, 1] }) as { lo: number[]; hi: number[] });
      const a = +face[2];
      const was = bc[face[1] as 'lo' | 'hi'][a];
      bc[face[1] as 'lo' | 'hi'][a] = Number(value);
      if (Number(value) === 0 || was === 0) bc[face[1] === 'lo' ? 'hi' : 'lo'][a] = Number(value);
      setPath(spec, spec.domain?.ndim === 2 && spec.domain.bc2d ? 'domain.bc2d' : 'domain.bc', bc);
      return app.applySpec(spec);
    }
    const byKey = flatten(OPTIONS).find((o) => o.linkKey === id);
    const path = CONTROL_ALIASES[id] ?? byKey?.path ?? id;
    const o = optionAt(spec, path);
    if (!o) throw new Error(`no control "${id}": controls() lists them`);
    const c = o.control;
    let v: unknown = value;
    if (c.type === 'number') {
      if (value === '' || value === null || value === undefined) v = c.nullable ? null : undefined;
      else {
        const x = Number(value);
        if (!Number.isFinite(x)) throw new Error(`"${id}" takes a number, not "${value}"`);
        v = Math.min(c.max ?? Infinity, Math.max(c.min ?? -Infinity, c.integer ? Math.round(x) : x));
      }
    } else if (c.type === 'select') {
      const k = c.choices.find((q) => String(q.value) === String(value));
      if (!k) throw new Error(`"${id}" has no option "${value}": use one of ${c.choices.map((q) => q.value).join(', ')}`);
      v = k.value;
    } else if (c.type === 'check') v = value === true || value === 1 || value === '1' || value === 'true';
    setPath(spec, path, v);
    await app.applySpec(spec);
  }

  /** Set a control (or press a button) the way a person would, and wait for what it starts. */
  async function control(id: string, value?: string | number | boolean) {
    const el = document.getElementById(id);
    if (!el || !document.getElementById('side')!.contains(el)) {
      // Not a control of the bar: a setting of the setup document, applied as the Setup dialog applies it.
      await setOption(id, value);
      await sleep(0);
      await app.idle();
      app.setRunning(false);
      return;
    }
    if (el instanceof HTMLButtonElement) el.click();
    else if (el instanceof HTMLSelectElement) {
      if (![...el.options].some((o) => o.value === String(value))) {
        throw new Error(`"${id}" has no option "${value}": use one of ${[...el.options].map((o) => o.value).join(', ')}`);
      }
      el.value = String(value);
      el.dispatchEvent(new Event('change'));
    } else if (el instanceof HTMLInputElement) {
      if (el.type === 'checkbox') el.checked = !!value;
      else el.value = String(value);
      el.dispatchEvent(new Event('input'));
      el.dispatchEvent(new Event('change'));
    } else throw new Error(`"${id}" is not a control`);
    await sleep(0);
    await app.idle();
    // A control that starts the run playing (a new problem, a reset) is held: the caller steps.
    if (id !== 'run') app.setRunning(false);
  }

  const api = {
    /** Resolves once WebGPU is up and the first run has been built. */
    ready,
    build: BUILD,

    /** Problems, fields, resolutions and the choices a setup can make. */
    describe() {
      return {
        problems: FAMILIES.flatMap((f) =>
          f.variants.map(({ label: variant, problem: p }) => ({
            id: p.id,
            label: p.label,
            family: f.label,
            variant,
            blurb: p.blurb,
            supports2d: !!p.supports2d,
            debris: !!p.debris,
            grid: p.grid ?? null,
            tEnd: Number.isFinite(p.tEnd) ? p.tEnd : null,
            species: p.species ?? [],
            params: (p.params ?? []).map((q) => ({ id: q.id, label: q.label, value: q.value, min: q.min, max: q.max, title: q.title ?? '', live: !!q.live, kind: q.kind ?? 'number' })),
          })),
        ),
        fields: FIELDS.map((name, index) => ({ index, name })),
        variables: [...VARIABLES, 'level'],
        resolutions: app.resolutions(),
        eos: Object.keys(EOS_LABELS),
        cooling: Object.keys(COOLING_LABELS),
        riemann: [...RIEMANN_KINDS],
        limiter: [...LIMITER_KINDS],
        gravity: [...GRAVITY_MODES],
        colormaps: Object.keys(COLORMAPS),
        // What a setup document may hold (docs/setups.md): the general options, the kinds of
        // part with their fields, and the built-in flows with their parameters.
        setup: {
          format: SETUP_FORMAT,
          options: flatten(OPTIONS).filter((o) => o.control.type !== 'group').map((o) => optionInfo(o, o.path)),
          parts: Object.values(PART_OPTIONS).map((k) => ({ kind: k.kind, label: k.label, title: k.tip, fields: flatten(k.fields).filter((o) => o.control.type !== 'group').map((o) => optionInfo({ ...o, linkKey: undefined }, o.path)) })),
          flows: Object.values(FLOWS).map((f) => ({ name: f.name, label: f.label, title: f.tip, wholeBox: f.wholeBox, fields: f.fields })),
        },
      };
    },

    /**
     * Check a setup document without running it: whether it can be read, what reading it left
     * out, and what the Setup dialog would list under it. An `error` is a rule broken or a
     * value that cannot work (`load` may still build it, and the run then misbehaves); a
     * `warning` is a choice to look at again (a softening under two cells, a disk heavy
     * against what it orbits).
     */
    check(document: unknown) {
      const { spec, report } = readSetup(document, (id) => PRESET_FILES[id]);
      if (!spec) return { ok: false, error: report.error ?? 'not a setup document', notes: report.notes, issues: [] };
      const issues = checkSetup(spec, { memory: app.gridMemory(spec) });
      return { ok: !issues.some((i) => i.level === 'error'), notes: report.notes, issues, document: spec };
    },

    /** A preset's whole setup document: a starting point for one of your own. */
    preset(id: string) {
      const spec = presetSpec(id);
      if (!spec) throw new Error(`unknown problem "${id}": describe() lists them`);
      return cloneSetup(spec);
    },

    /** Load a setup (a query string, a URL, or an object of query keys) as a new run, paused at t = 0. */
    load: (setup: string | Record<string, unknown>, document?: unknown) =>
      serial(async () => {
        // A setup document alone, or after a query that says the grid and what differs from it.
        if (typeof setup === 'object' && setup !== null && setup.format === SETUP_FORMAT) [setup, document] = ['', setup];
        const q = setupQuery(setup as string | Record<string, string | number | boolean>);
        if (document !== undefined) {
          const { spec, report } = readSetup(document, (id) => PRESET_FILES[id]);
          if (!spec) throw new Error(`not a setup document: ${report.error}`);
          q.set('problem', app.adopt(spec));
          q.delete('setup');
        }
        if (q.get('problem') && q.get('problem') !== app.cfg()?.problem.id && !api.describe().problems.some((p) => p.id === q.get('problem')) && document === undefined) throw new Error(`unknown problem "${q.get('problem')}": describe() lists them`);
        await built(() => app.build(decodeLink(q)));
        return state();
      }),

    /**
     * Start a run from an .athdf file, given as base64: one of ours with its settings, or another
     * code's. `warnings` lists what an older file of ours could not carry over as written.
     */
    loadAthdf: (base64: string) =>
      serial(async () => {
        let warnings: string[] = [];
        await built(async () => {
          warnings = await app.importAthdf(fromBase64(base64));
        });
        return { ...(await state()), warnings };
      }),

    /** The run: problem, grid, clock, and the query string that reproduces its setup. */
    state: () => serial(state),

    /** Advance the run: a number of steps, to a time, or whichever comes first. */
    run: (opts: RunOptions) => serial(() => run(opts)),
    step: (n = 1) => serial(() => run({ steps: n })),
    runTo: (t: number, timeoutMs?: number) => serial(() => run({ until: t, timeoutMs })),

    /** Start again from the initial conditions (sources, sinks and edits removed), paused. */
    restart: () =>
      serial(async () => {
        await app.restart();
        await app.idle();
        app.setRunning(false);
        return state();
      }),

    /** Let the page's own loop step the run, as the Run button does, or stop it. */
    play: () => serial(async () => app.setRunning(true)),
    pause: () => serial(async () => app.setRunning(false)),

    /** Totals, extremes and point masses, read back now. Code units. */
    diagnostics: () => serial(async () => diagJson(await current().diagnostics())),

    /**
     * Debris energies: the gas binned by its specific orbital energy about the problem's point
     * mass, what has left the box and what is in it (a surviving core apart), and the return
     * rate of the bound part, dM/dt. Code units, with the time and mass units in seconds and
     * kilograms where the problem has them. Only for problems that track debris.
     */
    debris: () =>
      serial(async () => {
        const cfg = app.cfg();
        const r = await current().debris();
        if (!r) throw new Error(`${cfg.problem.id} does not track debris energies: see describe(), problems[].debris`);
        const u = problemUnits(cfg.problem, cfg.params);
        return { ...r, units: u ? { time: u.time, mass: u.mass } : null };
      }),

    /** Each variable's least, greatest and volume-weighted mean value over the box. */
    stats: () =>
      serial(async () => {
        const { snap } = await sampled();
        const per = snap.blockSize[0] * snap.blockSize[1] * snap.blockSize[2];
        const out: Record<string, { min: number; max: number; mean: number }> = {};
        for (const [name, data] of Object.entries(snap.data)) {
          if (!data) continue;
          let min = Infinity, max = -Infinity, sum = 0, vol = 0;
          for (let b = 0; b < snap.levels.length; b++) {
            const w = 2 ** (-snap.levels[b] * snap.ndim);
            for (let c = b * per; c < (b + 1) * per; c++) {
              const v = data[c];
              if (v < min) min = v;
              if (v > max) max = v;
              sum += v * w;
              vol += w;
            }
          }
          out[name] = { min, max, mean: sum / vol };
        }
        return { t: snap.time, steps: snap.steps, variables: out };
      }),

    /**
     * A variable on a plane through the box, on a res x res grid: `values[j * res + i]`, with i
     * along the first of the other two axes (x, or y for a slice along x) and j along the second.
     */
    slice: (opts: { variable?: string; axis?: number; pos?: number; res?: number } = {}) =>
      serial(async () => {
        const { snap, sampler, species } = await sampled();
        const variable = opts.variable ?? 'rho';
        const axis = snap.ndim === 2 ? 2 : (opts.axis ?? 2);
        const pos = opts.pos ?? 0.5;
        const res = Math.max(2, Math.min(2048, Math.round(opts.res ?? Math.min(256, app.cfg().n))));
        const value = variableOf(snap, sampler, species, variable);
        const [a, b] = [0, 1, 2].filter((k) => k !== axis);
        const values = new Array<number>(res * res);
        const p: Point = [0, 0, 0];
        p[axis] = pos;
        let min = Infinity, max = -Infinity;
        for (let j = 0; j < res; j++) {
          for (let i = 0; i < res; i++) {
            p[a] = (i + 0.5) / res;
            p[b] = (j + 0.5) / res;
            const v = value(p);
            values[j * res + i] = v;
            if (v < min) min = v;
            if (v > max) max = v;
          }
        }
        return { t: snap.time, variable, axis, pos, res, axes: [a, b], min, max, values };
      }),

    /** The primitive state (and refinement level) at points given in fractions of the box. */
    probe: (points: Point[]) =>
      serial(async () => {
        const { snap, sampler, species } = await sampled();
        return {
          t: snap.time,
          points: points.map((p) => {
            const w = sampler.prim(p[0], p[1], p[2], species);
            return { at: p, rho: w.rho, press: w.p, vel: [w.vx, w.vy, w.vz], r0: w.s, level: sampler.level(p[0], p[1], p[2]), ...(w.X ? { X: Object.fromEntries(species.map((k, i) => [k, w.X![i]])) } : {}) };
          }),
        };
      }),

    /**
     * The view as a PNG data URL: at the canvas's size on the page, or at `width` x `height`
     * pixels (a square of `width` if only that is given; 1024 x 1024 when the page has no layout).
     */
    screenshot: (opts?: { width?: number; height?: number; overlays?: OverlayOptions }) => serial(() => screenshot(opts)),

    /**
     * Record the run as an MP4 movie: the view with a color bar and a triad over it, a frame
     * every `every` steps or near every `dt` of code time, until stopped (stopRecording(), the
     * end time, `until`, `frames`). The run steps as it would have unrecorded.
     */
    record: (opts?: RecordOptions) => serial(() => record(opts)),
    /** End the recording under way: the file is finished with the frames recorded so far. */
    stopRecording: async () => {
      if (recording) recording.stop = true;
      return !!recording;
    },
    /** The recording under way, or null. */
    recordingStatus: async (): Promise<RecordingStatus | null> => (recording ? { ...recording.status } : null),

    /** Change what the view shows (ViewState keys, plus colormap and autoRange). The run is untouched. */
    view: (changes: Partial<ViewState> & { colormap?: string; autoRange?: boolean } = {}) => serial(async () => setView(changes)),

    /** The state of the gas as an .athdf file (HDF5, readable by yt and Athena++'s tools), base64. */
    athdf: () =>
      serial(async () => {
        current();
        const { time, bytes, stem } = await app.athdf();
        return { name: `${stem}-t${+time.toPrecision(4)}.athdf`, time, bytes: bytes.length, base64: await toBase64(bytes) };
      }),

    /** The panel's controls as they stand: id, kind, label, value, options. */
    controls: () => serial(async () => controls()),
    /** Set one of the panel's controls (or press a button), as a person would, mid-run. */
    control: (id: string, value?: string | number | boolean) => serial(() => control(id, value)),

    /** Everything the page has reported going wrong, oldest first. */
    errors: () => [...errorLog()],
  };
  Object.assign(window, { shaderous: api });
  return api;
}

export type ShaderousApi = ReturnType<typeof installApi>;
