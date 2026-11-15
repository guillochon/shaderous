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
import { SnapshotSampler, VARIABLES, type Snapshot } from '../io/snapshot';
import type { Diagnostics, SimConfig, Simulation } from '../sim';
import { EOS_LABELS } from '../eos/eos';
import { COOLING_LABELS } from '../cooling/cooling';
import { FAMILIES, FIELDS, GRAVITY_MODES } from '../problems';
import { problemUnits } from '../problems/types';
import type { ViewState } from '../render/renderer';
import { COLORMAPS } from '../render/transfer';
import { paintOverlays, type OverlayInfo, type OverlayOptions } from '../render/overlays';
import { download } from '../io/checkpoints';
import { BUILD, errorLog } from './errors';

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
  draw(size: [number, number] | null): void;
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
  /** Checkpoints are on: the clock is held at each (cfg.tStop) until it is written. */
  checkpointsOn(): boolean;
  /** Write the checkpoint the clock is held at, and arm the next (as the page does when it plays). */
  writeCheckpoint(): Promise<void>;
  /** The page's measured step rate when it last played, in steps per second (0 if unknown). */
  stepRate(): number;
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

/** Options of record(). */
export interface RecordOptions {
  /** A frame every this many steps (the default cadence)... */
  every?: number;
  /**
   * ...or a frame near every multiple of this much code time: at the step boundary nearest to
   * it, so that no step is shortened and the run is the one it would have been unrecorded.
   */
  dt?: number;
  /** Stop after this many frames (the first, at the start, included). */
  frames?: number;
  /** Stop when the clock reaches this time (landed on exactly, as run() does). */
  until?: number;
  /** Give up after this much wall time, finishing the file with what was recorded. */
  timeoutMs?: number;
  /** Playback rate, frames per second (default 30). */
  fps?: number;
  /** The movie's size in pixels (default 1920 x 1080), rounded down to even numbers. */
  width?: number;
  height?: number;
  /** Bits per second (default about 0.2 bits per pixel per frame). */
  bitrate?: number;
  /** The color bar and the triad, both on by default. */
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
      adapter: app.adapter(),
    };
  }

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
    let lastDt = 0;
    const result = (d: Diagnostics, start: number, stopped: RunResult['stopped']): RunResult => ({
      t: d.t, dt: d.dt, steps: d.steps, taken: d.steps - start, seconds: (performance.now() - t0) / 1000, stopped, lastDt: d.dt > 0 ? d.dt : lastDt,
    });
    try {
      let d = await waitPrepared(s, t0, timeoutMs);
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
      let batch = 1;
      let rounds = 0;
      for (;;) {
        if (app.sim() !== s) throw new Error('the run was replaced while stepping');
        const left = steps === undefined ? Infinity : steps - (d.steps - start);
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
        if (timeoutMs !== undefined && performance.now() - t0 > timeoutMs) return result(d, start, 'timeout');
        const gpu = app.gpu();
        const enc = gpu.device.createCommandEncoder();
        const taken = s.encodeSteps(enc, Math.min(batch, left));
        gpu.device.queue.submit([enc.finish()]);
        s.submitted();
        if (!taken) {
          // An adaptive run waiting on a refinement check, or a star still settling.
          const resume = s.stepsResume();
          await (resume ?? sleep(5));
          continue;
        }
        const b0 = performance.now();
        d = await s.diagnostics();
        rounds++;
        if (d.dt > 0) lastDt = d.dt;
        if (!(d.t === d.t) || !(d.dt === d.dt)) throw new Error(`the clock is NaN after ${d.steps} steps`);
        // Batches of about a tenth of a second of GPU time: few readbacks, prompt returns.
        const ms = performance.now() - b0;
        if (ms < 60) batch = Math.min(batch * 2, 256);
        else if (ms > 200) batch = Math.max(1, batch >> 1);
      }
    } finally {
      if ((until !== undefined || frameAt !== undefined) && app.sim() === s) {
        cfg.tStop = stopWas;
        cfg.tFrame = undefined;
        s.cfg = cfg;
        app.rearm((await s.diagnostics()).t);
      }
      app.setDriving(false);
    }
  }

  async function waitPrepared(s: Simulation, t0: number, timeoutMs?: number): Promise<Diagnostics> {
    while (s.preparing) {
      if (timeoutMs !== undefined && performance.now() - t0 > timeoutMs) throw new Error('timed out while preparing the run (a star settling, or shaders compiling)');
      await sleep(20);
    }
    return s.diagnostics();
  }

  /** The state of the gas, read back now, and a sampler over it. */
  async function sampled() {
    const s = current();
    const snap = await capture(s);
    const species = app.cfg().problem.species ?? [];
    return { snap, sampler: new SnapshotSampler(snap), species };
  }

  /**
   * Draw the view with fresh readouts, at `size` (or the canvas's own with null), and hand the
   * canvas to `read` in the same task as the draw: the canvas holds the frame only until the
   * task ends.
   */
  async function drawFresh<T>(s: Simulation, size: [number, number] | null, read: (c: HTMLCanvasElement) => T): Promise<T> {
    if (!s.preparing) app.refresh(await s.diagnostics());
    // Fields ranged by the display pass need one frame drawn before their range is known.
    app.draw(size);
    await app.gpu().device.queue.onSubmittedWorkDone();
    if (!s.preparing) app.refresh(await s.diagnostics());
    app.draw(size);
    return read(app.canvas);
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
        if (!ov?.colorbar && !ov?.triad) return canvas.toDataURL('image/png');
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
    const { MovieWriter, movieSize } = await import('../io/movie');
    const fps = Math.max(1, Math.min(240, opts.fps ?? 30));
    const max = Math.min(4096, app.gpu().device.limits.maxTextureDimension2D);
    const [width, height] = movieSize(opts.width ?? 1920, opts.height ?? (opts.width !== undefined ? Math.round((opts.width * 9) / 16) : 1080), max);
    const size: [number, number] = [width, height];
    // A frame per this many steps: by default, about 30 frames for each second the run took
    // stepping on its own, so that recording costs little.
    const every = opts.dt !== undefined ? 0 : Math.round(opts.every ?? Math.max(1, Math.round(app.stepRate() / 30)));
    // Ten minutes of movie at most: the file is held in memory.
    const maxFrames = Math.min(opts.frames ?? Infinity, fps * 600);
    const overlays: OverlayOptions = { colorbar: opts.overlays?.colorbar ?? true, triad: opts.overlays?.triad ?? true };
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
      await drawFresh(s, size, (canvas) => {
        const g = writer.ctx;
        g.drawImage(canvas, 0, 0, width, height);
        paintOverlays(g, width, height, app.overlayInfo(), overlays);
      });
      await writer.add();
      d = await s.diagnostics();
      frameSteps = d.steps;
      state.status = { frames: writer.frames, seconds: writer.seconds, t: d.t };
      app.recordProgress(state.status);
    };
    try {
      await frame();
      // The time cadence's next frame time: the first multiple of dt past the clock.
      let k = opts.dt !== undefined ? Math.floor(d.t / opts.dt + 1e-6) + 1 : 0;
      for (;;) {
        if (writer.frames >= maxFrames) { stopped = 'frames'; break; }
        if (state.stop) { stopped = 'stopped'; break; }
        if (opts.until !== undefined && d.t >= opts.until * (1 - 1e-6)) { stopped = 'time'; break; }
        const left = opts.timeoutMs === undefined ? undefined : opts.timeoutMs - (performance.now() - t0);
        if (left !== undefined && left <= 0) { stopped = 'timeout'; break; }
        if (app.sim() !== s) { stopped = 'replaced'; break; }
        let r: RunResult;
        try {
          r = opts.dt !== undefined ? await run({ frameAt: k * opts.dt, until: opts.until, timeoutMs: left }) : await run({ steps: every, until: opts.until, timeoutMs: left });
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
        if (opts.dt !== undefined) {
          // The next frame time that lies more than half a step past the clock (as near as the
          // last step tells; one that does not is skipped above).
          const dt = opts.dt;
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

  const VIEW_KEYS = ['mode', 'field', 'species', 'log', 'lo', 'hi', 'sliceAxis', 'slicePos', 'minRho', 'opacity', 'yaw', 'pitch', 'dist', 'zoom', 'cx', 'cy', 'showBlocks', 'contours', 'nContours', 'fog', 'surfaceOpacity', 'light', 'box', 'fieldLines', 'fieldLineCount'];

  function setView(changes: Partial<ViewState> & { colormap?: string; autoRange?: boolean }) {
    const { colormap, autoRange, ...v } = changes;
    for (const k of Object.keys(v)) if (!VIEW_KEYS.includes(k)) throw new Error(`unknown view setting "${k}": use one of ${[...VIEW_KEYS, 'colormap', 'autoRange'].join(', ')}`);
    if (v.field !== undefined && !(v.field >= 0 && v.field < FIELDS.length)) throw new Error(`field is an index into FIELDS, 0 to ${FIELDS.length - 1}`);
    if (v.mode !== undefined && !['volume', 'slice', 'projection'].includes(v.mode)) throw new Error('mode is volume, slice or projection');
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
    return out;
  }

  /** Set a control (or press a button) the way a person would, and wait for what it starts. */
  async function control(id: string, value?: string | number | boolean) {
    const el = document.getElementById(id);
    if (!el || !document.getElementById('side')!.contains(el)) throw new Error(`no control "${id}": controls() lists them`);
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
        riemann: ['hllc', 'hllclm', 'hll'],
        limiter: ['koren', 'mc', 'vanleer', 'minmod'],
        gravity: [...GRAVITY_MODES],
        colormaps: Object.keys(COLORMAPS),
      };
    },

    /** Load a setup (a query string, a URL, or an object of query keys) as a new run, paused at t = 0. */
    load: (setup: string | Record<string, string | number | boolean>) =>
      serial(async () => {
        const q = setupQuery(setup);
        if (q.get('problem') && !api.describe().problems.some((p) => p.id === q.get('problem'))) throw new Error(`unknown problem "${q.get('problem')}": describe() lists them`);
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
