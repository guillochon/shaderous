// Session logs: what was done to a run while it ran (the editing tools, settings changed, the
// adaptive mesh's re-meshes), each with the step it was applied at, so that the run can be
// replayed exactly (ui/api.ts, replay()) and what went wrong in it looked at again.
//
// What is logged is what reached the simulation, not the pointer's path: a brush's operation,
// a point mass as placed, the sources as set. The parts of a tool that read the run as it was
// (the densest gas a dab of mass is measured against, the speed a drag sets) are worked out
// once, live, and the log keeps their result, so a replay does not depend on them.
import type { BrushOp, Outline, SourceSpec } from '../sim';
import type { PointMass } from '../problems/types';
import type { SetupSpec } from '../setup/spec';

export const SESSION_FORMAT = 'shaderous-session';
/** 2: re-meshes as the steps the tree took, not the indicators (a hundred times smaller). */
export const SESSION_VERSION = 2;

type Vec3 = [number, number, number];

/** What a point mass placed by a tool is for (app.ts `placed`). */
export type MassRole = 'sink' | 'source' | 'query';

/** A star as the star tool places it. */
export interface StarAction {
  /** center (fractions of the box) and velocity (code units) */
  pos: Vec3;
  vel: Vec3;
  /** polytropic index, central density, radius (fraction of the box) */
  n: number;
  rhoc: number;
  radius: number;
  /** its magnetic field, and the plasma beta at its center */
  field: 'none' | 'dipole' | 'random';
  beta: number;
}

export type Action =
  /** a brush's operation, as the passes take it (edit/brush_core.wgsl, edit/field_brush_core.wgsl) */
  | { kind: 'brush'; op: BrushOp }
  /** a point mass added, and what for */
  | { kind: 'mass'; spec: PointMass; role: MassRole }
  /** point masses taken out of the run, by their places among its masses */
  | { kind: 'remove'; indices: number[] }
  /** a tracer held where it is, or let go to follow the forces on it, by its place among the run's masses */
  | { kind: 'fix'; index: number; fixed: boolean }
  /** the sources and sinks, all of them as set */
  | { kind: 'sources'; list: SourceSpec[] }
  /** the gas's total mass told to self-gravity (after a tool added or removed gas) */
  | { kind: 'massChanged'; mass: number }
  /** a star placed with the star tool (settled, and given its field, as the tool does) */
  | { kind: 'star'; star: StarAction }
  /**
   * Settings changed mid-run: link keys and their new values (null: back to the default), for
   * reading, and the run's setup document after the change, which a replay applies as the
   * Setup dialog does
   */
  | { kind: 'set'; changes: Record<string, string | null>; spec: SetupSpec }
  /**
   * A re-mesh of the adaptive mesh, as the steps the tree took: [what, level, coordinates...] of
   * each block it refined or merged, in order (amr/tree.ts AdaptStep). The decision is read back
   * from the GPU while the run goes on, so the step it lands on depends on timing; a replay
   * takes the same steps at the step logged, which gives the same mesh. (A log of the first
   * version kept `indicators` instead: every leaf's refinement indicator, base64 float32, in the
   * order BlockTree.leaves() lists them; a replay still takes those.)
   */
  | { kind: 'regrid'; steps?: number[][]; indicators?: string }
  /**
   * Per-block timesteps paused their cycles for `pause` single steps, because the last cycles
   * saved too few block updates. The saving is read back from the GPU while the run goes on, so
   * the step a pause lands on depends on timing; a replay pauses at the step logged instead of
   * measuring it again.
   */
  | { kind: 'bins'; pause: number }
  /**
   * The clock's stop changed: the time the step that reaches it is shortened to land on (the
   * end time, a checkpoint, a scripted run's `until`; null for none) and the frame time it is
   * held at (a movie's). A replay sets them at the same step, so its steps are the run's.
   */
  | { kind: 'clock'; stop: number | null; frame: number | null };

export interface Entry {
  /** steps taken when it was applied (as diagnostics() counts them), and the time then */
  steps: number;
  t: number;
  action: Action;
}

/** The run's diagnostics as the page read them back, for looking at what happened. */
export interface TraceRow {
  steps: number;
  t: number;
  dt: number;
  mass: number;
  eTot: number;
  maxMach: number;
  floored: number;
  bMax?: number;
  betaMin?: number;
  divB?: number;
}

export interface SessionLog {
  format: typeof SESSION_FORMAT;
  version: number;
  /** the page's build */
  build: string;
  /** the setup the run started from, as the page's address has it */
  query: string;
  /** false when the run started from something a replay cannot rebuild (an imported file) */
  replayable: boolean;
  /** why it is not replayable */
  why?: string;
  /** when the run started (ISO 8601) */
  started: string;
  entries: Entry[];
  trace: TraceRow[];
}

/** Most diagnostics rows a log keeps (the oldest go first). */
const MAX_TRACE = 50_000;

/** Collects a run's log. Each entry's step is read back from the GPU, so it is filled in later. */
export class SessionRecorder {
  private log: SessionLog | null = null;
  private stamps: Promise<unknown>[] = [];

  /** Begin the log of a run that starts now from `query`. */
  start(query: string, build: string, replayable = true, why?: string) {
    this.log = { format: SESSION_FORMAT, version: SESSION_VERSION, build, query, replayable, ...(why ? { why } : {}), started: new Date().toISOString(), entries: [], trace: [] };
    this.stamps = [];
  }

  /** Stop logging (no run, or one that cannot be described). */
  stop() {
    this.log = null;
    this.stamps = [];
  }

  get active(): boolean {
    return this.log !== null;
  }

  /** Log an action applied now; `stamp` resolves to the steps taken and the time when it was. */
  add(action: Action, stamp: Promise<{ steps: number; t: number }>) {
    const log = this.log;
    if (!log) return;
    // A deep copy: the caller's objects may change afterwards (the sources' list does).
    const entry: Entry = { steps: -1, t: 0, action: JSON.parse(JSON.stringify(action)) };
    log.entries.push(entry);
    const done = stamp.then(
      (s) => {
        entry.steps = s.steps;
        entry.t = s.t;
      },
      () => {},
    );
    this.stamps.push(done);
  }

  trace(row: TraceRow) {
    const log = this.log;
    if (!log) return;
    const last = log.trace[log.trace.length - 1];
    if (last && last.steps === row.steps) return;
    log.trace.push(row);
    if (log.trace.length > MAX_TRACE) log.trace.splice(0, log.trace.length - MAX_TRACE);
  }

  /** The log as it stands, once every entry's step has been read back (a copy). */
  async snapshot(): Promise<SessionLog | null> {
    await Promise.all(this.stamps);
    return this.log ? JSON.parse(JSON.stringify(this.log)) : null;
  }
}

/** Whether a value read as a session log is one. */
export function isSessionLog(x: unknown): x is SessionLog {
  const o = x as SessionLog;
  return !!o && o.format === SESSION_FORMAT && typeof o.query === 'string' && Array.isArray(o.entries);
}

/**
 * The settings that differ between two setup queries (ui/link.ts): each key that is new or has
 * changed with its new value, and each key that went away with null (back to its default).
 */
export function queryDiff(before: string, after: string): Record<string, string | null> {
  const a = new URLSearchParams(before);
  const b = new URLSearchParams(after);
  const out: Record<string, string | null> = {};
  for (const [k, v] of b) if (a.get(k) !== v) out[k] = v;
  for (const k of a.keys()) if (!b.has(k)) out[k] = null;
  return out;
}

/** The pull tool's settings: brush radius and softness (fractions of the box), and its grip. */
export interface PullSettings {
  radius: number;
  softness: number;
  grip: number;
}

/**
 * One frame of the pull tool: from where the field under the pointer has been carried to
 * (`from`) toward the pointer (`to`), at most as far as a frame may carry it. Across the brush's
 * soft rim its weight w falls from 1 to 0, so a drag d squeezes the field ahead by
 * 1 / (1 - grip |grad w| d) (and stretches it behind): d is kept to 0.15 of the rim's width over
 * the grip, which holds that to 1.3 at most (smoothstep's steepest slope is 1.5 over the width),
 * and to 30 cells of the finest level (`n` across the box), which the passes cut into substeps
 * of half a cell. The rim is at least 0.3 of the radius wide: a hard edge would tear the field.
 * Returns the brush operation (null when there is nothing to carry), where the field has been
 * carried to after it, and whether that is `to`.
 */
export function pullStep(from: Vec3, to: Vec3, s: PullSettings, n: number, flat: boolean): { op: BrushOp | null; at: Vec3; done: boolean } {
  const soft = Math.max(s.softness, 0.3);
  const grip = Math.min(Math.max(s.grip, 0.01), 0.9);
  let d = [0, 1, 2].map((i) => to[i] - from[i]) as Vec3;
  if (flat) d[2] = 0;
  const len = Math.hypot(...d);
  const cap = Math.min((0.15 * soft * s.radius) / grip, 0.5 * s.radius, 30 / (n * grip));
  if (len > cap) d = d.map((v) => (v * cap) / len) as Vec3;
  if (len < 1e-6) return { op: null, at: from, done: true };
  const op: BrushOp = {
    kind: 'pull', radius: s.radius, softness: soft, amount: grip,
    pos: [0, 1, 2].map((i) => from[i] + 0.5 * d[i]) as Vec3, vel: d,
  };
  return { op, at: [0, 1, 2].map((i) => from[i] + d[i]) as Vec3, done: len <= cap };
}

/** The operations of a drag of the pull tool from `from` to `to`, frame by frame as the tool takes it. */
export function pullOps(from: Vec3, to: Vec3, s: PullSettings, n: number, flat: boolean): BrushOp[] {
  const ops: BrushOp[] = [];
  let at = from;
  for (let i = 0; i < 100_000; i++) {
    const r = pullStep(at, to, s, n, flat);
    if (r.op) ops.push(r.op);
    at = r.at;
    if (r.done) break;
  }
  return ops;
}

/** An obstacle's shape in a plane of the box: coordinates are fractions of the box along the plane's two axes. */
export type PlaneShape =
  | { shape: 'rect'; center: [number, number]; half: [number, number] }
  | { shape: 'ellipse'; center: [number, number]; half: [number, number] }
  | { shape: 'polygon'; points: [number, number][] };

/**
 * The outline of an obstacle drawn in the plane normal to `axis` (z in 2D) and cut through the
 * box along that axis, as the obstacle tool's outline is cut along the line of sight. Its
 * matrix takes a point of the box to its coordinates along the plane's two axes, in order (y
 * and z for a plane normal to x, x and z for y, x and y for z).
 */
export function planeOutline(s: PlaneShape, axis: 0 | 1 | 2): Outline {
  const [ia, ib] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
  const rx = [0, 0, 0, 0];
  const ry = [0, 0, 0, 0];
  rx[ia] = 1;
  ry[ib] = 1;
  const rw = [0, 0, 0, 1];
  const matrix = [0, 1, 2, 3].flatMap((c) => [rx[c], ry[c], 0, rw[c]]);
  if (s.shape === 'ellipse') return { matrix, ellipse: [s.center[0], s.center[1], s.half[0], s.half[1]] };
  if (s.shape === 'rect') {
    const [cx, cy] = s.center;
    const [hx, hy] = s.half;
    return { matrix, polygon: [cx - hx, cy - hy, cx + hx, cy - hy, cx + hx, cy + hy, cx - hx, cy + hy] };
  }
  return { matrix, polygon: s.points.flat() };
}

/** Float32 values as base64 (little-endian), and back. */
export function f32ToBase64(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
export function base64ToF32(b64: string): Float32Array {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
