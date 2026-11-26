// The profile tool's arithmetic and its plot (plans/profile-tool.md): a field along a line through
// the box. The line's ends are points of the box in fractions of its computed part, as locate()
// in app.ts gives them; the GPU samples the field at points along it (render/project.wgsl
// profile). Nothing here looks at the page but the plot, which draws into a canvas it is given.

import { FIELDS, MASS_FRACTION_FIELD } from '../problems/types';
import { fmt3 } from './diag_plot';

export type Vec3 = [number, number, number];

/** Fewest and most samples along a line, by default and when asked for. */
export const PROFILE_MIN_SAMPLES = 64;
export const PROFILE_DEFAULT_MAX = 4096;
export const PROFILE_MAX_SAMPLES = 16384;

/**
 * The part of the segment from `a` to `b` inside the box [0, 1]^dims (Liang-Barsky), or null if
 * it misses the box or is a point. In 2D the third coordinate is taken as 0.
 */
export function clipSegment(a0: readonly number[], b0: readonly number[], dims: 2 | 3): [Vec3, Vec3] | null {
  const a: Vec3 = [a0[0], a0[1], dims === 3 ? (a0[2] ?? 0) : 0];
  const b: Vec3 = [b0[0], b0[1], dims === 3 ? (b0[2] ?? 0) : 0];
  if (![...a, ...b].every(Number.isFinite)) return null;
  let t0 = 0;
  let t1 = 1;
  for (let k = 0; k < dims; k++) {
    const d = b[k] - a[k];
    for (const [p, q] of [[-d, a[k]], [d, 1 - a[k]]]) {
      if (p === 0) {
        if (q < 0) return null;
      } else {
        const r = q / p;
        if (p < 0) t0 = Math.max(t0, r);
        else t1 = Math.min(t1, r);
      }
    }
  }
  if (!(t1 > t0)) return null;
  const at = (t: number): Vec3 => a.map((v, k) => Math.min(1, Math.max(0, v + t * (b[k] - v)))) as Vec3;
  const [p, q] = [at(t0), at(t1)];
  return p.every((v, k) => v === q[k]) ? null : [p, q];
}

/**
 * How many samples to take along a line `cells` cells of the finest level long: two a cell, held
 * between PROFILE_MIN_SAMPLES and PROFILE_DEFAULT_MAX, or `n` if given (2 to PROFILE_MAX_SAMPLES).
 */
export function sampleCount(cells: number, n?: number): number {
  if (n !== undefined) {
    if (!Number.isFinite(n)) throw new Error('n must be a number');
    return Math.max(2, Math.min(PROFILE_MAX_SAMPLES, Math.round(n)));
  }
  return Math.max(PROFILE_MIN_SAMPLES, Math.min(PROFILE_DEFAULT_MAX, Math.ceil(2 * cells)));
}

/** `n` points evenly along the segment from `a` to `b`, both ends included. */
export function samplePoints(a: Vec3, b: Vec3, n: number): Vec3[] {
  return Array.from({ length: n }, (_, i) => {
    const t = n > 1 ? i / (n - 1) : 0;
    return a.map((v, k) => v + t * (b[k] - v)) as Vec3;
  });
}

/** The variables slice() and probe() take, as the view's fields that show them. */
const VARIABLE_FIELDS: Record<string, number> = { rho: 0, press: 3, vel1: 5, vel2: 6, vel3: 7, r0: 2, temp: 1, level: 14 };

/**
 * A field asked for by index into FIELDS, by its name (any case), by a variable name of slice()
 * (rho, press, vel1..3, r0, temp, level), or by a species' name (its mass fraction). Undefined
 * gives `fallback`.
 */
export function resolveField(spec: string | number | undefined, species: readonly string[], fallback: { field: number; species: number }): { field: number; species: number } {
  if (spec === undefined || spec === '') return fallback;
  const fail = (): never => {
    throw new Error(`unknown field "${spec}": use an index into describe().fields, one of their names, ${Object.keys(VARIABLE_FIELDS).join(', ')}${species.length ? `, or a species (${species.join(', ')})` : ''}`);
  };
  if (typeof spec === 'number' || /^\d+$/.test(spec)) {
    const i = Number(spec);
    if (!Number.isInteger(i) || i < 0 || i >= FIELDS.length) fail();
    if (i === MASS_FRACTION_FIELD && !species.length) fail();
    return { field: i, species: i === MASS_FRACTION_FIELD ? fallback.species : 0 };
  }
  const s = spec.trim();
  const k = species.indexOf(s);
  if (k >= 0) return { field: MASS_FRACTION_FIELD, species: k };
  if (s in VARIABLE_FIELDS) return { field: VARIABLE_FIELDS[s], species: 0 };
  const i = FIELDS.findIndex((f) => f.toLowerCase() === s.toLowerCase());
  if (i < 0 || i === MASS_FRACTION_FIELD) fail();
  return { field: i, species: 0 };
}

/** A profile as CSV: distance, value and refinement level, one sample a row. */
export function profileCsv(s: ArrayLike<number>, values: ArrayLike<number>, levels: ArrayLike<number>, head: { distance: string; value: string }): string {
  const q = (x: string) => (/[",\n]/.test(x) ? `"${x.replace(/"/g, '""')}"` : x);
  const rows = [`${q(head.distance)},${q(head.value)},level`];
  for (let i = 0; i < s.length; i++) rows.push(`${s[i]},${values[i]},${levels[i]}`);
  return rows.join('\n') + '\n';
}

/** Distance along the line in code length, from the ends' positions in code length. */
export function distances(a: readonly number[], b: readonly number[], n: number): Float64Array {
  const len = Math.hypot(...a.map((v, k) => b[k] - v));
  return Float64Array.from({ length: n }, (_, i) => (n > 1 ? (len * i) / (n - 1) : 0));
}

/** Tick values for an axis from lo to hi: 1, 2 or 5 times a power of ten apart, about `want` of them. */
export function ticks(lo: number, hi: number, want = 4): number[] {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const raw = (hi - lo) / want;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
  const out: number[] = [];
  // From whole multiples of the step, rounded off: no 0.6000000000000001.
  for (let i = Math.ceil(lo / step - 1e-9); i <= Math.floor(hi / step + 1e-9); i++) out.push(+(i * step).toPrecision(12));
  return out;
}

/** What the plot draws: distance and value of each sample (displayed units), with its options. */
export interface ProfilePlot {
  s: ArrayLike<number>;
  values: ArrayLike<number>;
  /** solid samples, left out of the line */
  solid?: ArrayLike<number>;
  log: boolean;
  /** the y range to draw, eased by the caller; null to fit the values */
  range?: [number, number] | null;
  /** a sample to mark (the one hovered), or -1 */
  mark?: number;
  /** refinement levels, whose changes are marked along the bottom */
  levels?: ArrayLike<number>;
  colors: { line: string; dim: string; frame: string; mark: string };
}

/** The plot's margins, in CSS pixels. */
export const PLOT_MARGIN = { left: 50, right: 8, top: 6, bottom: 18 };

/** The y range of the values to plot (logs of the positive ones with `log`), or null if none. */
export function valueRange(values: ArrayLike<number>, log: boolean, solid?: ArrayLike<number>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (solid?.[i]) continue;
    const v = log ? (values[i] > 0 ? Math.log10(values[i]) : NaN) : values[i];
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!(hi >= lo)) return null;
  if (hi === lo) {
    const pad = log ? 0.5 : Math.abs(lo) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

/** Draw the plot into a 2D context of CSS size w x h. */
export function drawProfilePlot(ctx: CanvasRenderingContext2D, w: number, h: number, p: ProfilePlot) {
  const m = PLOT_MARGIN;
  const [x0, y0, pw, ph] = [m.left, m.top, w - m.left - m.right, h - m.top - m.bottom];
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = p.colors.frame;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.fillStyle = p.colors.dim;
  const n = p.s.length;
  const range = p.range ?? valueRange(p.values, p.log, p.solid);
  if (n < 2 || !range || pw < 10 || ph < 10) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.log && n >= 2 ? 'no positive values to show on a log scale' : 'nothing to show', x0 + pw / 2, y0 + ph / 2);
    return;
  }
  const [lo, hi] = range;
  const [sa, sb] = [p.s[0], p.s[n - 1]];
  const px = (s: number) => x0 + ((s - sa) / (sb - sa || 1)) * pw;
  const py = (v: number) => y0 + ph - ((v - lo) / (hi - lo)) * ph;
  const yOf = (v: number) => (p.log ? (v > 0 ? Math.log10(v) : NaN) : v);

  // Ticks: on a log scale at whole decades when there are a few, else as on a linear one.
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yt = p.log && hi - lo >= 1.5 ? ticks(Math.ceil(lo), Math.floor(hi), 4).filter(Number.isInteger) : ticks(lo, hi, 4);
  for (const t of yt) {
    const y = Math.round(py(t)) + 0.5;
    ctx.strokeStyle = p.colors.frame;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + pw, y);
    ctx.stroke();
    ctx.fillText(fmt3(p.log ? 10 ** t : t), x0 - 4, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of ticks(sa, sb, Math.max(2, Math.floor(pw / 50)))) ctx.fillText(fmt3(t), px(t), y0 + ph + 3);

  // Where the refinement level changes: a short mark on the bottom edge.
  if (p.levels) {
    ctx.strokeStyle = p.colors.dim;
    ctx.beginPath();
    for (let i = 1; i < n; i++) {
      if (p.levels[i] === p.levels[i - 1]) continue;
      const x = Math.round(px((p.s[i] + p.s[i - 1]) / 2)) + 0.5;
      ctx.moveTo(x, y0 + ph - 4);
      ctx.lineTo(x, y0 + ph);
    }
    ctx.stroke();
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, pw, ph);
  ctx.clip();
  ctx.strokeStyle = p.colors.line;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < n; i++) {
    const v = yOf(p.values[i]);
    if (p.solid?.[i] || !Number.isFinite(v)) {
      pen = false;
      continue;
    }
    const [x, y] = [px(p.s[i]), py(v)];
    if (pen) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
    pen = true;
  }
  ctx.stroke();
  ctx.restore();

  const k = p.mark ?? -1;
  if (k >= 0 && k < n) {
    const x = Math.round(px(p.s[k])) + 0.5;
    ctx.strokeStyle = p.colors.mark;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    const v = yOf(p.values[k]);
    if (Number.isFinite(v) && !p.solid?.[k]) {
      ctx.fillStyle = p.colors.mark;
      ctx.beginPath();
      ctx.arc(x, Math.min(y0 + ph, Math.max(y0, py(v))), 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

/** The sample nearest a point x (CSS pixels) of a plot w wide, or -1 off the plot. */
export function sampleAt(x: number, w: number, s: ArrayLike<number>): number {
  const m = PLOT_MARGIN;
  const pw = w - m.left - m.right;
  const n = s.length;
  if (n < 2 || x < m.left - 4 || x > m.left + pw + 4) return -1;
  const t = Math.min(1, Math.max(0, (x - m.left) / pw));
  const target = s[0] + t * (s[n - 1] - s[0]);
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s[mid] < target) lo = mid;
    else hi = mid;
  }
  return Math.abs(s[lo] - target) <= Math.abs(s[hi] - target) ? lo : hi;
}
