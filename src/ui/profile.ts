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
export function clipSegment(a0: readonly number[], b0: readonly number[], dims: 1 | 2 | 3): [Vec3, Vec3] | null {
  // (A 1D run's y is where the line is drawn across the repeated axis: the middle if not given.)
  const a: Vec3 = [a0[0], a0[1] ?? 0.5, dims === 3 ? (a0[2] ?? 0) : 0];
  const b: Vec3 = [b0[0], b0[1] ?? 0.5, dims === 3 ? (b0[2] ?? 0) : 0];
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

/**
 * Profiles as CSV: distance, then each field's value (`head.values`, one column a field, in the
 * order given; a line without a field's samples leaves its column empty), then the refinement
 * level, one sample a row. With several lines, a first column numbers them (from 1), their rows
 * one line after another.
 */
export function profileCsv(lines: { s: ArrayLike<number>; values: (ArrayLike<number> | null | undefined)[]; levels: ArrayLike<number> }[], head: { distance: string; values: string[] }): string {
  const q = (x: string) => (/[",\n]/.test(x) ? `"${x.replace(/"/g, '""')}"` : x);
  const several = lines.length > 1;
  const rows = [`${several ? 'line,' : ''}${[head.distance, ...head.values].map(q).join(',')},level`];
  lines.forEach((l, j) => {
    for (let i = 0; i < l.s.length; i++) {
      const vals = head.values.map((_, k) => l.values[k]?.[i] ?? '');
      rows.push(`${several ? `${j + 1},` : ''}${l.s[i]},${vals.join(',')},${l.levels[i]}`);
    }
  });
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

/** The lines' colours, in the order lines take them (the first as the single line has always been). */
export const PROFILE_COLORS = ['#ffe066', '#33d6ff', '#ff6fae', '#7dff7a', '#ff9f43', '#b18cff'];
/** The most lines on show at once. */
export const MAX_PROFILES = PROFILE_COLORS.length;
/**
 * The fields' line patterns, in the order the panel lists them (as canvas line dashes): the
 * first solid, then dashed, dotted, dash-dot and so on, again from the second past the last.
 */
export const PROFILE_DASHES: number[][] = [[], [6, 3], [1.5, 2.5], [8, 2.5, 1.5, 2.5], [12, 3], [3, 2, 1, 2, 1, 2], [4, 4, 1, 4]];
export const profileDash = (k: number) => (k < PROFILE_DASHES.length ? PROFILE_DASHES[k] : PROFILE_DASHES[1 + ((k - 1) % (PROFILE_DASHES.length - 1))]);

/** One line's samples in the plot: distance and value of each (displayed units). */
export interface ProfileSeries {
  s: ArrayLike<number>;
  values: ArrayLike<number>;
  /** solid samples, left out of the line */
  solid?: ArrayLike<number>;
  /** refinement levels, whose changes are marked along the bottom */
  levels?: ArrayLike<number>;
  color: string;
}

/** What the plot draws: the lines' samples against distance from each line's start, with its options. */
export interface ProfilePlot {
  series: ProfileSeries[];
  log: boolean;
  /** the y range to draw, eased by the caller; null to fit the values */
  range?: [number, number] | null;
  /** a distance to mark (the one hovered), or null */
  mark?: number | null;
  /**
   * More fields along the same lines, each in its pattern (profileDash, from the second): the
   * first of them against an axis on the right, the rest each on its own range, unlabelled.
   */
  more?: { series: ProfileSeries[]; log: boolean; range?: [number, number] | null }[];
  colors: { dim: string; frame: string; mark: string };
}

/** The plot's margins, in CSS pixels. */
export const PLOT_MARGIN = { left: 50, right: 8, top: 6, bottom: 18 };

/** The y range of the values to plot (logs of the positive ones with `log`), or null if none. */
export function valueRange(values: ArrayLike<number>, log: boolean, solid?: ArrayLike<number>): [number, number] | null {
  return seriesRange([{ values, solid }], log);
}

/** The y range of several lines' values together, as valueRange. */
export function seriesRange(series: { values: ArrayLike<number>; solid?: ArrayLike<number> }[], log: boolean): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const { values, solid } of series) {
    for (let i = 0; i < values.length; i++) {
      if (solid?.[i]) continue;
      const v = log ? (values[i] > 0 ? Math.log10(values[i]) : NaN) : values[i];
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!(hi >= lo)) return null;
  if (hi === lo) {
    const pad = log ? 0.5 : Math.abs(lo) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

/** The x axis's end: the longest line's length (each line starts at distance 0). */
export const plotLength = (series: { s: ArrayLike<number> }[]) => Math.max(0, ...series.map((q) => (q.s.length ? q.s[q.s.length - 1] : 0)));

/** The plot's margins, in CSS pixels; with a second field the right one holds its axis. */
const plotMargin = (second: boolean) => ({ ...PLOT_MARGIN, right: second ? PLOT_MARGIN.left : PLOT_MARGIN.right });

/** One value axis: its range and the map from a value to a height. */
interface Axis {
  lo: number;
  hi: number;
  log: boolean;
  y: (v: number) => number;
  /** a sample's value as plotted (its log on a log axis), NaN where it cannot be */
  of: (v: number) => number;
}

/** Draw the plot into a 2D context of CSS size w x h. */
export function drawProfilePlot(ctx: CanvasRenderingContext2D, w: number, h: number, p: ProfilePlot) {
  const more = (p.more ?? []).map((f) => ({ ...f, series: f.series.filter((q) => q.s.length >= 2) }));
  const m = plotMargin(more.length > 0);
  const [x0, y0, pw, ph] = [m.left, m.top, w - m.left - m.right, h - m.top - m.bottom];
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = p.colors.frame;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.fillStyle = p.colors.dim;
  const series = p.series.filter((q) => q.s.length >= 2);
  const range = p.range ?? seriesRange(series, p.log);
  const sb = plotLength([...series, ...more.flatMap((f) => f.series)]);
  if (!series.length || !range || !(sb > 0) || pw < 10 || ph < 10) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(p.log && series.length ? 'no positive values to show on a log scale' : 'nothing to show', x0 + pw / 2, y0 + ph / 2);
    return;
  }
  const px = (s: number) => x0 + (s / sb) * pw;
  const axis = (r: [number, number], log: boolean): Axis => ({
    lo: r[0],
    hi: r[1],
    log,
    y: (v: number) => y0 + ph - ((v - r[0]) / (r[1] - r[0])) * ph,
    of: (v: number) => (log ? (v > 0 ? Math.log10(v) : NaN) : v),
  });
  const left = axis(range, p.log);
  // Each further field's axis, on its own range (the first of them labelled on the right).
  const axes = more.map((f) => {
    const r = f.series.length ? (f.range ?? seriesRange(f.series, f.log)) : null;
    return r ? axis(r, f.log) : null;
  });
  const right = axes[0] ?? null;

  // Ticks: on a log scale at whole decades when there are a few, else as on a linear one. The
  // grid follows the left axis; the right one has its labels alone, with a tick on the frame.
  const axisTicks = (a: Axis) => (a.log && a.hi - a.lo >= 1.5 ? ticks(Math.ceil(a.lo), Math.floor(a.hi), 4).filter(Number.isInteger) : ticks(a.lo, a.hi, 4));
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (const t of axisTicks(left)) {
    const y = Math.round(left.y(t)) + 0.5;
    ctx.strokeStyle = p.colors.frame;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + pw, y);
    ctx.stroke();
    ctx.fillText(fmt3(left.log ? 10 ** t : t), x0 - 4, y);
  }
  if (right) {
    ctx.textAlign = 'left';
    ctx.strokeStyle = p.colors.dim;
    for (const t of axisTicks(right)) {
      const y = Math.round(right.y(t)) + 0.5;
      ctx.beginPath();
      ctx.setLineDash([2, 2]);
      ctx.moveTo(x0 + pw - 5, y);
      ctx.lineTo(x0 + pw, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(fmt3(right.log ? 10 ** t : t), x0 + pw + 4, y);
    }
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of ticks(0, sb, Math.max(2, Math.floor(pw / 50)))) ctx.fillText(fmt3(t), px(t), y0 + ph + 3);

  // Where the refinement level changes: a short mark on the bottom edge, in the line's colour
  // when there are several.
  for (const q of series) {
    if (!q.levels) continue;
    ctx.strokeStyle = series.length > 1 ? q.color : p.colors.dim;
    ctx.beginPath();
    for (let i = 1; i < q.s.length; i++) {
      if (q.levels[i] === q.levels[i - 1]) continue;
      const x = Math.round(px((q.s[i] + q.s[i - 1]) / 2)) + 0.5;
      ctx.moveTo(x, y0 + ph - 4);
      ctx.lineTo(x, y0 + ph);
    }
    ctx.stroke();
  }

  // The curves: the first field solid, the second dashed, each line in its colour.
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, pw, ph);
  ctx.clip();
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  const curves = (qs: ProfileSeries[], a: Axis, dash: number[]) => {
    ctx.setLineDash(dash);
    for (const q of qs) {
      ctx.strokeStyle = q.color;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < q.s.length; i++) {
        const v = a.of(q.values[i]);
        if (q.solid?.[i] || !Number.isFinite(v)) {
          pen = false;
          continue;
        }
        const [x, y] = [px(q.s[i]), a.y(v)];
        if (pen) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  };
  curves(series, left, []);
  more.forEach((f, k) => {
    const a = axes[k];
    if (a) curves(f.series, a, profileDash(k + 1));
  });
  ctx.restore();

  const at = p.mark ?? null;
  if (at !== null && at >= 0 && at <= sb) {
    const x = Math.round(px(at)) + 0.5;
    ctx.strokeStyle = p.colors.mark;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    const dots = (qs: ProfileSeries[], a: Axis, hollow: boolean) => {
      for (const q of qs) {
        const k = sampleNear(q.s, at);
        if (k < 0) continue;
        const v = a.of(q.values[k]);
        if (!Number.isFinite(v) || q.solid?.[k]) continue;
        const c = series.length > 1 || hollow ? q.color : p.colors.mark;
        ctx.beginPath();
        ctx.arc(Math.round(px(q.s[k])) + 0.5, Math.min(y0 + ph, Math.max(y0, a.y(v))), 3, 0, 2 * Math.PI);
        if (hollow) {
          ctx.strokeStyle = c;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          ctx.fillStyle = c;
          ctx.fill();
        }
      }
    };
    dots(series, left, false);
    more.forEach((f, k) => {
      const a = axes[k];
      if (a) dots(f.series, a, true);
    });
  }
}

/**
 * The distance under a point x (CSS pixels) of a plot w wide whose axis ends at `length`, or null
 * off the plot. `second`: the plot has a second field's axis on its right.
 */
export function distanceAt(x: number, w: number, length: number, second = false): number | null {
  const m = plotMargin(second);
  const pw = w - m.left - m.right;
  if (!(length > 0) || pw <= 0 || x < m.left - 4 || x > m.left + pw + 4) return null;
  return Math.min(1, Math.max(0, (x - m.left) / pw)) * length;
}

/**
 * The sample of a line nearest a distance from its start, or -1 if the distance lies beyond the
 * line's ends by more than half a sample spacing (a line shorter than the longest).
 */
export function sampleNear(s: ArrayLike<number>, target: number): number {
  const n = s.length;
  if (n < 1) return -1;
  const half = n > 1 ? (s[n - 1] - s[0]) / (n - 1) / 2 : 0;
  if (target < s[0] - half || target > s[n - 1] + half) return -1;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s[mid] < target) lo = mid;
    else hi = mid;
  }
  return Math.abs(s[lo] - target) <= Math.abs(s[hi] - target) ? lo : hi;
}

/**
 * The lines of a link's `prof`: each `x0,y0,x1,y1` (2D) or `x0,y0,z0,x1,y1,z1` (3D) in fractions
 * of the box, `;` between lines; in 1D `x0,x1`, the ends along the one axis (or r, R), which is all
 * a 1D line says (plans/one-dimensional.md).
 */
export function encodeProfileLines(lines: { a: readonly number[]; b: readonly number[] }[], dims: 1 | 2 | 3): string {
  const f = (v: number) => String(+v.toFixed(4));
  return lines.map((l) => [...l.a.slice(0, dims), ...l.b.slice(0, dims)].map(f).join(',')).join(';');
}

/** The lines of a link's `prof` (encodeProfileLines), clipped to the box; null if any is not one. */
export function decodeProfileLines(raw: string, dims: 1 | 2 | 3): [Vec3, Vec3][] | null {
  const parts = raw.split(';').filter(Boolean);
  if (!parts.length || parts.length > MAX_PROFILES) return null;
  const out: [Vec3, Vec3][] = [];
  for (const part of parts) {
    const v = part.split(',').map((x) => (x.trim() === '' ? NaN : Number(x)));
    // (A 1D link may also carry 2D ends: their y is kept, where the line is drawn.)
    const k = dims === 1 && v.length === 4 ? 2 : dims;
    if (v.length !== 2 * k || !v.every((x) => Number.isFinite(x) && x >= 0 && x <= 1)) return null;
    const c = clipSegment(v.slice(0, k), v.slice(k), dims);
    if (!c) return null;
    out.push(c);
  }
  return out;
}
