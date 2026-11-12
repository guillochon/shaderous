// History of the diagnostics readouts, and the line plots drawn from it: a sparkline beside each
// number (the last SPARK_STEPS steps) and a larger plot of the whole run in a popup.

/** Steps the sparklines cover. */
export const SPARK_STEPS = 1000;
/** Fewest readouts a sparkline shows, when the run takes many steps per readout. */
const SPARK_MIN_SAMPLES = 8;
/** Past this many rows the older half of the history is thinned to every other row. */
const MAX_ROWS = 20000;

/** Three significant figures; counts of 1000 and over take a k or M suffix instead of an exponent. */
export function fmt3(v: number, count = false): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (count) {
    if (a < 1000) return String(Math.round(v));
    if (a < 999_500) return `${+(v / 1e3).toPrecision(3)}k`;
    return `${+(v / 1e6).toPrecision(3)}M`;
  }
  if (a >= 0.01 && a < 999.5) return v.toPrecision(3);
  return v.toExponential(2).replace('e+', 'e');
}

/**
 * One row per readout: the step count, then each diagnostic. Rows come in order of steps; a run
 * that goes back (a reset) starts the history again.
 */
export class DiagHistory {
  rows: number[][] = [];
  /** column of the step count */
  constructor(private stepCol: number) {}

  clear() {
    this.rows = [];
  }

  /** Add a readout. While the run is paused (no new steps) the last row is replaced instead. */
  push(row: number[]) {
    const last = this.rows[this.rows.length - 1];
    const s = row[this.stepCol];
    if (last && s < last[this.stepCol]) this.clear();
    if (last && s === last[this.stepCol]) this.rows[this.rows.length - 1] = row;
    else this.rows.push(row);
    if (this.rows.length > MAX_ROWS) {
      // Thin the older half: the recent past stays at full resolution, the distant past coarser.
      const half = this.rows.length >> 1;
      this.rows = this.rows.slice(0, half).filter((_, i) => i % 2 === 0).concat(this.rows.slice(half));
    }
  }

  /** Steps and values of column `col` over the last SPARK_STEPS steps (or the last few readouts). */
  recent(col: number): [number[], number[]] {
    const r = this.rows;
    if (!r.length) return [[], []];
    const end = r[r.length - 1][this.stepCol];
    let i = r.length - 1;
    while (i > 0 && (r[i - 1][this.stepCol] >= end - SPARK_STEPS || r.length - i < SPARK_MIN_SAMPLES)) i--;
    return this.column(col, i);
  }

  /** Steps and values of column `col` over the whole run. */
  all(col: number): [number[], number[]] {
    return this.column(col, 0);
  }

  /** Rows may be shorter than others (a column added in mid-run) or hold NaN: those are skipped. */
  private column(col: number, from: number): [number[], number[]] {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = from; i < this.rows.length; i++) {
      const y = this.rows[i][col];
      if (y === undefined || !Number.isFinite(y)) continue;
      xs.push(this.rows[i][this.stepCol]);
      ys.push(y);
    }
    return [xs, ys];
  }

  /**
   * Columns `cols` of the history as CSV, under the given names (8 significant figures: the
   * readouts are float32). A value a row lacks is left empty.
   */
  csv(cols: number[], header: string[]): string {
    const cell = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? '' : String(+v.toPrecision(8)));
    return [header.join(','), ...this.rows.map((r) => cols.map((i) => cell(r[i])).join(','))].join('\n') + '\n';
  }
}

/** Sign of the least-squares slope: 1 rising, -1 falling, 0 flat (or too few points). */
export function trend(xs: number[], ys: number[]): -1 | 0 | 1 {
  const n = xs.length;
  if (n < 2) return 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    lo = Math.min(lo, ys[i]);
    hi = Math.max(hi, ys[i]);
  }
  // Changes at the level of float32 rounding are no trend.
  if (!(sxx > 0) || hi - lo <= 1e-6 * Math.max(Math.abs(hi), Math.abs(lo))) return 0;
  return sxy > 0 ? 1 : sxy < 0 ? -1 : 0;
}

let colors: { good: string; bad: string; dim: string; line: string; text: string } | undefined;
function palette() {
  if (!colors) {
    const cs = getComputedStyle(document.documentElement);
    const get = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
    colors = { good: get('--good', '#5fd08a'), bad: get('--bad', '#ff6b6b'), dim: get('--dim', '#8792a8'), line: get('--line', '#262c3a'), text: get('--text', '#d6dbe6') };
  }
  return colors;
}

/** The color of a line with this trend. */
export function trendColor(t: -1 | 0 | 1): string {
  const c = palette();
  return t > 0 ? c.good : t < 0 ? c.bad : c.dim;
}

/** Size the canvas's backing store to its CSS box at the device's pixel ratio; returns the context and CSS size. */
function prepare(canvas: HTMLCanvasElement): [CanvasRenderingContext2D, number, number] | null {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null;
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

/** Path of the line through (xs, ys) in the box [x0, x0 + w] x [y0, y0 + h]. */
function trace(ctx: CanvasRenderingContext2D, xs: number[], ys: number[], x0: number, y0: number, w: number, h: number) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const y of ys) {
    if (!Number.isFinite(y)) continue;
    lo = Math.min(lo, y);
    hi = Math.max(hi, y);
  }
  const xa = xs[0];
  const xb = xs[xs.length - 1];
  const sx = xb > xa ? w / (xb - xa) : 0;
  const flat = !(hi > lo);
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) continue;
    const px = x0 + (sx ? (xs[i] - xa) * sx : (i / Math.max(1, xs.length - 1)) * w);
    const py = flat ? y0 + h / 2 : y0 + h - ((ys[i] - lo) / (hi - lo)) * h;
    if (started) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    started = true;
  }
  ctx.stroke();
  return [lo, hi];
}

/** A sparkline, colored by its trend. */
export function drawSpark(canvas: HTMLCanvasElement, xs: number[], ys: number[]) {
  const p = prepare(canvas);
  if (!p) return;
  const [ctx, w, h] = p;
  if (xs.length < 2) return;
  ctx.lineWidth = 1.25;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = trendColor(trend(xs, ys));
  trace(ctx, xs, ys, 1, 1.5, w - 2, h - 3);
}

/** The whole run: the line, colored by its trend, with the range of the values and of the steps. */
export function drawFull(canvas: HTMLCanvasElement, xs: number[], ys: number[], count: boolean) {
  const p = prepare(canvas);
  if (!p) return;
  const [ctx, w, h] = p;
  const c = palette();
  const left = 46;
  const bottom = 14;
  const top = 4;
  const right = 4;
  ctx.strokeStyle = c.line;
  ctx.lineWidth = 1;
  ctx.strokeRect(left + 0.5, top + 0.5, w - left - right - 1, h - top - bottom - 1);
  ctx.font = '10px ui-monospace, Consolas, monospace';
  ctx.fillStyle = c.dim;
  if (xs.length < 2) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('not enough steps yet', (left + w - right) / 2, (top + h - bottom) / 2);
    return;
  }
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = trendColor(trend(xs, ys));
  const [lo, hi] = trace(ctx, xs, ys, left + 2, top + 3, w - left - right - 4, h - top - bottom - 6);
  ctx.fillStyle = c.dim;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(fmt3(hi, count), left - 4, top);
  ctx.textBaseline = 'bottom';
  ctx.fillText(fmt3(lo, count), left - 4, h - bottom);
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'left';
  ctx.fillText(`step ${fmt3(xs[0], true)}`, left, h);
  ctx.textAlign = 'right';
  ctx.fillText(fmt3(xs[xs.length - 1], true), w - right, h);
}
