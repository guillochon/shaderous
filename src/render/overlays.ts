// A color bar, an axis triad and the status line, painted over a frame of the view (movies and
// screenshots) or beside a map that must not be covered (screenshots at the grid's resolution).
// Pure 2D canvas drawing: everything the overlays show
// comes in as an OverlayInfo, so nothing here looks at the page.
import { sampleColormap } from './transfer';

export interface OverlayOptions {
  colorbar?: boolean;
  triad?: boolean;
  /** The page's status line (grid, simulation time, point masses, mesh), upper left. */
  status?: boolean;
}

export interface OverlayInfo {
  colormap: string;
  /** The displayed range, in the displayed units. */
  lo: number;
  hi: number;
  /** A logarithmic color scale. */
  log: boolean;
  /** The color bar's title: the field's name, and its unit if it has one. */
  title: string;
  /**
   * What the view shows: the volume, or a plane (a slice or projection of a 3D run, or a 2D run,
   * whose plane is always x-y).
   */
  mode: 'volume' | 'plane';
  /** A plane's normal: 0, 1 or 2 (2 for a 2D run). */
  sliceAxis: number;
  /** The camera's right and up vectors in box coordinates (volume view; cameraFrame()). */
  right: number[];
  up: number[];
  /** Two dimensions: no z axis in the triad. */
  twoD: boolean;
  /** The light background: dark text and lines. */
  light: boolean;
  /** The status line's text. */
  status?: string;
  /**
   * A 1D run (plans/one-dimensional.md): what the view repeats, after a bold "1D" ("repeated
   * along y", "spherically symmetric"). Painted on every frame, whatever overlays are asked for,
   * so a picture of a 1D run cannot pass for a 2D one.
   */
  oneD?: string;
}

/** One axis of the triad, in pixels from its origin, y down. */
export interface TriadArrow {
  axis: number;
  label: string;
  dx: number;
  dy: number;
  /** Toward the viewer is negative: arrows are drawn in descending order of depth. */
  depth: number;
}

export interface Tick {
  value: number;
  /** Place along the bar, 0 at the bottom (lo) and 1 at the top (hi). */
  frac: number;
  label: string;
}

const AXES = ['x', 'y', 'z'];
/** One color per axis, as the guides through a tool are drawn (render/renderer.ts). */
const AXIS_COLORS = ['rgb(255,107,107)', 'rgb(115,230,128)', 'rgb(115,166,255)'];
/** The plane a slice along each axis shows: (horizontal, vertical), as render/project.wgsl maps it. */
export const PLANE_AXES: [number, number][] = [
  [1, 2],
  [0, 2],
  [0, 1],
];

const SUPERSCRIPT: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const power = (k: number) => `10${String(k).split('').map((c) => SUPERSCRIPT[c] ?? c).join('')}`;

/** A value as a short label: three significant figures, exponents for the very large and small. */
export function formatValue(v: number): string {
  if (v === 0 || !Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) {
    const [m, e] = v.toExponential(2).split('e');
    const mant = String(+m);
    return mant === '1' ? power(+e) : mant === '-1' ? `−${power(+e)}` : `${mant}×${power(+e)}`;
  }
  return String(+v.toPrecision(3)).replace('-', '−');
}

/** Round values between lo and hi, spaced 1, 2 or 5 times a power of ten: about `want` of them. */
function niceSteps(lo: number, hi: number, want: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [];
  const raw = span / want;
  const p = 10 ** Math.floor(Math.log10(raw));
  // The 1, 2 or 5 nearest the spacing asked for (on a log scale).
  const step = [1, 2, 5, 10].map((m) => m * p).reduce((a, b) => (Math.abs(Math.log(b / raw)) < Math.abs(Math.log(a / raw)) ? b : a));
  const out: number[] = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(Math.abs(k * step) < step * 1e-9 ? 0 : +(k * step).toPrecision(12));
  return out;
}

/**
 * The color bar's ticks: its two ends, and round values between (decades on a log scale) that
 * keep clear of the ends.
 */
export function colorbarTicks(lo: number, hi: number, log: boolean, want = 5): Tick[] {
  if (!(hi > lo) || (log && !(lo > 0))) return [{ value: lo, frac: 0, label: formatValue(lo) }];
  const L = log ? Math.log10(lo) : lo;
  const H = log ? Math.log10(hi) : hi;
  const frac = (v: number) => ((log ? Math.log10(v) : v) - L) / (H - L);
  const ends: Tick[] = [
    { value: lo, frac: 0, label: formatValue(lo) },
    { value: hi, frac: 1, label: formatValue(hi) },
  ];
  const inner: { value: number; label: string }[] = [];
  if (log && H - L >= 1.5) {
    // Every decade, or every few when there are many.
    const every = Math.max(1, Math.ceil((H - L) / want));
    for (let k = Math.ceil(L / every - 1e-9) * every; k <= H + 1e-9; k += every) inner.push({ value: 10 ** k, label: power(k) });
  } else {
    for (const value of niceSteps(lo, hi, want - 1)) if (!log || value > 0) inner.push({ value, label: formatValue(value) });
  }
  const gap = 0.08;
  const mids = inner.map((t) => ({ ...t, frac: frac(t.value) })).filter((t) => t.frac > gap && t.frac < 1 - gap);
  return [ends[0], ...mids, ends[1]];
}

/**
 * The triad's arrows, `len` pixels long at most. The volume view projects each box axis onto the
 * camera's right and up (orthographically); a plane shows its own two axes.
 */
export function triadArrows(info: Pick<OverlayInfo, 'mode' | 'sliceAxis' | 'right' | 'up' | 'twoD'>, len: number): TriadArrow[] {
  if (info.twoD || info.mode === 'plane') {
    const [a, b] = PLANE_AXES[info.twoD ? 2 : info.sliceAxis];
    return [
      { axis: a, label: AXES[a], dx: len, dy: 0, depth: 0 },
      { axis: b, label: AXES[b], dx: 0, dy: -len, depth: 0 },
    ];
  }
  const { right: r, up: u } = info;
  // Forward, into the screen: up x right.
  const fwd = [u[1] * r[2] - u[2] * r[1], u[2] * r[0] - u[0] * r[2], u[0] * r[1] - u[1] * r[0]];
  return [0, 1, 2]
    .map((k) => ({ axis: k, label: AXES[k], dx: r[k] * len, dy: -u[k] * len, depth: fwd[k] }))
    .sort((p, q) => q.depth - p.depth);
}

/**
 * Whether a plane's normal points out of the screen (toward the viewer) or into it: its two
 * axes, horizontal then vertical, crossed. x-y and y-z show +z and +x coming out; x-z shows +y
 * going in.
 */
export function normalOutOfScreen(sliceAxis: number): boolean {
  return sliceAxis !== 1;
}

/**
 * The room the overlays take at the left (the triad) and the right (the color bar) of a frame
 * whose shorter side is `s` pixels, as paintTriad() and paintColorbar() lay them out: a frame
 * with this much beside the picture keeps them off it. `measure` gives the width of the color
 * bar's title at a font size; without it the width is estimated.
 */
export function overlayGutters(s: number, info: Pick<OverlayInfo, 'mode' | 'twoD' | 'title'>, opts: OverlayOptions, measure?: (text: string, px: number) => number): { left: number; right: number } {
  const px = Math.max(10, Math.round(0.024 * s));
  const edge = Math.round(0.04 * s);
  let left = 0;
  let right = 0;
  if (opts.triad) {
    const len = Math.round(0.07 * s);
    const reach = info.mode === 'volume' && !info.twoD ? len + px : px;
    // The origin, an arrow, and its label.
    left = edge + reach + len + 1.35 * px + 0.5 * px;
  }
  if (opts.colorbar) {
    const barW = Math.max(6, Math.round(0.022 * s));
    // The bar, its ticks, and labels as long as "1.23×10⁻⁵".
    const ticks = edge + barW + Math.round(barW * 0.45) + 0.35 * px + 4.5 * px;
    // The title, flush with the bar's right edge: a long one may still reach over the picture.
    const title = edge + (measure ? measure(info.title, px) : 0.55 * px * info.title.length);
    right = Math.max(ticks, Math.min(title, 0.3 * s)) + 0.5 * px;
  }
  return { left: Math.ceil(left), right: Math.ceil(right) };
}

const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
/** The overlays' text size for a frame whose reference length is `s` pixels. */
export const textSize = (s: number) => Math.max(10, Math.round(0.024 * s));
const inks = (light: boolean) => ({ ink: light ? '#14161c' : '#eef1f7', edge: light ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.75)' });

/** Paint the overlays asked for onto a frame `w` x `h` pixels. */
export function paintOverlays(g: CanvasRenderingContext2D, w: number, h: number, info: OverlayInfo, opts: OverlayOptions) {
  // Sizes follow the frame's shorter side, so a 4K movie's labels look as large as a 720p one's.
  const s = Math.min(w, h);
  const { ink, edge } = inks(info.light);
  const px = textSize(s);
  g.save();
  g.lineJoin = 'round';
  g.lineCap = 'round';
  if (opts.colorbar) {
    const barH = Math.round(0.6 * h);
    const barW = Math.max(6, Math.round(0.022 * s));
    const x1 = w - Math.round(0.04 * s);
    const y0 = Math.round((h - barH) / 2) + Math.round(px * 0.8);
    paintColorbar(g, { x: x1 - barW, y: y0, w: barW, h: barH }, 'left', px, s, info, ink, edge);
  }
  if (opts.triad) {
    const len = Math.round(0.07 * s);
    // A plane's arrows point right and up; the volume's may point any way, labels and all.
    const reach = info.mode === 'volume' && !info.twoD ? len + px : px;
    paintTriad(g, Math.round(0.04 * s) + reach, h - Math.round(0.04 * s) - reach, len, px, Math.max(2, s / 360), info, ink, edge);
  }
  if (opts.status && info.status) {
    const m = Math.round(0.03 * s);
    // Kept clear of the color bar's title on the right.
    paintStatus(g, info.status, m, m, w - 2 * m - 0.18 * s, px, ink, edge);
  }
  if (info.oneD) {
    const m = Math.round(0.03 * s);
    paintChip(g, info.oneD, m, m + (opts.status && info.status ? Math.round(1.8 * px) : 0), px);
  }
  g.restore();
}

/** The 1D chip (OverlayInfo.oneD) from (x, y), its top left: a bold "1D" and what the view repeats, on a dark rounded tag. */
export function paintChip(g: CanvasRenderingContext2D, text: string, x: number, y: number, px: number) {
  g.save();
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  const bold = `700 ${px}px ${FONT}`;
  const plain = `${px}px ${FONT}`;
  g.font = bold;
  const w1 = g.measureText('1D').width;
  g.font = plain;
  const rest = ` · ${text}`;
  const w2 = g.measureText(rest).width;
  const pad = 0.55 * px;
  const h = 1.5 * px;
  const w = w1 + w2 + 2 * pad;
  g.beginPath();
  g.roundRect(x, y, w, h, h / 2);
  g.fillStyle = 'rgba(12, 28, 52, 0.85)';
  g.fill();
  g.lineWidth = Math.max(1, px / 12);
  g.strokeStyle = '#5d8fd6';
  g.stroke();
  g.font = bold;
  g.fillStyle = '#ffffff';
  g.fillText('1D', x + pad, y + h / 2);
  g.font = plain;
  g.fillStyle = '#cfe2ff';
  g.fillText(rest, x + pad + w1, y + h / 2);
  g.restore();
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A picture with the overlays beside a map (or a frame) rather than over it: see besideLayout(). */
export interface BesideLayout {
  /** The whole picture, in pixels. */
  width: number;
  height: number;
  /** Where the map's first pixel goes. */
  map: { x: number; y: number };
  /** The color bar itself; its title is above it and its tick labels to its right. */
  bar: Rect | null;
  /** The triad's origin, and its arrows' length. */
  triad: { x: number; y: number; len: number } | null;
  /** The status line's top left corner, and the width it is cut to. */
  status: { x: number; y: number; maxWidth: number } | null;
  /** Text size, and the reference length sizes follow. */
  px: number;
  s: number;
  /** What each overlay covers, labels and all (for tests: none of it meets the map). */
  regions: (Rect & { name: string })[];
}

/** How far the triad reaches from its origin, labels and all. */
function triadReach(len: number, px: number, info: Pick<OverlayInfo, 'mode' | 'twoD'>) {
  if (info.mode === 'volume' && !info.twoD) {
    const r = len + 1.4 * px;
    return { left: r, right: r, up: r, down: r };
  }
  // A plane's arrows go right and up; a slice of a 3D run has its normal's label below left.
  const near = info.twoD ? 0.7 * px : 1.7 * px;
  return { left: near, right: len + 1.4 * px, up: len + 1.4 * px, down: near };
}

/**
 * Where everything goes in a picture whose map is `mapW` x `mapH` pixels and must not be
 * covered: the status line in a strip above, the color bar and the axes in a gutter to the
 * right. `widths`: the widths in pixels, at textSize(s), of the widest tick label, the color
 * bar's title, and the status line. Sizes follow s = max(mapH, 600), so a small grid's labels
 * still read. With no overlay asked for there are no margins.
 */
export function besideLayout(
  mapW: number,
  mapH: number,
  widths: { ticks: number; title: number; status: number },
  opts: OverlayOptions,
  info: Pick<OverlayInfo, 'mode' | 'twoD'> = { mode: 'plane', twoD: false },
): BesideLayout {
  const s = Math.max(mapH, 600);
  const px = textSize(s);
  const none: BesideLayout = { width: mapW, height: mapH, map: { x: 0, y: 0 }, bar: null, triad: null, status: null, px, s, regions: [] };
  if (!opts.colorbar && !opts.triad && !opts.status) return none;
  const m = Math.round(0.03 * s);
  const lineH = Math.ceil(1.25 * px);
  // A strip above the map for the status line and the color bar's title, on one line.
  const top = opts.status || opts.colorbar ? m + Math.ceil(2 * px) : m;
  const map = { x: m, y: top };
  const regions: BesideLayout['regions'] = [];
  const gutter = !!(opts.colorbar || opts.triad);
  const gx = m + mapW + m;
  let colW = 0;
  let bar: Rect | null = null;
  if (opts.colorbar) {
    const barW = Math.max(6, Math.round(0.022 * s));
    const tick = Math.round(barW * 0.45);
    bar = { x: gx, y: top, w: barW, h: Math.round(0.6 * s) };
    const labels = barW + tick + 0.35 * px + widths.ticks;
    colW = Math.max(labels, widths.title);
    // The title above, and the end labels half a line beyond the bar's ends.
    regions.push({ name: 'colorbar', x: gx, y: top - Math.ceil(2 * px), w: colW, h: bar.h + Math.ceil(2 * px) + Math.ceil(0.7 * px) });
  }
  let triad: BesideLayout['triad'] = null;
  let bottom = top + mapH;
  let triadW = 0;
  if (opts.triad) {
    const len = Math.round(0.07 * s);
    const r = triadReach(len, px, info);
    triadW = r.left + r.right;
    // Bottom-aligned with the map, or below the bar where the map is shorter than both need.
    const need = (bar ? bar.y + bar.h + Math.ceil(0.7 * px) + m : top) + r.up + r.down;
    bottom = Math.max(bottom, Math.ceil(need));
    triad = { x: Math.round(gx + r.left), y: Math.round(bottom - r.down), len };
    regions.push({ name: 'triad', x: triad.x - r.left, y: triad.y - r.up, w: triadW, h: r.up + r.down });
  } else if (bar) {
    bottom = Math.max(bottom, bar.y + bar.h + Math.ceil(0.7 * px));
  }
  const width = gutter ? gx + Math.ceil(Math.max(colW, triadW)) + m : m + mapW + m;
  const height = bottom + m;
  let status: BesideLayout['status'] = null;
  if (opts.status) {
    status = { x: m, y: m, maxWidth: width - 2 * m };
    regions.push({ name: 'status', x: m, y: m, w: Math.min(widths.status, width - 2 * m), h: lineH });
  }
  return { width, height, map, bar, triad, status, px, s, regions };
}

/**
 * The largest canvas a browser is trusted to make: 16384 a side and 2^28 pixels (Chrome's
 * limits; Safari's are lower, which the read-back of the size catches).
 */
export const CANVAS_MAX_SIDE = 16384;
export const CANVAS_MAX_PIXELS = 2 ** 28;

/** A 2D canvas of this size, or an error that names it when the browser cannot make one. */
export function bigCanvas(w: number, h: number): { canvas: HTMLCanvasElement; g: CanvasRenderingContext2D } {
  const too = () => new Error(`a picture of ${w} × ${h} pixels is larger than the browser can make a canvas`);
  if (w > CANVAS_MAX_SIDE || h > CANVAS_MAX_SIDE || w * h > CANVAS_MAX_PIXELS) throw too();
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.width === w && canvas.height === h ? canvas.getContext('2d') : null;
  if (!g) throw too();
  return { canvas, g };
}

/**
 * A new canvas: the background, the map `mapW` x `mapH` at whole pixels, and the overlays asked
 * for in margins around it (besideLayout()). `background` is a CSS color.
 */
export function composeBeside(map: CanvasImageSource, mapW: number, mapH: number, info: OverlayInfo, opts: OverlayOptions, background: string): { canvas: HTMLCanvasElement; layout: BesideLayout } {
  const s = Math.max(mapH, 600);
  const px = textSize(s);
  const ruler = document.createElement('canvas').getContext('2d')!;
  ruler.font = `${px}px ${FONT}`;
  const ticks = opts.colorbar ? Math.max(0, ...colorbarTicks(info.lo, info.hi, info.log).map((t) => ruler.measureText(t.label).width)) : 0;
  const title = opts.colorbar ? ruler.measureText(info.title).width : 0;
  const statusW = opts.status && info.status ? ruler.measureText(info.status).width : 0;
  const L = besideLayout(mapW, mapH, { ticks, title, status: statusW }, { ...opts, status: opts.status && !!info.status }, info);
  const { canvas, g } = bigCanvas(L.width, L.height);
  g.fillStyle = background;
  g.fillRect(0, 0, L.width, L.height);
  g.imageSmoothingEnabled = false;
  g.drawImage(map, 0, 0, mapW, mapH, L.map.x, L.map.y, mapW, mapH);
  const { ink, edge } = inks(info.light);
  g.save();
  g.lineJoin = 'round';
  g.lineCap = 'round';
  if (L.bar) paintColorbar(g, L.bar, 'right', L.px, L.s, info, ink, edge);
  if (L.triad) paintTriad(g, L.triad.x, L.triad.y, L.triad.len, L.px, Math.max(2, L.s / 360), info, ink, edge);
  if (L.status && info.status) paintStatus(g, info.status, L.status.x, L.status.y, L.status.maxWidth, L.px, ink, edge);
  // (On the map itself, in its corner: nothing of the map is hidden that the repeat does not show again.)
  if (info.oneD) paintChip(g, info.oneD, L.map.x + Math.round(0.6 * L.px), L.map.y + Math.round(0.6 * L.px), L.px);
  g.restore();
  return { canvas, layout: L };
}

/** Text with a thin contrasting edge, so it reads over bright gas. */
function label(g: CanvasRenderingContext2D, text: string, x: number, y: number, ink: string, edge: string, px: number) {
  g.lineWidth = Math.max(2, px * 0.22);
  g.strokeStyle = edge;
  g.strokeText(text, x, y);
  g.fillStyle = ink;
  g.fillText(text, x, y);
}

/**
 * The color bar in `bar`, its tick labels to the `labels` side of it, and its title above it,
 * flush with the bar's edge on the other side. `s`: the reference length line widths follow.
 */
function paintColorbar(g: CanvasRenderingContext2D, bar: Rect, labels: 'left' | 'right', px: number, s: number, info: OverlayInfo, ink: string, edge: string) {
  const { x: x0, y: y0, w: barW, h: barH } = bar;
  const x1 = x0 + barW;
  const y1 = y0 + barH;
  for (let i = 0; i < barH; i++) {
    const [r, gg, b] = sampleColormap(info.colormap, 1 - i / Math.max(barH - 1, 1));
    g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
    g.fillRect(x0, y0 + i, barW, 1);
  }
  g.lineWidth = Math.max(1, s / 900);
  g.strokeStyle = edge;
  g.strokeRect(x0 - 0.5, y0 - 0.5, barW + 1, barH + 1);

  g.font = `${px}px ${FONT}`;
  const left = labels === 'left';
  g.textAlign = left ? 'right' : 'left';
  g.textBaseline = 'middle';
  const tick = Math.round(barW * 0.45);
  for (const t of colorbarTicks(info.lo, info.hi, info.log)) {
    const y = Math.round(y1 - t.frac * barH);
    g.lineWidth = Math.max(1.5, s / 600);
    g.strokeStyle = ink;
    g.beginPath();
    g.moveTo(left ? x0 - tick : x1 + tick, y);
    g.lineTo(left ? x0 : x1, y);
    g.stroke();
    label(g, t.label, left ? x0 - tick - px * 0.35 : x1 + tick + px * 0.35, y, ink, edge, px);
  }
  g.textAlign = left ? 'right' : 'left';
  g.textBaseline = 'bottom';
  label(g, info.title, left ? x1 : x0, y0 - px * 0.7, ink, edge, px);
}

/** The status line from (x, y), its top left; cut at a separator to fit in `maxWidth`. */
function paintStatus(g: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, px: number, ink: string, edge: string) {
  g.font = `${px}px ${FONT}`;
  g.textAlign = 'left';
  g.textBaseline = 'top';
  let t = text;
  while (t.length > 8 && g.measureText(t).width > maxWidth) t = t.slice(0, t.lastIndexOf('  •  ') > 0 ? t.lastIndexOf('  •  ') : t.length - 2);
  label(g, t, x, y, ink, edge, px);
}

/** The triad from its origin (ox, oy): arrows `len` long, `lw` wide. */
function paintTriad(g: CanvasRenderingContext2D, ox: number, oy: number, len: number, px: number, lw: number, info: OverlayInfo, ink: string, edge: string) {
  g.font = `600 ${px}px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  const arrows = triadArrows(info, len);
  for (const a of arrows) {
    const l = Math.hypot(a.dx, a.dy);
    const color = AXIS_COLORS[a.axis];
    if (l < 0.12 * len) {
      // Pointing at the viewer or away: a dot, or a cross.
      g.fillStyle = color;
      g.beginPath();
      g.arc(ox, oy, lw * 1.6, 0, 2 * Math.PI);
      g.fill();
      label(g, a.label, ox + px * 0.8, oy - px * 0.8, ink, edge, px);
      continue;
    }
    const ex = ox + a.dx;
    const ey = oy + a.dy;
    const ux = a.dx / l;
    const uy = a.dy / l;
    const head = Math.min(l * 0.4, lw * 4.5);
    for (const [style, width] of [[edge, lw + 2.5], [color, lw]] as const) {
      g.strokeStyle = style;
      g.lineWidth = width;
      g.beginPath();
      g.moveTo(ox, oy);
      g.lineTo(ex, ey);
      g.moveTo(ex - head * (ux - 0.5 * uy), ey - head * (uy + 0.5 * ux));
      g.lineTo(ex, ey);
      g.lineTo(ex - head * (ux + 0.5 * uy), ey - head * (uy - 0.5 * ux));
      g.stroke();
    }
    label(g, a.label, ex + ux * px * 0.85, ey + uy * px * 0.85, ink, edge, px);
  }
  if (info.mode === 'plane' && !info.twoD) {
    // The plane's normal at the origin: a dot in a circle coming out of the screen, a cross going in.
    const k = info.sliceAxis;
    const r = px * 0.55;
    const color = AXIS_COLORS[k];
    g.lineWidth = lw;
    g.strokeStyle = color;
    g.fillStyle = color;
    g.beginPath();
    g.arc(ox, oy, r, 0, 2 * Math.PI);
    g.stroke();
    if (normalOutOfScreen(k)) {
      g.beginPath();
      g.arc(ox, oy, lw * 1.1, 0, 2 * Math.PI);
      g.fill();
    } else {
      const d = r * 0.62;
      g.beginPath();
      g.moveTo(ox - d, oy - d);
      g.lineTo(ox + d, oy + d);
      g.moveTo(ox + d, oy - d);
      g.lineTo(ox - d, oy + d);
      g.stroke();
    }
    label(g, AXES[k], ox - r - px * 0.6, oy + r + px * 0.45, ink, edge, px);
  }
}
