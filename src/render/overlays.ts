// A color bar and an axis triad painted over a frame of the view, for movies and screenshots
// (plans/mp4-export-plan.md, section 6). Pure 2D canvas drawing: everything the overlays show
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

/** Paint the overlays asked for onto a frame `w` x `h` pixels. */
export function paintOverlays(g: CanvasRenderingContext2D, w: number, h: number, info: OverlayInfo, opts: OverlayOptions) {
  // Sizes follow the frame's shorter side, so a 4K movie's labels look as large as a 720p one's.
  const s = Math.min(w, h);
  const ink = info.light ? '#14161c' : '#eef1f7';
  const edge = info.light ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.75)';
  g.save();
  g.lineJoin = 'round';
  g.lineCap = 'round';
  if (opts.colorbar) paintColorbar(g, w, h, s, info, ink, edge);
  if (opts.triad) paintTriad(g, h, s, info, ink, edge);
  if (opts.status && info.status) paintStatus(g, w, s, info.status, ink, edge);
  g.restore();
}

/** Text with a thin contrasting edge, so it reads over bright gas. */
function label(g: CanvasRenderingContext2D, text: string, x: number, y: number, ink: string, edge: string, px: number) {
  g.lineWidth = Math.max(2, px * 0.22);
  g.strokeStyle = edge;
  g.strokeText(text, x, y);
  g.fillStyle = ink;
  g.fillText(text, x, y);
}

function paintColorbar(g: CanvasRenderingContext2D, w: number, h: number, s: number, info: OverlayInfo, ink: string, edge: string) {
  const px = Math.max(10, Math.round(0.024 * s));
  const barH = Math.round(0.6 * h);
  const barW = Math.max(6, Math.round(0.022 * s));
  const x1 = w - Math.round(0.04 * s);
  const x0 = x1 - barW;
  const y0 = Math.round((h - barH) / 2) + Math.round(px * 0.8);
  const y1 = y0 + barH;
  for (let i = 0; i < barH; i++) {
    const [r, gg, b] = sampleColormap(info.colormap, 1 - i / Math.max(barH - 1, 1));
    g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
    g.fillRect(x0, y0 + i, barW, 1);
  }
  g.lineWidth = Math.max(1, s / 900);
  g.strokeStyle = edge;
  g.strokeRect(x0 - 0.5, y0 - 0.5, barW + 1, barH + 1);

  g.font = `${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  const tick = Math.round(barW * 0.45);
  for (const t of colorbarTicks(info.lo, info.hi, info.log)) {
    const y = Math.round(y1 - t.frac * barH);
    g.lineWidth = Math.max(1.5, s / 600);
    g.strokeStyle = ink;
    g.beginPath();
    g.moveTo(x0 - tick, y);
    g.lineTo(x0, y);
    g.stroke();
    label(g, t.label, x0 - tick - px * 0.35, y, ink, edge, px);
  }
  g.textAlign = 'right';
  g.textBaseline = 'bottom';
  label(g, info.title, x1, y0 - px * 0.7, ink, edge, px);
}

function paintStatus(g: CanvasRenderingContext2D, w: number, s: number, text: string, ink: string, edge: string) {
  const px = Math.max(10, Math.round(0.024 * s));
  const m = Math.round(0.03 * s);
  g.font = `${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  g.textAlign = 'left';
  g.textBaseline = 'top';
  // Kept clear of the color bar's title on the right.
  let t = text;
  while (t.length > 8 && g.measureText(t).width > w - 2 * m - 0.18 * s) t = t.slice(0, t.lastIndexOf('  •  ') > 0 ? t.lastIndexOf('  •  ') : t.length - 2);
  label(g, t, m, m, ink, edge, px);
}

function paintTriad(g: CanvasRenderingContext2D, h: number, s: number, info: OverlayInfo, ink: string, edge: string) {
  const len = Math.round(0.07 * s);
  const px = Math.max(10, Math.round(0.024 * s));
  // A plane's arrows point right and up; the volume's may point any way, labels and all.
  const reach = info.mode === 'volume' && !info.twoD ? len + px : px;
  const ox = Math.round(0.04 * s) + reach;
  const oy = h - Math.round(0.04 * s) - reach;
  const lw = Math.max(2, s / 360);
  g.font = `600 ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
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
