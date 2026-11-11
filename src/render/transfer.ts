// Transfer function: a colour map plus a piecewise-linear opacity curve, baked to 256 RGBA texels.

type RGB = [number, number, number];

// Anchor colours, evenly spaced; interpolated linearly.
export const COLORMAPS: Record<string, RGB[]> = {
  inferno: [
    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
    [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
    [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
    [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ],
  'blue-red': [
    [33, 60, 130], [70, 120, 190], [140, 180, 225], [215, 228, 238], [245, 245, 245],
    [244, 215, 195], [233, 150, 115], [200, 80, 65], [150, 25, 35],
  ],
  gray: [[0, 0, 0], [255, 255, 255]],
};

export function sampleColormap(name: string, t: number): RGB {
  const a = COLORMAPS[name] ?? COLORMAPS.inferno;
  const x = Math.min(1, Math.max(0, t)) * (a.length - 1);
  const i = Math.min(a.length - 2, Math.floor(x));
  const f = x - i;
  return [0, 1, 2].map((c) => a[i][c] + (a[i + 1][c] - a[i][c]) * f) as RGB;
}

export interface OpacityPoint {
  x: number; // field value in [0, 1]
  a: number; // opacity in [0, 1]
}

export const defaultOpacity = (): OpacityPoint[] => [
  { x: 0, a: 0 },
  { x: 0.25, a: 0.02 },
  { x: 0.6, a: 0.25 },
  { x: 1, a: 0.9 },
];

export function sampleOpacity(pts: OpacityPoint[], t: number): number {
  if (t <= pts[0].x) return pts[0].a;
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i].x) {
      const w = (t - pts[i - 1].x) / Math.max(pts[i].x - pts[i - 1].x, 1e-9);
      return pts[i - 1].a + (pts[i].a - pts[i - 1].a) * w;
    }
  }
  return pts[pts.length - 1].a;
}

export function bakeTransfer(colormap: string, pts: OpacityPoint[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const [r, g, b] = sampleColormap(colormap, t);
    // Squaring gives finer control over faint, extended gas.
    const a = sampleOpacity(pts, t) ** 2;
    out.set([r, g, b, Math.round(255 * a)], i * 4);
  }
  return out;
}

/**
 * Small canvas editor for the opacity curve: drag points, double-click to add or remove.
 * Horizontal, the field value runs left to right and opacity upward; vertical (a colour bar),
 * the value runs upward and opacity to the right.
 */
export class OpacityEditor {
  points = defaultOpacity();
  colormap = 'inferno';
  /** The curve is drawn and can be edited (it matters only where the gas is drawn translucent). */
  editable = true;
  private drag = -1;

  constructor(
    private canvas: HTMLCanvasElement,
    private onChange: () => void,
    private vertical = false,
  ) {
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.editable) return;
      this.drag = this.hit(e);
      if (this.drag >= 0) canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.drag < 0) return;
      const [x, a] = this.toCurve(e);
      const p = this.points[this.drag];
      const lo = this.drag === 0 ? 0 : this.points[this.drag - 1].x + 0.01;
      const hi = this.drag === this.points.length - 1 ? 1 : this.points[this.drag + 1].x - 0.01;
      // End points stay pinned to the ends of the range.
      p.x = this.drag === 0 ? 0 : this.drag === this.points.length - 1 ? 1 : Math.min(hi, Math.max(lo, x));
      p.a = a;
      this.changed();
    });
    canvas.addEventListener('pointerup', () => (this.drag = -1));
    canvas.addEventListener('dblclick', (e) => {
      if (!this.editable) return;
      const i = this.hit(e);
      if (i > 0 && i < this.points.length - 1) {
        this.points.splice(i, 1);
      } else if (i < 0) {
        const [x, a] = this.toCurve(e);
        this.points.push({ x, a });
        this.points.sort((p, q) => p.x - q.x);
      }
      this.changed();
    });
    this.draw();
  }

  setEditable(on: boolean) {
    if (on === this.editable) return;
    this.editable = on;
    this.drag = -1;
    this.draw();
  }

  set(points: OpacityPoint[], colormap: string) {
    this.points = points;
    this.colormap = colormap;
    this.draw();
  }

  private changed() {
    this.draw();
    this.onChange();
  }

  private toCurve(e: PointerEvent | MouseEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    const u = (e.clientX - r.left) / r.width;
    const v = 1 - (e.clientY - r.top) / r.height;
    const [x, a] = this.vertical ? [v, u] : [u, v];
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, a))];
  }

  /** A point of the curve in pixels from the canvas corner. */
  private at(p: OpacityPoint, w: number, h: number): [number, number] {
    return this.vertical ? [p.a * w, (1 - p.x) * h] : [p.x * w, (1 - p.a) * h];
  }

  private hit(e: PointerEvent | MouseEvent): number {
    const r = this.canvas.getBoundingClientRect();
    return this.points.findIndex((p) => {
      const [px, py] = this.at(p, r.width, r.height);
      return Math.hypot(px - (e.clientX - r.left), py - (e.clientY - r.top)) < 9;
    });
  }

  draw() {
    const c = this.canvas;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    const w = c.clientWidth || (this.vertical ? 72 : 276);
    const h = c.clientHeight || (this.vertical ? 200 : 70);
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.cursor = this.editable ? 'crosshair' : 'default';
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const len = this.vertical ? h : w;
    for (let i = 0; i < len; i++) {
      const t = i / Math.max(len - 1, 1);
      const [r, gg, b] = sampleColormap(this.colormap, this.vertical ? 1 - t : t);
      g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
      if (this.vertical) g.fillRect(0, i, w, 1);
      else g.fillRect(i, 0, 1, h);
    }
    if (!this.editable) return;
    if (!this.vertical) {
      g.fillStyle = 'rgba(10,12,18,0.55)';
      g.fillRect(0, 0, w, h);
    }
    const pts = this.points.map((p) => this.at(p, w, h));
    // A dark edge keeps the curve visible over the bright end of a colour map.
    for (const [style, width] of [['rgba(0,0,0,0.6)', 3.5], ['#fff', 1.5]] as const) {
      g.strokeStyle = style;
      g.lineWidth = width;
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.stroke();
    }
    for (const [x, y] of pts) {
      g.beginPath();
      g.arc(x, y, 4, 0, 2 * Math.PI);
      g.fillStyle = '#fff';
      g.fill();
      g.strokeStyle = 'rgba(0,0,0,0.6)';
      g.lineWidth = 1;
      g.stroke();
    }
  }
}
