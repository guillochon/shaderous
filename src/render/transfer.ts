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

/** Small canvas editor for the opacity curve: drag points, double-click to add or remove. */
export class OpacityEditor {
  points = defaultOpacity();
  colormap = 'inferno';
  private drag = -1;

  constructor(
    private canvas: HTMLCanvasElement,
    private onChange: () => void,
  ) {
    canvas.addEventListener('pointerdown', (e) => {
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
    const x = (e.clientX - r.left) / r.width;
    const a = 1 - (e.clientY - r.top) / r.height;
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, a))];
  }

  private hit(e: PointerEvent | MouseEvent): number {
    const r = this.canvas.getBoundingClientRect();
    return this.points.findIndex(
      (p) => Math.hypot(p.x * r.width - (e.clientX - r.left), (1 - p.a) * r.height - (e.clientY - r.top)) < 9,
    );
  }

  draw() {
    const c = this.canvas;
    const w = (c.width = c.clientWidth || 276);
    const h = (c.height = c.clientHeight || 70);
    const g = c.getContext('2d')!;
    for (let x = 0; x < w; x++) {
      const [r, gg, b] = sampleColormap(this.colormap, x / (w - 1));
      g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
      g.fillRect(x, 0, 1, h);
    }
    g.fillStyle = 'rgba(10,12,18,0.55)';
    g.fillRect(0, 0, w, h);
    g.strokeStyle = '#fff';
    g.lineWidth = 1.5;
    g.beginPath();
    this.points.forEach((p, i) => (i ? g.lineTo(p.x * w, (1 - p.a) * h) : g.moveTo(p.x * w, (1 - p.a) * h)));
    g.stroke();
    for (const p of this.points) {
      g.beginPath();
      g.arc(p.x * w, (1 - p.a) * h, 4, 0, 2 * Math.PI);
      g.fillStyle = '#fff';
      g.fill();
    }
  }
}
