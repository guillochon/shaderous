// The dialog's preview: the setup drawn from its document alone, on 2D canvases, from above
// (x-y) and from the side (x-z). The box with its faces colored by what the gas meets there,
// the point masses with their softening and sink, stars to scale, disks, the shapes of regions,
// obstacles and inflows, and the paths the bodies would take as point masses for a chosen time.
// No GPU work: it cannot disturb the run behind the dialog.
//
// A body whose position is its own (not placed by an orbit) can be dragged in either view.
import { buildSetupBodies, type BuiltSetup } from '../../setup/check';
import { BC_CHOICES } from '../../setup/options';
import type { DiskPart, Part, SetupSpec, Shape, Vec3 } from '../../setup/spec';
import { h } from './forms';

// ---- the bodies as point masses ----

export interface NBody {
  /** G M */
  gm: number;
  pos: number[];
  vel: number[];
  /** Plummer softening of its pull */
  soft?: number;
  /** it pulls on nothing */
  massless?: boolean;
  /** nothing moves it */
  fixed?: boolean;
}

export interface Orbits {
  /** each body's positions along the way, the start and the end included */
  paths: number[][][];
  /** the bodies at the end */
  bodies: NBody[];
  steps: number;
}

const pulls = (b: NBody) => !b.massless && b.gm > 0;
const eps2 = (a: NBody, b: NBody) => Math.max(a.soft ?? 0, b.soft ?? 0) ** 2;

function accelerations(bodies: NBody[], acc: number[][]): void {
  for (const a of acc) a[0] = a[1] = a[2] = 0;
  for (let i = 0; i < bodies.length; i++)
    for (let j = i + 1; j < bodies.length; j++) {
      const [a, b] = [bodies[i], bodies[j]];
      if (!pulls(a) && !pulls(b)) continue;
      const d = [b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2]];
      const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + eps2(a, b);
      if (!(r2 > 0)) continue;
      const w = 1 / (r2 * Math.sqrt(r2));
      for (let k = 0; k < 3; k++) {
        if (pulls(b)) acc[i][k] += b.gm * w * d[k];
        if (pulls(a)) acc[j][k] -= a.gm * w * d[k];
      }
    }
}

/** The shortest time any pair takes to change: a step is a fraction of it. */
function pace(bodies: NBody[]): number {
  let t = Infinity;
  for (let i = 0; i < bodies.length; i++)
    for (let j = i + 1; j < bodies.length; j++) {
      const [a, b] = [bodies[i], bodies[j]];
      const gm = (pulls(a) ? a.gm : 0) + (pulls(b) ? b.gm : 0);
      if (!(gm > 0)) continue;
      const r2 = (b.pos[0] - a.pos[0]) ** 2 + (b.pos[1] - a.pos[1]) ** 2 + (b.pos[2] - a.pos[2]) ** 2 + eps2(a, b);
      if (r2 > 0) t = Math.min(t, Math.sqrt((r2 * Math.sqrt(r2)) / gm));
    }
  return t;
}

/**
 * Bodies as softened point masses under their own gravity for a time, in float64: a
 * kick-drift-kick leapfrog whose step is a fraction `eta` of the quickest pair's own time. The
 * bodies given are left as they are.
 */
export function integrateOrbits(start: NBody[], time: number, opts: { eta?: number; maxSteps?: number; samples?: number } = {}): Orbits {
  const { eta = 0.02, maxSteps = 20000, samples = 400 } = opts;
  const bodies = start.map((b) => ({ ...b, pos: [b.pos[0] ?? 0, b.pos[1] ?? 0, b.pos[2] ?? 0], vel: b.fixed ? [0, 0, 0] : [b.vel[0] ?? 0, b.vel[1] ?? 0, b.vel[2] ?? 0] }));
  const paths = bodies.map((b) => [[...b.pos]]);
  if (!(time > 0) || !Number.isFinite(time) || !bodies.length) return { paths, bodies, steps: 0 };
  const acc = bodies.map(() => [0, 0, 0]);
  accelerations(bodies, acc);
  const floor = time / maxSteps;
  const every = time / samples;
  let t = 0;
  let next = every;
  let steps = 0;
  while (t < time && steps < maxSteps) {
    let dt = Math.max(Math.min(eta * pace(bodies), every), floor);
    if (t + dt >= time * (1 - 1e-12)) dt = time - t;
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      if (b.fixed) continue;
      for (let k = 0; k < 3; k++) {
        b.vel[k] += 0.5 * dt * acc[i][k];
        b.pos[k] += dt * b.vel[k];
      }
    }
    accelerations(bodies, acc);
    for (let i = 0; i < bodies.length; i++) {
      if (bodies[i].fixed) continue;
      for (let k = 0; k < 3; k++) bodies[i].vel[k] += 0.5 * dt * acc[i][k];
    }
    t += dt;
    steps++;
    if (t >= next || t >= time) {
      bodies.forEach((b, i) => paths[i].push([...b.pos]));
      while (next <= t) next += every;
    }
  }
  return { paths, bodies, steps };
}

/** Kinetic plus potential energy of the bodies, in units where a body's mass is its G M. */
export function orbitEnergy(bodies: NBody[]): number {
  let e = 0;
  bodies.forEach((b, i) => {
    if (!pulls(b)) return;
    e += 0.5 * b.gm * (b.vel[0] ** 2 + b.vel[1] ** 2 + b.vel[2] ** 2);
    for (let j = i + 1; j < bodies.length; j++) {
      const c = bodies[j];
      if (!pulls(c)) continue;
      const r2 = (b.pos[0] - c.pos[0]) ** 2 + (b.pos[1] - c.pos[1]) ** 2 + (b.pos[2] - c.pos[2]) ** 2 + eps2(b, c);
      e -= (b.gm * c.gm) / Math.sqrt(r2);
    }
  });
  return e;
}

// ---- the scene: what a document puts in the box ----

export interface SceneBody extends NBody {
  /** its part's place in the document, and its id */
  part: number;
  id: string;
  kind: 'mass' | 'star';
  /** softening and sink of a point mass, radius of a star (0 where it has none) */
  soft: number;
  sink: number;
  radius: number;
  /** its position is its own: not placed by an orbit */
  draggable: boolean;
  /** a star the document alone could not build: drawn as a marker, its size unknown */
  unbuilt: boolean;
}

export interface Scene {
  box: number;
  ndim: 2 | 3;
  dx: number;
  faces: { lo: number[]; hi: number[] };
  bodies: SceneBody[];
  disks: { part: DiskPart; center: number[]; gm: number }[];
  shapes: { kind: 'region' | 'obstacle' | 'inflow'; shape: Shape }[];
}

/** What the preview draws, from the document alone. Never throws. */
export function sceneOf(spec: SetupSpec): Scene {
  let built: BuiltSetup;
  try {
    built = buildSetupBodies(spec);
  } catch {
    built = { G: 1, box: spec.domain?.box ?? 1, ndim: spec.domain?.ndim === 2 ? 2 : 3, dx: 0, ids: [], bodies: null, skipped: [] };
  }
  const parts: Part[] = spec.parts ?? [];
  const { box, ndim, dx, ids } = built;
  const d = spec.domain ?? {};
  const bc = (ndim === 2 && d.bc2d) || d.bc;
  const side = (s: readonly number[] | undefined) => [0, 1, 2].map((a) => (typeof s?.[a] === 'number' ? s[a] : 1));
  const placed = new Set(parts.flatMap((p) => (p.kind === 'orbit' && Array.isArray(p.of) ? p.of : [])));
  const bodies: SceneBody[] = [];
  const b = built.bodies;
  if (b) {
    for (const s of b.stars) {
      bodies.push({ part: ids.indexOf(s.id), id: s.id, kind: 'star', gm: built.G * s.M, pos: [...s.center], vel: [...s.vel], soft: 0, sink: 0, radius: s.R, draggable: !placed.has(s.id), unbuilt: false });
    }
    b.masses.forEach((m, k) => {
      const id = b.massIds[k];
      const soft = m.soft ?? 2 * dx;
      bodies.push({ part: ids.indexOf(id), id, kind: 'mass', gm: m.gm, pos: [...m.pos], vel: [...m.vel], soft, sink: m.sinkRadius ?? soft, radius: 0, draggable: !placed.has(id), unbuilt: false, massless: m.massless, fixed: m.fixed });
    });
  }
  for (const i of built.skipped) {
    const p = parts[i];
    if (p?.kind !== 'star') continue;
    bodies.push({ part: i, id: ids[i], kind: 'star', gm: built.G * (p.mass ?? 0), pos: [...(p.pos ?? [0, 0, 0])], vel: [...(p.vel ?? [0, 0, 0])], soft: 0, sink: 0, radius: 0, draggable: !placed.has(ids[i]), unbuilt: true });
  }
  const disks = parts.flatMap((p) => {
    if (p.kind !== 'disk') return [];
    const about = bodies.filter((k) => Array.isArray(p.about) && p.about.includes(k.id));
    const gm = about.reduce((s, k) => s + k.gm, 0);
    const center = [0, 1, 2].map((a) => (gm > 0 ? about.reduce((s, k) => s + k.gm * k.pos[a], 0) / gm : 0));
    return [{ part: p, center, gm }];
  });
  const shapes = parts.flatMap((p) => (p.kind === 'region' || p.kind === 'obstacle' || p.kind === 'inflow') && p.shape ? [{ kind: p.kind, shape: p.shape }] : []);
  return { box, ndim, dx, faces: { lo: side(bc?.lo), hi: side(bc?.hi) }, bodies, disks, shapes };
}

/** A time over which the bodies visibly move, where the document names none: the period of a circular orbit as wide as they are apart. */
export function guessTime(scene: Scene): number {
  const heavy = scene.bodies.filter(pulls);
  const gm = heavy.reduce((s, b) => s + b.gm, 0);
  let far = 0;
  for (const a of scene.bodies) for (const b of heavy) far = Math.max(far, Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1], a.pos[2] - b.pos[2]));
  if (gm > 0 && far > 0) return 2 * Math.PI * Math.sqrt(far ** 3 / gm);
  // Nothing pulls: the time the fastest body takes to cross the box.
  const fast = Math.max(0, ...scene.bodies.map((b) => Math.hypot(b.vel[0], b.vel[1], b.vel[2])));
  return fast > 0 ? scene.box / fast : 0;
}

// ---- drawing ----

const BC_COLORS = ['#6cb6ff', '#8792a8', '#ffb454', '#5fd08a'];
const BODY_COLORS = ['#6cb6ff', '#ff9d5c', '#5fd08a', '#e58fff', '#ffd666', '#7fe3e0', '#ff8fa3', '#b6c2ff'];
const SHAPE_STYLE: Record<'region' | 'obstacle' | 'inflow', { fill: string; stroke: string }> = {
  region: { fill: 'rgba(108, 182, 255, 0.16)', stroke: 'rgba(108, 182, 255, 0.7)' },
  obstacle: { fill: 'rgba(135, 146, 168, 0.55)', stroke: '#aab3c5' },
  inflow: { fill: 'rgba(95, 208, 138, 0.22)', stroke: 'rgba(95, 208, 138, 0.8)' },
};

interface View {
  canvas: HTMLCanvasElement;
  /** the world axes across and up the canvas */
  axes: [number, number];
  names: [string, string];
}

/** A convex polygon cut to the side of a line where n . p < at. */
function clip(poly: number[][], n: [number, number], at: number): number[][] {
  const out: number[][] = [];
  const f = (p: number[]) => n[0] * p[0] + n[1] * p[1] - at;
  poly.forEach((p, i) => {
    const q = poly[(i + 1) % poly.length];
    const [fp, fq] = [f(p), f(q)];
    if (fp < 0) out.push(p);
    if (fp < 0 !== fq < 0) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  });
  return out;
}

function draw(view: View, scene: Scene, paths: number[][][], extent: number, legend: boolean, hover: number): void {
  const { canvas } = view;
  const size = canvas.width;
  const ctx = canvas.getContext('2d');
  if (!ctx || !size) return;
  const dpr = size / Math.max(canvas.clientWidth, 1);
  const [a, b] = view.axes;
  const scale = size / (2 * extent);
  const X = (x: number) => size / 2 + x * scale;
  const Y = (y: number) => size / 2 - y * scale;
  const half = scene.box / 2;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#0d0f14';
  ctx.fillRect(0, 0, size, size);
  ctx.lineJoin = 'round';
  const px = (v: number) => v * dpr;

  // The inside of the box, a shade lighter than what is beyond it.
  ctx.fillStyle = '#131722';
  ctx.fillRect(X(-half), Y(half), 2 * half * scale, 2 * half * scale);

  const window_: number[][] = [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]];
  const polygon = (poly: number[][]) => {
    if (poly.length < 3) return;
    ctx.beginPath();
    poly.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  };

  // Disks: an annulus from above; from the side, as thick as their scale height.
  ctx.lineWidth = px(1);
  for (const d of scene.disks) {
    const { rIn, rOut, cs } = d.part;
    if (!(rOut > 0) || !(rIn >= 0)) continue;
    ctx.fillStyle = 'rgba(255, 180, 84, 0.14)';
    ctx.strokeStyle = 'rgba(255, 180, 84, 0.65)';
    const [cx, cy] = [d.center[a], d.center[b]];
    if (b === 1) {
      ctx.beginPath();
      ctx.arc(X(cx), Y(cy), rOut * scale, 0, 2 * Math.PI);
      ctx.moveTo(X(cx) + rIn * scale, Y(cy));
      ctx.arc(X(cx), Y(cy), rIn * scale, 0, 2 * Math.PI, true);
      ctx.fill('evenodd');
      ctx.stroke();
    } else {
      const height = (r: number) => Math.max(Math.min(d.gm > 0 ? (cs * r ** 1.5) / Math.sqrt(d.gm) : 0, r), 1.5 / scale);
      for (const s of [-1, 1]) {
        const lo = Math.min(rIn, rOut);
        polygon([[cx + s * lo, cy - height(lo)], [cx + s * rOut, cy - height(rOut)], [cx + s * rOut, cy + height(rOut)], [cx + s * lo, cy + height(lo)]]);
      }
    }
  }

  // Regions, obstacles and inflows, in the order of the parts.
  for (const s of scene.shapes) {
    ctx.fillStyle = SHAPE_STYLE[s.kind].fill;
    ctx.strokeStyle = SHAPE_STYLE[s.kind].stroke;
    const sh = s.shape;
    if (sh.type === 'sphere') {
      const c = sh.center ?? [0, 0, 0];
      const r = sh.radiusCells !== undefined ? sh.radiusCells * (scene.dx || scene.box / 128) : (sh.radius ?? 0.1 * scene.box);
      ctx.beginPath();
      ctx.arc(X(c[a]), Y(c[b]), Math.max(r * scale, px(1.5)), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    } else if (sh.type === 'box') {
      // A side left open runs out of the picture.
      const lo = (k: number) => (typeof sh.min?.[k] === 'number' ? sh.min[k] : -2 * extent);
      const hi = (k: number) => (typeof sh.max?.[k] === 'number' ? sh.max[k] : 2 * extent);
      polygon([[lo(a), lo(b)], [hi(a), lo(b)], [hi(a), hi(b)], [lo(a), hi(b)]]);
    } else if (sh.type === 'half' && Array.isArray(sh.normal)) {
      const norm = Math.hypot(...sh.normal) || 1;
      const n: [number, number] = [sh.normal[a] / norm, sh.normal[b] / norm];
      const at = sh.at ?? 0;
      // A plane seen face on covers the view or none of it.
      if (Math.hypot(n[0], n[1]) < 1e-9) {
        if (at > 0) polygon(window_);
      } else polygon(clip(window_, n, at));
    }
  }

  // The box: each edge in the color of what the gas meets at that face.
  const edge = (kind: number, x0: number, y0: number, x1: number, y1: number) => {
    ctx.strokeStyle = BC_COLORS[kind] ?? BC_COLORS[1];
    ctx.setLineDash(kind === 0 ? [px(5), px(3)] : []);
    ctx.beginPath();
    ctx.moveTo(X(x0), Y(y0));
    ctx.lineTo(X(x1), Y(y1));
    ctx.stroke();
  };
  ctx.lineWidth = px(2);
  ctx.lineCap = 'square';
  edge(scene.faces.lo[a], -half, -half, -half, half);
  edge(scene.faces.hi[a], half, -half, half, half);
  edge(scene.faces.lo[b], -half, -half, half, -half);
  edge(scene.faces.hi[b], -half, half, half, half);
  ctx.setLineDash([]);
  ctx.lineCap = 'butt';

  // The paths the bodies would take.
  ctx.lineWidth = px(1.25);
  paths.forEach((path, i) => {
    if (path.length < 2) return;
    ctx.strokeStyle = BODY_COLORS[i % BODY_COLORS.length];
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    path.forEach((p, k) => (k ? ctx.lineTo(X(p[a]), Y(p[b])) : ctx.moveTo(X(p[a]), Y(p[b]))));
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  ctx.font = `${px(10)}px system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  scene.bodies.forEach((body, i) => {
    const color = BODY_COLORS[i % BODY_COLORS.length];
    const [x, y] = [X(body.pos[a]), Y(body.pos[b])];
    ctx.lineWidth = px(body.part === hover ? 2 : 1);
    if (body.kind === 'star' && !body.unbuilt) {
      ctx.fillStyle = 'rgba(255, 214, 102, 0.3)';
      ctx.strokeStyle = '#ffd666';
      ctx.beginPath();
      ctx.arc(x, y, Math.max(body.radius * scale, px(3)), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    } else if (body.kind === 'star') {
      // Its size is not known here: a cross in a dashed ring.
      ctx.strokeStyle = '#ffd666';
      ctx.setLineDash([px(2), px(2)]);
      ctx.beginPath();
      ctx.arc(x, y, px(7), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x - px(4), y - px(4));
      ctx.lineTo(x + px(4), y + px(4));
      ctx.moveTo(x - px(4), y + px(4));
      ctx.lineTo(x + px(4), y - px(4));
      ctx.stroke();
    } else {
      if (body.soft > 0) {
        ctx.strokeStyle = color;
        ctx.setLineDash([px(3), px(3)]);
        ctx.beginPath();
        ctx.arc(x, y, body.soft * scale, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (body.sink > 0) {
        ctx.strokeStyle = '#ff9d5c';
        ctx.beginPath();
        ctx.arc(x, y, body.sink * scale, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, px(body.part === hover ? 4 : 3), 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = '#d6dbe6';
    ctx.fillText(body.id, Math.min(x + px(7), size - px(40)), Math.max(px(8), y - px(8)));
  });

  // Which way is which, and what the edge colors mean.
  ctx.fillStyle = '#8792a8';
  ctx.fillText(`${view.names[0]} →`, size - px(26), size - px(9));
  ctx.fillText(`${view.names[1]} ↑`, px(5), px(9));
  if (legend) {
    const kinds = [...new Set([...scene.faces.lo, ...scene.faces.hi].slice(0, 6))].filter((k) => BC_CHOICES[k]).sort();
    // On a dark patch of its own: the corner it sits in is where the box's edges meet.
    ctx.fillStyle = 'rgba(13, 15, 20, 0.8)';
    const wide = Math.max(...kinds.map((k) => ctx.measureText(BC_CHOICES[k].label).width));
    ctx.fillRect(px(18), size - px(22) - kinds.length * px(12), wide + px(28), kinds.length * px(12) + px(4));
    kinds.forEach((k, row) => {
      const y = size - px(26) - (kinds.length - 1 - row) * px(12);
      ctx.strokeStyle = BC_COLORS[k];
      ctx.lineWidth = px(2);
      ctx.setLineDash(k === 0 ? [px(5), px(3)] : []);
      ctx.beginPath();
      ctx.moveTo(px(22), y);
      ctx.lineTo(px(36), y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#8792a8';
      ctx.fillText(BC_CHOICES[k].label, px(40), y);
    });
  }
}

// ---- the panel ----

export interface Preview {
  el: HTMLElement;
  /** Draw this document. */
  update(spec: SetupSpec): void;
}

/** A length rounded to a step a person would type: about a five-hundredth of the box. */
function tidy(v: number, box: number): number {
  const step = 10 ** Math.floor(Math.log10(box / 500));
  return +(Math.round(v / step) * step).toPrecision(12);
}

/**
 * The preview panel. `onMove` is called while a body is dragged, with its part's place in the
 * document and its new position: the dialog writes it as the part's `pos` and updates.
 */
export function createPreview(opts: { onMove(part: number, pos: Vec3): void }): Preview {
  const canvas = (label: string) => h('canvas', { class: 'sp-view', role: 'img', 'aria-label': label });
  const views: View[] = [
    { canvas: canvas('The setup seen from above: x across, y up'), axes: [0, 1], names: ['x', 'y'] },
    { canvas: canvas('The setup seen from the side: x across, z up'), axes: [0, 2], names: ['x', 'z'] },
  ];
  const time = h('input', { type: 'number', id: 'sp-time', min: 0, step: 'any', 'data-factor': 2, inputmode: 'decimal' });
  const note = h('p', { class: 'sp-note' });
  const captions = [h('div', { class: 'sp-cap', text: 'From above (x, y)' }), h('div', { class: 'sp-cap', text: 'From the side (x, z)' })];
  const el = h('div', { class: 'sp' },
    captions[0], views[0].canvas, captions[1], views[1].canvas,
    h('label', { class: 'f', for: 'sp-time', title: 'How long the paths of the bodies are followed, in code units of time. The bodies move as point masses: no gas pulls on them here. Empty: the setup’s dynamical time, or one orbit' }, h('span', { text: 'Paths for time' }), time),
    note);

  let scene: Scene | null = null;
  let paths: number[][][] = [];
  let extent = 0.55;
  let hover = -1;
  let drag: { part: number; view: View; pos: number[] } | null = null;
  let spec_: SetupSpec | null = null;

  // Each canvas is as many device pixels as it is shown at.
  const fit = () => {
    for (const v of views) {
      const w = Math.round(v.canvas.clientWidth * (window.devicePixelRatio || 1));
      if (w > 0 && v.canvas.width !== w) v.canvas.width = v.canvas.height = w;
    }
  };

  const redraw = () => {
    fit();
    if (!scene) return;
    views.forEach((v, i) => {
      if (!v.canvas.hidden) draw(v, scene!, paths, extent, i === 0, hover);
    });
  };

  const compute = () => {
    if (!spec_) return;
    try {
      scene = sceneOf(spec_);
      const typed = time.value.trim() === '' ? NaN : Number(time.value);
      const guess = spec_.run?.tDyn ?? guessTime(scene);
      time.placeholder = guess > 0 ? String(+guess.toPrecision(4)) : 'none';
      const t = Number.isFinite(typed) ? typed : guess;
      paths = scene.bodies.length > 1 || scene.bodies.some((b) => Math.hypot(...b.vel) > 0) ? integrateOrbits(scene.bodies, t).paths : scene.bodies.map(() => []);
      // The picture takes in the box and every body's start; while one is dragged it holds still.
      if (!drag) {
        const half = scene.box / 2;
        const reach = Math.max(0, ...scene.bodies.map((b) => Math.max(...b.pos.map(Math.abs)) + b.radius));
        extent = 1.08 * Math.max(half, Math.min(reach, 20 * half));
      }
      const flat = scene.ndim === 2;
      views[1].canvas.hidden = captions[1].hidden = flat;
      const unbuilt = scene.bodies.filter((b) => b.unbuilt).length;
      const movable = scene.bodies.some((b) => b.draggable);
      note.textContent = [
        movable ? 'Drag a body to move it.' : scene.bodies.length ? 'These bodies are placed by an orbit: change the orbit to move them.' : '',
        unbuilt ? `${unbuilt === 1 ? 'A star is' : `${unbuilt} stars are`} drawn as a cross: the size follows from the equation of state, which the preview does not have.` : '',
      ].filter(Boolean).join(' ');
    } catch (e) {
      // A document that cannot be drawn is the issues list's to explain.
      scene = null;
      note.textContent = `No preview: ${e instanceof Error ? e.message : String(e)}`;
      for (const v of views) v.canvas.getContext('2d')?.clearRect(0, 0, v.canvas.width, v.canvas.height);
    }
    redraw();
  };

  time.addEventListener('change', compute);

  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(redraw);
    for (const v of views) ro.observe(v.canvas);
  }

  for (const view of views) {
    const c = view.canvas;
    const world = (e: PointerEvent): [number, number] => {
      const r = c.getBoundingClientRect();
      const s = r.width / (2 * extent);
      return [(e.clientX - r.left - r.width / 2) / s, -(e.clientY - r.top - r.height / 2) / s];
    };
    /** The body under the pointer that can be moved: the nearest within reach of a finger. */
    const pick = (e: PointerEvent): SceneBody | null => {
      if (!scene) return null;
      const r = c.getBoundingClientRect();
      const s = r.width / (2 * extent);
      const [x, y] = world(e);
      let best: SceneBody | null = null;
      let bestD = Infinity;
      for (const body of scene.bodies) {
        if (!body.draggable) continue;
        const d = Math.hypot(body.pos[view.axes[0]] - x, body.pos[view.axes[1]] - y);
        if (d <= Math.max(12 / s, body.radius) && d < bestD) [best, bestD] = [body, d];
      }
      return best;
    };
    c.addEventListener('pointerdown', (e) => {
      const body = pick(e);
      if (!body) return;
      drag = { part: body.part, view, pos: [...body.pos] };
      try {
        c.setPointerCapture(e.pointerId);
      } catch {
        // A pointer that cannot be captured still drags while it stays over the view.
      }
      c.classList.add('dragging');
      e.preventDefault();
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag || drag.view !== view) {
        const over = pick(e)?.part ?? -1;
        c.classList.toggle('grab', over >= 0);
        if (over !== hover) {
          hover = over;
          redraw();
        }
        return;
      }
      if (!scene) return;
      const [x, y] = world(e);
      const hold = (v: number) => tidy(Math.max(-extent, Math.min(extent, v)), scene!.box);
      drag.pos[view.axes[0]] = hold(x);
      drag.pos[view.axes[1]] = hold(y);
      opts.onMove(drag.part, [drag.pos[0], drag.pos[1], drag.pos[2]]);
    });
    const drop = (e: PointerEvent) => {
      if (!drag || drag.view !== view) return;
      drag = null;
      c.classList.remove('dragging');
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
      compute();
    };
    c.addEventListener('pointerup', drop);
    c.addEventListener('pointercancel', drop);
    c.addEventListener('pointerleave', () => {
      if (drag || hover < 0) return;
      hover = -1;
      redraw();
    });
  }

  return {
    el,
    update(spec) {
      spec_ = spec;
      compute();
    },
  };
}
