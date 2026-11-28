// The dialog's preview: the setup drawn from its document alone, on 2D canvases, from above
// (x-y) and from the side (x-z). The box with its faces colored by what the gas meets there,
// the point masses with their softening and sink, stars to scale, disks, the shapes of regions,
// obstacles and inflows, and the paths the bodies would take as point masses for a chosen time.
// No GPU work: it cannot disturb the run behind the dialog.
//
// The refinement zones (mesh.zones) are shaded by tier, as the run's mesh sees them: each point
// in the zone listed first that holds it (amr/zones.ts).
//
// Anything drawn can be clicked, which shows its settings: a body, a shape, a face of the box, a zone.
// A body whose position is its own (not placed by an orbit) can be dragged in either view, as can
// a shape; an obstacle that is a box or a half-space also turns, by the handle it shows when picked.
import { meshSilhouette } from '../../geometry/store';
import { zoneAt, type Zone } from '../../amr/zones';
import { zoneOf } from '../../setup/zones';
import { BOUNDARIES } from '../../problems/boundaries';
import { boxFrame, boxPivot, eulerOf, rotationMatrix } from '../../setup/parts/region';
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
  ndim: 1 | 2 | 3;
  dx: number;
  faces: { lo: number[]; hi: number[] };
  bodies: SceneBody[];
  disks: { part: DiskPart; center: number[]; gm: number }[];
  /** with its part's place in the document */
  shapes: { part: number; kind: 'region' | 'obstacle' | 'inflow'; shape: Shape }[];
  /** the refinement zones that could be placed, with their places in mesh.zones */
  zones: { index: number; zone: Zone }[];
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
  const shapes = parts.flatMap((p, part) => (p.kind === 'region' || p.kind === 'obstacle' || p.kind === 'inflow') && p.shape && typeof p.shape === 'object' ? [{ part, kind: p.kind, shape: p.shape }] : []);
  const zones = b ? (spec.mesh?.zones ?? []).flatMap((z, index) => {
    const zone = zoneOf(z, b);
    return typeof zone === 'string' ? [] : [{ index, zone }];
  }) : [];
  return { box, ndim, dx, faces: { lo: side(bc?.lo), hi: side(bc?.hi) }, bodies, disks, shapes, zones };
}

/** A tier's color in the preview, by its rank among the setup's tiers (the first the warmest). */
const TIER_COLORS = ['#e8590c', '#f59f00', '#e9d74a', '#d06fb0', '#4dabf7', '#69db7c', '#b197fc', '#adb5bd'];
const tierRank = (scene: Scene) => [...new Set(scene.zones.map((z) => z.zone.tier))].sort((p, q) => p - q);

/** The zones of a view, shaded by tier (lighter where hovered or picked), as an image of the view's window. */
function zoneLayer(view: View, scene: Scene, extent: number, hover: string, picked: string): HTMLCanvasElement | null {
  if (!scene.zones.length || typeof document === 'undefined') return null;
  const [a, b] = view.axes;
  const R = Math.max(32, Math.min(320, view.canvas.width));
  const key = JSON.stringify([scene.zones, extent, a, b, R, hover.startsWith('zone.') ? hover : '', picked.startsWith('zone.') ? picked : '']);
  if (view.zones?.key === key) return view.zones.image;
  const image = document.createElement('canvas');
  image.width = image.height = R;
  const ctx = image.getContext('2d');
  if (!ctx) return null;
  const data = ctx.createImageData(R, R);
  const ranks = tierRank(scene);
  const shapes = scene.zones.map((z) => z.zone);
  const rgb = (hex: string) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));
  const colors = scene.zones.map((z) => rgb(TIER_COLORS[Math.min(ranks.indexOf(z.zone.tier), TIER_COLORS.length - 1)]));
  const lit = scene.zones.map((z) => hover === `zone.${z.index}` || picked === `zone.${z.index}`);
  const p = [0, 0, 0];
  for (let j = 0; j < R; j++)
    for (let i = 0; i < R; i++) {
      p[a] = -extent + ((i + 0.5) * 2 * extent) / R;
      p[b] = extent - ((j + 0.5) * 2 * extent) / R;
      const k = zoneAt(shapes, p);
      if (k < 0) continue;
      const o = 4 * (j * R + i);
      const c = colors[k];
      data.data[o] = c[0];
      data.data[o + 1] = c[1];
      data.data[o + 2] = c[2];
      data.data[o + 3] = lit[k] ? 150 : 95;
    }
  ctx.putImageData(data, 0, 0);
  view.zones = { key, image };
  return image;
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

const BC_COLORS = BOUNDARIES.map((b) => b.color);
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
  /** the zones' shaded layer, as last drawn (zoneLayer) */
  zones?: { key: string; image: HTMLCanvasElement };
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

/** Points of the view's plane that wrap these, counterclockwise. */
function hull(pts: number[][]): number[][] {
  const p = [...pts].sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: number[][]) => {
    const out: number[][] = [];
    for (const q of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(p), ...half([...p].reverse())];
}

/** Whether a point is inside a convex polygon whose corners run counterclockwise. */
function within(poly: number[][], x: number, y: number): boolean {
  if (poly.length < 3) return false;
  return poly.every((p, i) => {
    const q = poly[(i + 1) % poly.length];
    return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]) >= 0;
  });
}

/** How a shape looks in a view: a circle, or a polygon cut to the window. Null where it shows nothing. */
type Outline = { circle: [number, number, number] } | { poly: number[][] } | null;

function outlineOf(sh: Shape, view: View, scene: Scene, extent: number): Outline {
  const [a, b] = view.axes;
  const window_: number[][] = [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]];
  if (sh.type === 'sphere') {
    const c = sh.center ?? [0, 0, 0];
    const r = sh.radiusCells !== undefined ? sh.radiusCells * (scene.dx || scene.box / 128) : (sh.radius ?? 0.1 * scene.box);
    return { circle: [c[a] ?? 0, c[b] ?? 0, r] };
  }
  if (sh.type === 'box') {
    // A side left open runs out of the picture. A turned box: its corners turned, seen from the view.
    const far = 4 * extent;
    const frame = boxFrame(sh, scene.ndim === 2);
    const lo = (k: number) => (typeof sh.min?.[k] === 'number' ? sh.min[k] : (frame?.pivot[k] ?? 0) - far);
    const hi = (k: number) => (typeof sh.max?.[k] === 'number' ? sh.max[k] : (frame?.pivot[k] ?? 0) + far);
    const corners: number[][] = [];
    for (let i = 0; i < 8; i++) {
      let p = [i & 1 ? hi(0) : lo(0), i & 2 ? hi(1) : lo(1), i & 4 ? hi(2) : lo(2)];
      if (frame) {
        const c = frame.pivot;
        const q = frame.world([p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
        p = [q[0] + c[0], q[1] + c[1], q[2] + c[2]];
      }
      corners.push([p[a], p[b]]);
    }
    let poly = hull(corners);
    for (const [n, at] of [[[1, 0], extent], [[-1, 0], extent], [[0, 1], extent], [[0, -1], extent]] as [[number, number], number][]) poly = clip(poly, n, at);
    return poly.length >= 3 ? { poly } : null;
  }
  if (sh.type === 'mesh') {
    // A mesh file, once loaded: the hull of its vertices as the view sees them (geometry/store.ts).
    const pts = meshSilhouette(sh, { box: scene.box, ndim: scene.ndim }, [a, b]);
    if (!pts || pts.length < 3) return null;
    let poly = hull(pts);
    for (const [n, at] of [[[1, 0], extent], [[-1, 0], extent], [[0, 1], extent], [[0, -1], extent]] as [[number, number], number][]) poly = clip(poly, n, at);
    return poly.length >= 3 ? { poly } : null;
  }
  if (sh.type === 'half' && Array.isArray(sh.normal)) {
    const norm = Math.hypot(...sh.normal) || 1;
    const n: [number, number] = [(sh.normal[a] ?? 0) / norm, (sh.normal[b] ?? 0) / norm];
    const at = sh.at ?? 0;
    // A plane seen face on covers the view or none of it.
    if (Math.hypot(n[0], n[1]) < 1e-9) return at > 0 ? { poly: window_ } : null;
    const poly = clip(window_, n, at);
    return poly.length >= 3 ? { poly } : null;
  }
  return null;
}

/** What the pointer can take hold of: a body or a shape (by its part), a shape's turning handle, a face of the box. */
type Target = { kind: 'part'; part: number } | { kind: 'turn'; part: number } | { kind: 'face'; side: 'lo' | 'hi'; axis: number } | { kind: 'zone'; zone: number };
const keyOf = (t: Target | null) => (!t ? '' : t.kind === 'face' ? `face.${t.side}.${t.axis}` : t.kind === 'zone' ? `zone.${t.zone}` : `${t.kind}.${t.part}`);

/** A shape that turns in the preview: an obstacle that is a box or a half-space. */
const turns = (s: Scene['shapes'][number]) => s.kind === 'obstacle' && (s.shape.type === 'box' || (s.shape.type === 'half' && Array.isArray(s.shape.normal)));

/**
 * Where a turnable shape turns about in a view, and where its handle is (in world units of the
 * view's plane): a box about its center, the handle above it; a half-space about the foot of its
 * plane, the handle along its normal.
 */
function handleOf(s: Scene['shapes'][number], view: View, scene: Scene, extent: number, scale: number): { pivot: [number, number]; at: [number, number] } | null {
  const [a, b] = view.axes;
  const sh = s.shape;
  if (sh.type === 'box') {
    const o = outlineOf(sh, view, scene, extent);
    if (!o || !('poly' in o)) return null;
    const c = boxPivot(sh);
    const top = Math.max(...o.poly.map((p) => p[1]));
    // Above the box, or inside the top of the picture where the box runs out of it.
    return { pivot: [c[a], c[b]], at: [c[a], Math.min(top + 16 / scale, extent - 10 / scale)] };
  }
  if (sh.type === 'half' && Array.isArray(sh.normal)) {
    const norm = Math.hypot(...sh.normal) || 1;
    const n = [(sh.normal[a] ?? 0) / norm, (sh.normal[b] ?? 0) / norm];
    const len = Math.hypot(n[0], n[1]);
    if (len < 1e-6) return null;
    const at = sh.at ?? 0;
    const foot: [number, number] = [n[0] * at, n[1] * at];
    return { pivot: foot, at: [foot[0] + ((n[0] / len) * 28) / scale, foot[1] + ((n[1] / len) * 28) / scale] };
  }
  return null;
}

function draw(view: View, scene: Scene, paths: number[][][], extent: number, legend: boolean, hover: string, picked: string): void {
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

  // The refinement zones, shaded by tier, under everything else.
  const layer = zoneLayer(view, scene, extent, hover, picked);
  if (layer) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(layer, 0, 0, size, size);
  }

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
    const o = outlineOf(s.shape, view, scene, extent);
    if (!o) continue;
    const lit = hover === `part.${s.part}` || picked === `part.${s.part}`;
    ctx.fillStyle = SHAPE_STYLE[s.kind].fill;
    ctx.strokeStyle = lit ? '#e8ecf4' : SHAPE_STYLE[s.kind].stroke;
    ctx.lineWidth = px(lit ? 2 : 1);
    if ('circle' in o) {
      const [cx, cy, r] = o.circle;
      ctx.beginPath();
      ctx.arc(X(cx), Y(cy), Math.max(r * scale, px(1.5)), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    } else polygon(o.poly);
  }
  // The picked obstacle's handle, to turn it by.
  const pickedShape = scene.shapes.find((s) => picked === `part.${s.part}` && turns(s));
  const handle = pickedShape && handleOf(pickedShape, view, scene, extent, scale);
  if (pickedShape && handle) {
    const lit = hover === `turn.${pickedShape.part}`;
    ctx.strokeStyle = '#e8ecf4';
    ctx.lineWidth = px(1);
    ctx.setLineDash([px(2), px(2)]);
    ctx.beginPath();
    ctx.moveTo(X(handle.pivot[0]), Y(handle.pivot[1]));
    ctx.lineTo(X(handle.at[0]), Y(handle.at[1]));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = lit ? '#ffd666' : '#c9b8ff';
    ctx.beginPath();
    ctx.arc(X(handle.at[0]), Y(handle.at[1]), px(lit ? 6 : 5), 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  // The box: each edge in the color of what the gas meets at that face.
  const edge = (kind: number, x0: number, y0: number, x1: number, y1: number, key: string) => {
    ctx.strokeStyle = BC_COLORS[kind] ?? BC_COLORS[1];
    ctx.lineWidth = px(hover === key || picked === key ? 4 : 2);
    ctx.setLineDash(kind === 0 ? [px(5), px(3)] : []);
    ctx.beginPath();
    ctx.moveTo(X(x0), Y(y0));
    ctx.lineTo(X(x1), Y(y1));
    ctx.stroke();
  };
  ctx.lineWidth = px(2);
  ctx.lineCap = 'square';
  edge(scene.faces.lo[a], -half, -half, -half, half, `face.lo.${a}`);
  edge(scene.faces.hi[a], half, -half, half, half, `face.hi.${a}`);
  edge(scene.faces.lo[b], -half, -half, half, -half, `face.lo.${b}`);
  edge(scene.faces.hi[b], -half, half, half, half, `face.hi.${b}`);
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
    ctx.lineWidth = px(hover === `part.${body.part}` || picked === `part.${body.part}` ? 2 : 1);
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
      ctx.arc(x, y, px(hover === `part.${body.part}` || picked === `part.${body.part}` ? 4 : 3), 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = '#d6dbe6';
    ctx.fillText(body.id, Math.min(x + px(7), size - px(40)), Math.max(px(8), y - px(8)));
  });

  // Which way is which, and what the edge colors mean.
  ctx.fillStyle = '#8792a8';
  ctx.fillText(`${view.names[0]} →`, size - px(26), size - px(9));
  ctx.fillText(`${view.names[1]} ↑`, px(5), px(9));
  if (legend && scene.zones.length) {
    // The tiers, top left: a swatch and the tier.
    const ranks = tierRank(scene);
    ctx.font = `${px(10)}px system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(13, 15, 20, 0.8)';
    ctx.fillRect(px(6), px(6), px(62), ranks.length * px(13) + px(6));
    ranks.forEach((t, row) => {
      const y = px(15) + row * px(13);
      ctx.fillStyle = TIER_COLORS[Math.min(row, TIER_COLORS.length - 1)];
      ctx.fillRect(px(11), y - px(4), px(10), px(8));
      ctx.fillStyle = '#8792a8';
      ctx.fillText(`tier ${t}`, px(26), y);
    });
  }
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
 * The preview panel. `onEdit` is called while something is dragged, with the path in the document
 * of what changed and its new value (a body's `pos`, a part's `shape`): the dialog writes it and
 * updates. `onSelect` is called with the path of what a click was on: the dialog shows it.
 */
export function createPreview(opts: { onEdit(path: string, value: unknown): void; onSelect(path: string): void }): Preview {
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
  /** what the pointer is over, and what was last taken hold of (keyOf) */
  let hover = '';
  let picked = '';
  /** a press on the views: what it took hold of, where, and as the document had it then */
  let drag: {
    view: View;
    target: Target;
    start: [number, number];
    client: [number, number];
    moved: boolean;
    pos?: number[];
    shape?: Shape;
    pivot?: [number, number];
  } | null = null;
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
      if (!v.canvas.hidden) draw(v, scene!, paths, extent, i === 0, hover, picked);
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
      // The picture takes in the box and every body's start; while something is dragged it holds still.
      if (!drag?.moved) {
        const half = scene.box / 2;
        const reach = Math.max(0, ...scene.bodies.map((b) => Math.max(...b.pos.map(Math.abs)) + b.radius));
        extent = 1.08 * Math.max(half, Math.min(reach, 20 * half));
      }
      const flat = scene.ndim === 2;
      views[1].canvas.hidden = captions[1].hidden = flat;
      const unbuilt = scene.bodies.filter((b) => b.unbuilt).length;
      const movable = scene.bodies.some((b) => b.draggable);
      const placed = scene.bodies.length && !movable;
      note.textContent = [
        'Click anything drawn to show its settings.',
        movable || scene.shapes.length ? `Drag ${movable && scene.shapes.length ? 'a body or a shape' : movable ? 'a body' : 'a shape'} to move it${scene.shapes.some(turns) ? '; a picked obstacle turns by its handle' : ''}.` : '',
        placed ? 'These bodies are placed by an orbit: change the orbit to move them.' : '',
        unbuilt ? `${unbuilt === 1 ? 'A star is' : `${unbuilt} stars are`} drawn as a cross: the size follows from the equation of state, which the preview does not have.` : '',
        scene.zones.length ? 'The refinement zones are shaded by tier, where the box\u2019s middle plane cuts them.' : '',
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

  /** Where a click on what was taken hold of leads in the settings. */
  const pathOf = (t: Target): string => {
    if (t.kind === 'face') return `domain.bc.${t.side}.${t.axis}`;
    if (t.kind === 'zone') return `mesh.zones.${t.zone}`;
    if (t.kind === 'part') return `parts.${t.part}`;
    const sh = scene?.shapes.find((s) => s.part === t.part)?.shape;
    return `parts.${t.part}.shape.${sh?.type === 'half' ? 'normal' : 'rotate'}`;
  };

  for (const view of views) {
    const c = view.canvas;
    const [a, b] = view.axes;
    /** World units per CSS pixel of this view. */
    const unit = () => (2 * extent) / Math.max(c.getBoundingClientRect().width, 1);
    const world = (e: PointerEvent): [number, number] => {
      const r = c.getBoundingClientRect();
      const s = r.width / (2 * extent);
      return [(e.clientX - r.left - r.width / 2) / s, -(e.clientY - r.top - r.height / 2) / s];
    };

    /**
     * What is under the pointer: the picked obstacle's handle, then a body (the nearest within
     * reach of a finger), a face of the box from just outside it, a shape (the last drawn on top),
     * a face from just inside, and last a half-space, which may cover half the view.
     */
    const pick = (e: PointerEvent): Target | null => {
      if (!scene) return null;
      const u = unit();
      const [x, y] = world(e);
      const pickedShape = scene.shapes.find((s) => picked === `part.${s.part}` && turns(s));
      const handle = pickedShape && handleOf(pickedShape, view, scene, extent, 1 / u);
      if (pickedShape && handle && Math.hypot(handle.at[0] - x, handle.at[1] - y) <= 10 * u) return { kind: 'turn', part: pickedShape.part };

      let best: SceneBody | null = null;
      let bestD = Infinity;
      for (const body of scene.bodies) {
        const d = Math.hypot(body.pos[a] - x, body.pos[b] - y);
        if (d <= Math.max(12 * u, body.radius) && d < bestD) [best, bestD] = [body, d];
      }
      if (best) return { kind: 'part', part: best.part };

      // A face of the box: just outside its edge it is the face's, as a thin shape along the
      // edge may be on its inside.
      const half = scene.box / 2;
      const near = 6 * u;
      const along = (v: number) => v >= -half - near && v <= half + near;
      const face = (outside: boolean): Target | null => {
        const by = (d: number) => (outside ? d >= 0 : d >= -near) && d <= near;
        if (by(-half - x) && along(y)) return { kind: 'face', side: 'lo', axis: a };
        if (by(x - half) && along(y)) return { kind: 'face', side: 'hi', axis: a };
        if (by(-half - y) && along(x)) return { kind: 'face', side: 'lo', axis: b };
        if (by(y - half) && along(x)) return { kind: 'face', side: 'hi', axis: b };
        return null;
      };
      const out = face(true);
      if (out) return out;

      const hit = (s: Scene['shapes'][number]) => {
        const o = outlineOf(s.shape, view, scene!, extent);
        if (!o) return false;
        if ('circle' in o) return Math.hypot(o.circle[0] - x, o.circle[1] - y) <= Math.max(o.circle[2], 6 * u);
        return within(o.poly, x, y);
      };
      const shapes = [...scene.shapes].reverse();
      const solid = shapes.find((s) => s.shape.type !== 'half' && hit(s));
      if (solid) return { kind: 'part', part: solid.part };

      const inner = face(false);
      if (inner) return inner;

      const plane = shapes.find((s) => s.shape.type === 'half' && hit(s));
      if (plane) return { kind: 'part', part: plane.part };

      // Last, a refinement zone: the one that holds the gas there.
      const p = [0, 0, 0];
      p[a] = x;
      p[b] = y;
      const k = zoneAt(scene.zones.map((z) => z.zone), p);
      return k >= 0 ? { kind: 'zone', zone: scene.zones[k].index } : null;
    };

    /** Whether what was taken hold of moves when dragged. */
    const movable = (t: Target) => {
      if (!scene || t.kind === 'face' || t.kind === 'zone') return false;
      if (t.kind === 'turn') return true;
      const body = scene.bodies.find((k) => k.part === t.part);
      return body ? body.draggable : scene.shapes.some((s) => s.part === t.part);
    };

    const setHover = (t: Target | null) => {
      c.classList.toggle('grab', !!t && movable(t));
      c.classList.toggle('point', !!t && !movable(t));
      const k = keyOf(t);
      if (k !== hover) {
        hover = k;
        redraw();
      }
    };

    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !scene) return;
      const target = pick(e);
      const k = keyOf(target);
      // What is pressed on is picked (an obstacle then shows its handle); a press on nothing drops it.
      if (target?.kind !== 'turn') {
        if (picked !== k) {
          picked = k;
          redraw();
        }
      }
      if (!target) return;
      const shape = target.kind === 'part' || target.kind === 'turn' ? scene.shapes.find((s) => s.part === target.part)?.shape : undefined;
      const body = target.kind === 'part' ? scene.bodies.find((k) => k.part === target.part) : undefined;
      const start = world(e);
      drag = { view, target, start, client: [e.clientX, e.clientY], moved: false, pos: body && [...body.pos], shape: shape && structuredClone(shape) };
      if (target.kind === 'turn' && shape) drag.pivot = handleOf({ part: target.part, kind: 'obstacle', shape }, view, scene, extent, 1 / unit())?.pivot;
      try {
        c.setPointerCapture(e.pointerId);
      } catch {
        // A pointer that cannot be captured still drags while it stays over the view.
      }
      if (movable(target)) c.classList.add('dragging');
      e.preventDefault();
    });

    c.addEventListener('pointermove', (e) => {
      if (!drag || drag.view !== view) {
        setHover(pick(e));
        return;
      }
      if (!scene || !movable(drag.target)) return;
      // A press that has not gone far is still a click.
      if (!drag.moved && Math.hypot(e.clientX - drag.client[0], e.clientY - drag.client[1]) < 4) return;
      drag.moved = true;
      const [x, y] = world(e);
      const box = scene.box;
      const t = drag.target;

      if (t.kind === 'part' && drag.pos) {
        // A body: where the pointer takes it, within the picture.
        const hold = (v: number) => tidy(Math.max(-extent, Math.min(extent, v)), box);
        const pos = [...drag.pos];
        pos[a] = hold(drag.pos[a] + x - drag.start[0]);
        pos[b] = hold(drag.pos[b] + y - drag.start[1]);
        opts.onEdit(`parts.${t.part}.pos`, [pos[0], pos[1], pos[2]]);
        return;
      }
      const sh = drag.shape;
      if (!sh) return;

      if (t.kind === 'part') {
        // A shape: moved by as much as the pointer, its open sides left open.
        const d = [0, 0, 0];
        d[a] = tidy(x - drag.start[0], box);
        d[b] = tidy(y - drag.start[1], box);
        const shift = (v: readonly (number | null)[] | undefined, k: number) => (typeof v?.[k] === 'number' ? tidy(v[k] + d[k], box) : (v?.[k] ?? null));
        let next: Shape;
        if (sh.type === 'sphere' || sh.type === 'mesh') next = { ...sh, center: [0, 1, 2].map((k) => tidy((sh.center?.[k] ?? 0) + d[k], box)) as Vec3 };
        else if (sh.type === 'box') next = { ...sh, min: [0, 1, 2].map((k) => shift(sh.min, k)) as Vec3, max: [0, 1, 2].map((k) => shift(sh.max, k)) as Vec3 };
        else {
          const norm = Math.hypot(...sh.normal) || 1;
          next = { ...sh, at: tidy((sh.at ?? 0) + (sh.normal[0] * d[0] + sh.normal[1] * d[1] + sh.normal[2] * d[2]) / norm, box) };
        }
        opts.onEdit(`parts.${t.part}.shape`, next);
        return;
      }

      if (t.kind === 'turn' && drag.pivot) {
        // An obstacle turned about the axis the view looks along, by the angle the pointer swept
        // about its pivot: in whole degrees, or in fifteens with Shift.
        const p = drag.pivot;
        const sweep = Math.atan2(y - p[1], x - p[0]) - Math.atan2(drag.start[1] - p[1], drag.start[0] - p[0]);
        const step = e.shiftKey ? 15 : 1;
        const deg = Math.round(((sweep * 180) / Math.PI) / step) * step;
        // Counterclockwise on the screen is about a x b: +z from above, -y from the side.
        const turn = [0, 0, 0];
        if (a === 0 && b === 1) turn[2] = deg;
        else turn[1] = -deg;
        const T = rotationMatrix(turn);
        const mul = (A: number[][], B: number[][]) => A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
        const round = (v: number, places: number) => +v.toFixed(places) + 0;
        let next: Shape;
        if (sh.type === 'box') {
          const angles = scene.ndim === 2 ? [0, 0, (sh.rotate?.[2] ?? 0) + turn[2]] : eulerOf(mul(T, rotationMatrix(sh.rotate ?? [0, 0, 0])));
          const r = angles.map((v) => round(((((v + 180) % 360) + 360) % 360) - 180, 1)) as Vec3;
          next = { ...sh, rotate: r };
          if (r.every((v) => v === 0)) delete (next as { rotate?: Vec3 }).rotate;
        } else if (sh.type === 'half') {
          const n = sh.normal;
          next = { ...sh, normal: [0, 1, 2].map((i) => round(T[i][0] * n[0] + T[i][1] * n[1] + T[i][2] * n[2], 4)) as Vec3 };
        } else return;
        opts.onEdit(`parts.${t.part}.shape`, next);
      }
    });

    const drop = (e: PointerEvent) => {
      if (!drag || drag.view !== view) return;
      const { target, moved } = drag;
      drag = null;
      c.classList.remove('dragging');
      if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
      // A click shows the settings of what it was on.
      if (!moved && e.type === 'pointerup') opts.onSelect(pathOf(target));
      compute();
    };
    c.addEventListener('pointerup', drop);
    c.addEventListener('pointercancel', drop);
    c.addEventListener('pointerleave', () => {
      if (drag || !hover) return;
      hover = '';
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
