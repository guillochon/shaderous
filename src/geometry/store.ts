// Mesh files a run's setup refers to, and their placement in the box (plans/wind-tunnel.md 3.3).
//
// The setup is compiled synchronously (setup/compile.ts), so a mesh's bytes must be at hand
// before: whoever loads a setup (the page, the runner, the MCP server, an adaptive mesh's
// initialization workers) puts each file it refers to here first, by the name the setup gives
// (putMeshFile). Parsed meshes and their distance fields are kept, by file, node and spacing.
import { parseMeshFile, extensionOf } from './index';
import { apply, bounds, flatten, invert, measure, mul, type Mat4, type Mesh } from './mesh';
import { buildSdf, sampleSdf, type DistanceField, type MeshReport } from './sdf';
import { outlineBounds, parseAirfoil, parseSvgPath, windingNumber, type Outline } from './outline';
import { builtinBody } from './builtin';

type Vec3 = [number, number, number];

/** The mesh shape of a setup (setup/spec.ts Shape, type 'mesh'). */
export interface MeshShape {
  type: 'mesh';
  file: string;
  node?: string;
  scale?: number;
  fit?: number;
  center?: readonly number[];
  rotate?: readonly number[];
  up?: 'x' | 'y' | 'z';
  /** what `center` places: the bounding box's middle (default) or the file's origin */
  anchor?: 'box' | 'origin';
}

const files = new Map<string, Uint8Array>();
const parsed = new Map<string, Mesh>();
const fields = new Map<string, { field: DistanceField; report: MeshReport }>();

/** Register a mesh file's bytes under the name a setup gives it. */
export function putMeshFile(name: string, bytes: Uint8Array) {
  if (files.get(name) === bytes) return;
  files.set(name, bytes);
  for (const k of [...parsed.keys()]) if (k.startsWith(`${name}|`)) parsed.delete(k);
  for (const k of [...fields.keys()]) if (k.startsWith(`${name}|`)) fields.delete(k);
}
export const hasMeshFile = (name: string): boolean => files.has(name) || name.startsWith('builtin:');
export const meshFileBytes = (name: string): Uint8Array | undefined => files.get(name);

/** The file names a setup's shapes refer to. */
export function meshFilesOf(spec: { parts?: readonly unknown[] }): string[] {
  const out = new Set<string>();
  for (const p of spec.parts ?? []) {
    const sh = (p as { shape?: { type?: string; file?: unknown } }).shape;
    if (sh?.type === 'mesh' && typeof sh.file === 'string' && sh.file) out.add(sh.file);
  }
  return [...out];
}

/** The bytes of every file a setup refers to that is registered, for a worker. */
export function meshFilesFor(spec: { parts?: readonly unknown[] }): Record<string, Uint8Array> {
  const out: Record<string, Uint8Array> = {};
  for (const f of meshFilesOf(spec)) {
    const b = files.get(f);
    if (b) out[f] = b;
  }
  return out;
}

/** The names a setup refers to that are not registered. */
export const missingMeshFiles = (spec: { parts?: readonly unknown[] }): string[] => meshFilesOf(spec).filter((f) => !hasMeshFile(f));

function bytesOf(file: string): Uint8Array {
  const b = files.get(file);
  if (!b) throw new Error(`the mesh file "${file}" is not loaded: give it beside the setup (runner, MCP) or drop it on the Setup dialog`);
  return b;
}

/** Whether a file is an outline (2D: an airfoil's coordinates, an SVG path) rather than triangles. */
export const isOutlineFile = (file: string): boolean => ['dat', 'svg'].includes(extensionOf(file));

function meshOf(file: string, node?: string): Mesh {
  const key = `${file}|${node ?? ''}`;
  let m = parsed.get(key);
  if (!m) {
    // (A built-in body, geometry/builtin.ts, is made here: no file.)
    const made = builtinBody(file);
    if (file.startsWith('builtin:') && !made) throw new Error(`there is no built-in body "${file.slice(8)}"`);
    m = flatten(made ?? parseMeshFile(file, bytesOf(file), (uri) => files.get(uri) ?? null), node);
    parsed.set(key, m);
  }
  return m;
}

/** The turn that takes the file's up axis to the box's (z in 3D, y in 2D). */
function upMatrix(up: 'x' | 'y' | 'z', ndim: 1 | 2 | 3): Mat4 {
  const m = new Float64Array(16);
  m[15] = 1;
  // Columns are where the file's x, y, z go.
  const target = ndim === 2 ? 1 : 2;
  const cols: Record<string, number[][]> = {
    // 3D (to z): y up turns about x by +90 (y to z, z to -y); x up about y by -90 (x to z, z to -x).
    'y3': [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    'x3': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    'z3': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    // 2D (to y): z up turns about x by -90 (z to y, y to -z); x up about z by +90 (x to y, y to -x).
    'z2': [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    'x2': [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    'y2': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  };
  const c = cols[`${up}${target === 2 ? 3 : 2}`];
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) m[4 * j + i] = c[j][i];
  return m;
}

/** R = Rz Ry Rx of degrees about x, then y, then z, as a matrix. */
function eulerMatrix(deg: readonly number[]): Mat4 {
  const [cx, cy, cz] = [0, 1, 2].map((a) => Math.cos(((deg[a] ?? 0) * Math.PI) / 180));
  const [sx, sy, sz] = [0, 1, 2].map((a) => Math.sin(((deg[a] ?? 0) * Math.PI) / 180));
  const R = [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
  const m = new Float64Array(16);
  m[15] = 1;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[4 * c + r] = R[r][c];
  return m;
}

const translate = (x: number, y: number, z: number): Mat4 => {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
};
const scaling = (s: number): Mat4 => {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = s;
  m[15] = 1;
  return m;
};

/** A body from a mesh file, placed in the box: the test of a point and what it is. */
export interface PlacedMesh {
  /** whether a point of the box is inside */
  inside(x: number, y: number, z: number): boolean;
  /** the body's signed distance at a point of the box, in the box's lengths */
  distance(x: number, y: number, z: number): number;
  /** from the file's frame to the box's */
  toBox: Mat4;
  /** the factor on the file's lengths */
  scale: number;
  center: Vec3;
  /** bounding box in the box's frame */
  min: Vec3;
  max: Vec3;
  /** in the box's lengths: volume, area, the shadow along x, y, z; thinnest part in finest cells */
  report: MeshReport & { frontalBox: Vec3 };
}

/**
 * Place a mesh shape: up axis, scale (or fit: the longest side a fraction of the box), turn,
 * and its bounding box's middle at `center`. Its distance field is made at the finest cell
 * size (`dx`), in the file's frame, and kept for later runs at the same spacing.
 */
export function placeMesh(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3; dx: number }): PlacedMesh {
  if (isOutlineFile(shape.file)) return placeOutline(shape, ctx);
  const { mesh, toBox, scale: s, center } = placement(shape, ctx);
  const toFile = invert(toBox);
  const h = ctx.dx / s;
  const key = `${shape.file}|${shape.node ?? ''}|${h.toPrecision(6)}`;
  let f = fields.get(key);
  if (!f) {
    f = buildSdf(mesh, { h });
    fields.set(key, f);
  }
  return finish(shape, ctx, mesh, toBox, toFile, s, center, f);
}

/** Where a mesh shape's file goes in the box: its triangles, the matrix, the scale and the center. */
function placement(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3 }): { mesh: Mesh; toBox: Mat4; scale: number; center: Vec3 } {
  const mesh = meshOf(shape.file, shape.node);
  const U = upMatrix(shape.up ?? 'z', ctx.ndim);
  const ub = bounds({ ...mesh, positions: transformed(mesh.positions, U) });
  // What `center` places: the middle of the bounding box, or the file's origin (turned up).
  const c0 = shape.anchor === 'origin' ? [0, 0, 0] : [0, 1, 2].map((a) => (ub.min[a] + ub.max[a]) / 2);
  const longest = Math.max(...[0, 1, 2].map((a) => ub.max[a] - ub.min[a])) || 1;
  const s = shape.scale !== undefined && shape.scale > 0 ? shape.scale : ((shape.fit !== undefined && shape.fit > 0 ? shape.fit : 0.25) * ctx.box) / longest;
  const center = [0, 1, 2].map((a) => Number(shape.center?.[a] ?? 0)) as Vec3;
  // box <- translate(center) R scale(s) translate(-c0) U <- file
  const toBox = mul(translate(...center), mul(eulerMatrix(shape.rotate ?? [0, 0, 0]), mul(scaling(s), mul(translate(-c0[0], -c0[1], -c0[2]), U))));
  return { mesh, toBox, scale: s, center };
}

/**
 * A placed mesh's vertices in the box, projected on two axes, at most about `max` of them: what
 * the setup preview draws its silhouette from (no distance field is made). Null for an outline
 * or a file not loaded.
 */
export function meshSilhouette(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3 }, axes: [number, number], max = 4000): number[][] | null {
  if (!hasMeshFile(shape.file) || isOutlineFile(shape.file)) return null;
  try {
    const { mesh, toBox } = placement(shape, ctx);
    const nv = mesh.positions.length / 3;
    const every = Math.max(1, Math.floor(nv / max));
    const out: number[][] = [];
    for (let v = 0; v < nv; v += every) {
      const p = apply(toBox, mesh.positions[3 * v], mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]);
      out.push([p[axes[0]], p[axes[1]]]);
    }
    return out;
  } catch {
    return null;
  }
}

function finish(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3; dx: number }, mesh: Mesh, toBox: Mat4, toFile: Mat4, s: number, center: Vec3, f: { field: DistanceField; report: MeshReport }): PlacedMesh {
  const h = ctx.dx / s;
  const { field, report } = f;
  const distance = (x: number, y: number, z: number) => {
    const p = apply(toFile, x, y, z);
    return sampleSdf(field, p[0], p[1], p[2]) * s;
  };
  const world = transformed(mesh.positions, toBox);
  const wb = bounds({ ...mesh, positions: world });
  const m = measure(mesh);
  return {
    inside: (x, y, z) => distance(x, y, z) < 0,
    distance,
    toBox,
    scale: s,
    center,
    min: wb.min,
    max: wb.max,
    report: {
      ...report,
      volume: m.volume * s ** 3, insideVolume: report.insideVolume * s ** 3, area: m.area * s * s,
      min: wb.min, max: wb.max,
      frontal: report.frontal.map((a) => a * s * s) as Vec3,
      frontalBox: [0, 1, 2].map((a) => shadow(world, mesh.indices, a, ctx.dx / 2)) as Vec3,
      thinnest: (report.thinnest * h * s) / ctx.dx,
    },
  };
}

/**
 * Points on a placed mesh's surface (3D), in the box's frame, no two neighbors on a triangle
 * further apart than `spacing`: each triangle's vertices, edges and inside on a grid of that
 * pitch. What an adaptive mesh's refinement near a body is decided from (setup/parts/surfaces.ts).
 */
export function meshSurfacePoints(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3 }, spacing: number): Float64Array {
  const { mesh, toBox } = placement(shape, ctx);
  const W = transformed(mesh.positions, toBox);
  const idx = mesh.indices;
  const out: number[] = [];
  const v = (i: number, a: number) => W[3 * idx[i] + a];
  for (let t = 0; t < idx.length; t += 3) {
    const e = Math.max(...[[0, 1], [1, 2], [2, 0]].map(([p, q]) => Math.hypot(v(t + p, 0) - v(t + q, 0), v(t + p, 1) - v(t + q, 1), v(t + p, 2) - v(t + q, 2))));
    const m = Math.max(1, Math.ceil(e / spacing));
    for (let i = 0; i <= m; i++)
      for (let j = 0; j <= m - i; j++) {
        const a = i / m, b = j / m, c = 1 - a - b;
        for (let k = 0; k < 3; k++) out.push(a * v(t, k) + b * v(t + 1, k) + c * v(t + 2, k));
      }
  }
  return Float64Array.from(out);
}

function transformed(P: Float32Array, m: Mat4): Float32Array {
  const out = new Float32Array(P.length);
  for (let i = 0; i < P.length; i += 3) out.set(apply(m, P[i], P[i + 1], P[i + 2]), i);
  return out;
}

/**
 * The area of the shadow of the triangles along axis `a`: their projections rasterized at
 * spacing `res` (pixel centers), the pixels any covers counted.
 */
export function shadow(P: Float32Array, idx: Uint32Array, a: number, res: number): number {
  const u = (a + 1) % 3, v = (a + 2) % 3;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const i of idx) {
    lo = [Math.min(lo[0], P[3 * i + u]), Math.min(lo[1], P[3 * i + v])];
    hi = [Math.max(hi[0], P[3 * i + u]), Math.max(hi[1], P[3 * i + v])];
  }
  const nu = Math.max(1, Math.ceil((hi[0] - lo[0]) / res)), nv = Math.max(1, Math.ceil((hi[1] - lo[1]) / res));
  if (nu * nv > 4e7) return shadow(P, idx, a, res * 2);
  const hit = new Uint8Array(nu * nv);
  for (let t = 0; t < idx.length; t += 3) {
    const xs = [0, 1, 2].map((k) => (P[3 * idx[t + k] + u] - lo[0]) / res - 0.5);
    const ys = [0, 1, 2].map((k) => (P[3 * idx[t + k] + v] - lo[1]) / res - 0.5);
    const area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0]);
    if (area === 0) continue;
    for (let j = Math.max(0, Math.ceil(Math.min(...ys))); j <= Math.min(nv - 1, Math.floor(Math.max(...ys))); j++)
      for (let i = Math.max(0, Math.ceil(Math.min(...xs))); i <= Math.min(nu - 1, Math.floor(Math.max(...xs))); i++) {
        let inside = true;
        for (let k = 0; k < 3 && inside; k++) {
          const k2 = (k + 1) % 3;
          const e = (xs[k2] - xs[k]) * (j - ys[k]) - (ys[k2] - ys[k]) * (i - xs[k]);
          if (e * area < 0) inside = false;
        }
        if (inside) hit[i + nu * j] = 1;
      }
  }
  let n = 0;
  for (const x of hit) n += x;
  return n * res * res;
}

/**
 * A 2D outline (an airfoil, an SVG path) placed in the plane: fit, turn about z (rotate[2]),
 * center. In 3D it is extruded along z over the box's depth unless `scale` sets nothing else;
 * its report's volume is its area.
 */
function placeOutline(shape: MeshShape, ctx: { box: number; ndim: 1 | 2 | 3; dx: number }): PlacedMesh {
  const text = new TextDecoder().decode(bytesOf(shape.file));
  const o: Outline = extensionOf(shape.file) === 'svg' ? parseSvgPath(text) : parseAirfoil(text);
  const b = outlineBounds(o);
  const c0 = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  const longest = Math.max(b[2] - b[0], b[3] - b[1]) || 1;
  const s = shape.scale !== undefined && shape.scale > 0 ? shape.scale : ((shape.fit !== undefined && shape.fit > 0 ? shape.fit : 0.25) * ctx.box) / longest;
  const center = [0, 1, 2].map((a) => Number(shape.center?.[a] ?? 0)) as Vec3;
  const th = ((shape.rotate?.[2] ?? 0) * Math.PI) / 180;
  const [ct, st] = [Math.cos(th), Math.sin(th)];
  const local = (x: number, y: number) => {
    const dx = x - center[0], dy = y - center[1];
    return [(ct * dx + st * dy) / s + c0[0], (-st * dx + ct * dy) / s + c0[1]];
  };
  const inside = (x: number, y: number) => {
    const p = local(x, y);
    return windingNumber(o, p[0], p[1]) !== 0;
  };
  const corners = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]].map(([x, y]) => {
    const dx = (x - c0[0]) * s, dy = (y - c0[1]) * s;
    return [center[0] + ct * dx - st * dy, center[1] + st * dx + ct * dy];
  });
  const min: Vec3 = [Math.min(...corners.map((c) => c[0])), Math.min(...corners.map((c) => c[1])), -ctx.box / 2];
  const max: Vec3 = [Math.max(...corners.map((c) => c[0])), Math.max(...corners.map((c) => c[1])), ctx.box / 2];
  const toBox = mul(translate(...center), mul(eulerMatrix([0, 0, shape.rotate?.[2] ?? 0]), mul(scaling(s), translate(-c0[0], -c0[1], 0))));
  // Shadows: counted on the plane, along x and y, at half a cell.
  const res = ctx.dx / 2;
  const span = (a: 0 | 1) => {
    const lo = min[1 - a], hi = max[1 - a];
    let n = 0;
    for (let q = lo + res / 2; q < hi; q += res) {
      let hit = false;
      for (let p = min[a] + res / 2; p < max[a] && !hit; p += res) hit = a === 0 ? inside(p, q) : inside(q, p);
      if (hit) n++;
    }
    return n * res;
  };
  let area = 0;
  for (let y = min[1] + res / 2; y < max[1]; y += res) for (let x = min[0] + res / 2; x < max[0]; x += res) if (inside(x, y)) area += res * res;
  const report = {
    triangles: 0, volume: area, insideVolume: area, area: 0, min, max, frontal: [span(0), span(1), 0] as Vec3, frontalBox: [span(0), span(1), 0] as Vec3,
    thinnest: Infinity, thinnestAt: [0, 0, 0] as Vec3, stray: 0, flipped: 0, points: 0, seconds: 0,
  };
  return { inside: (x, y) => inside(x, y), distance: (x, y) => (inside(x, y) ? -ctx.dx : ctx.dx), toBox, scale: s, center, min, max, report };
}

// ---- the page: files kept in the origin's private file system (OPFS), by their SHA-256, so
// that a reload, or a link to a setup that names them, finds them again.

const OPFS_DIR = 'shaderous-meshes';
const INDEX = 'index.json';

async function meshDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR, { create: true });
  } catch {
    return null;
  }
}

async function readIndex(dir: FileSystemDirectoryHandle): Promise<Record<string, string>> {
  try {
    return JSON.parse(await (await (await dir.getFileHandle(INDEX)).getFile()).text());
  } catch {
    return {};
  }
}

async function writeFile(dir: FileSystemDirectoryHandle, name: string, data: BlobPart) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

/** The SHA-256 of some bytes, as hex. */
export async function sha256(bytes: Uint8Array): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Register a file and keep it in this browser (OPFS) under its hash. Returns the hash. */
export async function keepMeshFile(name: string, bytes: Uint8Array): Promise<string> {
  putMeshFile(name, bytes);
  const hash = await sha256(bytes);
  const dir = await meshDir();
  if (dir) {
    try {
      await writeFile(dir, hash, bytes as Uint8Array<ArrayBuffer>);
      const index = await readIndex(dir);
      index[name] = hash;
      await writeFile(dir, INDEX, JSON.stringify(index));
    } catch {
      // (No room, or no private file system: the file lasts as long as the page.)
    }
  }
  return hash;
}

/**
 * Find a setup's mesh files the page does not hold: a URL is fetched, a name looked up among
 * the files kept in this browser. Returns the names still missing.
 */
export async function ensureMeshFiles(spec: { parts?: readonly unknown[] }): Promise<string[]> {
  let missing = missingMeshFiles(spec);
  if (!missing.length) return [];
  for (const f of missing.filter((m) => /^(https?:|blob:|data:)/.test(m))) {
    try {
      const r = await fetch(f);
      if (r.ok) putMeshFile(f, new Uint8Array(await r.arrayBuffer()));
    } catch {
      // (Left missing: the caller says which.)
    }
  }
  missing = missingMeshFiles(spec);
  const dir = missing.length ? await meshDir() : null;
  if (dir) {
    const index = await readIndex(dir);
    for (const f of missing) {
      const hash = index[f] ?? (/^[0-9a-f]{64}$/.test(f) ? f : undefined);
      if (!hash) continue;
      try {
        putMeshFile(f, new Uint8Array(await (await (await dir.getFileHandle(hash)).getFile()).arrayBuffer()));
      } catch {
        // (Not kept here.)
      }
    }
  }
  return missingMeshFiles(spec);
}
