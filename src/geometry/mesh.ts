// Triangle meshes read from files (plans/wind-tunnel.md 3.1): what every parser returns, and
// what is done with one before its distance field is made (sdf.ts): parts and their transforms,
// bounds, volume and area.

/** A 4x4 matrix, column-major as glTF writes it: m[col * 4 + row]. */
export type Mat4 = Float64Array;

/** One named piece of a file: a glTF node, an OBJ group. Its triangles are in its own frame. */
export interface MeshPart {
  name: string;
  /** first triangle and how many (indices first * 3 to (first + count) * 3) */
  first: number;
  count: number;
  /** from the part's frame to its parent's (the file's, for a part with no parent) */
  transform: Mat4;
  /** index of the parent part, or -1 */
  parent: number;
}

export interface Mesh {
  /** x, y, z of each vertex */
  positions: Float32Array;
  /** three vertex indices per triangle */
  indices: Uint32Array;
  /** at least one; a file without names has one part, "mesh" */
  parts: MeshPart[];
  /** the file's length unit in meters, where it says (glTF: 1) */
  meters?: number;
}

export const identity = (): Mat4 => {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
};

/** a * b (apply b first). */
export function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

/** A point through a matrix. */
export function apply(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
}

/** A rigid or affine matrix's inverse (general 4x4 with the last row 0 0 0 1). */
export function invert(m: Mat4): Mat4 {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!det) throw new Error('a transform that flattens space has no inverse');
  const r = [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
  const o = identity();
  for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) o[col * 4 + row] = r[row][col];
  const t = [m[12], m[13], m[14]];
  for (let row = 0; row < 3; row++) o[12 + row] = -(r[row][0] * t[0] + r[row][1] * t[1] + r[row][2] * t[2]);
  return o;
}

/** glTF's translation, rotation (quaternion x, y, z, w) and scale as one matrix, T R S. */
export function trs(t: number[] = [0, 0, 0], q: number[] = [0, 0, 0, 1], s: number[] = [1, 1, 1]): Mat4 {
  const [x, y, z, w] = q;
  const m = identity();
  m[0] = (1 - 2 * (y * y + z * z)) * s[0];
  m[1] = 2 * (x * y + z * w) * s[0];
  m[2] = 2 * (x * z - y * w) * s[0];
  m[4] = 2 * (x * y - z * w) * s[1];
  m[5] = (1 - 2 * (x * x + z * z)) * s[1];
  m[6] = 2 * (y * z + x * w) * s[1];
  m[8] = 2 * (x * z + y * w) * s[2];
  m[9] = 2 * (y * z - x * w) * s[2];
  m[10] = (1 - 2 * (x * x + y * y)) * s[2];
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

/** A turn by `deg` degrees about a unit axis (Rodrigues), about the origin. */
export function axisAngle(axis: readonly number[], deg: number): Mat4 {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const h = ((deg * Math.PI) / 180) / 2;
  const s = Math.sin(h) / n;
  return trs([0, 0, 0], [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(h)]);
}

/** From part `i`'s frame to the file's: its transform and every parent's. */
export function worldMatrix(parts: readonly MeshPart[], i: number): Mat4 {
  let m = parts[i].transform;
  for (let p = parts[i].parent, guard = 0; p >= 0 && guard < parts.length; p = parts[p].parent, guard++) m = mul(parts[p].transform, m);
  return m;
}

/** Whether part `i` is `root` or below it. */
export function within(parts: readonly MeshPart[], i: number, root: number): boolean {
  for (let p = i, guard = 0; p >= 0 && guard <= parts.length; p = parts[p].parent, guard++) if (p === root) return true;
  return false;
}

/**
 * The triangles of the part named `node` and the parts below it (or all of them), in the file's
 * frame, as one part: what a static body is made of. Throws if the file has no part of that name.
 */
export function flatten(mesh: Mesh, node?: string): Mesh {
  let root = -1;
  if (node !== undefined && node !== '') {
    root = mesh.parts.findIndex((p) => p.name === node);
    if (root < 0) throw new Error(`the file has no part named "${node}" (it has: ${mesh.parts.map((p) => p.name).join(', ')})`);
  }
  const pos: number[] = [];
  const idx: number[] = [];
  mesh.parts.forEach((p, i) => {
    if (root >= 0 && !within(mesh.parts, i, root)) return;
    const m = worldMatrix(mesh.parts, i);
    const map = new Map<number, number>();
    for (let t = p.first * 3; t < (p.first + p.count) * 3; t++) {
      const v = mesh.indices[t];
      let k = map.get(v);
      if (k === undefined) {
        k = pos.length / 3;
        map.set(v, k);
        pos.push(...apply(m, mesh.positions[3 * v], mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]));
      }
      idx.push(k);
    }
  });
  const count = idx.length / 3;
  if (!count) throw new Error(node ? `the part "${node}" has no triangles` : 'the file has no triangles');
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: [{ name: node || 'mesh', first: 0, count, transform: identity(), parent: -1 }], meters: mesh.meters };
}

/** Bounding box of the vertices the triangles use. */
export function bounds(mesh: Mesh): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const v of mesh.indices)
    for (let a = 0; a < 3; a++) {
      const x = mesh.positions[3 * v + a];
      if (x < min[a]) min[a] = x;
      if (x > max[a]) max[a] = x;
    }
  return { min, max };
}

/** Enclosed volume by the divergence theorem (exact for a closed, consistently wound mesh) and surface area. */
export function measure(mesh: Mesh): { volume: number; area: number } {
  let vol = 0;
  let area = 0;
  const P = mesh.positions;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [i, j, k] = [3 * mesh.indices[t], 3 * mesh.indices[t + 1], 3 * mesh.indices[t + 2]];
    const ax = P[i], ay = P[i + 1], az = P[i + 2];
    const bx = P[j] - ax, by = P[j + 1] - ay, bz = P[j + 2] - az;
    const cx = P[k] - ax, cy = P[k + 1] - ay, cz = P[k + 2] - az;
    const nx = by * cz - bz * cy, ny = bz * cx - bx * cz, nz = bx * cy - by * cx;
    area += 0.5 * Math.hypot(nx, ny, nz);
    vol += (ax * nx + ay * ny + az * nz) / 6;
  }
  return { volume: Math.abs(vol), area };
}

/**
 * The same triangles with a consistent winding, outward. Vertices are first welded (positions
 * equal to a millionth of the mesh's size are one vertex: STL repeats each vertex in every
 * triangle). Then, over each connected piece, a triangle's neighbor across an edge is flipped
 * where the two run along the edge the same way, and a piece whose enclosed volume comes out
 * negative is flipped whole. The generalized winding number survives holes and pieces that
 * pass through each other, but not this: a triangle wound the wrong way subtracts its solid
 * angle, and a sphere with half of its triangles flipped has a winding number near 0 inside.
 * Returns the mesh, the number of triangles flipped, the pieces (joined across edges of two
 * triangles), and the edges that do not have two triangles (0 for one closed, manifold mesh).
 */
export function orient(mesh: Mesh): { mesh: Mesh; flipped: number; pieces: number; open: number } {
  const P = mesh.positions;
  const n = mesh.indices.length / 3;
  const { min, max } = bounds(mesh);
  const q = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1e-30) * 1e-6;
  const ids = new Map<string, number>();
  const weld = new Uint32Array(P.length / 3);
  for (let v = 0; v < P.length / 3; v++) {
    const k = `${Math.round(P[3 * v] / q)},${Math.round(P[3 * v + 1] / q)},${Math.round(P[3 * v + 2] / q)}`;
    let id = ids.get(k);
    if (id === undefined) ids.set(k, (id = ids.size));
    weld[v] = id;
  }
  const tri = new Uint32Array(mesh.indices);
  const W = (t: number, k: number) => weld[tri[3 * t + k]];
  // Each undirected edge's triangles.
  const edges = new Map<number, number[]>();
  const key = (a: number, b: number) => (a < b ? a * 4294967296 + b : b * 4294967296 + a);
  for (let t = 0; t < n; t++)
    for (let k = 0; k < 3; k++) {
      const e = key(W(t, k), W(t, (k + 1) % 3));
      const l = edges.get(e);
      if (l) l.push(t);
      else edges.set(e, [t]);
    }
  /** Whether triangle t runs from a to b along one of its edges. */
  const runs = (t: number, a: number, b: number) => [0, 1, 2].some((k) => W(t, k) === a && W(t, (k + 1) % 3) === b);
  const flip = (t: number) => {
    const x = tri[3 * t + 1];
    tri[3 * t + 1] = tri[3 * t + 2];
    tri[3 * t + 2] = x;
  };
  const seen = new Uint8Array(n);
  let flipped = 0;
  let pieces = 0;
  let open = 0;
  // (An edge of three or more triangles, where pieces touch, joins nothing: the winding is
  // carried across manifold edges only, and such an edge counts as open.)
  for (const l of edges.values()) if (l.length !== 2) open++;
  for (let s = 0; s < n; s++) {
    if (seen[s]) continue;
    pieces++;
    const piece = [s];
    seen[s] = 1;
    for (let i = 0; i < piece.length; i++) {
      const t = piece[i];
      for (let k = 0; k < 3; k++) {
        const a = W(t, k), b = W(t, (k + 1) % 3);
        const across = edges.get(key(a, b))!;
        if (across.length !== 2) continue;
        for (const u of across) {
          if (seen[u]) continue;
          seen[u] = 1;
          if (runs(u, a, b)) flip(u);
          piece.push(u);
        }
      }
    }
    // Which way the piece faces: a closed piece outward (its enclosed volume positive); an open
    // one (a fragment between non-manifold edges, a surface with holes) the way most of its
    // triangles were given, since its volume says nothing.
    let vol = 0;
    let turned = 0;
    let rim = 0;
    for (const t of piece) {
      const [i, j, k] = [3 * tri[3 * t], 3 * tri[3 * t + 1], 3 * tri[3 * t + 2]];
      vol += P[i] * (P[j + 1] * P[k + 2] - P[j + 2] * P[k + 1]) - P[i + 1] * (P[j] * P[k + 2] - P[j + 2] * P[k]) + P[i + 2] * (P[j] * P[k + 1] - P[j + 1] * P[k]);
      if (tri[3 * t + 1] !== mesh.indices[3 * t + 1]) turned++;
      for (let k2 = 0; k2 < 3; k2++) if (edges.get(key(W(t, k2), W(t, (k2 + 1) % 3)))!.length !== 2) rim++;
    }
    if (rim === 0 ? vol < 0 : 2 * turned > piece.length) for (const t of piece) flip(t);
  }
  for (let t = 0; t < n; t++) if (tri[3 * t + 1] !== mesh.indices[3 * t + 1]) flipped++;
  return { mesh: { ...mesh, indices: tri }, flipped, pieces, open };
}

/** Fan a polygon (vertex indices) into triangles. */
export function fan(poly: readonly number[], out: number[]) {
  for (let i = 1; i + 1 < poly.length; i++) out.push(poly[0], poly[i], poly[i + 1]);
}

/** One part covering every triangle. */
export const wholePart = (count: number, name = 'mesh'): MeshPart => ({ name, first: 0, count, transform: identity(), parent: -1 });
