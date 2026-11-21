// Wavefront OBJ (plans/wind-tunnel.md 3.1): `v` positions and `f` faces (v, v/vt, v//vn, v/vt/vn;
// negative indices count back from the last vertex), polygons fanned into triangles. Each `o`
// or `g` begins a part of that name (faces before any name are the part "mesh"). Parts have no
// transforms of their own: OBJ has none.
import { identity, fan, type Mesh, type MeshPart } from './mesh';

export function parseObj(bytes: Uint8Array | string): Mesh {
  const text = typeof bytes === 'string' ? bytes : new TextDecoder().decode(bytes);
  const pos: number[] = [];
  const idx: number[] = [];
  const parts: MeshPart[] = [];
  let name = 'mesh';
  let start = 0;
  const close = () => {
    const count = idx.length / 3 - start;
    if (count > 0) parts.push({ name, first: start, count, transform: identity(), parent: -1 });
    start = idx.length / 3;
  };
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line[0] === '#') continue;
    const w = line.split(/\s+/);
    if (w[0] === 'v') pos.push(+w[1], +w[2], +w[3]);
    else if (w[0] === 'f') {
      const nv = pos.length / 3;
      const poly = w.slice(1).map((s) => {
        const i = parseInt(s.split('/')[0], 10);
        const k = i < 0 ? nv + i : i - 1;
        if (!(k >= 0 && k < nv)) throw new Error(`OBJ: a face refers to vertex ${i}, of ${nv}`);
        return k;
      });
      fan(poly, idx);
    } else if (w[0] === 'o' || w[0] === 'g') {
      close();
      name = w.slice(1).join(' ') || name;
    }
  }
  close();
  if (!idx.length) throw new Error('not an OBJ file: no faces were found');
  // Several groups of one name are one part (an OBJ may come back to a group): merged in order.
  const merged = mergeNames(parts, idx);
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: merged };
}

/** Parts of one name that are not one run of triangles: the triangles reordered so that each name is. */
function mergeNames(parts: MeshPart[], idx: number[]): MeshPart[] {
  const names = [...new Set(parts.map((p) => p.name))];
  if (names.length === parts.length) return parts;
  const out: MeshPart[] = [];
  const reordered: number[] = [];
  for (const n of names) {
    const first = reordered.length / 3;
    for (const p of parts.filter((q) => q.name === n)) for (let t = 3 * p.first; t < 3 * (p.first + p.count); t++) reordered.push(idx[t]);
    out.push({ name: n, first, count: reordered.length / 3 - first, transform: identity(), parent: -1 });
  }
  for (let i = 0; i < reordered.length; i++) idx[i] = reordered[i];
  return out;
}
