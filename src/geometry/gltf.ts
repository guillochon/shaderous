// glTF 2.0, as .gltf (JSON, its buffers embedded as data URIs or in files beside it) or .glb
// (plans/wind-tunnel.md 3.1). Every node is a part, named after the node (or "node <i>"), with
// its own transform (its matrix, or translation, rotation and scale) and its parent: the node
// tree is the chain of a body's moving parts. A node's triangles are those of its mesh's
// primitives in mode 4 (triangles; other modes are skipped), in the node's own frame. glTF's
// lengths are meters. Compressed geometry (Draco, meshopt) is refused, with the reason.
import { identity, trs, type Mat4, type Mesh, type MeshPart } from './mesh';

interface Accessor { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }
interface BufferView { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }
interface Primitive { attributes: Record<string, number>; indices?: number; mode?: number }
interface Node { name?: string; mesh?: number; children?: number[]; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }
interface Gltf {
  asset?: { version?: string };
  buffers?: { uri?: string; byteLength: number }[];
  bufferViews?: BufferView[];
  accessors?: Accessor[];
  meshes?: { name?: string; primitives: Primitive[] }[];
  nodes?: Node[];
  scenes?: { nodes?: number[] }[];
  scene?: number;
  extensionsRequired?: string[];
  extensionsUsed?: string[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const COMPRESSED = ['KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_meshopt_compression'];

/**
 * @param resolve the bytes of a buffer given by a URI that is not a data URI (a .bin beside a
 *   .gltf), or null if there is no such file
 */
export function parseGltf(bytes: Uint8Array, resolve?: (uri: string) => Uint8Array | null): Mesh {
  let json: Gltf;
  let bin: Uint8Array | null = null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength >= 12 && dv.getUint32(0, true) === 0x46546c67) {
    // GLB: a header, then chunks of (length, type, data): JSON first, then an optional BIN.
    let o = 12;
    let text = '';
    while (o + 8 <= bytes.byteLength) {
      const len = dv.getUint32(o, true);
      const type = dv.getUint32(o + 4, true);
      const data = bytes.subarray(o + 8, o + 8 + len);
      if (type === 0x4e4f534a) text = new TextDecoder().decode(data);
      else if (type === 0x004e4942) bin = data;
      o += 8 + len;
    }
    json = JSON.parse(text);
  } else json = JSON.parse(new TextDecoder().decode(bytes));
  const compressed = [...(json.extensionsRequired ?? []), ...(json.extensionsUsed ?? [])].find((e) => COMPRESSED.includes(e));
  if (compressed) throw new Error(`glTF: the file's geometry is compressed (${compressed}), which is not read yet: export it again without compression`);

  const buffers = (json.buffers ?? []).map((b, i) => {
    if (b.uri === undefined) {
      if (!bin) throw new Error(`glTF: buffer ${i} has no data`);
      return bin;
    }
    const m = b.uri.match(/^data:[^,]*?(;base64)?,(.*)$/s);
    if (m) {
      if (!m[1]) return new TextEncoder().encode(decodeURIComponent(m[2]));
      const s = atob(m[2]);
      const out = new Uint8Array(s.length);
      for (let k = 0; k < s.length; k++) out[k] = s.charCodeAt(k);
      return out;
    }
    const r = resolve?.(b.uri);
    if (!r) throw new Error(`glTF: its buffer "${b.uri}" is a separate file; give it too (or use .glb, which holds everything)`);
    return r;
  });

  const read = (a: number): number[] => {
    const acc = json.accessors![a];
    const n = COMPONENTS[acc.type] ?? 1;
    const out = new Array<number>(acc.count * n).fill(0);
    if (acc.bufferView === undefined) return out;
    const view = json.bufferViews![acc.bufferView];
    const buf = buffers[view.buffer];
    const size = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[acc.componentType];
    if (!size) throw new Error(`glTF: accessor ${a} has a component type (${acc.componentType}) that is not read`);
    const stride = view.byteStride || size * n;
    const d = new DataView(buf.buffer, buf.byteOffset + (view.byteOffset ?? 0) + (acc.byteOffset ?? 0));
    for (let i = 0; i < acc.count; i++)
      for (let c = 0; c < n; c++) {
        const o = i * stride + c * size;
        let x: number;
        switch (acc.componentType) {
          case 5120: x = d.getInt8(o); break;
          case 5121: x = d.getUint8(o); break;
          case 5122: x = d.getInt16(o, true); break;
          case 5123: x = d.getUint16(o, true); break;
          case 5125: x = d.getUint32(o, true); break;
          default: x = d.getFloat32(o, true);
        }
        out[i * n + c] = x;
      }
    return out;
  };

  const nodes = json.nodes ?? [];
  const parent = new Array<number>(nodes.length).fill(-1);
  nodes.forEach((nd, i) => nd.children?.forEach((c) => (parent[c] = i)));
  // Only the scene's nodes and those below them (all nodes if the file names no scene).
  const scene = json.scenes?.[json.scene ?? 0];
  const keep = new Set<number>();
  const visit = (i: number) => {
    if (keep.has(i)) return;
    keep.add(i);
    nodes[i].children?.forEach(visit);
  };
  if (scene?.nodes) scene.nodes.forEach(visit);
  else nodes.forEach((_, i) => visit(i));

  const pos: number[] = [];
  const idx: number[] = [];
  const parts: MeshPart[] = [];
  const partOf = new Map<number, number>();
  const order = [...keep].sort((a, b) => a - b);
  for (const i of order) partOf.set(i, partOf.size);
  for (const i of order) {
    const nd = nodes[i];
    const first = idx.length / 3;
    for (const prim of nd.mesh !== undefined ? json.meshes![nd.mesh].primitives : []) {
      if ((prim.mode ?? 4) !== 4 || prim.attributes.POSITION === undefined) continue;
      const base = pos.length / 3;
      const p = read(prim.attributes.POSITION);
      for (const x of p) pos.push(x);
      const nv = p.length / 3;
      const ind = prim.indices !== undefined ? read(prim.indices) : Array.from({ length: nv }, (_, k) => k);
      for (const k of ind) idx.push(base + k);
    }
    const transform: Mat4 = nd.matrix ? Float64Array.from(nd.matrix) : nd.translation || nd.rotation || nd.scale ? trs(nd.translation, nd.rotation, nd.scale) : identity();
    parts.push({ name: nd.name ?? `node ${i}`, first, count: idx.length / 3 - first, transform, parent: parent[i] >= 0 && partOf.has(parent[i]) ? partOf.get(parent[i])! : -1 });
  }
  if (!idx.length) throw new Error('glTF: the file has no triangles');
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts, meters: 1 };
}
