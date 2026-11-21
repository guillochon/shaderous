// STL, binary and ASCII (plans/wind-tunnel.md 3.1). Binary is told apart by the size its header
// implies (80 bytes, a count, 50 bytes a triangle), not by the word `solid`, which many binary
// files also begin with. STL has no units and no parts: one part, "mesh". Vertices are not
// shared between triangles; nothing here needs them to be.
import { wholePart, type Mesh } from './mesh';

export function isBinaryStl(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 84) return false;
  const n = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(80, true);
  return 84 + 50 * n === bytes.byteLength;
}

export function parseStl(bytes: Uint8Array): Mesh {
  return isBinaryStl(bytes) ? binary(bytes) : ascii(new TextDecoder().decode(bytes));
}

function binary(bytes: Uint8Array): Mesh {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = v.getUint32(80, true);
  const positions = new Float32Array(9 * n);
  const indices = new Uint32Array(3 * n);
  for (let t = 0; t < n; t++) {
    const o = 84 + 50 * t + 12; // past the facet normal, which is not used
    for (let k = 0; k < 9; k++) positions[9 * t + k] = v.getFloat32(o + 4 * k, true);
    indices.set([3 * t, 3 * t + 1, 3 * t + 2], 3 * t);
  }
  return { positions, indices, parts: [wholePart(n)] };
}

function ascii(text: string): Mesh {
  const pos: number[] = [];
  const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
  for (let m = re.exec(text); m; m = re.exec(text)) pos.push(+m[1], +m[2], +m[3]);
  if (!pos.length || pos.length % 9 !== 0 || pos.some((x) => !Number.isFinite(x))) throw new Error('not an STL file: no triangles of three vertices were found');
  const n = pos.length / 9;
  const indices = new Uint32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) indices[i] = i;
  return { positions: new Float32Array(pos), indices, parts: [wholePart(n)] };
}
