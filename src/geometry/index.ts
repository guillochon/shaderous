// Mesh files to meshes (plans/wind-tunnel.md section 3): the reader for a file by its name.
import { parseGltf } from './gltf';
import { parseObj } from './obj';
import { parsePly } from './ply';
import { parseStl } from './stl';
import type { Mesh } from './mesh';

export * from './mesh';
export { parseGltf } from './gltf';
export { parseObj } from './obj';
export { parsePly } from './ply';
export { parseStl, isBinaryStl } from './stl';
export { parseAirfoil, parseSvgPath, windingNumber, outlineBounds, outlineArea, type Outline } from './outline';
export { buildSdf, sampleSdf, type DistanceField, type MeshReport, type SdfOptions } from './sdf';

/** File types read as triangles. */
export const MESH_EXTENSIONS = ['stl', 'obj', 'ply', 'gltf', 'glb'] as const;
/** File types read as 2D outlines. */
export const OUTLINE_EXTENSIONS = ['dat', 'svg'] as const;

export const extensionOf = (name: string): string => (name.match(/\.([a-z0-9]+)(?:[?#].*)?$/i)?.[1] ?? '').toLowerCase();

/**
 * Read a mesh file. The type is the name's extension, or (without one) what the bytes look
 * like. `resolve` gives a .gltf's separate buffers.
 */
export function parseMeshFile(name: string, bytes: Uint8Array, resolve?: (uri: string) => Uint8Array | null): Mesh {
  let ext = extensionOf(name);
  if (!(MESH_EXTENSIONS as readonly string[]).includes(ext)) ext = sniff(bytes);
  switch (ext) {
    case 'stl': return parseStl(bytes);
    case 'obj': return parseObj(bytes);
    case 'ply': return parsePly(bytes);
    case 'gltf': case 'glb': return parseGltf(bytes, resolve);
    default: throw new Error(`"${name}" is not a mesh file this reads (${MESH_EXTENSIONS.join(', ')}). STEP and IGES hold exact surfaces, not triangles: export STL or glTF from the CAD program`);
  }
}

function sniff(bytes: Uint8Array): string {
  const head = new TextDecoder().decode(bytes.subarray(0, 256));
  if (bytes.byteLength >= 4 && head.startsWith('glTF')) return 'glb';
  if (head.startsWith('ply')) return 'ply';
  if (/^\s*\{/.test(head) && head.includes('"asset"')) return 'gltf';
  if (/^\s*(#|v |o |g |mtllib)/m.test(head) && !head.startsWith('solid')) return 'obj';
  return 'stl';
}
