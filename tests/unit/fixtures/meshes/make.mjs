// Writes the mesh fixtures of tests/unit/geometry.test.ts: a unit cube (centered on the origin,
// wound counterclockwise seen from outside) in each format, and a glTF of three nodes in a
// chain. Run with node from the repository's root; the files are committed.
import { writeFileSync } from 'node:fs';
import path from 'node:path';

const dir = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const V = [
  [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
  [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
];
// Six quads, counterclockwise from outside.
const Q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
const T = Q.flatMap((q) => [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);

// STL, ASCII and binary.
let ascii = 'solid cube\n';
for (const t of T) {
  ascii += ' facet normal 0 0 0\n  outer loop\n';
  for (const i of t) ascii += `   vertex ${V[i].join(' ')}\n`;
  ascii += '  endloop\n endfacet\n';
}
writeFileSync(path.join(dir, 'cube_ascii.stl'), ascii + 'endsolid cube\n');
const bin = Buffer.alloc(84 + 50 * T.length);
bin.write('solid binary cube, whose header starts as an ASCII file would', 0);
bin.writeUInt32LE(T.length, 80);
T.forEach((t, k) => t.forEach((i, j) => V[i].forEach((x, a) => bin.writeFloatLE(x, 84 + 50 * k + 12 + 12 * j + 4 * a))));
writeFileSync(path.join(dir, 'cube.stl'), bin);

// OBJ with quads, two groups, a negative index and v/vt/vn faces.
let obj = '# a cube of quads\nmtllib none.mtl\n' + V.map((v) => `v ${v.join(' ')}`).join('\n') + '\nvt 0 0\nvn 0 0 1\n';
obj += 'o lower\n' + Q.slice(0, 3).map((q) => 'f ' + q.map((i) => `${i + 1}/1/1`).join(' ')).join('\n') + '\n';
obj += 'g upper\n' + Q.slice(3).map((q) => 'f ' + q.map((i) => `${i - 8}`).join(' ')).join('\n') + '\n';
writeFileSync(path.join(dir, 'cube.obj'), obj);

// PLY, ASCII and binary little-endian, with an extra vertex property to skip.
const plyHead = (fmt) => `ply\nformat ${fmt} 1.0\ncomment cube\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nelement face 6\nproperty list uchar int vertex_indices\nend_header\n`;
writeFileSync(path.join(dir, 'cube_ascii.ply'), plyHead('ascii') + V.map((v) => `${v.join(' ')} 255`).join('\n') + '\n' + Q.map((q) => `4 ${q.join(' ')}`).join('\n') + '\n');
const body = Buffer.alloc(8 * 13 + 6 * 17);
let o = 0;
for (const v of V) {
  for (const x of v) (body.writeFloatLE(x, o), (o += 4));
  body.writeUInt8(255, o++);
}
for (const q of Q) {
  body.writeUInt8(4, o++);
  for (const i of q) (body.writeInt32LE(i, o), (o += 4));
}
writeFileSync(path.join(dir, 'cube_bin.ply'), Buffer.concat([Buffer.from(plyHead('binary_little_endian')), body]));

// glTF: one cube mesh used by three nodes in a chain. "base" at the origin; its child "arm"
// 2 along x and turned 90 degrees about z; its child "tip" 1 along the arm's own x.
const pos = Buffer.alloc(8 * 12);
V.forEach((v, i) => v.forEach((x, a) => pos.writeFloatLE(x, 12 * i + 4 * a)));
const ind = Buffer.alloc(36 * 2);
T.flat().forEach((i, k) => ind.writeUInt16LE(i, 2 * k));
const data = Buffer.concat([pos, ind]);
const s = Math.SQRT1_2;
const gltf = (uri) => ({
  asset: { version: '2.0' },
  scene: 0,
  scenes: [{ nodes: [0] }],
  nodes: [
    { name: 'base', mesh: 0, children: [1] },
    { name: 'arm', mesh: 0, translation: [2, 0, 0], rotation: [0, 0, s, s], children: [2] },
    { name: 'tip', mesh: 0, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1] },
  ],
  meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
  buffers: [{ ...(uri ? { uri } : {}), byteLength: data.length }],
  bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 96 }, { buffer: 0, byteOffset: 96, byteLength: 72 }],
  accessors: [
    { bufferView: 0, componentType: 5126, count: 8, type: 'VEC3', min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
    { bufferView: 1, componentType: 5123, count: 36, type: 'SCALAR' },
  ],
});
writeFileSync(path.join(dir, 'chain.gltf'), JSON.stringify(gltf(`data:application/octet-stream;base64,${data.toString('base64')}`), null, 1));
const json = Buffer.from(JSON.stringify(gltf(null)));
const jsonPad = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 0x20)]);
const binPad = Buffer.concat([data, Buffer.alloc((4 - (data.length % 4)) % 4)]);
const glb = Buffer.alloc(12 + 8 + jsonPad.length + 8 + binPad.length);
glb.writeUInt32LE(0x46546c67, 0);
glb.writeUInt32LE(2, 4);
glb.writeUInt32LE(glb.length, 8);
glb.writeUInt32LE(jsonPad.length, 12);
glb.writeUInt32LE(0x4e4f534a, 16);
jsonPad.copy(glb, 20);
glb.writeUInt32LE(binPad.length, 20 + jsonPad.length);
glb.writeUInt32LE(0x004e4942, 24 + jsonPad.length);
binPad.copy(glb, 28 + jsonPad.length);
writeFileSync(path.join(dir, 'chain.glb'), glb);
