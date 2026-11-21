// PLY, ASCII and binary of either byte order (plans/wind-tunnel.md 3.1): the vertices' x, y, z
// (any numeric type, other properties skipped) and the faces' vertex lists (fanned). Other
// elements are skipped. One part, "mesh".
import { fan, wholePart, type Mesh } from './mesh';

const SIZES: Record<string, number> = {
  char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8,
};

interface Prop { name: string; type: string; list?: { count: string; item: string } }
interface Elem { name: string; count: number; props: Prop[] }

export function parsePly(bytes: Uint8Array): Mesh {
  // The header is ASCII and ends with "end_header" and a newline.
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.byteLength, 65536)));
  const end = head.indexOf('end_header');
  if (!head.startsWith('ply') || end < 0) throw new Error('not a PLY file: no "ply" header');
  let bodyStart = end + 'end_header'.length;
  while (bodyStart < head.length && head[bodyStart] !== '\n') bodyStart++;
  bodyStart++;
  let format = '';
  const elems: Elem[] = [];
  for (const line of head.slice(0, end).split(/\r?\n/)) {
    const w = line.trim().split(/\s+/);
    if (w[0] === 'format') format = w[1];
    else if (w[0] === 'element') elems.push({ name: w[1], count: +w[2], props: [] });
    else if (w[0] === 'property' && elems.length) {
      const e = elems[elems.length - 1];
      if (w[1] === 'list') e.props.push({ name: w[4], type: 'list', list: { count: w[2], item: w[3] } });
      else e.props.push({ name: w[2], type: w[1] });
    }
  }
  const pos: number[] = [];
  const idx: number[] = [];
  const take = (e: Elem, values: (p: Prop) => number | number[]) => {
    if (e.name === 'vertex') {
      const v: Record<string, number> = {};
      for (const p of e.props) {
        const x = values(p);
        if (typeof x === 'number') v[p.name] = x;
      }
      pos.push(v.x ?? 0, v.y ?? 0, v.z ?? 0);
    } else if (e.name === 'face') {
      for (const p of e.props) {
        const x = values(p);
        if (Array.isArray(x) && (p.name === 'vertex_indices' || p.name === 'vertex_index')) fan(x, idx);
      }
    } else for (const p of e.props) values(p);
  };
  if (format === 'ascii') {
    const words = new TextDecoder().decode(bytes.subarray(bodyStart)).trim().split(/\s+/);
    let k = 0;
    const next = () => +words[k++];
    for (const e of elems)
      for (let i = 0; i < e.count; i++)
        take(e, (p) => {
          if (!p.list) return next();
          const n = next();
          const out: number[] = [];
          for (let j = 0; j < n; j++) out.push(next());
          return out;
        });
  } else if (format === 'binary_little_endian' || format === 'binary_big_endian') {
    const little = format === 'binary_little_endian';
    const v = new DataView(bytes.buffer, bytes.byteOffset + bodyStart, bytes.byteLength - bodyStart);
    let o = 0;
    const read = (type: string): number => {
      let x: number;
      switch (type) {
        case 'char': case 'int8': x = v.getInt8(o); break;
        case 'uchar': case 'uint8': x = v.getUint8(o); break;
        case 'short': case 'int16': x = v.getInt16(o, little); break;
        case 'ushort': case 'uint16': x = v.getUint16(o, little); break;
        case 'int': case 'int32': x = v.getInt32(o, little); break;
        case 'uint': case 'uint32': x = v.getUint32(o, little); break;
        case 'float': case 'float32': x = v.getFloat32(o, little); break;
        case 'double': case 'float64': x = v.getFloat64(o, little); break;
        default: throw new Error(`PLY: unknown property type "${type}"`);
      }
      o += SIZES[type];
      return x;
    };
    for (const e of elems)
      for (let i = 0; i < e.count; i++)
        take(e, (p) => {
          if (!p.list) return read(p.type);
          const n = read(p.list.count);
          const out: number[] = [];
          for (let j = 0; j < n; j++) out.push(read(p.list.item));
          return out;
        });
  } else throw new Error(`PLY: unknown format "${format}"`);
  const nv = pos.length / 3;
  if (!idx.length) throw new Error('PLY: the file has no faces (a point cloud cannot be a body)');
  if (idx.some((i) => !(i >= 0 && i < nv))) throw new Error('PLY: a face refers to a vertex the file does not have');
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: [wholePart(idx.length / 3)] };
}
