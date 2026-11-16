// numpy's .npy format (version 1.0) for float32 arrays, with no dependency: what np.load reads
// with its shape and dtype. https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

/** The header of a little-endian float32 array of `shape`, in C order, padded to 64 bytes. */
export function npyHeader(shape) {
  const dims = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': ${dims}, }`;
  const fixed = 10; // magic (6), version (2), header length (2)
  const total = Math.ceil((fixed + dict.length + 1) / 64) * 64;
  const head = Buffer.alloc(total, 0x20);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1;
  head[7] = 0;
  head.writeUInt16LE(total - fixed, 8);
  head.write(dict, fixed, 'latin1');
  head[total - 1] = 0x0a;
  return head;
}

/** A .npy file of a Float32Array laid out in C order with `shape`. */
export function npyBytes(values, shape) {
  if (!(values instanceof Float32Array)) throw new Error('npy: values must be a Float32Array');
  if (shape.reduce((a, b) => a * b, 1) !== values.length) throw new Error(`npy: shape [${shape}] does not hold ${values.length} values`);
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('npy: this machine is not little-endian');
  return Buffer.concat([npyHeader(shape), Buffer.from(values.buffer, values.byteOffset, values.byteLength)]);
}

/** Read a .npy file of little-endian float32 values: { shape, values }. */
export function readNpy(buf) {
  if (buf.subarray(0, 6).toString('latin1') !== '\x93NUMPY') throw new Error('npy: not a .npy file');
  const [major] = buf.subarray(6, 8);
  const len = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const start = major === 1 ? 10 : 12;
  const header = buf.subarray(start, start + len).toString('latin1');
  if (!/'descr':\s*'<f4'/.test(header) || /'fortran_order':\s*True/.test(header)) throw new Error(`npy: only little-endian float32 in C order (${header.trim()})`);
  const shape = [...header.match(/'shape':\s*\(([^)]*)\)/)[1].matchAll(/\d+/g)].map((m) => Number(m[0]));
  const data = buf.subarray(start + len);
  const values = new Float32Array(data.length / 4);
  new Uint8Array(values.buffer).set(data);
  return { shape, values };
}
