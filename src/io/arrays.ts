// Arrays packed as bytes in base64, for read-backs too large for a JSON list of numbers
// (slice() with encoding 'base64'). Pure: no DOM, so Node tests and reads them too.

/** A typed array as base64 of its bytes, with what numpy needs to read it back. */
export interface EncodedArray {
  dtype: 'float32';
  /** Slowest-varying axis first: [rows, columns] for a slice. */
  shape: number[];
  /** Row-major, as numpy's default. */
  order: 'C';
  endian: 'little';
  base64: string;
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

type NativeBase64 = { toBase64?(): string };
type NativeBase64Ctor = { fromBase64?(s: string): Uint8Array };

/** Bytes as base64: natively where the engine has it, else through btoa a chunk at a time. */
export function bytesToBase64(bytes: Uint8Array): string {
  const native = (bytes as NativeBase64).toBase64;
  if (native) return native.call(bytes);
  // Chunks of a multiple of 3 bytes encode on their own without padding, so they join.
  const CHUNK = 3 * 16384;
  const parts: string[] = [];
  for (let i = 0; i < bytes.length; i += CHUNK) {
    parts.push(btoa(String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK) as unknown as number[])));
  }
  return parts.join('');
}

export function base64ToBytes(b64: string): Uint8Array {
  const native = (Uint8Array as NativeBase64Ctor).fromBase64;
  if (native) return native(b64);
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const count = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

function checkShape(shape: readonly number[], length: number) {
  if (!shape.every((n) => Number.isInteger(n) && n >= 0)) throw new Error(`array shape must be whole numbers, got [${shape.join(', ')}]`);
  if (count(shape) !== length) throw new Error(`array shape [${shape.join(', ')}] holds ${count(shape)} values, not ${length}`);
}

/** Pack `values`, laid out in C order with the given shape. */
export function encodeArray(values: Float32Array, shape: readonly number[]): EncodedArray {
  if (!LITTLE_ENDIAN) throw new Error('arrays are encoded little-endian, and this machine is not');
  checkShape(shape, values.length);
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  return { dtype: 'float32', shape: [...shape], order: 'C', endian: 'little', base64: bytesToBase64(bytes) };
}

/** The values of an encoded array, in a fresh (so aligned) Float32Array. */
export function decodeArray(a: EncodedArray): Float32Array {
  if (a.dtype !== 'float32') throw new Error(`unsupported dtype "${a.dtype}": only float32`);
  if (a.order !== 'C') throw new Error(`unsupported order "${a.order}": only C`);
  if (a.endian !== 'little' || !LITTLE_ENDIAN) throw new Error('only little-endian arrays on a little-endian machine');
  const bytes = base64ToBytes(a.base64);
  if (bytes.length % 4) throw new Error(`${bytes.length} bytes is not a whole number of float32 values`);
  const out = new Float32Array(bytes.length / 4);
  new Uint8Array(out.buffer).set(bytes);
  checkShape(a.shape, out.length);
  return out;
}
