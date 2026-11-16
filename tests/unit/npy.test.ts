// The .npy writer of the headless tools (tools/headless/npy.mjs) against files numpy wrote
// (tests/unit/fixtures, by np.save of float32 arrays).
import { describe, expect, it } from 'vitest';
import { npyBytes, readNpy } from '../../tools/headless/npy.mjs';

// The project has no Node types: the one function needed, typed here.
const fs: { readFileSync(u: URL): Uint8Array } = await import(/* @vite-ignore */ 'node:fs' as string);
const fixture = (name: string) => Array.from(fs.readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));

describe('.npy files', () => {
  it('writes a 2D array byte for byte as numpy does', () => {
    const v = Float32Array.from({ length: 12 }, (_, i) => i * 0.5 - 1);
    expect(Array.from(npyBytes(v, [3, 4]))).toEqual(fixture('slice-3x4.npy'));
  });

  it('writes a 1D array, NaN and infinity included, as numpy does', () => {
    const v = Float32Array.from([1, -2, Infinity, NaN, 0]);
    expect(Array.from(npyBytes(v, [5]))).toEqual(fixture('vector-5.npy'));
  });

  it('pads the header to 64 bytes, and reads back what it writes', () => {
    const v = Float32Array.from({ length: 2048 * 2048 }, (_, i) => i);
    const file = npyBytes(v, [2048, 2048]);
    expect((file.length - v.byteLength) % 64).toBe(0);
    const back = readNpy(file);
    expect(back.shape).toEqual([2048, 2048]);
    expect(back.values[2048 * 2048 - 1]).toBe(2048 * 2048 - 1);
    expect(readNpy(npyBytes(new Float32Array(12), [3, 4])).shape).toEqual([3, 4]);
  });

  it('refuses a shape that does not hold the values', () => {
    expect(() => npyBytes(new Float32Array(6), [2, 4])).toThrow(/does not hold 6/);
  });
});
