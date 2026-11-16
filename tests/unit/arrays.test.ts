import { describe, expect, it } from 'vitest';
import { base64ToBytes, bytesToBase64, decodeArray, encodeArray } from '../../src/io/arrays';

describe('encoded arrays', () => {
  it('round-trips values, with the shape and dtype fields', () => {
    const v = Float32Array.from({ length: 12 }, (_, i) => i * 0.1 - 0.3);
    const a = encodeArray(v, [3, 4]);
    expect(a).toMatchObject({ dtype: 'float32', shape: [3, 4], order: 'C', endian: 'little' });
    expect(a.base64.length).toBe(Math.ceil((12 * 4) / 3) * 4);
    expect(Array.from(decodeArray(a))).toEqual(Array.from(v));
  });

  it('keeps NaN, infinities and negative zero', () => {
    const v = Float32Array.from([NaN, Infinity, -Infinity, -0, 3.4e38, 1e-45]);
    const back = decodeArray(encodeArray(v, [6]));
    expect(Number.isNaN(back[0])).toBe(true);
    expect(back[1]).toBe(Infinity);
    expect(back[2]).toBe(-Infinity);
    expect(Object.is(back[3], -0)).toBe(true);
    expect(back[4]).toBe(v[4]);
    expect(back[5]).toBe(v[5]);
  });

  it('writes little-endian bytes', () => {
    expect(Array.from(base64ToBytes(encodeArray(Float32Array.from([1]), [1]).base64))).toEqual([0, 0, 0x80, 0x3f]);
  });

  it('encodes a view at its offset, and arrays longer than one chunk', () => {
    const big = Float32Array.from({ length: 70000 }, (_, i) => Math.sin(i));
    const view = big.subarray(5, 5 + 3 * 20000);
    expect(Array.from(decodeArray(encodeArray(view, [3, 20000])))).toEqual(Array.from(view));
  });

  it('matches btoa of the same bytes taken whole', () => {
    const bytes = Uint8Array.from({ length: 100001 }, (_, i) => (i * 37) & 255);
    let binary = '';
    for (const b of bytes) binary += String.fromCharCode(b);
    expect(bytesToBase64(bytes)).toBe(btoa(binary));
  });

  it('refuses a shape that does not fit, and a wrong byte length', () => {
    expect(() => encodeArray(new Float32Array(6), [2, 4])).toThrow(/holds 8 values, not 6/);
    const a = encodeArray(new Float32Array(4), [4]);
    expect(() => decodeArray({ ...a, shape: [5] })).toThrow(/holds 5 values, not 4/);
    expect(() => decodeArray({ ...a, base64: bytesToBase64(new Uint8Array(6)) })).toThrow(/not a whole number/);
  });
});
