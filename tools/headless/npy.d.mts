// Types of npy.mjs, for the unit tests that import it.
export function npyHeader(shape: readonly number[]): Uint8Array;
export function npyBytes(values: Float32Array, shape: readonly number[]): Uint8Array;
export function readNpy(buf: Uint8Array): { shape: number[]; values: Float32Array };
