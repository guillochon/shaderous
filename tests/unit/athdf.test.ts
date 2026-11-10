import { describe, expect, it } from 'vitest';
import h5wasm from 'h5wasm/node';
import { readAthdf, writeAthdf, type H5 } from '../../src/io/athdf';
import { SnapshotSampler, VARIABLES, type Snapshot } from '../../src/io/snapshot';

const h5 = h5wasm as unknown as H5;

/** Two root blocks along x; the second is refined once in the plane (a 2D mesh, 4-cell blocks). */
function sample(): Snapshot {
  const blocks = [
    [0, 0, 0, 0],
    [1, 2, 0, 0], [1, 3, 0, 0], [1, 2, 1, 0], [1, 3, 1, 0],
  ];
  const cells = 4 * 4;
  const data = Object.fromEntries(VARIABLES.map((v, q) => [v, Float32Array.from({ length: blocks.length * cells }, (_, i) => q * 1000 + i + 0.25)]));
  return {
    ndim: 2, rootSize: [8, 4, 1], blockSize: [4, 4, 1], lo: [-1, -0.5, -0.5], hi: [1, 0.5, 0.5],
    time: 1.5, steps: 42,
    levels: Int32Array.from(blocks.map((b) => b[0])),
    locations: Int32Array.from(blocks.flatMap((b) => b.slice(1))),
    data,
    meta: { program: 'shaderous', build: 'test', query: 'problem=sedov&dim=2&n=128', perturber: { pos: [1, 2, 3], vel: [4, 5, 6] } },
  };
}

describe('Athena++ HDF5 snapshots', () => {
  it('round-trips blocks, variables, grid and settings', async () => {
    const s = sample();
    const back = await readAthdf(await writeAthdf(s, h5), h5);
    expect(back.ndim).toBe(2);
    expect(back.rootSize).toEqual(s.rootSize);
    expect(back.blockSize).toEqual(s.blockSize);
    expect(back.lo).toEqual(s.lo);
    expect(back.hi).toEqual(s.hi);
    expect([back.time, back.steps]).toEqual([1.5, 42]);
    expect(Array.from(back.levels)).toEqual(Array.from(s.levels));
    expect(Array.from(back.locations)).toEqual(Array.from(s.locations));
    for (const v of VARIABLES) expect(Array.from(back.data[v]!)).toEqual(Array.from(s.data[v]!));
    expect(back.meta).toEqual(s.meta);
  });

  it('writes the layout other tools expect', async () => {
    const bytes = await writeAthdf(sample(), h5);
    (await h5.ready).FS.writeFile('check.athdf', bytes);
    const f = new h5wasm.File('check.athdf', 'r');
    expect((f.get('prim') as { shape: number[] }).shape).toEqual([6, 5, 1, 4, 4]);
    expect((f.get('x1f') as { shape: number[] }).shape).toEqual([5, 5]);
    expect((f.get('LogicalLocations') as { shape: number[] }).shape).toEqual([5, 3]);
    expect(f.attrs.DatasetNames.value).toEqual(['prim']);
    expect(f.attrs.VariableNames.value).toEqual([...VARIABLES]);
    expect(Array.from(f.attrs.NumVariables.value as Int32Array)).toEqual([6]);
    // The refined block at (2, 0) starts at x = 0 and its cells are 0.125 wide.
    const x1f = (f.get('x1f') as { value: Float32Array }).value;
    expect(Array.from(x1f.slice(5, 10))).toEqual([0, 0.125, 0.25, 0.375, 0.5]);
    f.close();
    (await h5.ready).FS.unlink('check.athdf');
  });

  it('samples the finest block that covers a point', async () => {
    const s = new SnapshotSampler(sample());
    expect(s.level(0.1, 0.5, 0.5)).toBe(0);
    expect(s.level(0.9, 0.9, 0.5)).toBe(1);
    // Root block, cell (1, 2): fractions 0.5 * (1.5 / 4), (2.5 / 4).
    expect(s.prim(0.1875, 0.625, 0).rho).toBe(2 * 4 + 1 + 0.25);
    // Refined block 3 at (2, 1): its cell (0, 0) sits at fractions (0.5 + 1/32, 0.5 + 1/16).
    expect(s.prim(0.5 + 1 / 32, 0.5 + 1 / 16, 0).rho).toBe(3 * 16 + 0.25);
  });

  it('reads back compressed files, which are smaller', async () => {
    const s = sample();
    // Mostly uniform data, as a quiet region of a run would be.
    for (const v of VARIABLES) s.data[v]!.fill(1.5, 16);
    const plain = await writeAthdf(s, h5);
    const packed = await writeAthdf(s, h5, 1);
    expect(packed.length).toBeLessThan(plain.length);
    const back = await readAthdf(packed, h5);
    for (const v of VARIABLES) expect(Array.from(back.data[v]!)).toEqual(Array.from(s.data[v]!));
  });

  it('rejects files that are not snapshots', async () => {
    await expect(readAthdf(new Uint8Array([1, 2, 3, 4]), h5)).rejects.toThrow(/not an HDF5 file/);
  });
});
