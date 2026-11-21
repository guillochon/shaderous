import { describe, expect, it } from 'vitest';
import h5wasm from 'h5wasm/node';

import { library, openAthdf, readAthdfFile, writeAthdfFile, type H5 } from '../../src/io/athdf';
import { countAtCap, planView, readForView } from '../../src/io/view';
import type { Snapshot } from '../../src/io/snapshot';

const h5 = h5wasm as unknown as H5;
const BS = 16;

/**
 * A 2D mesh of 16-cell blocks: two root blocks along x, the second refined once and its first
 * child once more, with a divergence-free face field from a vector potential.
 */
function sample(): Snapshot {
  const blocks: number[][] = [[0, 0, 0]];
  for (const [i, j] of [[3, 0], [2, 1], [3, 1]]) blocks.push([1, i, j]);
  for (const [i, j] of [[4, 0], [5, 0], [4, 1], [5, 1]]) blocks.push([2, i, j]);
  const n = blocks.length;
  const cells = BS * BS;
  const rootSize = [2 * BS, BS, 1];
  const L = 2; // box [0,2] x [0,1]
  const data: Record<string, Float32Array> = {};
  for (const v of ['rho', 'press', 'vel1', 'vel2', 'vel3', 'r0']) data[v] = new Float32Array(n * cells);
  const W = BS + 1;
  const per = W * W;
  const faces = new Float32Array(n * per * 3);
  const A = (x: number, y: number) => Math.sin(2 * x + 0.3) * Math.cos(3 * y) + x * y;
  blocks.forEach(([l, bi, bj], b) => {
    const dx = L / (rootSize[0] << l);
    for (let j = 0; j < BS; j++)
      for (let i = 0; i < BS; i++) {
        const x = (bi * BS + i + 0.5) * dx;
        const y = (bj * BS + j + 0.5) * dx;
        const c = b * cells + j * BS + i;
        data.rho[c] = 1 + 0.5 * Math.sin(3 * x) * Math.cos(2 * y);
        data.press[c] = 2 + x;
        data.vel1[c] = Math.cos(x + y);
        data.vel2[c] = y - x;
        data.vel3[c] = 0.1;
        data.r0[c] = x * y;
      }
    for (let j = 0; j < W; j++)
      for (let i = 0; i < W; i++) {
        const x = (bi * BS + i) * dx;
        const y = (bj * BS + j) * dx;
        const t = (b * per + j * W + i) * 3;
        faces[t] = (A(x, y + dx) - A(x, y)) / dx;
        faces[t + 1] = -(A(x + dx, y) - A(x, y)) / dx;
      }
  });
  return {
    ndim: 2, rootSize, blockSize: [BS, BS, 1], lo: [0, 0, -0.5], hi: [2, 1, 0.5], time: 3, steps: 7,
    levels: Int32Array.from(blocks.map((b) => b[0])),
    locations: Int32Array.from(blocks.flatMap((b) => [b[1], b[2], 0])),
    data, faces,
  };
}

/** Sums of mass, x momentum and passive-scalar mass, by cell area. */
function totals(s: Snapshot) {
  const cells = BS * BS;
  let m = 0;
  let px = 0;
  let q = 0;
  for (let b = 0; b < s.levels.length; b++) {
    const a = 1 / 4 ** s.levels[b];
    for (let c = b * cells; c < (b + 1) * cells; c++) {
      m += a * s.data.rho![c];
      px += a * s.data.rho![c] * s.data.vel1![c];
      q += a * s.data.rho![c] * s.data.r0![c];
    }
  }
  return { m, px, q };
}

/** Largest |div B| dx over the cells of every block, in units of the field. */
function maxDiv(s: Snapshot) {
  const W = BS + 1;
  let worst = 0;
  for (let b = 0; b < s.levels.length; b++)
    for (let j = 0; j < BS; j++)
      for (let i = 0; i < BS; i++) {
        const f = (ii: number, jj: number, a: number) => s.faces![((b * W * W) + jj * W + ii) * 3 + a];
        worst = Math.max(worst, Math.abs(f(i + 1, j, 0) - f(i, j, 0) + f(i, j + 1, 1) - f(i, j, 1)));
      }
  return worst;
}

describe('view-only loading', () => {
  it('counts the blocks a cap leaves, ancestors included', () => {
    const s = sample();
    expect(countAtCap(s, 2)).toEqual({ leaves: 8, nodes: 10 });
    expect(countAtCap(s, 1)).toEqual({ leaves: 5, nodes: 6 });
    expect(countAtCap(s, 0)).toEqual({ leaves: 2, nodes: 2 });
    expect(planView(s, 6)?.level).toBe(1);
    expect(planView(s, 1)).toBeNull();
  });

  it('reads a file whole when nothing is merged', async () => {
    const s = sample();
    // (Files of their own: the other tests' workers write theirs in the same folder.)
    const file = `view-whole-${Math.random().toString(36).slice(2)}.athdf`;
    await writeAthdfFile(file, s, h5);
    const lib = await library(h5);
    const r = openAthdf(lib, file);
    const v = readForView(r, 2, { chunk: 3 });
    r.close();
    const whole = await readAthdfFile(file, h5);
    (await library(h5)).FS.unlink(file);
    for (const k of ['rho', 'press', 'vel1', 'r0']) expect(Array.from(v.data[k]!)).toEqual(Array.from(whole.data[k]!));
    expect(Array.from(v.faces!)).toEqual(Array.from(whole.faces!));
  });

  it('merges finer blocks into their ancestors, keeping mass, momentum and div B = 0', async () => {
    const s = sample();
    const file = `view-merge-${Math.random().toString(36).slice(2)}.athdf`;
    await writeAthdfFile(file, s, h5);
    const lib = await library(h5);
    const before = totals(s);
    expect(maxDiv(s)).toBeLessThan(1e-4);
    for (const cap of [1, 0]) {
      const r = openAthdf(lib, file);
      const v = readForView(r, cap, { chunk: 2 });
      r.close();
      expect(v.levels.length).toBe(cap === 1 ? 5 : 2);
      expect(Math.max(...v.levels)).toBe(cap);
      const after = totals(v);
      expect(after.m).toBeCloseTo(before.m, 4);
      expect(after.px).toBeCloseTo(before.px, 4);
      expect(after.q).toBeCloseTo(before.q, 4);
      expect(maxDiv(v)).toBeLessThan(1e-4);
      expect(v.time).toBe(3);
    }
    (await library(h5)).FS.unlink(file);
  });
});
