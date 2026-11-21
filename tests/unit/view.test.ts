import { describe, expect, it } from 'vitest';
import h5wasm from 'h5wasm/node';

import { library, openAthdf, readAthdfFile, writeAthdfFile, type H5 } from '../../src/io/athdf';
import { blocksOfUniform, countAtCap, planView, probeFile, readForView, streamForView } from '../../src/io/view';
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

/** A uniform 2D grid of n x n cells on [0, 1]^2 with a periodic, divergence-free face field. */
function uniform(n: number): Snapshot {
  const data: Record<string, Float32Array> = {};
  for (const v of ['rho', 'press', 'vel1', 'vel2', 'vel3', 'Bcc1', 'Bcc2', 'Bcc3', 'Bf1', 'Bf2', 'Bf3']) data[v] = new Float32Array(n * n);
  const dx = 1 / n;
  const A = (x: number, y: number) => Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y) / (2 * Math.PI);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const c = j * n + i;
      const x = (i + 0.5) * dx;
      const y = (j + 0.5) * dx;
      data.rho[c] = 1 + 0.5 * Math.sin(2 * Math.PI * x) * Math.sin(4 * Math.PI * y);
      data.press[c] = 1 + y;
      data.vel1[c] = Math.cos(2 * Math.PI * (x + y));
      data.Bf1[c] = (A(i * dx, (j + 1) * dx) - A(i * dx, j * dx)) / dx;
      data.Bf2[c] = -(A((i + 1) * dx, j * dx) - A(i * dx, j * dx)) / dx;
    }
  return {
    ndim: 2, rootSize: [n, n, 1], blockSize: [n, n, 1], lo: [0, 0, -0.5], hi: [1, 1, 0.5], time: 1, steps: 1,
    levels: Int32Array.of(0), locations: Int32Array.of(0, 0, 0), data,
    meta: { program: 'shaderous', build: 'test', query: 'problem=sedov&dim=2&n=32', mhd: { n, nz: 1, upper: [[], [], []] } },
  };
}

describe('view-only loading in parts', () => {
  it('streams merged blocks in batches, the same as reading them whole', async () => {
    const s = sample();
    const file = `view-stream-${Math.random().toString(36).slice(2)}.athdf`;
    await writeAthdfFile(file, s, h5);
    const lib = await library(h5);
    const r = openAthdf(lib, file);
    const whole = readForView(r, 1, { chunk: 2 });
    const parts: number[] = [];
    let blocks = 0;
    streamForView(r, 1, { chunk: 3, batch: 2, emit: (b) => (parts.push(b.levels.length), (blocks += b.levels.length)) });
    expect(blocks).toBe(whole.levels.length);
    expect(Math.max(...parts)).toBeLessThanOrEqual(2);
    // A probe reads the file's own cell, whatever the view merged.
    const [p] = probeFile(r, [[0.80, 0.30, 0]]);
    expect(p.level).toBe(1);
    const x = 0.8 * 2;
    const y = 0.3;
    const dx = 2 / (32 << 1);
    const cx = (Math.floor(x / dx) + 0.5) * dx;
    const cy = (Math.floor(y / dx) + 0.5) * dx;
    expect(p.values.rho).toBeCloseTo(1 + 0.5 * Math.sin(3 * cx) * Math.cos(2 * cy), 5);
    r.close();
    lib.FS.unlink(file);
  });

  it('reads a uniform grid as blocks of 16, downsampled by merging, its field divergence-free', async () => {
    const n = 64;
    const s = uniform(n);
    const file = `view-uniform-${Math.random().toString(36).slice(2)}.athdf`;
    await writeAthdfFile(file, s, h5);
    const lib = await library(h5);
    const r = blocksOfUniform(openAthdf(lib, file));
    const head = r.header();
    expect(head.rootSize).toEqual([16, 16, 1]);
    expect(head.blockSize).toEqual([16, 16, 1]);
    expect(r.blocks).toBe(16);
    expect(Math.max(...head.levels)).toBe(2);
    expect(r.variables).not.toContain('Bf1');
    expect(r.hasFaces).toBe(true);
    const full = readForView(r, 2, { chunk: 5 });
    // Each block's cells are the grid's.
    for (let b = 0; b < full.levels.length; b++) {
      const [bi, bj] = [full.locations[3 * b], full.locations[3 * b + 1]];
      for (const [i, j] of [[0, 0], [15, 3], [7, 15]]) expect(full.data.rho![b * 256 + j * 16 + i]).toBe(s.data.rho![(bj * 16 + j) * n + bi * 16 + i]);
    }
    expect(maxDiv(full)).toBeLessThan(1e-4);
    for (const cap of [1, 0]) {
      const v = readForView(r, cap, { chunk: 7 });
      expect(v.levels.length).toBe(cap === 1 ? 4 : 1);
      let m0 = 0;
      for (const x of s.data.rho!) m0 += x / (n * n);
      let m1 = 0;
      const cellsAt = (16 << cap) ** 2;
      for (const x of v.data.rho!) m1 += x / cellsAt;
      expect(m1).toBeCloseTo(m0, 5);
      expect(maxDiv(v)).toBeLessThan(1e-4);
    }
    r.close();
    lib.FS.unlink(file);
  });
});
