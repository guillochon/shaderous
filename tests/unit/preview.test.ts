import { describe, expect, it } from 'vitest';
import { fitLumps, previewOrbit, type Lump } from '../../src/gravity/preview';

const at = (o: { points: Float32Array }, i: number) => [...o.points.subarray(3 * i, 3 * i + 3)];
const last = (o: { points: Float32Array }) => at(o, o.points.length / 3 - 1);

/** A uniform ball of gas of unit mass and radius `r`, as lumps on a grid of `k` cells per box side. */
function ball(r: number, k: number, box = 1): Lump[] {
  const out: Lump[] = [];
  const h = box / k;
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++)
      for (let l = 0; l < k; l++) {
        const pos: [number, number, number] = [(i + 0.5) * h - box / 2, (j + 0.5) * h - box / 2, (l + 0.5) * h - box / 2];
        if (Math.hypot(...pos) < r) out.push({ m: 1, pos, h });
      }
  for (const l of out) l.m = 1 / out.length;
  return out;
}

describe('orbit preview', () => {
  it('is a straight line of the capped length when nothing pulls', () => {
    const o = previewOrbit({ bodies: [{ gm: 1, pos: [0, 0, 0], vel: [1, 0, 0], soft: 0.04 }], lumps: [], G: 1, frame: 0, box: 10 }, { length: 0.3 });
    expect(o.end).toBe('length');
    const p = last(o);
    expect(p[0]).toBeGreaterThanOrEqual(3);
    expect(p[0]).toBeLessThan(3.2);
    expect(p[1]).toBe(0);
    expect(o.times[o.times.length - 1]).toBeCloseTo(p[0], 5);
  });

  it('does not move a mass at rest with nothing to pull it', () => {
    const o = previewOrbit({ bodies: [{ gm: 1, pos: [0.1, 0, 0], vel: [0, 0, 0], soft: 0.04 }], lumps: [], G: 1, frame: 0, box: 1 });
    expect(o.end).toBe('rest');
    expect(o.points.length).toBe(3);
  });

  it('closes a circular orbit about a heavy mass', () => {
    const r = 0.2;
    const soft = 0.004;
    // The circular speed of the softened potential.
    const vc = Math.sqrt((r * r) / (r * r + soft * soft) ** 1.5);
    const period = (2 * Math.PI * r) / vc;
    const o = previewOrbit(
      {
        bodies: [
          { gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft },
          { gm: 1e-9, pos: [r, 0, 0], vel: [0, vc, 0], soft },
        ],
        lumps: [], G: 1, frame: 0, box: 1,
      },
      { length: 10, steps: 100000, time: period, points: 400 },
    );
    expect(o.end).toBe('time');
    const p = last(o);
    expect(Math.hypot(p[0] - r, p[1], p[2])).toBeLessThan(0.01 * r);
    // And stays on the circle all the way round.
    for (let i = 0; i < o.points.length / 3; i++) expect(Math.abs(Math.hypot(...at(o, i)) - r)).toBeLessThan(1e-3 * r);
  });

  it('feels a ball of gas as its mass at its center, from outside', () => {
    const lumps = ball(0.2, 16);
    const run = (x: number) => previewOrbit({ bodies: [{ gm: 1e-9, pos: [x, 0, 0], vel: [0, 0, 0], soft: 0.01 }], lumps, G: 1, frame: 0, box: 1 }, { time: 1e-2, steps: 100000 });
    // Released at rest, it falls by a t^2 / 2.
    const fall = (x: number) => (x - last(run(x))[0]) / (0.5 * 1e-4);
    expect(fall(0.4)).toBeGreaterThan(0.95 / 0.16);
    expect(fall(0.4)).toBeLessThan(1.05 / 0.16);
    // At the center the pulls cancel.
    expect(run(0).end).toBe('time');
    expect(Math.abs(fall(0))).toBeLessThan(0.05 / 0.16);
  });

  it('holds the center of the masses at rest in their frame', () => {
    const o = previewOrbit(
      {
        bodies: [
          { gm: 1, pos: [-0.1, 0, 0], vel: [0, -1.5, 0], soft: 0.01 },
          { gm: 1, pos: [0.1, 0, 0], vel: [0, 1.5, 0], soft: 0.01 },
        ],
        lumps: [{ m: 50, pos: [3, 0, 0], h: 0.1 }],
        G: 1, frame: 2, box: 1,
      },
      { time: 0.3, steps: 100000 },
    );
    // The lump would pull both away; in their frame the pair turns about the middle instead.
    const p = last(o);
    expect(Math.abs(Math.hypot(p[0], p[1]) - 0.1)).toBeLessThan(0.02);
  });

  it('stops once the mass has left the box', () => {
    const o = previewOrbit({ bodies: [{ gm: 1, pos: [0.4, 0, 0], vel: [1, 0, 0], soft: 0.04 }], lumps: [], G: 1, frame: 0, box: 1 });
    expect(o.end).toBe('left');
    expect(last(o)[0]).toBeGreaterThan(0.6);
    expect(last(o)[0]).toBeLessThan(0.63);
  });
});

describe('lumps for a preview', () => {
  const sum = (ls: Lump[]) => ls.reduce((s, l) => s + l.m, 0);

  it('keeps every lump when they fit, the thin gas too', () => {
    const lumps = [...ball(0.2, 16), { m: 1e-9, pos: [0.45, 0.45, 0.45] as [number, number, number], h: 1 / 16 }];
    const fit = fitLumps(lumps, 1, 3);
    expect(fit.length).toBe(lumps.length);
    expect(sum(fit)).toBeCloseTo(1 + 1e-9, 12);
  });

  it('merges a full box down to the cap, with all of its mass about the same center', () => {
    const lumps = ball(2, 32); // every cell of the box
    expect(lumps.length).toBe(32 ** 3);
    const fit = fitLumps(lumps, 1, 3, 768);
    expect(fit.length).toBeLessThanOrEqual(768);
    expect(fit.length).toBeGreaterThan(768 - 7);
    expect(sum(fit)).toBeCloseTo(1, 9);
    for (let c = 0; c < 3; c++) expect(Math.abs(fit.reduce((s, l) => s + l.m * l.pos[c], 0))).toBeLessThan(1e-9);
  });

  it('keeps a star fine and its surroundings coarse', () => {
    // A ball holding nearly all the mass, in thin gas that fills the box.
    const star = ball(0.15, 32);
    const thin = ball(2, 32).map((l) => ({ ...l, m: l.m * 1e-3 }));
    const fit = fitLumps([...star, ...thin], 1, 3, 768);
    expect(fit.length).toBeLessThanOrEqual(768);
    expect(sum(fit)).toBeCloseTo(1.001, 9);
    const inside = fit.filter((l) => Math.hypot(...l.pos) < 0.12);
    const outside = fit.filter((l) => Math.hypot(...l.pos) > 0.3);
    expect(Math.max(...inside.map((l) => l.h))).toBeLessThanOrEqual(1 / 16);
    expect(Math.min(...outside.map((l) => l.h))).toBeGreaterThanOrEqual(1 / 8);
  });

  it('cuts a plane in four', () => {
    const lumps: Lump[] = [];
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) lumps.push({ m: 1, pos: [(i + 0.5) / 64 - 0.5, (j + 0.5) / 64 - 0.5, 0], h: 1 / 64 });
    const fit = fitLumps(lumps, 1, 2, 256);
    expect(fit.length).toBe(256);
    expect(fit.every((l) => l.h === 1 / 16 && l.pos[2] === 0)).toBe(true);
  });
});
