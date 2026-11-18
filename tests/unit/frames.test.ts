import { describe, expect, it } from 'vitest';
import { fromRotating, rotTurn, toRotating } from '../../src/problems/rotating';
import { domainHi, domainLo } from '../../src/domain';
import { configFor, rotationOf, rotating } from '../../src/sim';
import { all, PERIODIC, type Problem } from '../../src/problems/types';

/** Rotation by angle th about the unit axis n, as a matrix. */
function rotation(n: number[], th: number): number[][] {
  const c = Math.cos(th);
  const s = Math.sin(th);
  const [x, y, z] = n;
  return [
    [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
  ];
}

const still: Problem = {
  id: 'still', label: 'Still', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
  init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }),
};

describe('rotating frame', () => {
  it('turns the momentum as a rotation matrix does', () => {
    const omega = [0.3, -0.7, 1.1];
    const w = Math.hypot(...omega);
    const n = omega.map((o) => o / w);
    for (const dt of [1e-6, 0.01, 0.3, 2]) {
      const m = [0.4, -1.3, 2.2];
      const R = rotation(n, -2 * w * dt);
      const want = R.map((row) => row[0] * m[0] + row[1] * m[1] + row[2] * m[2]);
      const got = rotTurn(m, omega, dt);
      for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(want[i], 13);
      expect(Math.hypot(...got)).toBeCloseTo(Math.hypot(...m), 13);
    }
  });

  it('solves dm/dt = -2 omega x m', () => {
    const omega = [0, 0, 2];
    const m = [1, 0, 0];
    const h = 1e-7;
    const got = rotTurn(m, omega, h);
    // -2 omega x m = (0, -4, 0)
    expect((got[1] - m[1]) / h).toBeCloseTo(-4, 5);
  });

  it('converts velocities between the frames', () => {
    const omega = [0, 0, 1.5];
    const x = [0.2, -0.4, 0.1];
    const v = [1, 2, 3];
    const r = toRotating(v, x, omega);
    expect(r[0]).toBeCloseTo(1 - 1.5 * 0.4, 14);
    expect(r[1]).toBeCloseTo(2 - 1.5 * 0.2, 14);
    expect(fromRotating(r, x, omega)).toEqual(v.map((a, i) => a + 0 * i));
  });

  it('packs the frame into Params: z alone in 2D, centrifugal by default', () => {
    const rot: Problem = { ...still, rotation: () => ({ omega: [1, 2, 3] }) };
    expect(rotationOf(configFor(rot, 16, { ndim: 3 }))).toEqual([1, 2, 3, 1]);
    expect(rotationOf(configFor(rot, 16, { ndim: 2 }))).toEqual([0, 0, 3, 1]);
    const f: Problem = { ...still, rotation: () => ({ omega: [0, 0, 1], centrifugal: false }) };
    expect(rotationOf(configFor(f, 16, { ndim: 2 }))).toEqual([0, 0, 1, 0]);
    expect(rotating(configFor(still, 16, {}))).toBe(false);
    expect(rotating(configFor({ ...still, rotation: () => ({ omega: [0, 0, 0] }) }, 16, {}))).toBe(false);
  });
});

describe('domain', () => {
  it('is centered on the origin, or starts there along mirrored axes', () => {
    expect(domainLo({ problem: { box: 4 } })).toEqual([-2, -2, -2]);
    expect(domainHi({ problem: { box: 4 } })).toEqual([2, 2, 2]);
    expect(domainLo({ problem: { box: 4 }, symmetry: [true, false, true] })).toEqual([0, -2, 0]);
    expect(domainHi({ problem: { box: 4 }, symmetry: [true, false, true] })).toEqual([4, 2, 4]);
  });
});
