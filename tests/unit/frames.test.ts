import { describe, expect, it } from 'vitest';
import { fromRotating, rotTurn, toRotating } from '../../src/problems/rotating';
import { domainHi, domainLo } from '../../src/domain';
import { configFor, rotationOf, rotating, symmetryUnsupported, unfoldDiagnostics, withSymmetry, type Diagnostics } from '../../src/sim';
import { problemById } from '../../src/problems';
import { decodeLink, encodeLink, viewDefaults } from '../../src/ui/link';
import { defaultOpacity } from '../../src/render/transfer';
import { runningSpec } from '../../src/setup/running';
import { mirrorAxes, mirrorName } from '../../src/domain';
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

describe('mirror symmetry', () => {
  it('mirrors only the axes the preset allows, never z in 2D, and makes their planes reflect', () => {
    const sedov = problemById('sedov');
    const c = configFor(sedov, 32, { symmetry: [true, true, true] });
    expect(c.symmetry).toEqual([true, true, true]);
    expect(c.bcLo).toEqual([2, 2, 2]);
    expect(c.bcHi).toEqual([1, 1, 1]);
    expect(configFor(sedov, 32, { ndim: 2, symmetry: [true, true, true] }).symmetry).toEqual([true, true, false]);
    const kh = problemById('kelvin-helmholtz');
    expect(configFor(kh, 32, { ndim: 2, symmetry: [true, true, false] }).symmetry).toEqual([false, false, false]);
    expect(withSymmetry(configFor(sedov, 32, {}), [false, true, false]).bcLo).toEqual([1, 2, 1]);
  });

  it('refuses what is not symmetric under the mirrors', () => {
    const sedov = problemById('sedov');
    expect(symmetryUnsupported(configFor(sedov, 32, { symmetry: [true, true, true] }))).toBeNull();
    const pm: typeof sedov = { ...sedov, masses: () => [{ gm: 1, pos: [0.1, 0, 0], vel: [0, 0, 0] }] };
    expect(symmetryUnsupported(configFor(pm, 32, { symmetry: [true, false, false] }))).toMatch(/off the x plane/);
    const spun: typeof sedov = { ...sedov, rotation: () => ({ omega: [0, 0, 1] }) };
    expect(symmetryUnsupported(configFor(spun, 32, { symmetry: [false, false, true] }))).toBeNull();
    expect(symmetryUnsupported(configFor(spun, 32, { symmetry: [true, false, false] }))).toMatch(/turning/);
  });

  it('reports the whole configuration', () => {
    const d = { mass: 1, eKin: 2, eInt: 3, eGrav: -1, eTot: 4, nuclearHeating: 0.5, momentum: [1, 2, 3], com: [0.1, 0.2, 0.3] } as unknown as Diagnostics;
    const u = unfoldDiagnostics(d, [true, false, true]);
    expect([u.mass, u.eKin, u.eInt, u.eGrav, u.eTot, u.nuclearHeating]).toEqual([4, 8, 12, -4, 16, 2]);
    expect(u.momentum).toEqual([0, 8, 0]);
    expect(u.com).toEqual([0, 0.2, 0]);
  });

  it('names mirror sets both ways', () => {
    expect(mirrorAxes('xz')).toEqual([true, false, true]);
    expect(mirrorName([true, false, true])).toBe('xz');
    expect(mirrorName([false, false, false])).toBe('none');
  });

  it('travels in links as sym, without the faces of its planes', () => {
    const problem = problemById('sedov');
    const cfg = configFor(problem, 64, { symmetry: [true, true, true] });
    const q = encodeLink({ cfg, view: viewDefaults(problem, 3), auto: true, colormap: 'inferno', opacity: defaultOpacity() });
    expect(q.get('sym')).toBe('xyz');
    expect(q.has('bc')).toBe(false);
    expect(decodeLink(q).cfg.symmetry).toEqual([true, true, true]);
  });

  it('is written into the document a run stands for', () => {
    const cfg = configFor(problemById('evrard'), 32, { symmetry: [true, true, true] });
    const spec = runningSpec(cfg);
    expect(spec.domain?.mirror).toBe('xyz');
    expect(spec.domain?.bc).toEqual({ lo: [1, 1, 1], hi: [1, 1, 1] });
  });
});
