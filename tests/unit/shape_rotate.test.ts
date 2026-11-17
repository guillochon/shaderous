// A box shape turned by `rotate`: the cells it holds, and the angles read back from its matrix.
import { describe, expect, it } from 'vitest';
import { eulerOf, insideOf, rotationMatrix } from '../../src/setup/parts/region';
import type { Shape } from '../../src/setup/spec';

const ctx3 = { box: 1, n: 64, ndim: 3 as const };
const ctx2 = { box: 1, n: 64, ndim: 2 as const };

describe('turned box shapes', () => {
  it('holds the same points as before when not turned', () => {
    const plain: Shape = { type: 'box', min: [-0.2, -0.05, -0.05], max: [0.2, 0.05, 0.05] };
    const zero: Shape = { ...plain, rotate: [0, 0, 360] };
    const a = insideOf(plain, ctx3).inside;
    const b = insideOf(zero, ctx3).inside;
    const points: [number, number, number][] = [[0, 0, 0], [0.15, 0, 0], [0, 0.15, 0], [0.19, 0.04, -0.04], [0.3, 0, 0]];
    for (const p of points) expect(b(...p)).toBe(a(...p));
  });

  it('turns a long thin box about its center by z', () => {
    // Along x, 0.4 long; turned 90 degrees about z it lies along y.
    const turned: Shape = { type: 'box', min: [0.1, -0.05, -0.05], max: [0.5, 0.05, 0.05], rotate: [0, 0, 90] };
    const inside = insideOf(turned, ctx3).inside;
    expect(inside(0.3, 0, 0)).toBe(true); // the center stays
    expect(inside(0.3, 0.15, 0)).toBe(true);
    expect(inside(0.3, -0.15, 0)).toBe(true);
    expect(inside(0.45, 0, 0)).toBe(false);
    expect(inside(0.15, 0, 0)).toBe(false);
    // The same in 2D, where z is 0 at every cell.
    const flat = insideOf(turned, ctx2).inside;
    expect(flat(0.3, 0.15, 0)).toBe(true);
    expect(flat(0.45, 0, 0)).toBe(false);
  });

  it('turns by 45 degrees: the corners of the square become its diamond', () => {
    const sq: Shape = { type: 'box', min: [-0.1, -0.1, -0.1], max: [0.1, 0.1, 0.1], rotate: [0, 0, 45] };
    const inside = insideOf(sq, ctx3).inside;
    expect(inside(0.13, 0, 0)).toBe(true); // half diagonal 0.141
    expect(inside(0.09, 0.09, 0)).toBe(false); // was a corner
  });

  it('ignores turns about x and y in 2D', () => {
    const s: Shape = { type: 'box', min: [-0.2, -0.05, -0.05], max: [0.2, 0.05, 0.05], rotate: [90, 90, 0] };
    const inside = insideOf(s, ctx2).inside;
    expect(inside(0.15, 0, 0)).toBe(true);
    expect(inside(0, 0.15, 0)).toBe(false);
  });

  it('reads back the angles of a rotation matrix', () => {
    for (const deg of [[10, 20, 30], [-45, 5, 170], [0, 0, -90], [30, -60, 0]]) {
      const back = eulerOf(rotationMatrix(deg));
      back.forEach((v, i) => expect(v).toBeCloseTo(deg[i], 9));
    }
  });
});
