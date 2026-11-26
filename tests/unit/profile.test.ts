import { describe, expect, it } from 'vitest';
import { clipSegment, distances, profileCsv, resolveField, sampleAt, sampleCount, samplePoints, ticks, valueRange, PLOT_MARGIN } from '../../src/ui/profile';
import { FIELDS, MASS_FRACTION_FIELD } from '../../src/problems/types';

describe('clipSegment', () => {
  it('keeps a segment inside the box as it is', () => {
    expect(clipSegment([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], 3)).toEqual([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]);
  });

  it('cuts a segment at the faces it crosses', () => {
    const c = clipSegment([-0.5, 0.5], [1.5, 0.5], 2)!;
    expect(c[0][0]).toBeCloseTo(0, 12);
    expect(c[1][0]).toBeCloseTo(1, 12);
    expect(c[0][1]).toBe(0.5);
    // In 2D the third coordinate is 0, whatever was given.
    expect(c[0][2]).toBe(0);
    const d = clipSegment([-1, -1, 0.5], [2, 2, 0.5], 3)!;
    expect(d[0].map((v) => +v.toFixed(12))).toEqual([0, 0, 0.5]);
    expect(d[1].map((v) => +v.toFixed(12))).toEqual([1, 1, 0.5]);
  });

  it('refuses a segment that misses the box, or a point', () => {
    expect(clipSegment([1.2, 0.1], [1.5, 0.9], 2)).toBeNull();
    expect(clipSegment([-0.5, 1.2], [1.2, 2], 2)).toBeNull();
    expect(clipSegment([0.3, 0.3, 0.3], [0.3, 0.3, 0.3], 3)).toBeNull();
    expect(clipSegment([0.3, NaN], [0.5, 0.5], 2)).toBeNull();
  });
});

describe('sampling', () => {
  it('takes two samples a cell, within bounds, or as many as asked', () => {
    expect(sampleCount(10)).toBe(64);
    expect(sampleCount(300)).toBe(600);
    expect(sampleCount(1e6)).toBe(4096);
    expect(sampleCount(10, 5)).toBe(5);
    expect(sampleCount(10, 1)).toBe(2);
    expect(sampleCount(10, 1e9)).toBe(16384);
    expect(() => sampleCount(10, NaN)).toThrow();
  });

  it('places samples evenly from end to end', () => {
    const p = samplePoints([0, 0, 0], [1, 0.5, 0], 5);
    expect(p).toHaveLength(5);
    expect(p[0]).toEqual([0, 0, 0]);
    expect(p[4]).toEqual([1, 0.5, 0]);
    expect(p[2]).toEqual([0.5, 0.25, 0]);
  });

  it('measures distance along the line', () => {
    const s = distances([0, 0], [3, 4], 6);
    expect(s[0]).toBe(0);
    expect(s[5]).toBe(5);
    expect(s[1]).toBeCloseTo(1, 12);
  });
});

describe('resolveField', () => {
  const fb = { field: 4, species: 0 };
  it('takes an index, a name, a variable or a species', () => {
    expect(resolveField(undefined, [], fb)).toEqual(fb);
    expect(resolveField(8, [], fb)).toEqual({ field: 8, species: 0 });
    expect(resolveField('8', [], fb)).toEqual({ field: 8, species: 0 });
    expect(resolveField('mach number', [], fb)).toEqual({ field: FIELDS.indexOf('Mach number'), species: 0 });
    expect(resolveField('rho', [], fb)).toEqual({ field: 0, species: 0 });
    expect(resolveField('press', [], fb)).toEqual({ field: 3, species: 0 });
    expect(resolveField('vel2', [], fb)).toEqual({ field: 6, species: 0 });
    expect(resolveField('he4', ['h1', 'he4'], fb)).toEqual({ field: MASS_FRACTION_FIELD, species: 1 });
  });

  it('refuses what it does not know', () => {
    expect(() => resolveField('nonsense', [], fb)).toThrow(/unknown field/);
    expect(() => resolveField(FIELDS.length, [], fb)).toThrow();
    expect(() => resolveField(MASS_FRACTION_FIELD, [], fb)).toThrow();
    expect(() => resolveField('Mass fraction', ['h1'], fb)).toThrow();
  });
});

describe('the plot', () => {
  it('pads the value range, and leaves out what it cannot show', () => {
    expect(valueRange([1, 2, 3], false)).toEqual([0.9, 3.1]);
    const [lo, hi] = valueRange([0.1, 10, -5, 0], true)!;
    expect(lo).toBeCloseTo(-1.1, 12);
    expect(hi).toBeCloseTo(1.1, 12);
    expect(valueRange([-1, 0], true)).toBeNull();
    expect(valueRange([5, 100], false, [0, 1])).toEqual([4.75, 5.25]);
  });

  it('puts ticks at round numbers', () => {
    expect(ticks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(ticks(0, 1, 4)).toEqual([0, 0.5, 1]);
    expect(ticks(-3, 7, 5)).toEqual([-2, 0, 2, 4, 6]);
    expect(ticks(1e-3, 3.1e-3, 4)).toEqual([0.001, 0.002, 0.003]);
    expect(ticks(1, 1)).toEqual([]);
  });

  it('finds the sample under the pointer', () => {
    const s = [0, 1, 2, 3, 4];
    const w = 200;
    const pw = w - PLOT_MARGIN.left - PLOT_MARGIN.right;
    expect(sampleAt(PLOT_MARGIN.left, w, s)).toBe(0);
    expect(sampleAt(PLOT_MARGIN.left + pw, w, s)).toBe(4);
    expect(sampleAt(PLOT_MARGIN.left + pw * 0.52, w, s)).toBe(2);
    expect(sampleAt(5, w, s)).toBe(-1);
  });

  it('writes CSV with its header quoted where it must be', () => {
    expect(profileCsv([0, 1], [2, 3], [0, 1], { distance: 'distance (cm)', value: 'X(He, 4)' })).toBe('distance (cm),"X(He, 4)",level\n0,2,0\n1,3,1\n');
  });
});
