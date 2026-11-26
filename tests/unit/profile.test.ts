import { describe, expect, it } from 'vitest';
import { clipSegment, decodeProfileLines, distanceAt, distances, encodeProfileLines, MAX_PROFILES, plotLength, profileCsv, resolveField, sampleCount, sampleNear, samplePoints, seriesRange, ticks, valueRange, PLOT_MARGIN } from '../../src/ui/profile';
import { decodeLink, encodeLink, viewDefaults, type LinkReport } from '../../src/ui/link';
import { configFor } from '../../src/sim';
import { problemOf } from '../../src/presets/lookup';
import { defaultOpacity } from '../../src/render/transfer';
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

  it('finds the distance under the pointer, and each line\'s sample there', () => {
    const w = 200;
    const pw = w - PLOT_MARGIN.left - PLOT_MARGIN.right;
    expect(distanceAt(PLOT_MARGIN.left, w, 4)).toBe(0);
    expect(distanceAt(PLOT_MARGIN.left + pw, w, 4)).toBe(4);
    expect(distanceAt(PLOT_MARGIN.left + pw * 0.52, w, 4)).toBeCloseTo(2.08);
    expect(distanceAt(5, w, 4)).toBeNull();
    const s = [0, 1, 2, 3, 4];
    expect(sampleNear(s, 0)).toBe(0);
    expect(sampleNear(s, 2.08)).toBe(2);
    expect(sampleNear(s, 4.4)).toBe(4);
    // Past a shorter line's end: none of its samples.
    expect(sampleNear(s, 4.6)).toBe(-1);
  });

  it('fits the y range to every line, and the x axis to the longest', () => {
    expect(seriesRange([{ values: [1, 2] }, { values: [3] }], false)).toEqual([0.9, 3.1]);
    expect(seriesRange([{ values: [1, 2] }, { values: [3] }], false)).toEqual(valueRange([1, 2, 3], false));
    expect(plotLength([{ s: [0, 0.5] }, { s: [0, 0.25, 0.75] }])).toBe(0.75);
  });

  it('writes CSV with its header quoted where it must be', () => {
    expect(profileCsv([{ s: [0, 1], values: [[2, 3]], levels: [0, 1] }], { distance: 'distance (cm)', values: ['X(He, 4)'] })).toBe('distance (cm),"X(He, 4)",level\n0,2,0\n1,3,1\n');
  });

  it('puts each field in a column of its own, empty where a line has none', () => {
    const csv = profileCsv([{ s: [0, 1], values: [[2, 3], [7, 8], [9, 10]], levels: [0, 0] }, { s: [0], values: [[5], null, [11]], levels: [1] }], { distance: 'distance', values: ['Density', 'Pressure', 'Mach number'] });
    expect(csv).toBe('line,distance,Density,Pressure,Mach number,level\n1,0,2,7,9,0\n1,1,3,8,10,0\n2,0,5,,11,1\n');
  });

  it('numbers the lines in CSV when there are several', () => {
    const csv = profileCsv([{ s: [0, 1], values: [[2, 3]], levels: [0, 0] }, { s: [0], values: [[5]], levels: [1] }], { distance: 'distance', values: ['Density'] });
    expect(csv).toBe('line,distance,Density,level\n1,0,2,0\n1,1,3,0\n2,0,5,1\n');
  });
});

describe('profile lines in a link', () => {
  it('writes and reads lines in 2D and 3D', () => {
    const lines = [{ a: [0.1, 0.2, 0], b: [0.9, 0.2, 0] }, { a: [0.5, 0, 0], b: [0.5, 1, 0] }];
    expect(encodeProfileLines(lines, 2)).toBe('0.1,0.2,0.9,0.2;0.5,0,0.5,1');
    expect(decodeProfileLines('0.1,0.2,0.9,0.2;0.5,0,0.5,1', 2)).toEqual([[[0.1, 0.2, 0], [0.9, 0.2, 0]], [[0.5, 0, 0], [0.5, 1, 0]]]);
    expect(decodeProfileLines('0.1,0.2,0.3,0.9,0.8,0.7', 3)).toEqual([[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]]);
    expect(encodeProfileLines([{ a: [1 / 3, 0, 0.5], b: [1, 1, 0.5] }], 3)).toBe('0.3333,0,0.5,1,1,0.5');
  });

  it('refuses what is not a line of the box', () => {
    expect(decodeProfileLines('0.1,0.2,0.9', 2)).toBeNull(); // too few numbers
    expect(decodeProfileLines('0.1,0.2,0.9,0.2', 3)).toBeNull(); // a 2D line in 3D
    expect(decodeProfileLines('0.1,0.2,1.5,0.2', 2)).toBeNull(); // outside the box
    expect(decodeProfileLines('0.5,0.5,0.5,0.5', 2)).toBeNull(); // a point
    expect(decodeProfileLines('0.1,x,0.9,0.2', 2)).toBeNull();
    expect(decodeProfileLines(Array(MAX_PROFILES + 1).fill('0,0,1,1').join(';'), 2)).toBeNull();
  });

  it('round-trips through a whole link, with the field and its scale', () => {
    const problem = problemOf('sedov');
    const cfg = configFor(problem, 256, { ndim: 2, amrLevels: 0 });
    const view = viewDefaults(problem, 2);
    const q = encodeLink({ cfg, view, auto: true, colormap: 'inferno', opacity: defaultOpacity(), profiles: { lines: [[[0.5, 0.5, 0], [1, 0.5, 0]], [[0.5, 0.5, 0], [1, 1, 0]]], fields: [{ field: 3, species: 0, log: false }] } });
    expect(q.get('prof')).toBe('0.5,0.5,1,0.5;0.5,0.5,1,1');
    expect(q.get('proff')).toBe('3');
    expect(q.get('proflog')).toBe('0');
    expect(q.has('profsp')).toBe(false);
    const report: LinkReport = { ignored: [] };
    const back = decodeLink(q, report);
    expect(report.ignored).toEqual([]);
    expect(back.profiles).toEqual({ lines: [[[0.5, 0.5, 0], [1, 0.5, 0]], [[0.5, 0.5, 0], [1, 1, 0]]], fields: [{ field: 3, log: false }] });
    // Several fields along the same lines, their scales where set.
    const fields = [{ field: 0, log: null }, { field: 3, log: true }, { field: 8, log: null }];
    const q2 = encodeLink({ cfg, view, auto: true, colormap: 'inferno', opacity: defaultOpacity(), profiles: { lines: [[[0.5, 0.5, 0], [1, 0.5, 0]]], fields } });
    expect([q2.get('proff'), q2.get('proflog'), q2.has('profsp')]).toEqual(['0,3,8', ',1,', false]);
    const r2: LinkReport = { ignored: [] };
    expect(decodeLink(q2, r2).profiles).toEqual({ lines: [[[0.5, 0.5, 0], [1, 0.5, 0]]], fields: [{ field: 0 }, { field: 3, log: true }, { field: 8 }] });
    expect(r2.ignored).toEqual([]);
    // A second field as links had it for a while (proff2), read after the first.
    const r3: LinkReport = { ignored: [] };
    expect(decodeLink(new URLSearchParams('problem=sedov&dim=2&n=256&prof=0.5,0.5,1,0.5&proff=0&proff2=3&proflog2=1'), r3).profiles?.fields).toEqual([{ field: 0 }, { field: 3, log: true }]);
    expect(r3.ignored).toEqual([]);
    // A list that is not one is reported.
    const r4: LinkReport = { ignored: [] };
    expect(decodeLink(new URLSearchParams('problem=sedov&dim=2&n=256&prof=0.5,0.5,1,0.5&proff=0,x'), r4).profiles?.fields).toBeUndefined();
    expect(r4.ignored.join()).toMatch(/^proff=/);
    // No lines, no keys.
    expect(encodeLink({ cfg, view, auto: true, colormap: 'inferno', opacity: defaultOpacity() }).has('prof')).toBe(false);
  });

  it('reports a line it cannot read and keeps the rest of the link', () => {
    const report: LinkReport = { ignored: [] };
    const back = decodeLink(new URLSearchParams('problem=sedov&dim=2&n=256&prof=0.5,0.5,2,0.5'), report);
    expect(back.profiles).toBeUndefined();
    expect(back.n).toBe(256);
    expect(report.ignored.join()).toMatch(/^prof=/);
  });
});
