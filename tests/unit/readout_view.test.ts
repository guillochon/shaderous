import { describe, expect, it } from 'vitest';
import { SeriesLog } from '../../src/ui/readout_view';

const v = (raw: number[], scale = 1, unit = '') => ({ label: 'Velocity', raw, scale, unit });

describe('the query readout series', () => {
  it('appends a reading per step and replaces it while paused', () => {
    const log = new SeriesLog();
    log.push(0, 10, new Map([['row:vel', v([1, 2])]]));
    log.push(0.1, 10, new Map([['row:vel', v([3, 4])]]));
    log.push(0.2, 11, new Map([['row:vel', v([5, 6])]]));
    expect(log.t).toEqual([0.1, 0.2]);
    expect(log.series('row:vel')![1]).toEqual([[3, 5], [4, 6]]);
  });

  it('a series that appears later is NaN before it; display scale applied', () => {
    const log = new SeriesLog();
    log.push(0, 1, new Map([['a', v([1])]]));
    log.push(1, 2, new Map([['a', v([2])], ['b', { label: 'Density', raw: [7], scale: 10, unit: 'g/cm3' }]]));
    const [, b] = log.series('b')!;
    expect(Number.isNaN(b[0][0])).toBe(true);
    expect(b[0][1]).toBe(70);
  });

  it('CSV: time, step, a column per component with its unit', () => {
    const log = new SeriesLog();
    log.push(0.5, 3, new Map([['row:vel', v([1, 2], 2, 'km/s')], ['eq:mass:flux', { label: 'Mass: carried, in or out', raw: [100], scale: 1, unit: '%' }]]));
    const [head, row] = log.csv().trim().split('\n');
    expect(head).toBe('t,step,Velocity x (km/s),Velocity y (km/s),"Mass: carried, in or out (%)"');
    expect(row).toBe('0.5,3,2,4,100');
  });
});
