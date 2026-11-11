import { describe, expect, it } from 'vitest';
import { FIELD_MENU, FIELDS } from '../../src/problems/types';
import { fieldMeasure, readout } from '../../src/ui/units_view';
import { TDE_UNITS } from '../../src/problems/tidal_disruption';
import { COMPOSITIONS, meanMolecularWeight } from '../../src/units/composition';
import { displayLabel, kelvinPerCodeT, unitOf } from '../../src/units/units';

describe('unit readouts', () => {
  it('assigns a measure to every view field', () => {
    expect(FIELDS.length).toBe(19);
    expect(fieldMeasure(18, 5 / 3)).toBe('time');
    expect(fieldMeasure(0, 5 / 3)).toBe('density');
    expect(fieldMeasure(1, 5 / 3)).toBe('temperature');
    expect(fieldMeasure(17, 5 / 3)).toBeNull();
    // Entropy p / rho^gamma: for gamma = 1 it is a specific energy.
    expect(fieldMeasure(9, 1)).toEqual([0, 2, -2]);
  });

  it('lists every field exactly once in the grouped menu', () => {
    const listed = FIELD_MENU.flatMap((g) => g.fields).sort((a, b) => a - b);
    expect(listed).toEqual(FIELDS.map((_, i) => i));
  });

  it('shows code values unchanged without units or in code', () => {
    const r = readout(undefined, undefined, 'cgs');
    expect(r.physical).toBe(false);
    expect(r.fmt(2, 'density')).toBe('2.00');
    expect(readout(TDE_UNITS, COMPOSITIONS.solar, 'code').label('density')).toBe('');
  });

  it('converts with the problem units', () => {
    const r = readout(TDE_UNITS, COMPOSITIONS.solar, 'cgs');
    expect(r.scale('density')).toBeCloseTo(unitOf(TDE_UNITS, 'density') * 1e-3, 12);
    expect(r.label('density')).toBe('g/cm³');
    expect(r.scale('temperature')).toBeCloseTo(kelvinPerCodeT(TDE_UNITS, meanMolecularWeight(COMPOSITIONS.solar)), 3);
    expect(r.label('temperature')).toBe('K');
    expect(r.scale(null)).toBe(1);
    expect(r.fmt(1, 'time')).toMatch(/ s$/);
  });

  it('labels a bare dimension from its exponents', () => {
    expect(displayLabel(TDE_UNITS, [1, -4, 0], 'si')).toBe('kg m⁻⁴');
    expect(displayLabel(TDE_UNITS, [-2 / 3, 4, -2], 'cgs')).toBe('g⁻⁰·⁶⁷ cm⁴ s⁻²');
  });
});
