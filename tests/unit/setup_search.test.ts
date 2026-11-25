import { describe, expect, it } from 'vitest';
import { bestRow, fuzzyMatch, searchRows, type SetupEntry } from '../../src/ui/setup_search';

const ENTRIES: SetupEntry[] = [
  { value: 'sod', label: 'Sod shock tube', group: 'Tests', variants: [{ id: 'sod', label: 'Standard' }] },
  { value: 'sedov', label: 'Sedov blast', group: 'Tests', variants: [{ id: 'sedov', label: '2D' }, { id: 'sedov-3d', label: '3D' }] },
  { value: 'kh', label: 'Kelvin-Helmholtz', group: 'Instabilities', variants: [{ id: 'kh', label: 'Standard' }, { id: 'kh-mhd', label: 'Magnetized' }] },
  { value: 'polytrope', label: 'Polytrope', group: 'Stars', variants: [{ id: 'polytrope', label: 'No field' }, { id: 'polytrope-dipole', label: 'Dipole field' }] },
];

describe('fuzzyMatch', () => {
  it('needs every word of the query', () => {
    expect(fuzzyMatch('sedov blast', 'Sedov blast')).not.toBeNull();
    expect(fuzzyMatch('sedov zebra', 'Sedov blast')).toBeNull();
  });
  it('takes word starts as initials', () => {
    expect(fuzzyMatch('kh', 'Kelvin-Helmholtz')!.hits).toEqual([0, 7]);
  });
  it('forgives a typo in a longer word', () => {
    expect(fuzzyMatch('polytorpe', 'Polytrope')).not.toBeNull();
    expect(fuzzyMatch('sedvo', 'Sedov blast')).not.toBeNull();
  });
  it('scores a prefix over a scattered match', () => {
    expect(fuzzyMatch('sod', 'Sod shock tube')!.score).toBeGreaterThan(fuzzyMatch('sod', 'Sedov blast')?.score ?? -Infinity);
  });
});

describe('searchRows', () => {
  it('shows the whole menu under its groups for no query', () => {
    const rows = searchRows(ENTRIES, '  ');
    expect(rows.map((r) => r.value)).toEqual(['sod', 'sedov', 'kh', 'polytrope']);
    expect(rows.map((r) => r.group)).toEqual(['Tests', undefined, 'Instabilities', 'Stars']);
  });
  it('puts the closest setup first', () => {
    expect(searchRows(ENTRIES, 'sedov')[0].value).toBe('sedov');
    expect(searchRows(ENTRIES, 'kelvin')[0].value).toBe('kh');
  });
  it('finds variants by their own names, under a dimmed setup', () => {
    const rows = searchRows(ENTRIES, 'dipole');
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ value: 'polytrope', dim: true });
    expect(rows[1]).toMatchObject({ value: 'polytrope', variant: 'polytrope-dipole' });
    expect(bestRow(rows, 'dipole')).toBe(1);
  });
  it('finds a variant by the setup and variant names together', () => {
    const rows = searchRows(ENTRIES, 'sedov 3d');
    expect(rows[bestRow(rows, 'sedov 3d')]).toMatchObject({ variant: 'sedov-3d' });
  });
  it('matches on the group name, less well than on the label', () => {
    expect(searchRows(ENTRIES, 'stars')[0].value).toBe('polytrope');
  });
  it('prefers initials to the letters inside a word', () => {
    const entries = [{ value: 'wd', label: 'Sub-Chandrasekhar', group: 'Stars', variants: [] }, ...ENTRIES];
    expect(searchRows(entries, 'kh')[0].value).toBe('kh');
  });
  it('does not strew a word across the setup and variant names', () => {
    // "dipole" in order through "Polytrope No field" is not a match of the variant No field.
    const rows = searchRows(ENTRIES, 'dipole');
    expect(rows.some((r) => r.variant === 'polytrope')).toBe(false);
  });
  it('shows nothing when nothing matches', () => {
    expect(searchRows(ENTRIES, 'qqqq')).toEqual([]);
  });
});
