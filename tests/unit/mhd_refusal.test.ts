// What the simulators refuse with a magnetic field (mhd/mhd.ts mhdUnsupported), found by the
// random setup testing (tools/random-test).
import { describe, expect, it } from 'vitest';
import { compileSetup } from '../../src/setup/compile';
import { emptySetup, type SetupSpec } from '../../src/setup/spec';
import { configFor, refusal } from '../../src/sim';

const magnetized = (species: string[] | 'aprox13'): SetupSpec => ({
  ...emptySetup(),
  domain: { ndim: 3, n: 64, box: 1 },
  physics: { species },
  parts: [{ kind: 'field', b: [0, 0.05, 0.05] }],
});

describe('a magnetic field with species that set the equation of state', () => {
  it('is refused on the adaptive mesh, whose MHD passes know no composition (random test seed 10113)', () => {
    for (const species of ['aprox13', ['he4', 'c12']] as const) {
      const { problem } = compileSetup(magnetized(species as string[] | 'aprox13'));
      expect(refusal(configFor(problem, 128, { ndim: 3, amrLevels: 2 }))).toMatch(/adaptive mesh/);
      expect(refusal(configFor(problem, 64, { ndim: 3, amrLevels: 0 }))).toBeNull();
    }
  });

  it('is not refused for species that only ride with the gas', () => {
    const { problem } = compileSetup(magnetized(['fuel', 'ash']));
    expect(refusal(configFor(problem, 128, { ndim: 3, amrLevels: 2 }))).toBeNull();
  });
});
