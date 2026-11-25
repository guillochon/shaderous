import { describe, expect, it } from 'vitest';
import { burnRates } from '../../src/ui/burn_rates';
import { NET_SPECIES, netUnits } from '../../src/burn/network';

describe('burn rates of the readout species equations', () => {
  it('one reaction: fuel burns to ash at rho X k exp(-Ea / T)', () => {
    const f = burnRates({ fuel: 'f', ash: 'a', q: 50, ea: 25, k: 7 }, { species: ['a', 'f'] }, () => 0.01)!;
    const r = f(2, 5, [0.4, 0.6])!;
    const w = 2 * 0.6 * 7 * Math.exp(-5);
    expect(r[1]).toBeCloseTo(-w, 12);
    expect(r[0]).toBeCloseTo(w, 12);
  });

  it('nothing burns without a burn, or in a run that only watches for it', () => {
    expect(burnRates(null, { species: ['a'] }, () => 0.01)).toBeNull();
    expect(burnRates({ network: 'aprox13' }, { species: ['he4', 'c12', 'o16'] }, () => 0.01)).toBeNull();
  });

  it('the network: helium burns in hot dense gas, and below its floors nothing does', () => {
    // cgs code units: density in g/cm^3, time in seconds.
    const units = { name: 'cgs', length: 0.01, mass: 1e-3, time: 1 };
    const u = netUnits(units);
    const f = burnRates({ network: 'aprox13' }, { species: NET_SPECIES, units }, () => 1e-6);
    expect(f).not.toBeNull();
    const X = NET_SPECIES.map((n) => (n === 'he4' ? 1 : 0));
    const cold = f!(1e6 / u.rho, 1e7 / u.T, X)!;
    expect(cold.every((v) => v === 0)).toBe(true);
    // Helium at 1e7 g/cm^3 and 2e9 K: it burns (he4 used up, heavier species made), mass kept.
    const hot = f!(1e7 / u.rho, 2e9 / u.T, X)!;
    expect(hot[0]).toBeLessThan(0);
    expect(hot.slice(1).some((v) => v > 0)).toBe(true);
    expect(Math.abs(hot.reduce((a, b) => a + b, 0))).toBeLessThan(1e-3 * Math.abs(hot[0]));
  });
});
