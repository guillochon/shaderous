import { describe, expect, it } from 'vitest';
import { ASTRO, SI } from '../../src/units/constants';
import { COMPOSITIONS, composition, meanMolecularWeight, nuclearAverages } from '../../src/units/composition';
import { codeG, display, displayTemperature, eosConstants, flowUnits, fromCode, gravityUnits, kelvinPerCodeT, toCode } from '../../src/units/units';
import { TDE, TDE_STAR, TDE_UNITS, tidalDisruption } from '../../src/problems/tidal_disruption';

describe('composition', () => {
  it('gives the textbook mean molecular weights', () => {
    expect(meanMolecularWeight(COMPOSITIONS.hydrogen)).toBeCloseTo(0.5, 12);
    expect(meanMolecularWeight(COMPOSITIONS.solar)).toBeCloseTo(0.599, 3);
    expect(meanMolecularWeight(COMPOSITIONS.solarNeutral)).toBeCloseTo(1.248, 3);
    expect(meanMolecularWeight(composition('he', { he4: 1 }))).toBeCloseTo(4 / 3, 12);
  });

  it('normalizes mass fractions and takes partial ionization per species', () => {
    const c = composition('h', { h1: 2 }, [0.5]);
    expect(c.X).toEqual([1]);
    expect(meanMolecularWeight(c)).toBeCloseTo(1 / 1.5, 12);
  });

  it('gives network averages', () => {
    const { abar, zbar, ye } = nuclearAverages(COMPOSITIONS.carbonOxygen);
    expect(ye).toBeCloseTo(0.5, 12);
    expect(abar).toBeCloseTo(1 / (0.5 / 12 + 0.5 / 16), 12);
    expect(zbar / abar).toBeCloseTo(ye, 12);
    // Mass fractions can be passed in, as a per-cell evaluation will.
    expect(nuclearAverages(COMPOSITIONS.carbonOxygen, [1, 0]).abar).toBeCloseTo(12, 12);
  });
});

describe('units', () => {
  it('sets G = 1 for gravity units and round-trips quantities', () => {
    const u = gravityUnits('solar', ASTRO.Rsun, ASTRO.Msun);
    expect(codeG(u)).toBeCloseTo(1, 12);
    expect(u.time).toBeCloseTo(1593, 0);
    expect(fromCode(u, 'density', toCode(u, 'density', 1410))).toBeCloseTo(1410, 9);
    expect(fromCode(u, 'velocity', 1)).toBeCloseTo(Math.sqrt(ASTRO.GMsun / ASTRO.Rsun), 3);
  });

  it('shows values in SI and cgs', () => {
    const u = flowUnits('ism', ASTRO.pc, 1.67e-21, 1e3);
    expect(display(u, 'density', 1, 'si')).toEqual({ value: expect.closeTo(1.67e-21, 30), unit: 'kg/m³' });
    expect(display(u, 'density', 1, 'cgs').value).toBeCloseTo(1.67e-24, 30);
    expect(display(u, 'pressure', 1, 'cgs').value / display(u, 'pressure', 1, 'si').value).toBeCloseTo(10, 9);
    expect(display(undefined, 'density', 3, 'cgs')).toEqual({ value: 3, unit: '' });
  });

  it('keeps the kelvin scale apart from the composition', () => {
    const u = flowUnits('kms', 1, 1, 1e3);
    // 1 km/s squared, times m_u / k_B: about 120 K for mu = 1.
    expect(kelvinPerCodeT(u, 1)).toBeCloseTo((SI.mu * 1e6) / SI.kB, 9);
    expect(kelvinPerCodeT(u, 0.5)).toBeCloseTo(kelvinPerCodeT(u, 1) / 2, 9);
  });

  it('puts the radiation constant in code units', () => {
    const u = gravityUnits('solar', ASTRO.Rsun, ASTRO.Msun);
    const { R, arad } = eosConstants(u, COMPOSITIONS.solar);
    expect(R).toBe(1);
    // a T^4 in code units, at code T = 1, matches a_SI T_K^4 in SI.
    const tk = kelvinPerCodeT(u, meanMolecularWeight(COMPOSITIONS.solar));
    expect(fromCode(u, 'pressure', arad) / (SI.arad * tk ** 4)).toBeCloseTo(1, 12);
  });
});

describe('tidal disruption in solar units', () => {
  it('is the Sun', () => {
    expect(tidalDisruption.units).toBe(TDE_UNITS);
    expect(tidalDisruption.G).toBeCloseTo(TDE.G, 12);
    // Central density of an n = 3/2 polytrope: 5.99 times the mean, 1410 kg/m^3 for the Sun.
    const rhoc = fromCode(TDE_UNITS, 'density', TDE_STAR.rhoc);
    expect(rhoc / 1409.8).toBeCloseTo(5.991, 2);
    // Central temperature: 0.539 G M mu m_u / (k_B R) (Chandrasekhar, n = 3/2).
    const Tc = displayTemperature(TDE_UNITS, meanMolecularWeight(COMPOSITIONS.solar), TDE_STAR.K * TDE_STAR.rhoc ** (1 / TDE.n), 'si').value;
    const virial = (ASTRO.GMsun * meanMolecularWeight(COMPOSITIONS.solar) * SI.mu) / (SI.kB * ASTRO.Rsun);
    expect(Tc / virial).toBeCloseTo(0.539, 2);
  });
});
