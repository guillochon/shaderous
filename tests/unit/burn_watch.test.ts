// Watching for burning (burn/watch.ts, burn/network.ts): the energy rate of a cell of he4, c12
// and o16 against the trained network, the setups that watch, and the switch to all thirteen
// species at a restart.
import { describe, expect, it } from 'vitest';
import { netEnergy, netEnergyRate, netStep, netWatchSlot, NET_SPECIES, watchesOnly, WATCH_SPECIES } from '../../src/burn/network';
import { expandSpecies, switchedSetup, SWITCH_AT } from '../../src/burn/watch';
import { checkSetup } from '../../src/setup/check';
import { compileSetup } from '../../src/setup/compile';
import type { SetupSpec } from '../../src/setup/spec';

const X = (he: number, c: number, o: number) => NET_SPECIES.map((s) => (s === 'he4' ? he : s === 'c12' ? c : s === 'o16' ? o : 0));

describe('the energy rate of a cell of helium, carbon and oxygen', () => {
  it('is what the trained network releases over a short step, within 30%', () => {
    for (const [rho, T, x] of [[1e6, 3e8, X(1, 0, 0)], [1e6, 6e8, X(1, 0, 0)], [3e5, 1e9, X(1, 0, 0)], [1e7, 1.5e9, X(0, 0.5, 0.5)], [1e6, 8e8, X(0.5, 0.25, 0.25)]] as const) {
      const eps = netEnergyRate(Math.log10(rho), Math.log10(T), x);
      const dt = 1e-6;
      const released = netEnergy(netStep(Math.log10(rho), Math.log10(T), Math.log10(dt), x));
      expect(released / (eps * dt)).toBeGreaterThan(0.7);
      expect(released / (eps * dt)).toBeLessThan(1.3);
    }
  });

  it('rises steeply with temperature, and is nothing in cold helium', () => {
    const at = (T: number) => netEnergyRate(6, Math.log10(T), X(1, 0, 0));
    expect(at(6e8) / at(3e8)).toBeGreaterThan(30);
    expect(at(2e7)).toBeLessThan(1e-3 * at(3e8));
  });

  it('is generated for the GPU from the eight reactions a cell of these three species runs', () => {
    const slot = netWatchSlot();
    expect(slot.match(/^ {2}e \+= /gm)).toHaveLength(8);
    expect(slot).toContain('fn net_watch_rate(rho: f32, T: f32, xhe: f32, xc: f32, xo: f32) -> f32');
  });
});

describe('setups that watch for burning', () => {
  const watcher = (): SetupSpec => ({
    format: 'shaderous-setup',
    version: 1,
    domain: { ndim: 3, n: 32, amr: 0, box: 0.2, units: { name: 'white dwarf', length: { unit: 'm', value: 1e7 }, mass: { unit: 'Msun', value: 1 } } },
    physics: { eos: 'helmholtz', gravity: 'off', composition: 'carbonOxygen', species: ['he4', 'c12', 'o16'], burn: { network: 'aprox13', switchAt: 0.2 } },
    background: { rho: 1, p: 0.05, X: { he4: 1 } },
  });

  it('are he4, c12 and o16 with the network’s burning, which the checks take', () => {
    expect(WATCH_SPECIES).toEqual(['he4', 'c12', 'o16']);
    expect(NET_SPECIES.slice(0, 3)).toEqual(WATCH_SPECIES);
    expect(watchesOnly(['he4', 'c12', 'o16'])).toBe(true);
    expect(watchesOnly([...NET_SPECIES])).toBe(false);
    expect(watchesOnly(['he4', 'c12'])).toBe(false);
    expect(checkSetup(watcher()).filter((i) => i.level === 'error')).toEqual([]);
    expect(compileSetup(watcher()).problem.burn!({})).toMatchObject({ network: 'aprox13', switchAt: 0.2 });
    expect(SWITCH_AT).toBe(0.1);
    // Other species with the network: not a watch, and refused.
    const odd = { ...watcher(), physics: { ...watcher().physics, species: ['he4', 'c12'] } };
    expect(checkSetup(odd).some((i) => i.level === 'error' && /burn/i.test(i.path ?? ''))).toBe(true);
  });

  it('switch to all thirteen species: the setup, and the snapshot’s arrays', () => {
    const spec = switchedSetup(watcher());
    expect(spec.physics!.species).toBe('aprox13');
    expect(spec.physics!.burn).toEqual({ network: 'aprox13', switchAt: 0.2 });
    expect(checkSetup(spec).filter((i) => i.level === 'error')).toEqual([]);
    expect(() => switchedSetup({ ...watcher(), physics: { ...watcher().physics, species: 'aprox13' } })).toThrow(/carries he4, c12, o16 alone/);

    const data: Record<string, Float32Array> = { rho: new Float32Array([1, 2]), he4: new Float32Array([0.9, 0.2]), c12: new Float32Array([0.1, 0.4]), o16: new Float32Array([0, 0.4]) };
    const added = expandSpecies(data);
    expect(added).toEqual(NET_SPECIES.slice(3));
    for (const k of NET_SPECIES.slice(3)) expect(Array.from(data[k])).toEqual([0, 0]);
    expect(Array.from(data.c12)).toEqual([Math.fround(0.1), Math.fround(0.4)]);
    expect(() => expandSpecies({ rho: new Float32Array(2) })).toThrow(/no he4, c12, o16/);
  });
});
