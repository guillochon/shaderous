import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { coolRate, defaultCoolingParams, integrateCooling, ismBalanceDensity, packCooling } from '../../src/cooling/cooling';
import { balanceTemperature } from '../../src/problems/thermal_instability';
import { binaryMasses } from '../../src/problems/binary';
import { configFor, initialMasses, MASS_FLOATS, packMasses, unpackMasses } from '../../src/sim';
import { tidalDisruption } from '../../src/problems/tidal_disruption';

const eos = makeEos('ideal', { ...defaultEosParams(), gamma: 5 / 3 });

describe('cooling (float64 mirror of hydro/cooling.wgsl)', () => {
  it('follows the power-law solution T^(1 - a) = T0^(1 - a) - (1 - a) (gamma - 1) rho L0 t', () => {
    for (const alpha of [0.5, -0.5, 2]) {
      const p = { ...defaultCoolingParams(), rate: 1, alpha, heat: 0 };
      const rho = 2;
      let e = eos.eintFromT(rho, 1);
      const dt = 0.002;
      const steps = 100;
      for (let i = 0; i < steps; i++) e = integrateCooling('power', p, eos, rho, e, dt);
      const k = (2 / 3) * rho;
      const exact = Math.pow(1 - (1 - alpha) * k * dt * steps, 1 / (1 - alpha));
      expect(Math.abs(eos.temperature(rho, e) - exact) / exact).toBeLessThan(1e-5);
    }
  });

  it('settles at thermal balance when the cooling time is far below the step, without overshoot', () => {
    const p = { ...defaultCoolingParams(), rate: 1e6, alpha: 0.5, heat: 5e5 };
    let e = eos.eintFromT(1, 1);
    for (let i = 0; i < 5; i++) {
      e = integrateCooling('power', p, eos, 1, e, 0.01);
      expect(eos.temperature(1, e)).toBeGreaterThan(0.2499);
    }
    expect(Math.abs(eos.temperature(1, e) - 0.25)).toBeLessThan(1e-4);
    // And from below: heating up to the same balance.
    e = eos.eintFromT(1, 0.01);
    for (let i = 0; i < 40; i++) e = integrateCooling('power', p, eos, 1, e, 0.01);
    expect(Math.abs(eos.temperature(1, e) - 0.25)).toBeLessThan(1e-4);
  });

  it('does not cool below the temperature floor, and leaves the gas alone when off', () => {
    const p = { ...defaultCoolingParams(), rate: 100, alpha: 0.5, heat: 0, tFloor: 0.3 };
    let e = eos.eintFromT(1, 1);
    for (let i = 0; i < 50; i++) e = integrateCooling('power', p, eos, 1, e, 0.01);
    expect(eos.temperature(1, e)).toBeCloseTo(0.3, 6);
    expect(integrateCooling('off', p, eos, 1, 1.5, 1)).toBe(1.5);
    expect(packCooling('off', p)).toEqual([0, 0, 0, 0]);
  });

  it('interstellar curve: balance density falls with temperature, and the preset starts in balance', () => {
    const p = { ...defaultCoolingParams(), heat: 50, tUnit: 1000 };
    expect(ismBalanceDensity(p, 0.05)).toBeGreaterThan(ismBalanceDensity(p, 1.5));
    expect(ismBalanceDensity(p, 1.5)).toBeGreaterThan(ismBalanceDensity(p, 8));
    for (const n of [1, 2, 5]) {
      const T = balanceTemperature(n);
      expect(Math.abs(coolRate('ism', p, n, T))).toBeLessThan(1e-9 * p.heat * n);
      // Unstable at constant pressure: d ln(Lambda) / d ln(T) < 1 there.
      const slope = Math.log((1 / ismBalanceDensity(p, T * 1.01)) / (1 / ismBalanceDensity(p, T))) / Math.log(1.01);
      expect(slope).toBeLessThan(1);
    }
  });
});

describe('point masses', () => {
  it('pack and unpack round-trip, with defaults from the cell size', () => {
    const cfg = configFor(tidalDisruption, 64);
    const m = initialMasses(cfg, 0.125);
    expect(m).toHaveLength(1);
    expect(m[0].soft).toBe(0.25);
    expect(m[0].sinkRadius).toBe(0.25);
    const f = packMasses(m);
    expect(f).toHaveLength(MASS_FLOATS);
    const back = unpackMasses(f, 1)[0];
    expect(back.gm).toBeCloseTo(m[0].gm, 4);
    expect(back.pos[0]).toBeCloseTo(m[0].pos[0], 5);
    expect(back.vel[1]).toBeCloseTo(m[0].vel[1], 5);
  });

  it('a snapshot’s masses are laid over the preset’s', () => {
    const cfg = configFor(tidalDisruption, 64, { restart: { time: 1, steps: 10, masses: [{ pos: [1, 2, 3], vel: [4, 5, 6], accreted: 0.5 }] } });
    const m = initialMasses(cfg, 0.125)[0];
    expect(m.pos).toEqual([1, 2, 3]);
    expect(m.accreted).toBe(0.5);
    expect(m.gm).toBe(100);
  });

  it('the binary starts at rest in its centre-of-mass frame', () => {
    for (const q of [1, 0.3]) {
      const [a, b] = binaryMasses(q, 0.2, 0.05);
      expect(a.gm + b.gm).toBeCloseTo(1, 12);
      for (const i of [0, 1]) {
        expect(a.gm * a.pos[i] + b.gm * b.pos[i]).toBeCloseTo(0, 12);
        expect(a.gm * a.vel[i] + b.gm * b.vel[i]).toBeCloseTo(0, 12);
      }
    }
  });
});
