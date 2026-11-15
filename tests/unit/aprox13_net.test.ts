import { describe, expect, it } from 'vitest';
import CASES from '../../src/burn/aprox13_net_cases.json';
import { NET_SPECIES, netEnergy, netFeatures, netSlot, netStep, netWeights } from '../../src/burn/network';

// Steps of the trained network computed by PyTorch (tools/burn/export_net.py), with the same
// half-precision weights: the float64 mirror must reproduce its inputs and its steps.
interface Case {
  logRho: number;
  logT: number;
  logDt: number;
  X: number[];
  features: number[];
  D: number[];
}
const cases = CASES as Case[];

describe('aprox13 network', () => {
  it('has the weights of 61 -> 4 x 128 -> 54', () => {
    expect(netWeights().length).toBe(61 * 128 + 128 + 3 * (128 * 128 + 128) + 128 * 54 + 54);
    expect(NET_SPECIES.length).toBe(13);
  });

  it('computes the inputs as training did', () => {
    for (const c of cases) {
      const f = netFeatures(c.logRho, c.logT, c.logDt, c.X);
      expect(f.length).toBe(61);
      for (let i = 0; i < 61; i++) expect(Math.abs(f[i] - c.features[i])).toBeLessThan(2e-4);
    }
  });

  it('steps the composition as PyTorch does', () => {
    for (const c of cases) {
      const D = netStep(c.logRho, c.logT, c.logDt, c.X);
      const size = c.D.reduce((s, d) => s + Math.abs(d), 0);
      const err = D.reduce((s, d, i) => s + Math.abs(d - c.D[i]), 0);
      expect(err).toBeLessThan(1e-6 + 2e-3 * size);
      // Mass is conserved whatever the weights.
      expect(Math.abs(D.reduce((s, d) => s + d, 0))).toBeLessThan(1e-12);
    }
  });

  it('releases energy burning carbon and oxygen', () => {
    const X = [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const D = netStep(7, Math.log10(3e9), -3, X);
    expect(D[1]).toBeLessThan(-0.1);
    expect(netEnergy(D)).toBeGreaterThan(1e17);
  });

  it('generates its WGSL', () => {
    const s = netSlot();
    expect(s).toContain('fn net_features');
    expect(s.match(/f\[\d+\] = \(clamp/g)!.length).toBe(32);
  });
});
