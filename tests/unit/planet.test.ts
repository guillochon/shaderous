import { beforeAll, describe, expect, it } from 'vitest';
import { loadAneos, type AneosTable } from '../../src/eos/aneos';
import { buildPlanet, EARTH } from '../../src/problems/planet';

let pair: AneosTable;
beforeAll(async () => {
  pair = await loadAneos('pyrolite+fe85si15');
}, 120000);

describe('planets', () => {
  it('an Earth of 30% iron alloy under pyrolite has about the radius of the Earth', () => {
    const M = EARTH.mass * 1e3;
    const pl = buildPlanet(pair, [{ xb: 1, fraction: 0.3 }, { xb: 0, fraction: 0.7 }], M, 2000);
    const R = pl.R / (EARTH.radius * 100);
    console.log('Earth: R', R.toFixed(4), 'rhoc', pl.rhoc.toFixed(2), 'pc GPa', (pl.p[0] / 1e10).toFixed(1), 'Tc', pl.T[0].toFixed(0));
    expect(R).toBeGreaterThan(0.93);
    expect(R).toBeLessThan(1.07);
    // The center: near 360 GPa in the Earth, and denser than 10 g/cm^3.
    expect(pl.p[0] / 1e10).toBeGreaterThan(280);
    expect(pl.p[0] / 1e10).toBeLessThan(450);
    expect(pl.rhoc).toBeGreaterThan(10);
    // The mass adds up, and the core is the inner 30% of it.
    let m = 0;
    let core = 0;
    for (let i = 1; i < pl.r.length; i++) {
      const dm = (4 / 3) * Math.PI * (pl.r[i] ** 3 - pl.r[i - 1] ** 3) * 0.5 * (pl.rho[i] + pl.rho[i - 1]);
      m += dm;
      if (pl.layer[i] === 0) core += dm;
    }
    expect(Math.abs(m / M - 1)).toBeLessThan(0.01);
    expect(Math.abs(core / M - 0.3)).toBeLessThan(0.01);
    // The pressure is the weight above, by differences.
    for (const i of [200, 800, 1500]) {
      let mi = 0;
      for (let k = 1; k <= i; k++) mi += (4 / 3) * Math.PI * (pl.r[k] ** 3 - pl.r[k - 1] ** 3) * 0.5 * (pl.rho[k] + pl.rho[k - 1]);
      const dpdr = (pl.p[i + 1] - pl.p[i - 1]) / (pl.r[i + 1] - pl.r[i - 1]);
      const want = (-6.6743e-8 * mi * pl.rho[i]) / pl.r[i] ** 2;
      if (pl.layer[i - 1] === pl.layer[i + 1]) expect(Math.abs(dpdr / want - 1)).toBeLessThan(0.02);
    }
  });

  it('a Mars-mass impactor is smaller and less pressed', () => {
    const pl = buildPlanet(pair, [{ xb: 1, fraction: 0.3 }, { xb: 0, fraction: 0.7 }], 0.12 * EARTH.mass * 1e3, 2000);
    const R = pl.R / (EARTH.radius * 100);
    console.log('Theia: R', R.toFixed(4), 'rhoc', pl.rhoc.toFixed(2), 'pc GPa', (pl.p[0] / 1e10).toFixed(1));
    expect(R).toBeGreaterThan(0.45);
    expect(R).toBeLessThan(0.62);
  });
});
