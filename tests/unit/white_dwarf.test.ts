import { describe, expect, it } from 'vitest';
import { physicalEosParams, type EosParams } from '../../src/eos/eos';
import { COMPOSITIONS } from '../../src/units/composition';
import { ASTRO } from '../../src/units/constants';
import { gravityUnits, codeG } from '../../src/units/units';
import { degenerateBarotrope, whiteDwarf } from '../../src/problems/white_dwarf';
import { collisionOrbit } from '../../src/problems/stellar_collision';

// Code units of 1e4 km and one solar mass, with G = 1.
const U = gravityUnits('wd', 1e7, ASTRO.Msun);
const scales = physicalEosParams(U, COMPOSITIONS.carbonOxygen) as EosParams;
const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

describe('cold white dwarfs', () => {
  it('have the mass asked for', () => {
    for (const M of [0.5, 0.8, 1.0, 1.3]) expect(rel(whiteDwarf(M, scales, codeG(U)).M, M)).toBeLessThan(1e-6);
  });

  it('follow the Chandrasekhar mass-radius relation (Nauenberg 1972 fit: 3% to 1 M_sun, 6% at 1.2)', () => {
    // R = 0.0112 R_sun [(M / M_ch)^(-2/3) - (M / M_ch)^(2/3)]^(1/2), M_ch = 5.836 / mu_e^2 = 1.459 for C/O.
    const Mch = 5.836 / 4;
    for (const M of [0.5, 0.8, 1.0, 1.2]) {
      const R = (0.0112 * ASTRO.Rsun * Math.sqrt((M / Mch) ** (-2 / 3) - (M / Mch) ** (2 / 3))) / 1e7;
      expect(rel(whiteDwarf(M, scales, codeG(U)).R, R)).toBeLessThan(M > 1.1 ? 0.06 : 0.03);
    }
  });

  it('are denser and smaller the heavier they are, with a smooth surface', () => {
    const a = whiteDwarf(0.8, scales, codeG(U));
    const b = whiteDwarf(1.0, scales, codeG(U));
    expect(b.rhoc).toBeGreaterThan(a.rhoc);
    expect(b.R).toBeLessThan(a.R);
    // About 3.5e7 g/cm^3 at the center of a 1 M_sun C/O dwarf (code density 2e6 g/cm^3).
    expect(b.rhoc * 1.989e6).toBeGreaterThan(2e7);
    expect(b.rhoc * 1.989e6).toBeLessThan(6e7);
    expect(b.rho(0.999 * b.R)).toBeLessThan(1e-4 * b.rhoc);
    expect(b.rho(b.R * 1.01)).toBe(0);
  });
});

describe('the collision orbit', () => {
  it('is parabolic about the center of mass, with the pericenter asked for', () => {
    const [m1, m2, R1, R2] = [1.0, 0.8, 0.54, 0.69];
    for (const b of [0, 0.3, 1, 1.5]) {
      const o = collisionOrbit(m1, m2, R1, R2, b, 1, 8);
      // Center of mass at rest at the origin.
      for (let a = 0; a < 3; a++) {
        expect(Math.abs(m1 * o.pos[0][a] + m2 * o.pos[1][a])).toBeLessThan(1e-12);
        expect(Math.abs(m1 * o.vel[0][a] + m2 * o.vel[1][a])).toBeLessThan(1e-12);
      }
      const r = o.pos[1].map((x, a) => x - o.pos[0][a]);
      const v = o.vel[1].map((x, a) => x - o.vel[0][a]);
      const d = Math.hypot(...r);
      // Zero orbital energy, and h^2 = 2 G M rp.
      expect(Math.abs(0.5 * (v[0] ** 2 + v[1] ** 2) - (m1 + m2) / d)).toBeLessThan(1e-12);
      const h = r[0] * v[1] - r[1] * v[0];
      expect(Math.abs(h * h - 2 * (m1 + m2) * b * (R1 + R2))).toBeLessThan(1e-12);
      // Falling together, apart by more than the two radii, and inside the box.
      expect(r[0] * v[0] + r[1] * v[1]).toBeLessThan(0);
      expect(d).toBeGreaterThan(R1 + R2);
      expect(Math.hypot(...o.pos[0]) + R1).toBeLessThan(4);
      expect(Math.hypot(...o.pos[1]) + R2).toBeLessThan(4);
    }
  });
});

describe('the degenerate barotrope', () => {
  it('inverts its enthalpy, h = int dp / rho', () => {
    const b = degenerateBarotrope(scales);
    for (const rho of [1e-6, 1e-3, 0.1, 1, 30, 1e3]) {
      // Integrate dp / rho from (near) zero density up to rho, in log rho.
      let h = 0;
      const steps = 20000;
      const lo = Math.log(1e-14);
      const dl = (Math.log(rho) - lo) / steps;
      for (let i = 0; i < steps; i++) {
        const r0 = Math.exp(lo + i * dl), r1 = Math.exp(lo + (i + 1) * dl);
        h += (b.pressure(r1) - b.pressure(r0)) / Math.sqrt(r0 * r1);
      }
      expect(rel(b.rhoOfH(h), rho)).toBeLessThan(1e-4);
    }
  });
});
