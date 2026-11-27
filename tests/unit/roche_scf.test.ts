import { describe, expect, it } from 'vitest';
import { rocheStar } from '../../src/problems/roche_scf';
import { binaryOmega, fillSeparation, innerLagrange, type RocheBody } from '../../src/problems/roche';
import { polytropeBarotrope } from '../../src/problems/settle';

// An n = 1 polytrope of mass 1 and radius 1 (G = 1): rho = rho_c sin(pi r) / (pi r).
const K = 2 / Math.PI;
const rhoc = Math.PI / 4;
const rho0 = (r: number) => (r >= 1 ? 0 : r < 1e-9 ? rhoc : (rhoc * Math.sin(Math.PI * r)) / (Math.PI * r));
const baro = polytropeBarotrope(1, K);
const pressure = (rho: number) => K * rho * rho;

// The star (mass 1) and a companion of 1.5, the star at `fill` of its Roche lobe.
function binary(fill: number) {
  const a = fillSeparation(1, 1, 1.5, fill);
  const omega = binaryOmega(2.5, a);
  const star = [(-1.5 / 2.5) * a, 0, 0];
  const bodies: RocheBody[] = [
    { gm: 1, at: star, soft: 0 },
    { gm: 1.5, at: [a / 2.5, 0, 0], soft: 0 },
  ];
  const L1 = innerLagrange(bodies, omega);
  return { star, omega, lobe: { bodies, phi: L1.phi }, toL1: L1.at[0] - star[0] };
}

// Mass and center of mass of the solved star, on a lattice over its reach.
function moments(s: { density(x: number, y: number, z: number): number }, L: number, n = 48) {
  const d = (2 * L) / n;
  let m = 0;
  let cx = 0;
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const x = -L + (i + 0.5) * d;
        const r = s.density(x, -L + (j + 0.5) * d, -L + (k + 0.5) * d);
        m += r;
        cx += r * x;
      }
  return { M: m * d ** 3, x: cx / m };
}

describe('a star solved in a binary\'s Roche potential (problems/roche_scf.ts)', () => {
  it('is the round star when the companion is far', () => {
    const b = binary(0.2);
    const reach = Math.min(1.5, b.toL1);
    const s = rocheStar({ center: b.star, M: 1, R: 1, reach, rho0, barotrope: baro, pressure, G: 1, omega: b.omega, lobe: b.lobe });
    // The grid is 32 cells across 3 radii: a tenth of a radius each.
    expect(Math.abs(s.density(0, 0, 0) - rhoc) / rhoc).toBeLessThan(0.05);
    expect(s.density(0.85, 0, 0)).toBeGreaterThan(0);
    expect(s.density(1.12, 0, 0)).toBe(0);
    expect(Math.abs(moments(s, reach).M - 1)).toBeLessThan(0.03);
  });

  it('near filling its lobe, reaches out to L1 and stays where the orbit put it', () => {
    const b = binary(0.97);
    const reach = Math.min(1.5, b.toL1);
    const s = rocheStar({ center: b.star, M: 1, R: 1, reach, rho0, barotrope: baro, pressure, G: 1, omega: b.omega, lobe: b.lobe });
    expect(s.change).toBeLessThan(1e-4);
    // The round star ends at 1; this one has gas within 5% of the way to L1, where the round
    // profile's pressure (the field's old taper) was zero.
    expect(rho0(0.95 * b.toL1)).toBe(0);
    expect(s.pressure(0.95 * b.toL1, 0, 0)).toBeGreaterThan(0);
    // Stretched toward L1 more than away from it, and held in place.
    const edge = (sgn: number) => {
      let r = 0;
      while (s.density(sgn * (r + 0.01), 0, 0) > 0) r += 0.01;
      return r;
    };
    expect(edge(1)).toBeGreaterThan(edge(-1) + 0.05);
    const mo = moments(s, reach);
    expect(Math.abs(mo.M - 1)).toBeLessThan(0.03);
    expect(Math.abs(mo.x)).toBeLessThan(0.03);
  });
});
