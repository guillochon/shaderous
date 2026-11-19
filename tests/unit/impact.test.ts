import { describe, expect, it } from 'vitest';
import { impactDebris, ROCHE } from '../../src/debris/impact';

/** Cells from a list of (x, y, z, dx, rho, vx, vy, vz). */
function cells(list: number[][]) {
  const n = list.length;
  const c = { count: n, x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n), dx: new Float64Array(n), u1: new Float64Array(4 * n) };
  list.forEach(([x, y, z, dx, rho, vx, vy, vz], q) => {
    c.x[q] = x; c.y[q] = y; c.z[q] = z; c.dx[q] = dx;
    c.u1.set([rho, rho * vx, rho * vy, rho * vz], 4 * q);
  });
  return c;
}

describe('what an impact leaves', () => {
  // A planet of mass 1 (a ball of cells of density 0.24, radius about 1), moving at (0.1, 0, 0).
  const rho = 3 / (4 * Math.PI);
  const dx = 0.1;
  const planet: number[][] = [];
  for (let i = -12; i <= 12; i++) for (let j = -12; j <= 12; j++) for (let k = -12; k <= 12; k++) {
    const [x, y, z] = [(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx];
    if (Math.hypot(x, y, z) < 1) planet.push([x + 5, y, z, dx, rho, 0.1, 0, 0]);
  }
  const M = planet.length * rho * dx ** 3;
  // Light parcels around it: on a circular orbit at 4, on one at 1.5 that dips inside, and one leaving.
  const parcel = (r: number, v: number, m: number) => [5 + r, 0, 0, 0.3, m / 0.027, 0.1, v, 0];
  const vc = (r: number) => Math.sqrt(M / r);
  const list = [...planet, parcel(4, vc(4), 1e-3), parcel(1.5, 0.5 * vc(1.5), 2e-3), parcel(3, 1.5 * Math.sqrt(2) * vc(3), 4e-3)];
  // A tracer: 1 in the orbiting parcel, 0.25 in the planet.
  const tracer = new Float64Array(list.length).fill(0.25);
  tracer[planet.length] = 1;

  it('sorts gas into the planet, the disk and what escapes', () => {
    const d = impactDebris(cells(list), 1, rho, [tracer]);
    expect(d.disk).toBeCloseTo(1e-3, 8);
    expect(d.escaping).toBeCloseTo(4e-3, 8);
    expect(d.planet).toBeCloseTo(M + 2e-3, 6);
    expect(d.center[0]).toBeCloseTo(5, 2);
    expect(d.velocity[0]).toBeCloseTo(0.1, 6);
    expect(d.radius).toBeCloseTo(Math.cbrt(d.planet / (M / 1)) * Math.cbrt((3 * M) / (4 * Math.PI * rho)), 6);
    expect(d.shares[1][0]).toBeCloseTo(1, 8);
    expect(d.shares[0][0]).toBeCloseTo(0.25, 6);
  });

  it('gives the disk its angular momentum, and the moon it makes', () => {
    const d = impactDebris(cells(list), 1, rho);
    // The parcel at 4 radii, on a circular orbit about the planet as found.
    const L = 1e-3 * Math.hypot(5 + 4 - d.center[0], 0) * vc(4);
    expect(d.diskL / L).toBeCloseTo(1, 2);
    const roche = Math.sqrt(d.planet * ROCHE * d.radius);
    expect(d.moon).toBeCloseTo((1.9 * d.diskL) / roche - 1.195 * d.disk, 10);
    // A disk beyond the Roche radius gathers into a moon of most of its mass.
    expect(d.moon / d.disk).toBeGreaterThan(0.9);
  });

  it('leaves thin gas out', () => {
    // Below a density of 0.05: the orbiting parcel (0.037), not the others.
    const d = impactDebris(cells(list), 1, rho, [], 0.05);
    expect(d.disk).toBe(0);
    expect(d.escaping).toBeCloseTo(4e-3, 8);
    const none = impactDebris(cells(list), 1, rho, [], 50);
    expect(none.planet).toBe(0);
    expect(none.moon).toBe(0);
  });
});
