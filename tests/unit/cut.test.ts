import { describe, expect, it } from 'vitest';
import { CUT_MIN, cutGeometry } from '../../src/hydro/cut';

// Partly filled cells' geometry on the GPU's grid (hydro/cut.ts), made on the CPU.
describe('cut cells: the geometry', () => {
  const circle = (cx: number, cy: number, r: number) => [{ mark: 1, distance: (x: number, y: number) => Math.hypot(x - cx, y - cy) - r }];

  it('holds the box less the body, to the sampling', () => {
    const n = 64, dx = 1 / n, r = 0.21;
    const g = cutGeometry(circle(0.013, -0.02, r), [n, n, 1], dx, [-0.5, -0.5, 0]);
    let gas = 0;
    for (let q = 0; q < n * n; q++) gas += g.cut[4 * q] * dx * dx;
    expect(Math.abs(gas - (1 - Math.PI * r * r))).toBeLessThan(2e-4);
    expect(g.cutCells).toBeGreaterThan(50);
  });

  it('closes every cell: no face open into a cell with no gas, no gas behind closed faces, no sliver', () => {
    const n = 48, dx = 1 / n;
    const g = cutGeometry(circle(0.04, 0.01, 0.17), [n, n, 1], dx, [-0.5, -0.5, 0]);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const q = j * n + i;
        const phi = g.cut[4 * q];
        expect(phi === 0 || phi >= CUT_MIN).toBe(true);
        const faces = [g.cut[4 * q + 1], g.cut[4 * q + 2], g.cut[4 * (j * n + ((i - 1 + n) % n)) + 1], g.cut[4 * (((j - 1 + n) % n) * n + i) + 2]];
        if (phi === 0) {
          for (const a of faces) expect(a).toBe(0);
          expect(g.solid[q]).toBe(1);
        } else {
          expect(faces.some((a) => a > 0)).toBe(true);
          expect(g.solid[q]).toBe(0);
        }
      }
  });

  it('opens a face the surface only grazes: a face parallel to the surface is all open or all shut', () => {
    // Gas where x < 0.301: the x faces lie parallel to the surface, whole gas or whole solid; a y
    // face across it is open in the share of it on the gas side.
    const n = 32, dx = 1 / n, xs = 0.301;
    const g = cutGeometry([{ mark: 1, distance: (x: number) => xs - x }], [n, n, 1], dx, [-0.5, -0.5, 0]);
    for (let i = 0; i < n; i++) {
      const q = 5 * n + i;
      const ax = g.cut[4 * q + 1];
      expect(ax === 0 || ax === 1).toBe(true);
      const lo = -0.5 + i * dx;
      const want = Math.min(1, Math.max(0, (xs - lo) / dx));
      const ay = g.cut[4 * q + 2];
      if (want >= CUT_MIN && g.cut[4 * q] > 0) expect(Math.abs(ay - want)).toBeLessThan(0.02);
    }
  });
});
