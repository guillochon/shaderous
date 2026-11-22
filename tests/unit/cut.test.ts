import { describe, expect, it } from 'vitest';
import { CUT_MIN, cutGeometry } from '../../src/hydro/cut';
import { sectionOutline } from '../../src/geometry/builtin';
import { compileSetup } from '../../src/setup/compile';
import type { SetupSpec } from '../../src/setup/spec';
import { SECTIONS, sectionSpec } from '../../src/validation/cut';

// (As generated.test.ts: the app's types have no node.)
const fs: { readFileSync(p: string, enc: string): string } = await import(/* @vite-ignore */ 'node:fs' as string);

// The blade sections as built-in wings (plans/wind-tunnel.md 8.1.6).
describe('blade sections', () => {
  const area = (p: [number, number][]) => p.reduce((s, a, k) => s + a[0] * p[(k + 1) % p.length][1] - p[(k + 1) % p.length][0] * a[1], 0) / 2;

  it('leaves the NACA 0012 outline as it was, point for point', () => {
    const t = 0.12, n = 160;
    const yt = (x: number) => 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    const xs = Array.from({ length: n + 1 }, (_, k) => 0.5 * (1 - Math.cos((Math.PI * k) / n)));
    const old = [...xs.map((x) => [x - 0.25, -yt(x)]), ...xs.slice(1, -1).reverse().map((x) => [x - 0.25, yt(x)])];
    expect(sectionOutline(t, n)).toEqual(old);
  });

  it('floors the thickness: the tip a rectangle, the round nose round', () => {
    const tip = sectionOutline(0.2, 160, { minT: 0.2 });
    expect(Math.abs(area(tip) - 0.2)).toBeLessThan(1e-3);
    const round = sectionOutline(0.2, 160, { minT: 0.15, roundNose: true });
    expect(Math.min(...round.filter((p) => p[0] < -0.2).map((p) => Math.abs(p[1])))).toBeLessThan(1e-9); // a point nose
    expect(Math.abs(round[0][1])).toBeLessThan(1e-12);
    for (const p of round) if (p[0] > 0.05) expect(Math.abs(p[1])).toBeGreaterThanOrEqual(0.075 - 1e-12);
  });

  it('the runner’s setup documents are the GPU tests’ (tools/cutcell/sections)', () => {
    for (const f of SECTIONS) {
      const doc = JSON.parse(fs.readFileSync(`tools/cutcell/sections/${f}.shaderous.json`, 'utf8'));
      expect(doc).toEqual(JSON.parse(JSON.stringify(sectionSpec(f, 32))));
      expect(() => compileSetup(doc)).not.toThrow();
    }
  });

  for (const [file, t, o] of [
    ['section-tip', 0.2, { minT: 0.2 }],
    ['section-075', 0.2, { minT: 0.15 }],
    ['section-round', 0.2, { minT: 0.15, roundNose: true }],
    ['naca0020', 0.2, {}],
  ] as const)
    it(`builtin:${file} on a 2D grid holds the outline's area`, () => {
      const box = 1.5, n = 96, dx = box / n;
      const spec = {
        format: 'shaderous-setup', version: 1, id: 'section', name: 'Section',
        domain: { ndim: 2, n, amr: 0, box, supports2d: true },
        physics: { eos: 'liquid', eosParams: { liqRho: 1, liqC: 10 }, gravity: 'off', G: 0 },
        background: { rho: 1, p: 0, v: [1, 0, 0] },
        parts: [{ kind: 'obstacle', id: 'wing', shape: { type: 'mesh', file: `builtin:${file}`, scale: 1, center: [0.0037, 0.0051, 0], anchor: 'origin', up: 'y' } }],
      } as unknown as SetupSpec;
      const { problem } = compileSetup(spec);
      const g = cutGeometry(problem.obstacleDistances!(n, 2), [n, n, 1], dx, [-box / 2, -box / 2, 0]);
      let gas = 0;
      for (let q = 0; q < n * n; q++) gas += g.cut[4 * q] * dx * dx;
      const want = area(sectionOutline(t, 160, o));
      expect(Math.abs(box * box - gas - want) / want).toBeLessThan(0.02);
    });
});

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
