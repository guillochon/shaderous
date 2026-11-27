// The reference state a gas is stored as its deviation from (src/hydro/refstate.ts, SimConfig.reference).
import { describe, expect, it } from 'vitest';
import { buildReference, refAt } from '../../src/hydro/refstate';
import { presetSpec } from '../../src/presets/index';
import { compileSetup } from '../../src/setup/compile';
import { configFor, refOn, refUnsupported } from '../../src/sim';

/** A 2D isothermal column of nx * ny cells in exact discrete balance with the field g (down y). */
function column(nx: number, ny: number, h: number, g: number) {
  const dx = 1 / ny;
  const a = 1 / (2 * h);
  const q = (1 - a) / (1 + a);
  const T = (g * dx) / (2 * a);
  const rho = new Float64Array(nx * ny);
  const p = new Float64Array(nx * ny);
  const r1 = Array.from({ length: ny }, (_, j) => q ** j);
  const p1 = new Array<number>(ny);
  p1[ny - 1] = T * r1[ny - 1];
  for (let j = ny - 2; j >= 0; j--) p1[j] = p1[j + 1] + (g * dx * (r1[j] + r1[j + 1])) / 2;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    rho[j * nx + i] = r1[j];
    p[j * nx + i] = p1[j];
  }
  return { rho, p, dx, r1, p1 };
}

describe('the reference state', () => {
  it('builds a column in exact balance with the trapezoid rule, equal to a balanced initial state', () => {
    const g = 3;
    const c = column(4, 32, 1.5, g);
    const ref = buildReference('column', [4, 32, 1], c.rho, c.p, 5 / 3, [0, -g, 0], c.dx);
    if (typeof ref === 'string') throw new Error(ref);
    expect(ref.axis).toBe(1);
    for (let j = 0; j + 1 < 32; j++) {
      const res = ref.p[j + 1] - ref.p[j] + (0.5 * (ref.rho[j] + ref.rho[j + 1]) * g * c.dx);
      expect(Math.abs(res)).toBeLessThan(1e-14 * ref.p[j]);
      expect(Math.abs(ref.p[j] - c.p1[j])).toBeLessThan(1e-13 * c.p1[j]);
    }
    // Every cell of a layer takes that layer's values; e = p / (gamma - 1).
    const [r, p, e] = refAt(ref, [4, 32, 1], 4 * 7 + 3);
    expect(r).toBe(ref.rho[7]);
    expect(p).toBe(ref.p[7]);
    expect(e).toBeCloseTo(p / (2 / 3), 12);
  });

  it('takes the mean of the initial state where it is uniform', () => {
    const rho = Float64Array.of(1, 2, 3, 4);
    const p = Float64Array.of(10, 10, 20, 20);
    const ref = buildReference('uniform', [2, 2, 1], rho, p, 1.4, [0, 0, 0], 0.5);
    if (typeof ref === 'string') throw new Error(ref);
    expect([ref.axis, ref.rho[0], ref.p[0]]).toEqual([3, 2.5, 15]);
  });

  it('is offered by the presets made for it, and refused where its passes do not reach', () => {
    for (const id of ['atmosphere', 'gresho-gas']) {
      const { problem, overrides } = compileSetup(presetSpec(id)!);
      const cfg = { ...configFor(problem, 64, { ndim: 2 }), ...overrides };
      expect(refUnsupported(cfg), id).toBeNull();
      expect(refOn(cfg), id).toBe(true);
    }
    const { problem, overrides } = compileSetup(presetSpec('atmosphere')!);
    const cfg = { ...configFor(problem, 64, { ndim: 2 }), ...overrides };
    expect(refUnsupported({ ...cfg, amrLevels: 2 })).toMatch(/uniform grid/);
    expect(refUnsupported({ ...cfg, pressure: 'implicit' })).toMatch(/explicit/);
    expect(refUnsupported({ ...cfg, reference: 'uniform' })).toMatch(/without gravity/);
    expect(refUnsupported({ ...cfg, bcLo: [0, 0, 0], bcHi: [0, 0, 0] })).toMatch(/periodic/);
  });
});
