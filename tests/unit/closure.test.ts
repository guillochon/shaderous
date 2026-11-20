// The turbulence closure's float64 mirror (src/closure/closure.ts) against cells computed by
// the code it was trained with (tools/closure/export.py wrote them with the weights).
import { describe, expect, it } from 'vitest';
import { closureCell, closureNetSlot, closureWeights, type ClosureCell } from '../../src/closure/closure';
import { CLOSURE_DATA } from '../../src/closure/closure_net';
import cases from '../../src/closure/closure_cases.json';
import { preset } from '../../src/presets/index';
import { closureOn, closureUnsupported, configFor, hydroSlots } from '../../src/sim';

interface Case {
  c: ClosureCell; xm: ClosureCell; xp: ClosureCell; ym: ClosureCell; yp: ClosureCell; cs: number;
  features: number[]; gate: number; a: number; b: number;
}

describe('the turbulence closure', () => {
  it('has the weights its layers need', () => {
    const { features, hidden: h, layers } = CLOSURE_DATA;
    expect(closureWeights().length).toBe(features * h + h + (layers - 1) * (h * h + h) + 4 * h + 4);
    expect(closureNetSlot()).toContain(`const CL_H: u32 = ${h}u;`);
  });

  it('gives the features, gate and coefficients the training code gives', () => {
    expect(cases.length).toBeGreaterThan(100);
    let open = 0;
    for (const k of cases as Case[]) {
      const got = closureCell(k.c, k.xm, k.xp, k.ym, k.yp, k.cs);
      got.features.forEach((f, i) => expect(f).toBeCloseTo(k.features[i], 9));
      expect(got.gate).toBeCloseTo(k.gate, 9);
      expect(got.a).toBeCloseTo(k.a, 7);
      expect(got.b).toBeCloseTo(k.b, 7);
      if (got.gate > 0) open++;
    }
    // Both kinds of cell are in the set: turbulence, and the shock and contact where it is shut.
    expect(open).toBeGreaterThan(50);
    expect(cases.length - open).toBeGreaterThan(20);
  });

  it('keeps its coefficients within their limits and is shut in one-dimensional flow', () => {
    const cell = (rho: number, vx: number, vy: number, p: number, s: number): ClosureCell => ({ rho, vx, vy, p, s });
    // A planar compression along x, however weak or strong: nothing varies along y.
    for (const dv of [1e-3, 0.01, 0.1, 1]) {
      const c = closureCell(cell(1, 0, 0, 1, 0.5), cell(1, dv, 0, 1, 0.4), cell(1, -dv, 0, 1, 0.6), cell(1, 0, 0, 1, 0.5), cell(1, 0, 0, 1, 0.5), 1.2);
      expect(c.gate).toBe(0);
      expect(c.a).toBe(0);
      expect(c.b).toBe(0);
    }
    // Gas at rest with a jump in density and dye (a contact): nothing is sheared.
    const rest = closureCell(cell(1.5, 0, 0, 1, 0.5), cell(1, 0, 0, 1, 0), cell(2, 0, 0, 1, 1), cell(1.5, 0, 0, 1, 0.5), cell(1.5, 0, 0, 1, 0.5), 1);
    expect(rest.gate).toBe(0);
    // A shear layer: open, and within the limits.
    const shear = closureCell(cell(1, 0.2, 0, 1, 0.5), cell(1, 0.2, -0.05, 1, 0.5), cell(1, 0.2, 0.05, 1, 0.5), cell(1, 0.1, 0, 1, 0.3), cell(1, 0.3, 0, 1, 0.7), 1.2);
    expect(shear.gate).toBeGreaterThan(0.9);
    expect(shear.a).toBeGreaterThanOrEqual(0);
    expect(shear.a).toBeLessThanOrEqual(CLOSURE_DATA.aMax);
    expect(shear.b).toBeGreaterThanOrEqual(0);
    expect(shear.b).toBeLessThanOrEqual(CLOSURE_DATA.bMax);
    // Constants: the limits themselves, times the gate; the same gates.
    const flat = closureCell(cell(1, 0.2, 0, 1, 0.5), cell(1, 0.2, -0.05, 1, 0.5), cell(1, 0.2, 0.05, 1, 0.5), cell(1, 0.1, 0, 1, 0.3), cell(1, 0.3, 0, 1, 0.7), 1.2, 'constant');
    expect(flat.gate).toBe(shear.gate);
    expect(flat.a).toBeCloseTo(CLOSURE_DATA.aMax * flat.gate, 12);
    expect(flat.b).toBeCloseTo(CLOSURE_DATA.bMax * flat.gate, 12);
    expect(closureCell(cell(1.5, 0, 0, 1, 0.5), cell(1, 0, 0, 1, 0), cell(2, 0, 0, 1, 1), cell(1.5, 0, 0, 1, 0.5), cell(1.5, 0, 0, 1, 0.5), 1, 'constant').a).toBe(0);
  });

  it('is compiled in only where a run can have it', () => {
    const kh = preset('kelvin-helmholtz').problem;
    const on = configFor(kh, 128, { ndim: 2, closure: 'network' });
    expect(closureUnsupported(on)).toBeNull();
    expect(hydroSlots(on).closure).toBe('hydro/closure_on.wgsl');
    expect(hydroSlots(configFor(kh, 128, { ndim: 2 })).closure).toBe('hydro/closure_off.wgsl');
    expect(hydroSlots(configFor(kh, 128, { ndim: 2, closure: 'constant' })).closure).toBe('hydro/closure_on.wgsl');
    expect(closureNetSlot('constant')).toContain('const CL_NET = false;');
    expect(closureUnsupported(configFor(kh, 64, { ndim: 3, closure: 'network' }))).toMatch(/two-dimensional/);
    expect(closureUnsupported(configFor(kh, 128, { ndim: 2, closure: 'network', amrLevels: 2 }))).toMatch(/adaptive mesh/);
    expect(closureOn(configFor(preset('driven-turbulence').problem, 128, { ndim: 2, closure: 'network' }))).toBe(false);
  });
});
