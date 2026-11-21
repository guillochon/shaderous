import { describe, expect, it } from 'vitest';
import { CutCell, cutDefaults, circle, rectangle, rigid, type Body } from '../../src/reference/cutcell';

// The float64 prototype of partly filled cells (src/reference/cutcell.ts; plans/wind-tunnel.md 8.1.1).
describe('cut cells (float64 prototype)', () => {
  const grid = (N: number, L = 2) => ({ ...cutDefaults, nx: N, ny: N, dx: L / N, x0: -L / 2, y0: -L / 2, nu: 0.01, subSteps: 4 });
  const still = () => ({ rho: 1, vx: 0, vy: 0 });

  it('cuts cells exactly: the gas shares add up to the box less the body, and every wall closes', () => {
    const shape = rectangle(0.37, 0.21);
    const s = new CutCell(grid(32), rigid(shape, [0.013, -0.041], [0, 0], 0.4), still);
    let gas = 0;
    for (let k = 0; k < s.n; k++) gas += s.g.phi[k] * s.cfg.dx ** 2;
    expect(gas).toBeCloseTo(4 - 0.37 * 0.21, 12);
    // The walls' vectors add up to nothing around a closed body.
    let wx = 0, wy = 0;
    for (const k of s.g.cut) {
      wx += s.g.wx[k];
      wy += s.g.wy[k];
    }
    expect(Math.abs(wx) + Math.abs(wy)).toBeLessThan(1e-12);
  });

  it('keeps liquid at rest around a body at rest, with no force on it', () => {
    const s = new CutCell(grid(32), rigid(circle(0.25), [0.01, 0.02]), still);
    for (let k = 0; k < 10; k++) s.step();
    let v = 0;
    for (let k = 0; k < s.n; k++) if (s.Q[k] > 0) v = Math.max(v, Math.hypot(s.Mx[k], s.My[k]) / s.Q[k]);
    expect(v).toBeLessThan(1e-12);
    expect(Math.abs(s.force.px) + Math.abs(s.force.py)).toBeLessThan(1e-12);
  });

  for (const moving of [false, true]) {
    it(`conserves mass, and the force on ${moving ? 'a moving' : 'a'} body is the momentum the liquid lost`, () => {
      const body = moving ? rigid(circle(0.25), [0.3, 0.02], [-1, 0.2]) : rigid(rectangle(0.5, 0.15), [0.01, 0.02], [0, 0], 0.3, 1.5);
      const s = new CutCell(grid(32), body, () => ({ rho: 1, vx: moving ? 0 : 1, vy: 0 }));
      const a = s.totals();
      const imp = [0, 0];
      for (let k = 0; k < 40; k++) {
        const dt = s.dtMax();
        s.step(dt);
        imp[0] += (s.force.px + s.force.vx) * dt;
        imp[1] += (s.force.py + s.force.vy) * dt;
      }
      const b = s.totals();
      expect(Math.abs(b.mass / a.mass - 1)).toBeLessThan(1e-13);
      expect(Math.abs(b.px - a.px + imp[0])).toBeLessThan(1e-12);
      expect(Math.abs(b.py - a.py + imp[1])).toBeLessThan(1e-12);
      expect(Math.abs(imp[0])).toBeGreaterThan(1e-3);
      expect(s.lost).toBeLessThan(1e-10);
    });
  }

  it('moves the liquid with a piston that spans the box (to 1%)', () => {
    const U = 0.5, tau = 0.5;
    const x = (t: number) => (t < tau ? U * (t / 2 - (tau / (2 * Math.PI)) * Math.sin((Math.PI * t) / tau)) : U * (t - tau / 2));
    const v = (t: number) => (t < tau ? (U / 2) * (1 - Math.cos((Math.PI * t) / tau)) : U);
    const body: Body = { shape: rectangle(0.2, 4), pos: (t) => [-0.53 + x(t), 0], angle: () => 0, vel: (t) => [v(t), 0], spin: () => 0 };
    const s = new CutCell(grid(32), body, still);
    const sp: number[] = [];
    while (s.t < 1.5) {
      s.step(Math.min(s.dtMax(), 1.5 - s.t + 1e-12));
      if (s.t > 0.8) sp.push(s.totals().px / s.totals().mass);
    }
    const mean = sp.reduce((a, b) => a + b, 0) / sp.length;
    expect(Math.abs(mean / U - 1)).toBeLessThan(0.01);
  });
});
