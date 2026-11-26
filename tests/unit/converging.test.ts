import { describe, expect, it } from 'vitest';
import { cjState, coggeshallExact, ehepExact, fickettExact, nohExact, nohShockRadius, uniformCollapseExact } from '../../src/reference/converging';
import { exactRiemann } from '../../src/reference/riemann_exact';
import { RIEMANN_TUBES } from '../../src/presets/names';
import { preset } from '../../src/presets/index';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { startOf } from '../../src/setup/parts/flows/coggeshall';

const G = 5 / 3;

describe('the Noh implosion, exactly', () => {
  for (const k of [1, 2, 3] as const) {
    it(`meets the shock jump conditions in ${['planar', 'cylindrical', 'spherical'][k - 1]} symmetry`, () => {
      const t = 0.6;
      const q = { rho0: 1, u0: 1 };
      const rs = nohShockRadius(G, 1, t);
      const s = rs / t;
      const a = nohExact(k, G, q, rs * (1 + 1e-12), t);
      const b = nohExact(k, G, q, rs * (1 - 1e-12), t);
      // mass, momentum and energy fluxes through the moving front
      expect(a.rho * (a.u - s)).toBeCloseTo(b.rho * (b.u - s), 10);
      expect(a.rho * (a.u - s) ** 2 + a.p).toBeCloseTo(b.rho * (b.u - s) ** 2 + b.p, 10);
      const h = (w: typeof a) => 0.5 * (w.u - s) ** 2 + (G / (G - 1)) * (w.p / w.rho);
      expect(h(a)).toBeCloseTo(h(b), 10);
      expect(b.rho).toBeCloseTo(4 ** k, 10);
    });
  }
});

describe('the uniform collapse, exactly', () => {
  it('conserves mass and follows its adiabat', () => {
    for (const k of [1, 2, 3] as const) {
      const q = { rho0: 1, e0: 1, tau: 1 };
      const at = (r: number, t: number) => uniformCollapseExact(k, G, q, r, t);
      const [r, t, h] = [0.3, 0.4, 1e-6];
      // d rho / dt + r^(1-k) d(r^(k-1) rho u) / dr = 0
      const dRho = (at(r, t + h).rho - at(r, t - h).rho) / (2 * h);
      const flux = (x: number) => x ** (k - 1) * at(x, t).rho * at(x, t).u;
      const div = (flux(r + h) - flux(r - h)) / (2 * h) / r ** (k - 1);
      expect(dRho + div).toBeCloseTo(0, 5);
      // p / rho^gamma stays as it started
      const w = at(r, t);
      expect(w.p / w.rho ** G).toBeCloseTo((G - 1) * q.e0 * q.rho0 ** (1 - G), 10);
    }
  });
});

describe('the ExactPack Riemann problems', () => {
  it('has the star states ExactPack’s own tests give', () => {
    const [einfeldt, contact, slow] = RIEMANN_TUBES.map((t) => exactRiemann(t.left, t.right, t.eosParams.gamma!));
    // Toro's test 2: two rarefactions, p* = 0.00189 at rest
    expect(einfeldt(0).p).toBeCloseTo(0.00189, 5);
    expect(einfeldt(0).u).toBeCloseTo(0, 10);
    // Toro's test 3 moved by -19.59745: the contact at rest, p* = 460.894
    expect(contact(-1e-9).p).toBeCloseTo(460.894, 2);
    expect(Math.abs(contact(-1e-9).u)).toBeLessThan(1e-3);
    // a single slow shock: the left state is the star state (ExactPack: p* = 10.333334047951963)
    expect(slow(-0.01).p).toBeCloseTo(10.333334047951963, 4);
    expect(slow(-0.01).rho).toBeCloseTo(3.857143, 4);
  });

  it('puts each preset’s two states either side of its interface', () => {
    for (const t of RIEMANN_TUBES) {
      const init = preset(t.id).problem.init(64, makeEos('ideal', { ...defaultEosParams(), gamma: t.eosParams.gamma! }), 3, {});
      const l = init(t.x0 - 0.01, 0, 0);
      const r = init(t.x0 + 0.01, 0, 0);
      expect([l.rho, l.vx, l.p]).toEqual([t.left.rho, t.left.u, t.left.p]);
      expect([r.rho, r.vx, r.p]).toEqual([t.right.rho, t.right.u, t.right.p]);
    }
  });
});

describe('the converging-flow presets', () => {
  const eos = makeEos('ideal', { ...defaultEosParams(), gamma: G });
  it('start the Noh gas falling at unit speed toward the center, line or plane', () => {
    // planar: toward x = 0 from both sides
    const planar = preset('noh-planar').problem.init(64, eos, 3, {});
    expect(planar(0.3, 0.2, -0.1).vx).toBe(-1);
    expect(planar(-0.3, 0.2, -0.1).vx).toBe(1);
    expect(planar(0.3, 0.2, -0.1).vy).toBe(0);
    // (R, z): toward the axis, with no z velocity
    const cyl = preset('noh-cylindrical').problem.init(64, eos, 2, {});
    expect([cyl(0.3, 0.2, 0).vx, cyl(0.3, 0.2, 0).vy]).toEqual([-1, 0]);
    // (r, theta): radially inward, v_theta zero
    const sph = preset('noh-spherical').problem.init(64, eos, 2, {});
    const w = sph(0.4, 1.1, 0);
    expect(w.vx).toBeCloseTo(-1, 12);
    expect(w.vy).toBeCloseTo(0, 12);
    // a Cartesian plane: toward its center
    const sq = preset('noh-cartesian').problem.init(64, eos, 2, {});
    const v = sq(0.3, 0.4, 0);
    expect([v.vx, v.vy]).toEqual([-0.6, -0.8].map((c) => expect.closeTo(c, 12)));
  });

  it('start the uniform collapse at u = -r with p = (gamma - 1) rho e', () => {
    const sph = preset('uniform-collapse-spherical').problem.init(64, eos, 2, {});
    const w = sph(0.4, 0.7, 0);
    expect(w.vx).toBeCloseTo(-0.4, 12);
    expect(w.vy).toBeCloseTo(0, 12);
    expect(w.p).toBeCloseTo(2 / 3, 12);
    const cyl = preset('uniform-collapse-cylindrical').problem.init(64, eos, 2, {});
    expect(cyl(0.25, -0.3, 0).vx).toBeCloseTo(-0.25, 12);
    expect(cyl(0.25, -0.3, 0).vy).toBe(0);
    const planar = preset('uniform-collapse-planar').problem.init(64, eos, 3, {});
    expect(planar(-0.2, 0.3, 0.1).vx).toBeCloseTo(0.2, 12);
  });
});

describe('Coggeshall’s similarity solutions, exactly', () => {
  // Residuals of the Euler equations in symmetry g, by centered differences:
  // continuity, momentum, and the entropy p / rho^gamma carried with the gas.
  const residuals = (f: (r: number, t: number) => { rho: number; u: number; p: number }, g: number, gamma: number, r: number, t: number) => {
    const h = 1e-5;
    const dt = (q: 'rho' | 'u' | 'p' | 's') => (val(q, r, t + h) - val(q, r, t - h)) / (2 * h);
    const dr = (q: 'rho' | 'u' | 'p' | 's') => (val(q, r + h, t) - val(q, r - h, t)) / (2 * h);
    function val(q: 'rho' | 'u' | 'p' | 's', x: number, s: number) {
      const w = f(x, s);
      return q === 's' ? w.p / w.rho ** gamma : w[q];
    }
    const w = f(r, t);
    const flux = (x: number) => x ** (g - 1) * f(x, t).rho * f(x, t).u;
    const mass = dt('rho') + (flux(r + h) - flux(r - h)) / (2 * h) / r ** (g - 1);
    const mom = dt('u') + w.u * dr('u') + dr('p') / w.rho;
    const ent = dt('s') + w.u * dr('s');
    // relative to the size of the terms
    return [mass / (w.rho * Math.abs(dr('u')) + 1e-12), mom / (Math.abs(dt('u')) + Math.abs(w.u * dr('u')) + 1e-12), ent / (w.p / w.rho ** gamma)];
  };
  for (const g of [1, 2, 3] as const) {
    it(`satisfy the Euler equations in geometry ${g}`, () => {
      const q = { rho0: 1.8, b: 1.2, T0: 1.4, Gamma: 40, tau: 1.25 };
      const cases: [1 | 2 | 6, number, number][] = [[1, 1.4, 1.3], [2, 1.4, 1.3], [6, (g + 2) / g, 0.6]];
      for (const [c, gamma, t] of cases)
        for (const r of [0.2, 0.5, 0.9])
          for (const e of residuals((x, s) => coggeshallExact(c, g, gamma, q, x, s), g, gamma, r, t)) expect(Math.abs(e)).toBeLessThan(1e-4);
    });
  }

  it('start the presets at the solution’s state at t0', () => {
    for (const [id, c, gamma] of [['coggeshall-1', 1, 1.4], ['coggeshall-2', 2, 1.4], ['coggeshall-6', 6, 5 / 3]] as const) {
      const e = makeEos('ideal', { ...defaultEosParams(), gamma });
      const init = preset(id).problem.init(64, e, 2, {});
      const part = preset(id).spec.parts![0] as unknown as { rho0: number; b: number; T0?: number; Gamma?: number; tau?: number };
      const w = init(0.5, 0.8, 0);
      const x = coggeshallExact(c, 3, gamma, part, 0.5, startOf(part as unknown as Record<string, number>));
      expect(w.rho).toBeCloseTo(x.rho, 10);
      expect(w.p).toBeCloseTo(x.p, 10);
      expect(w.vx).toBeCloseTo(x.u, 10);
    }
  });
});

describe('Fickett’s detonation and the escape of its products, exactly', () => {
  it('has the CJ state behind the front and the piston’s speed at the tail', () => {
    for (const gamma of [1.4, 3]) {
      const q = { D: 0.8, rho0: 1.875, up: 0.05 };
      const cj = cjState(gamma, q.D, q.rho0);
      const t = 2;
      const front = fickettExact(gamma, q, q.D * t * (1 - 1e-12), t);
      expect(front.rho).toBeCloseTo(cj.rho, 8);
      expect(front.p).toBeCloseTo(cj.p, 8);
      expect(front.u).toBeCloseTo(cj.u, 8);
      // the jump conditions from cold explosive at rest, releasing q
      expect(cj.rho * (cj.u - q.D)).toBeCloseTo(-q.rho0 * q.D, 10);
      expect(cj.rho * (cj.u - q.D) ** 2 + cj.p).toBeCloseTo(q.rho0 * q.D * q.D, 10);
      expect(0.5 * (cj.u - q.D) ** 2 + (gamma / (gamma - 1)) * (cj.p / cj.rho)).toBeCloseTo(0.5 * q.D ** 2 + cj.q, 10);
      // Chapman-Jouguet: the products leave the front at their sound speed
      expect(q.D - cj.u).toBeCloseTo(cj.c, 12);
      expect(fickettExact(gamma, q, 0.06 * t, t).u).toBeCloseTo(q.up, 12);
    }
  });

  it('is Fickett’s detonation until the explosive runs out, and continuous between its regions after', () => {
    const q = { D: 0.85, rho0: 1.6, up: 0.05, L: 1 };
    for (const t of [0.4, 1.0])
      for (let x = 0.06; x < q.D * t; x += 0.01) {
        const a = ehepExact(q, x, t);
        const b = fickettExact(3, q, x, t);
        expect(a.rho).toBeCloseTo(b.rho, 10);
        expect(a.u).toBeCloseTo(b.u, 10);
      }
    for (const t of [1.5, 2.5, 3.5, 4.5])
      for (let x = q.up * t + 1e-3; x < q.D * t - 1e-3; x += 1e-3) {
        const a = ehepExact(q, x, t);
        const b = ehepExact(q, x + 1e-7, t);
        expect(Math.abs(a.rho - b.rho)).toBeLessThan(1e-4);
        expect(Math.abs(a.u - b.u)).toBeLessThan(1e-4);
      }
  });

  it('satisfies the Euler equations in every region of the escape (gamma = 3)', () => {
    const q = { D: 0.85, rho0: 1.6, up: 0.05, L: 1 };
    const h = 1e-6;
    const f = (y: number, s: number) => ehepExact(q, y, s);
    const d = (g: (y: number, s: number) => number, x: number, t: number, dy: number, ds: number) => (g(x + dy, t + ds) - g(x - dy, t - ds)) / (2 * h);
    const rho = (y: number, s: number) => f(y, s).rho;
    const mom = (y: number, s: number) => f(y, s).rho * f(y, s).u;
    const u = (y: number, s: number) => f(y, s).u;
    const p = (y: number, s: number) => f(y, s).p;
    let checked = 0;
    for (const t of [0.8, 1.5, 2.5, 3.5, 4.5])
      for (let x = q.up * t + 0.01; x < q.D * t - 0.01; x += 0.0137) {
        const w = f(x, t);
        if (w.rho < 1e-3) continue;
        const mass = d(rho, x, t, 0, h) + d(mom, x, t, h, 0);
        const acc = d(u, x, t, 0, h) + w.u * d(u, x, t, h, 0) + d(p, x, t, h, 0) / w.rho;
        // within a step of a line between regions the derivatives jump
        const s = [-2 * h, 2 * h].map((e) => [f(x + e, t), f(x, t + e)]).flat();
        if (s.some((v) => Math.abs(v.rho - w.rho) > 1e-4 * w.rho * 50)) continue;
        expect(Math.abs(mass) / (w.rho * q.D)).toBeLessThan(1e-3);
        expect(Math.abs(acc) / (q.D * q.D)).toBeLessThan(1e-3);
        checked++;
      }
    expect(checked).toBeGreaterThan(100);
  });
});
