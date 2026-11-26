import { describe, expect, it } from 'vitest';
import { nohExact, nohShockRadius, uniformCollapseExact } from '../../src/reference/converging';
import { exactRiemann } from '../../src/reference/riemann_exact';
import { RIEMANN_TUBES } from '../../src/presets/names';
import { preset } from '../../src/presets/index';
import { defaultEosParams, makeEos } from '../../src/eos/eos';

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
