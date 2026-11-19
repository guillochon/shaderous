// A polytropic envelope around a point-mass core (src/problems/cored_polytrope.ts), and the
// star part that builds one (StarPart.core).
import { describe, expect, it } from 'vitest';
import { coredPolytrope } from '../../src/problems/cored_polytrope';
import { laneEmden, polytropeScales } from '../../src/problems/lane_emden';
import { buildSetupBodies, checkSetup } from '../../src/setup/check';
import { compileSetup } from '../../src/setup/compile';
import { makeEos, defaultEosParams } from '../../src/eos/eos';
import { emptySetup, type Part, type SetupSpec } from '../../src/setup/spec';

/** The envelope's mass, and the largest relative residual of dh/dr = -G (m + Mc s) / r^2 over its bulk. */
function measure(n: number, Menv: number, R: number, Mc: number, soft: number, G: number) {
  const c = coredPolytrope(n, Menv, R, Mc, soft, G);
  const N = 4000;
  const h = (r: number) => (n + 1) * c.K * Math.pow(c.rhoc * Math.pow(c.theta(r / R), n), 1 / n);
  let m = 0;
  let worst = 0;
  for (let i = 0; i < N; i++) {
    const r = ((i + 0.5) * R) / N;
    m += (4 * Math.PI * r * r * c.rhoc * Math.pow(c.theta(r / R), n) * R) / N;
    const re = ((i + 1) * R) / N;
    if (i % 100 === 99 && re < 0.9 * R) {
      const d = R / N;
      const dh = (h(re + d) - h(re - d)) / (2 * d);
      const g = (G * (m + (Mc * re ** 3) / Math.pow(re * re + soft * soft, 1.5))) / (re * re);
      worst = Math.max(worst, Math.abs(dh + g) / g);
    }
  }
  return { c, mass: m, worst };
}

describe('a polytropic envelope around a point-mass core', () => {
  it('has the mass and radius asked for, in balance with its own gravity and the core’s', () => {
    for (const [n, Menv, R, Mc, soft, G] of [[1.5, 0.6, 1, 0.4, 0.1, 1], [3, 0.6, 1, 0.4, 0.05, 1], [1.5, 2, 3, 0.5, 0.2, 2.5]]) {
      const { c, mass, worst } = measure(n, Menv, R, Mc, soft, G);
      expect(mass / Menv, `mass, n = ${n}`).toBeCloseTo(1, 4);
      expect(worst, `balance, n = ${n}`).toBeLessThan(2e-3);
      expect(c.theta(0)).toBe(1);
      expect(c.theta(0.999)).toBeLessThan(2e-3);
      expect(c.theta(1)).toBe(0);
      // Monotonic outward.
      for (let s = 0.01; s < 1; s += 0.01) expect(c.theta(s)).toBeLessThanOrEqual(c.theta(s - 0.01));
    }
  });

  it('is the plain polytrope when the core weighs next to nothing', () => {
    for (const n of [1.5, 3]) {
      const c = coredPolytrope(n, 1, 1, 1e-7, 0.05, 1);
      const le = polytropeScales(laneEmden(n), 1, 1, 1);
      expect(c.rhoc / le.rhoc).toBeCloseTo(1, 3);
      expect(c.K / le.K).toBeCloseTo(1, 3);
    }
  });

  it('is denser at the center the heavier and the more compact the core', () => {
    const rhoc = (Mc: number, soft: number) => coredPolytrope(1.5, 1 - Mc, 1, Mc, soft, 1).rhoc;
    expect(rhoc(0.4, 0.1)).toBeGreaterThan(rhoc(0.2, 0.1));
    expect(rhoc(0.4, 0.05)).toBeGreaterThan(rhoc(0.4, 0.1));
  });
});

describe('a star with a core', () => {
  const giant = (more: Partial<Extract<Part, { kind: 'star' }>> = {}): Part => ({ kind: 'star', id: 'giant', type: 'polytrope', n: 1.5, mass: 1, radius: 1, core: { mass: 0.4, soft: 0.1 }, pos: [-0.4, 0, 0], vel: [0, -0.3, 0], ...more });
  const setup = (parts: Part[]): SetupSpec => ({ ...emptySetup(), domain: { ndim: 3, n: 256, amr: 3, box: 8, bc: { lo: [3, 3, 3], hi: [3, 3, 3] } }, physics: { gravity: 'isolated', G: 1 }, parts });

  it('brings a point mass at its center, moving with it, after the parts’ own masses', () => {
    const spec = setup([giant(), { kind: 'mass', id: 'companion', gm: 0.5, pos: [0.8, 0, 0], soft: 0.1, sink: 0 }]);
    const b = buildSetupBodies(spec).bodies!;
    expect(b.massIds).toEqual(['companion', 'giant.core']);
    expect(b.masses[1]).toMatchObject({ gm: 0.4, pos: [-0.4, 0, 0], vel: [0, -0.3, 0], soft: 0.1, sinkRadius: 0 });
    // The star's mass is the whole star's; its gas is the rest.
    expect(b.stars[0].M).toBe(1);
    expect(b.stars[0].core).toEqual({ mass: 0.4, soft: 0.1 });
    // Its softening is a twentieth of the radius if not given.
    expect(buildSetupBodies(setup([giant({ core: { mass: 0.4 } })])).bodies!.masses[0].soft).toBeCloseTo(0.05, 12);
    expect(checkSetup(spec).filter((i) => i.level === 'error')).toEqual([]);
    // (Its own core is not "a point mass inside the star".)
    expect(checkSetup(spec).some((i) => /starts inside star/.test(i.text))).toBe(false);
  });

  it('compiles to a run with the core among its masses, settled with the core’s potential on the CPU', () => {
    const { problem } = compileSetup(setup([giant()]));
    const masses = problem.masses!({}, 1, 8 / 256, 3);
    expect(masses.length).toBe(1);
    expect(masses[0].gm).toBeCloseTo(0.4, 12);
    const set = problem.settle!({}, makeEos('ideal', defaultEosParams()));
    expect(set.stars[0].core).toEqual({ gm: 0.4, soft: 0.1 });
    expect(set.stars[0].barotrope.gpu).toBeUndefined();
    // A star without one is settled as before.
    const plain = compileSetup(setup([giant({ core: undefined })])).problem.settle!({}, makeEos('ideal', defaultEosParams()));
    expect(plain.stars[0].core).toBeUndefined();
    expect(plain.stars[0].barotrope.gpu).toBeDefined();
  });

  it('is refused with a central density, or a core as heavy as the star', () => {
    const errors = (p: Part) => checkSetup(setup([p])).filter((i) => i.level === 'error').map((i) => i.text);
    expect(errors(giant({ core: { mass: 1.2 } })).some((t) => /core’s mass/.test(t))).toBe(true);
    expect(errors(giant({ mass: undefined, rhoc: 2 })).some((t) => /not its central density/.test(t))).toBe(true);
    // And a softening under two finest cells (8 / 256 here) is pointed out.
    expect(checkSetup(setup([giant({ core: { mass: 0.4, soft: 0.03 } })])).some((i) => i.level === 'warning' && /under two cells/.test(i.text))).toBe(true);
  });
});
