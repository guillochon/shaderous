import { describe, expect, it } from 'vitest';
import { BURN_REFINE_SCALE, burnCell, burnIndicator, cjSpeed, znd } from '../../src/burn/burn';
import { defaultEosParams, makeEos } from '../../src/eos/eos';

// The standard parameters of detonation numerics: gamma 1.2, q 50, unburned rho = p = 1.
const G = 1.2;
const Q = 50;

describe('ZND detonation', () => {
  it('has the Chapman-Jouguet speed of the literature', () => {
    expect(cjSpeed(G, Q)).toBeCloseTo(6.80947, 4);
  });

  it('ends sonic at the CJ speed, and conserves mass, momentum and energy across it', () => {
    const D = cjSpeed(G, Q);
    const z = znd(G, Q, 25, D);
    for (const x of [0, 0.5, 1, 3, 1e3]) {
      const s = z.at(x);
      const w = D - s.u;
      expect(s.rho * w).toBeCloseTo(D, 9);
      expect(s.p + s.rho * w * w).toBeCloseTo(1 + D * D, 7);
      const h = (G / (G - 1)) * (s.p / s.rho) + 0.5 * w * w - Q * (1 - s.f);
      expect(h).toBeCloseTo(G / (G - 1) + 0.5 * D * D, 6);
    }
    const end = z.at(1e3);
    expect(end.f).toBeLessThan(1e-9);
    // Sonic relative to the front at the end of the reaction.
    expect(D - end.u).toBeCloseTo(Math.sqrt((G * end.p) / end.rho), 3);
    // Half of the fuel burned one unit behind the shock.
    expect(z.at(1).f).toBeCloseTo(0.5, 4);
  });

  it('matches the rate constant Henrick, Aslam & Powers (2006) give for Ea = 25', () => {
    // K = 35.955584760859722 there (half-reaction length 1).
    const z = znd(G, Q, 25, cjSpeed(G, Q));
    expect(z.K).toBeCloseTo(35.9556, 3);
    // von Neumann spike: the shock jump of a Mach D / sqrt(gamma) shock.
    const M2 = cjSpeed(G, Q) ** 2 / G;
    expect(z.vonNeumann.p).toBeCloseTo(1 + ((2 * G) / (G + 1)) * (M2 - 1), 6);
  });
});

describe('burnCell', () => {
  const T = (rho: number, e: number) => (0.2 * e) / rho;
  const spec = { fuel: 'fuel', ash: 'ash', q: 50, ea: 25, k: 30 };
  it('conserves energy: what is released is q times what burned', () => {
    const [f, e] = burnCell(spec, T, 2, 0.8, 400, 0.05);
    expect(f).toBeLessThan(0.8);
    expect(f).toBeGreaterThanOrEqual(0);
    expect(e - 400).toBeCloseTo(2 * 50 * (0.8 - f), 9);
  });
  it('does nothing in cold gas or without fuel', () => {
    expect(burnCell(spec, T, 1, 1, 1e-6, 1)[0]).toBeCloseTo(1, 12);
    expect(burnCell(spec, T, 1, 0, 400, 1)).toEqual([0, 400]);
  });
  it('converges to the exact burn at fixed temperature as dt -> 0', () => {
    const q0 = { ...spec, q: 1e-9 };
    const [f] = burnCell(q0, T, 1, 1, 100, 0.01);
    expect(f).toBeCloseTo(Math.exp(-30 * Math.exp(-25 / 20) * 0.01), 9);
  });
});

describe('refinement on the burning', () => {
  const D = cjSpeed(G, Q);
  const z = znd(G, Q, 25, D);
  const spec = { fuel: 'fuel', ash: 'ash', q: Q, ea: 25, k: z.K };
  const eos = makeEos('ideal', { ...defaultEosParams(), gamma: G });

  it('is BURN_REFINE_SCALE cells over the burning length c_s e / (rho q f k)', () => {
    const s = z.at(1);
    const T = s.p / s.rho;
    const l = (Math.sqrt((G * s.p) / s.rho) * (s.p / (G - 1))) / (s.rho * Q * s.f * z.K * Math.exp(-25 / T));
    expect(burnIndicator(spec, eos, s.rho, s.p, s.f, 0.01)).toBeCloseTo((BURN_REFINE_SCALE * 0.01) / l, 9);
  });

  it('asks the reaction zone of the steady detonation for about 30 cells per half-reaction length', () => {
    // The shortest burning length, about 2.5 half-reaction lengths just behind the half-burned point.
    // At a cell size small enough to stay under the cap of 1.
    const dx = 0.01;
    let peak = 0;
    for (let x = 0; x < 6; x += 0.01) {
      const s = z.at(x);
      peak = Math.max(peak, burnIndicator(spec, eos, s.rho, s.p, s.f, dx));
    }
    const cells = peak / dx / 0.8; // cells per half-reaction length where it crosses the default threshold
    expect(cells).toBeGreaterThan(25);
    expect(cells).toBeLessThan(40);
  });

  it('never outranks a shock (capped at 1)', () => {
    const s = z.at(1);
    expect(burnIndicator(spec, eos, s.rho, s.p, s.f, 1)).toBe(1);
  });

  it('asks nothing of cold fuel or burned gas', () => {
    expect(burnIndicator(spec, eos, 1, 1, 1, 1)).toBeLessThan(1e-6);
    expect(burnIndicator(spec, eos, 1.7, 20, 1e-9, 1)).toBeLessThan(1e-4);
    expect(burnIndicator(spec, eos, 1, 1, 0, 1)).toBe(0);
  });
});
