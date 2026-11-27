import { describe, expect, it } from 'vitest';
import { dipoleField, randomField, type StarProfile } from '../../src/mhd/star_fields';

// An n = 1 polytrope, p = (sin(pi r) / (pi r))^2 in units of the center's, radius 1.
const star: StarProfile = {
  pressure: (r) => (r >= 1 ? 0 : r === 0 ? 1 : (Math.sin(Math.PI * r) / (Math.PI * r)) ** 2),
  pc: 1,
  pcut: 0,
  radius: 1,
};

// B = curl A by central differences.
const curl = (A: (x: number, y: number, z: number) => number[], p: number[], h = 1e-4): number[] => {
  const d = (a: number, axis: number) => {
    const q1 = [...p];
    const q0 = [...p];
    q1[axis] += h;
    q0[axis] -= h;
    return (A(q1[0], q1[1], q1[2])[a] - A(q0[0], q0[1], q0[2])[a]) / (2 * h);
  };
  return [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)];
};

// The mean plasma beta, 2p / B^2 averaged as B^2, over a shell of points at radius r.
const shellBeta = (A: (x: number, y: number, z: number) => number[], r: number): number => {
  let b2 = 0;
  const n = 400;
  for (let i = 0; i < n; i++) {
    // Points spread over the sphere (a Fibonacci lattice).
    const z = 1 - (2 * (i + 0.5)) / n;
    const s = Math.sqrt(1 - z * z);
    const ph = i * Math.PI * (3 - Math.sqrt(5));
    const B = curl(A, [r * s * Math.cos(ph), r * s * Math.sin(ph), r * z]);
    b2 += B[0] ** 2 + B[1] ** 2 + B[2] ** 2;
  }
  return (2 * star.pressure(r)) / (b2 / n);
};

describe('the taper of the fields inside stars (mhd/star_fields.ts)', () => {
  it('keeps the dipole at the requested beta at the center, either way', () => {
    for (const e of [1, 0.5]) {
      const B = curl(dipoleField(star, 1000, e), [0, 0, 0]);
      expect((2 * star.pc) / (B[0] ** 2 + B[1] ** 2 + B[2] ** 2)).toBeCloseTo(1000, 0);
    }
  });

  it('with sqrt(p), keeps the random field\'s beta about the same out to the outer layers', () => {
    const lin = randomField(star, 1e5, 1, 1);
    const sq = randomField(star, 1e5, 1, 0.5);
    // At r = 0.85 the pressure is 2.6% of the center's.
    const ratio = (A: typeof lin) => shellBeta(A, 0.85) / shellBeta(A, 0.3);
    // With the pressure, beta climbs about as 1 / p (B ~ p, plus the taper's own gradient).
    expect(ratio(lin)).toBeGreaterThan(10);
    // With its square root, it stays within a factor of a few.
    expect(ratio(sq)).toBeLessThan(3);
    expect(ratio(sq)).toBeGreaterThan(0.05);
  });
});
