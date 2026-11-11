import { describe, expect, it } from 'vitest';
import { laneEmden, polytropeK, polytropeMass, polytropeScales } from '../../src/problems/lane_emden';
import { packBrush, STAR_TABLE, starBrush } from '../../src/sim';
import { POLY, POLY_K } from '../../src/problems/polytrope';

describe('star brush', () => {
  it('normalizes as the polytrope preset does', () => {
    // n = 1 has K = 2 G R^2 / pi whatever the central density.
    expect(polytropeK(laneEmden(1), POLY.rhoc, POLY.R, POLY.G)).toBeCloseTo(POLY_K, 4);
    expect(polytropeK(laneEmden(1), 7, POLY.R, POLY.G)).toBeCloseTo(POLY_K, 4);
  });

  it('inverts polytropeScales', () => {
    for (const n of [1, 1.5, 3]) {
      const poly = laneEmden(n);
      const { rhoc, K } = polytropeScales(poly, 2.5, 0.7, 1.3);
      expect(polytropeMass(poly, rhoc, 0.7)).toBeCloseTo(2.5, 6);
      expect(polytropeK(poly, rhoc, 0.7, 1.3)).toBeCloseTo(K, 6);
    }
  });

  it('packs the profile into the brush block', () => {
    const op = starBrush(1.5, 2, 0.1, 4, 1, [0.5, 0.25, 0.75], [1, 0, 0]);
    const t = op.star!.table;
    expect(t.length).toBe(STAR_TABLE);
    expect(t[0]).toBeCloseTo(1, 5);
    expect(t[STAR_TABLE - 1]).toBe(0);
    for (let i = 1; i < STAR_TABLE; i++) expect(t[i]).toBeLessThanOrEqual(t[i - 1]);
    const f = new Float32Array(packBrush(op));
    expect(new Uint32Array(f.buffer)[8]).toBe(6);
    expect(f[9]).toBe(2);
    expect(f[28]).toBeCloseTo(op.star!.K, 5);
    expect(f[29]).toBeCloseTo(1 + 1 / 1.5, 6);
    expect(f[32]).toBeCloseTo(1, 5);
  });
});
