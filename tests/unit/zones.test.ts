// Refinement zones (amr/zones.ts, mesh.zones): their shapes, which zone a point is in, a block's
// tier, what a setup's zones come to, and how the tree spends a short block budget by tier.
import { describe, expect, it } from 'vitest';
import { blockTier, inZone, pointTier, type Zone } from '../../src/amr/zones';
import { BlockTree, type Block } from '../../src/amr/tree';
import { innerLagrange } from '../../src/problems/roche';
import { compileSetup, zonesOfSetup } from '../../src/setup/compile';
import { buildBodies } from '../../src/setup/parts/bodies';
import type { SetupSpec } from '../../src/setup/spec';
import { emptySetup, type Part } from '../../src/setup/spec';

const sphere = (c: number[], r: number, tier: number, rIn = 0, cuts: Zone['cuts'] = []): Zone => ({ shape: 'sphere', c, r, rIn, tier, cuts });

describe('refinement zones', () => {
  it('a shell, cut to a half by a plane', () => {
    const bottom = sphere([1, 0, 0], 2, 2, 1, [{ axis: 1, value: 0 }]);
    expect(inZone(bottom, [1, -1.5, 0])).toBe(true);
    expect(inZone(bottom, [1, 1.5, 0])).toBe(false); // above the plane
    expect(inZone(bottom, [1, -0.5, 0])).toBe(false); // inside the inner radius
    expect(inZone(bottom, [1, -2.5, 0])).toBe(false); // outside the outer
    const top = { ...bottom, cuts: [{ axis: 1 as const, value: 0, above: true }] };
    expect(inZone(top, [1, 1.5, 0])).toBe(true);
  });

  it('a cone tangent to two spheres, bridging them', () => {
    // Spheres of radius 1 at x = 0 and 3 at x = 10: sin(alpha) = 0.2.
    const cone: Zone = { shape: 'cone', c1: [0, 0, 0], r1: 1, c2: [10, 0, 0], r2: 3, tier: 1, cuts: [] };
    const s = 0.2;
    const c = Math.sqrt(1 - s * s);
    const u1 = -1 * s;
    const u2 = 10 - 3 * s;
    const radius = (u: number) => c + ((u - u1) / (u2 - u1)) * (3 * c - c);
    for (const u of [0, 2, 5, 9]) {
      expect(inZone(cone, [u, 0.99 * radius(u), 0])).toBe(true);
      expect(inZone(cone, [u, 0, -1.01 * radius(u)])).toBe(false);
    }
    // The tangent circles lie on both spheres.
    expect(Math.hypot(u1, c)).toBeCloseTo(1, 12);
    expect(Math.hypot(u2 - 10, 3 * c)).toBeCloseTo(3, 12);
    // Cut above a plane tangent to the top of the first sphere.
    const cut = { ...cone, cuts: [{ axis: 1 as const, value: 1 }] };
    expect(inZone(cut, [5, 1.5, 0])).toBe(false);
    expect(inZone(cut, [5, -1.5, 0])).toBe(true);
  });

  it('a point is in the first zone that holds it: a cone listed last yields to the zones before it', () => {
    const zones: Zone[] = [sphere([10, 0, 0], 6, 2, 3), { shape: 'cone', c1: [0, 0, 0], r1: 1, c2: [10, 0, 0], r2: 6, tier: 1, cuts: [] }];
    expect(pointTier(zones, [2, 0, 0])).toBe(1); // the cone alone
    expect(pointTier(zones, [5.5, 0, 0])).toBe(2); // the cone inside the shell: the shell's tier
    expect(pointTier(zones, [20, 0, 0])).toBe(Infinity);
  });

  it('a block takes the best tier of its points, a small zone in a large block included', () => {
    const zones = [sphere([0.71, 0.33, 0.52], 0.03, 1)];
    expect(blockTier(zones, [0, 0, 0], 1, 3, 0.03)).toBe(1);
    expect(blockTier(zones, [1, 0, 0], 1, 3, 0.03)).toBe(Infinity);
  });
});

describe('the zones of a setup (mesh.zones)', () => {
  const donor: Part = { kind: 'star', id: 'donor', type: 'polytrope', n: 1.5, mass: 0.5, radius: 0.5 };
  const acc: Part = { kind: 'star', id: 'acc', type: 'polytrope', n: 1.5, mass: 1, radius: 0.35 };
  const spec: SetupSpec = {
    ...emptySetup(),
    domain: { ndim: 3, n: 256, amr: 2, box: 8 },
    physics: { gravity: 'isolated', G: 1 },
    parts: [donor, acc, { kind: 'orbit', of: ['donor', 'acc'], type: 'binary', fill: 0.9, corotating: true, track: true }],
    mesh: {
      zones: [
        { name: 'L1', shape: 'sphere', at: 'L1', r: 0.25, tier: 1 },
        { shape: 'sphere', at: 'acc', rIn: 1, r: 2, radii: true, below: { y: 0 }, tier: 2 },
        { shape: 'sphere', at: 'acc', rIn: 1, r: 2, radii: true, above: { y: 0 }, tier: 3 },
        { shape: 'sphere', at: 'acc', r: 1, radii: true, tier: 4 },
        { shape: 'sphere', at: 'donor', r: 1.5, radii: true, tier: 5 },
        { shape: 'cone', from: { at: 'L1', r: 0.25 }, to: { at: 'acc', r: 2, radii: true }, below: { y: 0.25 }, tier: 1 },
      ],
    },
  };

  it('places them at L1 and the stars, in the stars’ radii where asked', () => {
    const b = buildBodies(spec, { G: 1, dx: 0, ndim: 3, box: 8 });
    const L1 = innerLagrange(b.binary!.lobe!.bodies, b.binary!.omega).at;
    const a = b.stars.find((s) => s.id === 'acc')!;
    const zones = zonesOfSetup(spec);
    expect(zones).toHaveLength(6);
    const z0 = zones[0] as Extract<Zone, { shape: 'sphere' }>;
    z0.c.forEach((v, k) => expect(v).toBeCloseTo(L1[k] ?? 0, 12));
    const z1 = zones[1] as Extract<Zone, { shape: 'sphere' }>;
    expect(z1.c[0]).toBeCloseTo(a.center[0], 12);
    expect(z1.rIn).toBeCloseTo(a.R, 12);
    expect(z1.r).toBeCloseTo(2 * a.R, 12);
    expect(z1.cuts).toEqual([{ axis: 1, value: 0 }]);
    expect(zones[2].cuts).toEqual([{ axis: 1, value: 0, above: true }]);
    const cone = zones[5] as Extract<Zone, { shape: 'cone' }>;
    expect(cone.r1).toBe(0.25);
    expect(cone.r2).toBeCloseTo(2 * a.R, 12);
    // Between L1 and the accretor's shell, below the plane y = 0.25: the cone's tier 1. On the
    // stream's side of the shell: the shell's tier 2, as it is listed first. Above the plane: no zone.
    const mid = [0.5 * (L1[0] + a.center[0] - 2 * a.R), -0.1, 0];
    expect(pointTier(zones, mid)).toBe(1);
    expect(pointTier(zones, [a.center[0], -1.5 * a.R, 0])).toBe(2);
    expect(pointTier(zones, [mid[0], 0.4, 0])).toBe(Infinity);
    // The compiled problem carries them.
    expect(compileSetup(spec).problem.refineZones!({})).toEqual(zones);
    expect(compileSetup({ ...spec, mesh: {} }).problem.refineZones).toBeUndefined();
  });

  it('refuses L1 without a binary, and a star it does not have', () => {
    expect(() => zonesOfSetup({ ...spec, mesh: { zones: [{ shape: 'sphere', at: 'nobody', r: 1, tier: 1 }] } })).toThrow(/neither L1 nor a star/);
    const single: SetupSpec = { ...spec, parts: [donor], mesh: { zones: [{ shape: 'sphere', at: 'L1', r: 1, tier: 1 }] } };
    expect(() => zonesOfSetup(single)).toThrow(/circular binary/);
  });
});

describe('a short block budget spent by tier', () => {
  // A 2D tree over the unit square (2 x 2 roots, 3 levels: a finest block is 1/16 across) with
  // every block flagged alike and room for far fewer than all of them at the finest level.
  const tierAt = (zones: Zone[]) => (b: Block) => {
    const size = 1 / (2 << b.level);
    return blockTier(zones, [b.c[0] * size, b.c[1] * size, 0], size, 2, 1 / 64);
  };
  const levelAt = (t: BlockTree, p: number[]) => t.leaves().find((b) => {
    const size = 1 / (2 << b.level);
    return p[0] >= b.c[0] * size && p[0] < (b.c[0] + 1) * size && p[1] >= b.c[1] * size && p[1] < (b.c[1] + 1) * size;
  })!.level;
  const grow = (tier?: (b: Block) => number, ind: (b: Block) => number = () => 1) => {
    const t = new BlockTree(2, 2, 3, 60, [1, 1, 0], [1, 1, 0]);
    for (let i = 0; i < 12; i++) t.adapt(ind, 0.5, 0.2, tier);
    return t;
  };

  it('keeps a zone at the finest level, ahead of blocks in no zone', () => {
    const zones = [sphere([0.8, 0.8, 0], 0.08, 1)];
    // Without the zone, flagged alike everywhere, the corner at (0.8, 0.8) is not at the finest level.
    expect(levelAt(grow(), [0.8, 0.8])).toBeLessThan(3);
    const t = grow(tierAt(zones));
    expect(levelAt(t, [0.8, 0.8])).toBe(3);
    expect(levelAt(t, [0.86, 0.8])).toBe(3);
  });

  it('refines a zone whatever its indicator, and never merges it', () => {
    const zones = [sphere([0.3, 0.3, 0], 0.05, 1)];
    const t = grow(tierAt(zones), () => 0);
    expect(levelAt(t, [0.3, 0.3])).toBe(3);
    // Elsewhere, nothing flagged: no finer than 2:1 balance about the zone asks.
    expect(levelAt(t, [0.9, 0.9])).toBe(0);
    expect(levelAt(t, [0.9, 0.1])).toBeLessThan(2);
  });

  it('gives up the lower tier first when the budget cannot hold both', () => {
    // Two discs, each of which the budget can hold at the finest level on its own, not both.
    const a = sphere([0.25, 0.25, 0], 0.12, 1);
    const b = sphere([0.75, 0.75, 0], 0.12, 2);
    expect(levelAt(grow(tierAt([b])), [0.75, 0.75])).toBe(3);
    for (const order of [[a, b], [b, a]]) {
      const t = grow(tierAt(order));
      expect(levelAt(t, [0.25, 0.25])).toBe(3);
      expect(levelAt(t, [0.75, 0.75])).toBeLessThan(3);
    }
    // The tiers swapped: the other disc wins.
    const t = grow(tierAt([{ ...a, tier: 2 }, { ...b, tier: 1 }]));
    expect(levelAt(t, [0.75, 0.75])).toBe(3);
    expect(levelAt(t, [0.25, 0.25])).toBeLessThan(3);
  });
});
