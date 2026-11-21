import { describe, expect, it } from 'vitest';
import { bodySurfaces, shapeDistance } from '../../src/setup/parts/surfaces';
import type { ObstaclePart } from '../../src/setup/spec';

// Obstacles' surfaces on an adaptive mesh (setup/parts/surfaces.ts): which blocks come within a
// band of a body, from its shape.
describe('bodies refined from their shapes', () => {
  const ctx = { box: 1, n: 128, ndim: 3 as const, tEnd: Infinity, lo: [-0.5, -0.5, -0.5] };
  const h = 1 / 128;
  const block = (level: number) => 16 / (32 << level); // a block's side at a level, 2 levels below 32 cells

  it('gives exact distances for spheres, boxes (open sides included) and half-spaces', () => {
    const s = shapeDistance({ type: 'sphere', center: [0.1, 0, 0], radius: 0.2 }, ctx)!;
    expect(s(0.1, 0, 0)).toBeCloseTo(-0.2, 12);
    expect(s(0.4, 0, 0)).toBeCloseTo(0.1, 12);
    const b = shapeDistance({ type: 'box', min: [-0.1, -0.2, -0.3], max: [0.1, 0.2, 0.3] }, ctx)!;
    expect(b(0, 0, 0)).toBeCloseTo(-0.1, 12);
    expect(b(0.4, 0.6, 0)).toBeCloseTo(Math.hypot(0.3, 0.4), 12);
    const plate = shapeDistance({ type: 'box', min: [-0.01, null, null], max: [0.01, null, null] } as never, ctx)!;
    expect(plate(0.05, 0.4, -0.4)).toBeCloseTo(0.04, 12);
    const half = shapeDistance({ type: 'half', normal: [0, 0, 2], at: 0.1 }, ctx)!;
    expect(half(0, 0, 0.3)).toBeCloseTo(0.2, 12);
  });

  it('finds the blocks along a plate thinner than a coarse cell, and no others', () => {
    const plate: ObstaclePart = { kind: 'obstacle', shape: { type: 'box', min: [-0.45 / 32, null, null], max: [0.45 / 32, null, null] } as never };
    const [s] = bodySurfaces([plate], ctx);
    for (let level = 0; level <= 2; level++) {
      const size = block(level);
      for (let i = 0; i * size < 1; i++) {
        const lo = [-0.5 + i * size, 0, 0];
        // Within the band (4 finest cells) of either face, or straddling it.
        const gap = Math.max(0, lo[0] - 0.45 / 32, -0.45 / 32 - (lo[0] + size));
        const within = gap <= 4 * h;
        if (within) expect(s.near(lo, size)).toBe(true);
        if (gap > 4 * h + size) expect(s.near(lo, size)).toBe(false);
      }
    }
  });

  it('a mesh sphere is near wherever the built-in sphere is', () => {
    const shape = { type: 'sphere' as const, center: [0, 0, 0] as [number, number, number], radius: 0.2 };
    const [a] = bodySurfaces([{ kind: 'obstacle', shape }], ctx);
    const [m] = bodySurfaces([{ kind: 'obstacle', shape: { type: 'mesh', file: 'builtin:sphere', fit: 0.4, center: [0, 0, 0] } as never }], ctx);
    const d = shapeDistance(shape, ctx)!;
    let both = 0;
    for (let level = 0; level <= 2; level++) {
      const size = block(level);
      const n = Math.round(1 / size);
      for (let k = 0; k < n; k++)
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const lo = [-0.5 + i * size, -0.5 + j * size, -0.5 + k * size];
            const c = lo.map((v) => v + size / 2);
            // Blocks whose box surely holds part of the surface's band.
            const sure = Math.abs(d(c[0], c[1], c[2])) < size / 2;
            if (sure) {
              expect(m.near(lo, size)).toBe(true);
              both++;
            }
            // The mesh's points miss nothing the exact sphere has, give or take its facets.
            if (a.near(lo, size) === false && Math.abs(d(c[0], c[1], c[2])) > (size / 2) * Math.sqrt(3) + 6 * h) expect(m.near(lo, size)).toBe(false);
          }
    }
    expect(both).toBeGreaterThan(0);
  });

  it('refine "swept" covers where a spinning blade goes, "surface" only where it starts', () => {
    const blade = (refine: 'surface' | 'swept'): ObstaclePart => ({
      kind: 'obstacle',
      shape: { type: 'box', min: [-0.02, 0.05, -0.03], max: [0.02, 0.4, 0.03] },
      motion: { spin: { axis: [1, 0, 0], rate: 2 } },
      refine,
    });
    const [surface] = bodySurfaces([blade('surface')], ctx);
    const [swept] = bodySurfaces([blade('swept')], ctx);
    const size = block(2);
    // A finest block on the far side of the disk (the blade starts along +y).
    const far = [-size / 2, -0.3, -size / 2];
    expect(surface.near(far, size)).toBe(false);
    expect(swept.near(far, size)).toBe(true);
    // Off the disk, along the axis beyond the blade's thickness: neither.
    const off = [0.2, -0.3, -size / 2];
    expect(swept.near(off, size)).toBe(false);
  });

  it('skips obstacles with their own refinement off, and keeps the level asked for', () => {
    const list = bodySurfaces(
      [
        { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1 }, refine: 'off' },
        { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1, center: [0.2, 0, 0] }, refineLevel: 1 },
      ],
      ctx,
    );
    expect(list.length).toBe(1);
    expect(list[0].mark).toBe(2);
    expect(list[0].level).toBe(1);
  });
});
