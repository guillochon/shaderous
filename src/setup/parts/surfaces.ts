// Bodies' surfaces for an adaptive mesh (plans/wind-tunnel.md 4.3, 9.2): which blocks come within
// a band of an obstacle's surface, from the shape itself (its exact distance, or points on a
// mesh's triangles), not from the solid cells a coarse grid happens to have. A blade thinner than
// a coarse cell is in pieces on that grid, or missing; asked for from its own shape, every block
// along it is refined to the level it names, and its cells are painted at that level
// (amr/bodies.ts).
import type { ObstaclePart, Shape, Vec3 } from '../spec';
import { boxFrame, insideOf } from './region';
import { meshSurfacePoints, placeMesh } from '../../geometry/store';
import { chainPose, poseApply, poseUnapply, type Moving, type Pose } from '../../geometry/motion';
import { REFINE_MODES } from '../spec';
import { MAX_BODIES } from '../../problems/types';

export type RefineMode = (typeof REFINE_MODES)[number];

/** The band about a surface that is refined, in finest cells, unless an obstacle says. */
export const SURFACE_BAND_CELLS = 4;

export interface BodySurface {
  /** its cells' mark (amr: U2.w): 1 + its place among the obstacles */
  mark: number;
  /** whether a point of the box is inside it, where it starts */
  inside(x: number, y: number, z: number): boolean;
  /** the level (0 the coarsest) its band is refined to at least; undefined: the finest */
  level?: number;
  /**
   * Whether the block with lower corner `lo` and side `size` (in the box's coordinates) comes
   * within the band of the surface, or with `refine: "swept"` of anywhere the surface passes.
   * Conservative: may say yes for a block a little further away, never no for one within.
   */
  near(lo: readonly number[], size: number): boolean;
}

interface Ctx {
  box: number;
  /** finest cells across the box */
  n: number;
  ndim: 2 | 3;
  /** the run's end time, for what a body sweeps */
  tEnd: number;
  /** the domain's lower corner (domain.ts domainLo) */
  lo: readonly number[];
}

/** A shape's signed distance (negative inside), exact, where it has one in closed form. */
export function shapeDistance(shape: Shape, ctx: { box: number; n: number; ndim: 2 | 3 }): ((x: number, y: number, z: number) => number) | null {
  const flat = ctx.ndim === 2;
  if (shape.type === 'sphere') {
    const c = shape.center ?? [0, 0, 0];
    const r = shape.radiusCells !== undefined ? shape.radiusCells * (ctx.box / ctx.n) : (shape.radius ?? 0.1 * ctx.box);
    return (x, y, z) => Math.hypot(x - c[0], y - c[1], flat ? 0 : z - c[2]) - r;
  }
  if (shape.type === 'box') {
    const lo = [0, 1, 2].map((a) => (shape.min?.[a] === null || shape.min?.[a] === undefined ? -Infinity : shape.min[a]));
    const hi = [0, 1, 2].map((a) => (shape.max?.[a] === null || shape.max?.[a] === undefined ? Infinity : shape.max[a]));
    const frame = boxFrame(shape, flat);
    const axes = flat ? 2 : 3;
    return (x, y, z) => {
      let p = [x, y, flat ? 0 : z];
      if (frame) {
        const c = frame.pivot;
        const q = frame.local([p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
        p = [q[0] + c[0], q[1] + c[1], q[2] + c[2]];
      }
      let out = 0;
      let inMax = -Infinity;
      for (let a = 0; a < axes; a++) {
        const q = Math.max(lo[a] - p[a], p[a] - hi[a]); // -Infinity on an open axis
        if (q > 0) out += q * q;
        inMax = Math.max(inMax, q);
      }
      return out > 0 ? Math.sqrt(out) : inMax;
    };
  }
  if (shape.type === 'half') {
    const normal = shape.normal ?? [1, 0, 0];
    const norm = Math.hypot(...normal) || 1;
    const d = normal.map((c) => c / norm);
    const at = shape.at ?? 0;
    return (x, y, z) => x * d[0] + y * d[1] + (flat ? 0 : z * d[2]) - at;
  }
  return null;
}

/**
 * Points on a shape's surface where it has no distance in closed form (a mesh), in the box's
 * frame, at most `h` apart: in 3D on its triangles; in 2D (a mesh's section at z = 0, or an
 * outline) the cells of a grid of pitch h over its bounds that have the inside on one side and
 * not the other.
 */
function surfacePoints(shape: Extract<Shape, { type: 'mesh' }>, ctx: Ctx, h: number): Float64Array {
  if (ctx.ndim === 3) {
    try {
      return meshSurfacePoints(shape, ctx, h);
    } catch {
      // (an outline in 3D, an extrusion: from its inside, as in 2D, at each layer)
    }
  }
  const m = placeMesh(shape, { box: ctx.box, ndim: ctx.ndim, dx: ctx.box / ctx.n });
  const lo = m.min.map((v) => v - 2 * h);
  const hi = m.max.map((v) => v + 2 * h);
  const N = [0, 1, 2].map((a) => (a < ctx.ndim ? Math.max(1, Math.ceil((hi[a] - lo[a]) / h)) : 1));
  const zs = ctx.ndim === 3 ? N[2] : 1;
  const out: number[] = [];
  const pt = (i: number, j: number, k: number): Vec3 => [lo[0] + i * h, lo[1] + j * h, ctx.ndim === 3 ? lo[2] + k * h : 0];
  for (let k = 0; k < zs; k++)
    for (let j = 0; j < N[1]; j++) {
      let prev = m.inside(...pt(0, j, k));
      for (let i = 1; i < N[0]; i++) {
        const p = pt(i, j, k);
        const now = m.inside(...p);
        const below = j > 0 && m.inside(...pt(i, j - 1, k));
        if (now !== prev || now !== below) out.push(...p);
        prev = now;
      }
    }
  return Float64Array.from(out);
}

/** The poses a body passes through over the run (or one turn of its motions), close enough together that its points move at most `step` between them. */
function sweptPoses(chain: Moving[], i: number, ctx: Ctx, reach: number, step: number): { poses: Pose[]; gap: number } {
  let horizon = Number.isFinite(ctx.tEnd) ? ctx.tEnd : 0;
  if (!horizon) {
    // A run without an end: one period of the slowest of its motions, when they all repeat.
    for (let k = i, guard = 0; k >= 0 && guard <= chain.length; k = chain[k].parent, guard++) {
      const m = chain[k].motion;
      if (!m) continue;
      if (m.slide || m.path || (m.spin && typeof m.spin.rate !== 'number')) return { poses: [chainPose(chain, i, 0)], gap: 0 };
      if (m.spin) horizon = Math.max(horizon, (2 * Math.PI) / Math.max(Math.abs(m.spin.rate as number), 1e-12));
      if (m.swing) horizon = Math.max(horizon, m.swing.period);
    }
  }
  if (!horizon) return { poses: [chainPose(chain, i, 0)], gap: 0 };
  // Points within `reach` of the origin: how far they move between two poses.
  const probes: Vec3[] = [[reach, 0, 0], [-reach, 0, 0], [0, reach, 0], [0, -reach, 0], [0, 0, reach], [0, 0, -reach], [0, 0, 0]];
  for (let count = 64; ; count *= 2) {
    const poses = Array.from({ length: count + 1 }, (_, k) => chainPose(chain, i, (horizon * k) / count));
    let gap = 0;
    for (let k = 1; k < poses.length; k++)
      for (const p of probes) {
        const a = poseApply(poses[k - 1], p);
        const b = poseApply(poses[k], p);
        gap = Math.max(gap, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
      }
    if (gap <= step || count >= 4096) return { poses, gap };
  }
}

/**
 * Each obstacle's signed distance (negative inside) where it starts, and its cells' mark: exact for
 * spheres, boxes and half-spaces, a mesh's distance field (exact within a few finest cells of its
 * surface, clamped beyond) for a mesh. In 2D at z = 0.
 */
export function obstacleDistances(obstacles: ObstaclePart[], ctx: { box: number; n: number; ndim: 2 | 3 }): { mark: number; distance(x: number, y: number, z: number): number }[] {
  return obstacles.map((p, i) => {
    const mark = 1 + Math.min(i, MAX_BODIES - 1);
    const d = shapeDistance(p.shape, ctx);
    if (d) return { mark, distance: d };
    const m = placeMesh(p.shape as Extract<Shape, { type: 'mesh' }>, { box: ctx.box, ndim: ctx.ndim, dx: ctx.box / ctx.n });
    return { mark, distance: ctx.ndim === 2 ? (x: number, y: number) => m.distance(x, y, 0) : m.distance };
  });
}

/**
 * The surfaces of a setup's obstacles that ask for refinement (all but those with
 * `refine: "off"`), in the order of their marks.
 */
export function bodySurfaces(obstacles: ObstaclePart[], ctx: Ctx): BodySurface[] {
  const h = ctx.box / ctx.n;
  const byId = new Map(obstacles.map((p, i) => [p.id ?? `obstacle ${i + 1}`, i]));
  const chain: Moving[] = obstacles.map((p) => ({ motion: p.motion, parent: p.parent !== undefined ? (byId.get(p.parent) ?? -1) : -1 }));
  const out: BodySurface[] = [];
  obstacles.forEach((p, i) => {
    const mode: RefineMode = p.refine ?? 'surface';
    if (mode === 'off') return;
    const band = Math.max(0, p.refineBand ?? SURFACE_BAND_CELLS) * h;
    const { inside } = insideOf(p.shape, ctx);
    const dist = shapeDistance(p.shape, ctx);
    const moves = chain.some((_, k) => {
      for (let q = i, guard = 0; q >= 0 && guard <= chain.length; q = chain[q].parent, guard++) if (q === k && chain[q].motion) return true;
      return false;
    });
    // Its reach from the box's center, for how far its points move between sampled poses.
    const reach = Math.sqrt(ctx.ndim) * ctx.box;
    const { poses, gap } = mode === 'swept' && moves ? sweptPoses(chain, i, ctx, reach, Math.max(band, h) / 2) : { poses: [chainPose(chain, i, 0)], gap: 0 };
    const margin = band + gap / 2;
    const flatZ = ctx.ndim === 2;
    let near: BodySurface['near'];
    if (dist) {
      near = (lo, size) => {
        const c = [lo[0] + size / 2, lo[1] + size / 2, flatZ ? 0 : lo[2] + size / 2];
        const r = (size / 2) * Math.sqrt(ctx.ndim) + margin;
        let last = 0;
        for (const pose of poses) {
          const q = poseUnapply(pose, c);
          const d = dist(q[0], q[1], flatZ ? 0 : q[2]);
          if (Math.abs(d) <= r) return true;
          // The surface passed over the block between two poses.
          if (last && Math.sign(d) !== Math.sign(last)) return true;
          last = d;
        }
        return false;
      };
    } else {
      // A mesh: its surface's points, each pose's, binned by the blocks of each size they come
      // near (blocks of one size lie on one grid, from the domain's lower corner).
      const base = surfacePoints(p.shape as Extract<Shape, { type: 'mesh' }>, ctx, h);
      const bins = new Map<number, Set<number>>();
      const K = 1 << 16; // block coordinates are within (-K/2, K/2)
      const keyOf = (a: number, b: number, c: number) => ((c + K / 2) * K + (b + K / 2)) * K + (a + K / 2);
      const binsFor = (size: number) => {
        let s = bins.get(size);
        if (s) return s;
        s = new Set<number>();
        const o = ctx.lo;
        for (const pose of poses) {
          for (let k = 0; k < base.length; k += 3) {
            const q = poses.length === 1 ? [base[k], base[k + 1], base[k + 2]] : poseApply(pose, [base[k], base[k + 1], base[k + 2]]);
            const a0 = Math.floor((q[0] - margin - o[0]) / size), a1 = Math.floor((q[0] + margin - o[0]) / size);
            const b0 = Math.floor((q[1] - margin - o[1]) / size), b1 = Math.floor((q[1] + margin - o[1]) / size);
            const c0 = flatZ ? 0 : Math.floor((q[2] - margin - o[2]) / size), c1 = flatZ ? 0 : Math.floor((q[2] + margin - o[2]) / size);
            for (let cc = c0; cc <= c1; cc++) for (let bb = b0; bb <= b1; bb++) for (let aa = a0; aa <= a1; aa++) s.add(keyOf(aa, bb, cc));
          }
        }
        bins.set(size, s);
        return s;
      };
      near = (lo, size) => {
        const i3 = [0, 1, 2].map((a) => (a < ctx.ndim ? Math.round((lo[a] - ctx.lo[a]) / size) : 0));
        return binsFor(size).has(keyOf(i3[0], i3[1], i3[2]));
      };
    }
    out.push({ mark: 1 + Math.min(i, MAX_BODIES - 1), inside, level: p.refineLevel, near });
  });
  return out;
}
