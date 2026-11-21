// Gas regions, obstacles and inflows: shapes painted over what is under them.
import type { Eos } from '../../eos/eos';
import type { Prim } from '../../problems/types';
import type { ObstaclePart, RegionPart, Shape, Vec3 } from '../spec';
import { placeMesh } from '../../geometry/store';

export interface PaintContext {
  box: number;
  n: number;
  ndim: 2 | 3;
  eos: Eos;
  /** mass fractions in the run's species order, from fractions by name */
  fractions(X: Record<string, number>): number[];
}

type Inside = (x: number, y: number, z: number) => boolean;

/** Bounds of a box shape may be null: open on that side. */
const bound = (v: number | null | undefined, open: number) => (v === null || v === undefined ? open : v);

/** The rotation by these degrees about x, then y, then z: R = Rz Ry Rx, as rows. */
export function rotationMatrix(deg: readonly number[]): number[][] {
  const [cx, cy, cz] = [0, 1, 2].map((a) => Math.cos(((deg[a] ?? 0) * Math.PI) / 180));
  const [sx, sy, sz] = [0, 1, 2].map((a) => Math.sin(((deg[a] ?? 0) * Math.PI) / 180));
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
}

/** The degrees about x, then y, then z of a rotation matrix (the inverse of rotationMatrix). */
export function eulerOf(R: number[][]): Vec3 {
  const d = (r: number) => (r * 180) / Math.PI;
  const sy = Math.max(-1, Math.min(1, -R[2][0]));
  if (Math.abs(sy) > 1 - 1e-12) return [d(Math.atan2(-R[1][2], R[1][1])), d(Math.asin(sy)), 0];
  return [d(Math.atan2(R[2][1], R[2][2])), d(Math.asin(sy)), d(Math.atan2(R[1][0], R[0][0]))];
}

/** The point a box turns about: its center; on an axis open on one side its one bound, open on both 0. */
export function boxPivot(shape: Extract<Shape, { type: 'box' }>): Vec3 {
  return [0, 1, 2].map((a) => {
    const [l, h] = [shape.min?.[a], shape.max?.[a]];
    const fl = typeof l === 'number' && Number.isFinite(l);
    const fh = typeof h === 'number' && Number.isFinite(h);
    return fl && fh ? (l + h) / 2 : fl ? l : fh ? h : 0;
  }) as Vec3;
}

/**
 * The frame of a turned box: the point it turns about (its center; on an open axis its one bound,
 * or 0) and the rotation, about x, then y, then z, as `world` (box to world) and `local` (back).
 * Null for a box that is not turned. In 2D only the turn about z counts.
 */
export function boxFrame(shape: Extract<Shape, { type: 'box' }>, flat = false): { pivot: Vec3; world(v: number[]): Vec3; local(v: number[]): Vec3 } | null {
  const deg = [0, 1, 2].map((a) => (flat && a < 2 ? 0 : Number(shape.rotate?.[a]) || 0));
  if (deg.every((d) => d % 360 === 0)) return null;
  const R = rotationMatrix(deg);
  return {
    pivot: boxPivot(shape),
    world: (v) => [0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2]) as Vec3,
    local: (v) => [0, 1, 2].map((i) => R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2]) as Vec3,
  };
}

export function insideOf(shape: Shape, ctx: { box: number; n: number; ndim: 2 | 3 }): { inside: Inside; center: Vec3; radius: number } {
  if (shape.type === 'sphere') {
    const c = shape.center ?? [0, 0, 0];
    const r = shape.radiusCells !== undefined ? shape.radiusCells * (ctx.box / ctx.n) : (shape.radius ?? 0.1 * ctx.box);
    const centered = c[0] === 0 && c[1] === 0 && c[2] === 0;
    // In 2D z is 0 at every cell, and the sphere is a cylinder along it.
    const flat = ctx.ndim === 2;
    const inside: Inside = centered
      ? flat ? (x, y) => Math.hypot(x, y) < r : (x, y, z) => Math.hypot(x, y, z) < r
      : flat ? (x, y) => Math.hypot(x - c[0], y - c[1]) < r : (x, y, z) => Math.hypot(x - c[0], y - c[1], z - c[2]) < r;
    return { inside, center: c, radius: r };
  }
  if (shape.type === 'box') {
    // A side left out is open, as a null bound is.
    const lo = [0, 1, 2].map((a) => bound(shape.min?.[a], -Infinity));
    const hi = [0, 1, 2].map((a) => bound(shape.max?.[a], Infinity));
    const flat = ctx.ndim === 2;
    const within = (x: number, y: number, z: number) => x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1] && (flat || (z > lo[2] && z < hi[2]));
    const frame = boxFrame(shape, flat);
    if (!frame) return { inside: within, center: [0, 0, 0], radius: 0 };
    // A turned box: the point taken into the box's own frame, about its center.
    const { pivot: c, local } = frame;
    const inside: Inside = (x, y, z) => {
      const q = local([x - c[0], y - c[1], flat ? 0 : z - c[2]]);
      return within(q[0] + c[0], q[1] + c[1], q[2] + c[2]);
    };
    return { inside, center: [0, 0, 0], radius: 0 };
  }
  if (shape.type === 'mesh') {
    // A mesh file (geometry/store.ts): its distance field at the finest cell size. In 2D every
    // cell is at z = 0: the cross-section through the plane z = 0 of the placed mesh.
    const m = placeMesh(shape, { box: ctx.box, ndim: ctx.ndim, dx: ctx.box / ctx.n });
    return { inside: (x, y, z) => m.inside(x, y, ctx.ndim === 2 ? 0 : z), center: m.center, radius: 0 };
  }
  const normal = shape.normal ?? [1, 0, 0];
  const norm = Math.hypot(...normal) || 1;
  const d = normal.map((c) => c / norm);
  const at = shape.at ?? 0;
  return { inside: (x, y, z) => x * d[0] + y * d[1] + z * d[2] < at, center: [0, 0, 0], radius: 0 };
}

/**
 * The point a body's torque is taken about: a sphere's center, a box's pivot (its center), the
 * origin for a half-space.
 */
export function bodyCenter(shape: Shape): Vec3 {
  if (shape.type === 'sphere') return [...(shape.center ?? [0, 0, 0])] as Vec3;
  if (shape.type === 'box') return boxPivot(shape);
  if (shape.type === 'mesh') return [0, 1, 2].map((a) => shape.center?.[a] ?? 0) as Vec3;
  return [0, 0, 0];
}

type Painter = (w: Prim, x: number, y: number, z: number) => Prim;

/** A region's painter: the state under it, replaced inside its shape (or blended at a supersampled edge). */
export function regionPainter(part: RegionPart, ctx: PaintContext): Painter {
  const { inside, center, radius } = insideOf(part.shape, ctx);
  const dx = ctx.box / ctx.n;
  const X = part.X ? ctx.fractions(part.X) : undefined;
  // A total energy, spread evenly over the cells whose centers the region holds: the pressure
  // that energy density gives at the region's density. Counted for a sphere (a point explosion).
  let pEnergy: ((rho: number) => number) | null = null;
  if (part.energy !== undefined) {
    if (part.shape.type !== 'sphere') throw new Error('a region with a total energy must be a sphere');
    const reach = Math.ceil(radius / dx) + 1;
    const flat = ctx.ndim === 2;
    const [i0, j0, k0] = center.map((c) => Math.floor(c / dx));
    let count = 0;
    for (let k = flat ? 0 : -reach; k < (flat ? 1 : reach); k++)
      for (let j = -reach; j < reach; j++)
        for (let i = -reach; i < reach; i++) {
          const [a, b, c] = [i0 + i + 0.5, j0 + j + 0.5, flat ? 0 : k0 + k + 0.5];
          if (Math.hypot(a - center[0] / dx, b - center[1] / dx, flat ? 0 : c - center[2] / dx) * dx < radius) count++;
        }
    const eDep = part.energy / (Math.max(count, 1) * dx ** ctx.ndim);
    pEnergy = (rho) => ctx.eos.pressure(rho, eDep);
  }
  // The radiation of a cell: the region's where it is given and the cell is the region's
  // (`own`), else what was under it.
  const radiationOf = (out: Prim, w: Prim, own: boolean) => {
    const erad = own ? (part.erad ?? w.erad) : w.erad;
    const frad = own ? (part.frad ?? w.frad) : w.frad;
    if (erad !== undefined) out.erad = erad;
    if (frad !== undefined) out.frad = frad;
  };
  const set = (w: Prim): Prim => {
    const rho = part.rho ?? w.rho;
    const v = part.v;
    const out: Prim = {
      rho,
      vx: v ? v[0] : w.vx, vy: v ? v[1] : w.vy, vz: v ? v[2] : w.vz,
      p: pEnergy ? pEnergy(rho) : (part.p ?? w.p),
      s: part.s ?? w.s,
    };
    if (X ?? w.X) out.X = X ?? w.X;
    if (w.solid) out.solid = true;
    if (w.fixed) out.fixed = true;
    radiationOf(out, w, true);
    return out;
  };
  const sub = part.supersample;
  if (!sub || part.shape.type !== 'sphere') return (w, x, y, z) => (inside(x, y, z) ? set(w) : w);
  // Cells the surface crosses take the share of their volume inside it.
  const [cx, cy, cz] = center;
  const flat = ctx.ndim === 2;
  const o = (q: number) => ((q + 0.5) / sub - 0.5) * dx;
  const mix = (a: number, b: number | undefined, f: number) => (b === undefined ? a : a + f * (b - a));
  return (w, x, y, z) => {
    const r = Math.hypot(x - cx, y - cy, flat ? 0 : z - cz);
    let frac = r < radius ? 1 : 0;
    if (Math.abs(r - radius) < dx) {
      let hits = 0;
      for (let c = 0; c < (flat ? 1 : sub); c++)
        for (let b = 0; b < sub; b++)
          for (let a = 0; a < sub; a++) if (Math.hypot(x + o(a) - cx, y + o(b) - cy, flat ? 0 : z + o(c) - cz) < radius) hits++;
      frac = hits / (flat ? sub * sub : sub ** 3);
    }
    const out: Prim = {
      rho: mix(w.rho, part.rho, frac),
      vx: mix(w.vx, part.v?.[0], frac), vy: mix(w.vy, part.v?.[1], frac), vz: mix(w.vz, part.v?.[2], frac),
      p: mix(w.p, part.p, frac),
      s: mix(w.s, part.s, frac),
    };
    if (w.X || X) out.X = X && frac >= 0.5 ? X : w.X;
    if (w.solid) out.solid = true;
    if (w.fixed) out.fixed = true;
    radiationOf(out, w, frac >= 0.5);
    return out;
  };
}

/**
 * An obstacle: its cells are solid, at rest, and belong to body `body` (the obstacle's place
 * among the setup's obstacles: its forces are summed apart). An inflow: its cells (not solid
 * ones) keep their state for the whole run.
 */
export function obstaclePainter(part: ObstaclePart, ctx: PaintContext, body = 0): Painter {
  const { inside } = insideOf(part.shape, ctx);
  if (part.kind === 'obstacle') return (w, x, y, z) => (inside(x, y, z) ? { ...w, vx: 0, vy: 0, vz: 0, solid: true, fixed: false, body } : w);
  return (w, x, y, z) => (!w.solid && inside(x, y, z) ? { ...w, fixed: true } : w);
}
