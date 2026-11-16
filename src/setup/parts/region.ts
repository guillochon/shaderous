// Gas regions, obstacles and inflows: shapes painted over what is under them.
import type { Eos } from '../../eos/eos';
import type { Prim } from '../../problems/types';
import type { ObstaclePart, RegionPart, Shape, Vec3 } from '../spec';

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
    const lo = shape.min.map((v) => bound(v, -Infinity));
    const hi = shape.max.map((v) => bound(v, Infinity));
    const inside: Inside = (x, y, z) => x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1] && (ctx.ndim === 2 || (z > lo[2] && z < hi[2]));
    return { inside, center: [0, 0, 0], radius: 0 };
  }
  const norm = Math.hypot(...shape.normal);
  const d = shape.normal.map((c) => c / norm);
  const at = shape.at ?? 0;
  return { inside: (x, y, z) => x * d[0] + y * d[1] + z * d[2] < at, center: [0, 0, 0], radius: 0 };
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
    return out;
  };
}

/** An obstacle: its cells are solid, at rest. An inflow: its cells (not solid ones) keep their state for the whole run. */
export function obstaclePainter(part: ObstaclePart, ctx: PaintContext): Painter {
  const { inside } = insideOf(part.shape, ctx);
  if (part.kind === 'obstacle') return (w, x, y, z) => (inside(x, y, z) ? { ...w, vx: 0, vy: 0, vz: 0, solid: true, fixed: false } : w);
  return (w, x, y, z) => (!w.solid && inside(x, y, z) ? { ...w, fixed: true } : w);
}
