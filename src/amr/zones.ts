// Refinement zones (mesh.zones): parts of the box an adaptive mesh keeps at the finest level, each
// with a tier that says who gives up blocks first when the block budget runs out. Tier 1 is kept
// longest; a block in no zone has no tier (Infinity) and is given up before any zone's.
//
// A point belongs to the first zone in the list that holds it, so a zone listed later yields to
// those before it where they overlap (a cone that bridges two spheres yields to them). A
// block's tier is the best tier of the points inside it. The zones do not move: in the frame
// that follows a binary, L1 and the two stars stay where they start.
//
// All of this is the CPU's: the tree decides what to refine and what to merge (amr/tree.ts
// adapt), from each block's place alone. Nothing of it is on the GPU.

/** A plane that cuts a zone: only the side below (or above) `value` along `axis` is kept. */
export interface ZoneCut {
  axis: 0 | 1 | 2;
  value: number;
  /** keep the side above the plane, not below it */
  above?: boolean;
}

/** A zone's shape, worked out from the setup (setup/compile.ts). Coordinates are the box's. */
export type ZoneShape =
  | { shape: 'sphere'; c: number[]; rIn: number; r: number }
  /** the cone tangent to two spheres (their convex hull, less the spheres themselves) */
  | { shape: 'cone'; c1: number[]; r1: number; c2: number[]; r2: number };

export type Zone = ZoneShape & { tier: number; cuts: ZoneCut[] };

const sub = (a: number[], b: number[]) => [0, 1, 2].map((k) => (a[k] ?? 0) - (b[k] ?? 0));
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Whether a point is in a zone's shape, its cuts aside. */
function inShape(z: ZoneShape, p: number[]): boolean {
  if (z.shape === 'sphere') {
    const d = Math.hypot(...sub(p, z.c));
    return d <= z.r && d >= z.rIn;
  }
  // The cone tangent to both spheres: along the axis from c1 to c2 (length L), with
  // sin(alpha) = (r2 - r1) / L, it touches sphere 1 at u1 = -r1 sin(alpha) and sphere 2 at
  // L - r2 sin(alpha), its radius going linearly from r1 cos(alpha) to r2 cos(alpha) between.
  const ax = sub(z.c2, z.c1);
  const L = Math.hypot(...ax);
  if (!(L > Math.abs(z.r2 - z.r1))) return false;
  const e = ax.map((v) => v / L);
  const v = sub(p, z.c1);
  const u = dot(v, e);
  const rho = Math.sqrt(Math.max(dot(v, v) - u * u, 0));
  const s = (z.r2 - z.r1) / L;
  const c = Math.sqrt(1 - s * s);
  const u1 = -z.r1 * s;
  const u2 = L - z.r2 * s;
  if (u < u1 || u > u2) return false;
  return rho <= z.r1 * c + ((u - u1) / (u2 - u1)) * (z.r2 * c - z.r1 * c);
}

/** Whether a point is in a zone: its shape, on the kept side of every cut. */
export function inZone(z: Zone, p: number[]): boolean {
  for (const cut of z.cuts) {
    const x = p[cut.axis] ?? 0;
    if (cut.above ? x < cut.value : x > cut.value) return false;
  }
  return inShape(z, p);
}

/** The place in the list of the first zone holding a point, or -1. */
export function zoneAt(zones: readonly Zone[], p: number[]): number {
  for (let i = 0; i < zones.length; i++) if (inZone(zones[i], p)) return i;
  return -1;
}

/** The tier of a point: that of the first zone holding it, or Infinity. */
export function pointTier(zones: readonly Zone[], p: number[]): number {
  const i = zoneAt(zones, p);
  return i < 0 ? Infinity : zones[i].tier;
}

/** A sphere every point of the zone's shape lies in, for a quick test that a block is far from it. */
function bounds(z: Zone): { c: number[]; r: number } {
  if (z.shape === 'sphere') return { c: z.c, r: z.r };
  const c = [0, 1, 2].map((k) => 0.5 * ((z.c1[k] ?? 0) + (z.c2[k] ?? 0)));
  const L = Math.hypot(...sub(z.c2, z.c1));
  return { c, r: 0.5 * L + Math.max(z.r1, z.r2) };
}

/**
 * The tier of a block (its lower corner `lo` and side `side`, in 2D or 3D): the best of its
 * points', sampled at most `h` apart (half a finest block, so that no zone a finest block could
 * hold slips between the samples) and at least four to a side.
 */
export function blockTier(zones: readonly Zone[], lo: number[], side: number, ndim: 1 | 2 | 3, h: number): number {
  if (!zones.length) return Infinity;
  const near = zones.filter((z) => {
    const b = bounds(z);
    let d2 = 0;
    for (let k = 0; k < ndim; k++) {
      const q = Math.max(lo[k] - b.c[k], 0, b.c[k] - (lo[k] + side));
      d2 += q * q;
    }
    return d2 <= b.r * b.r;
  });
  if (!near.length) return Infinity;
  const n = Math.max(4, Math.ceil(side / h));
  const step = side / n;
  let best = Infinity;
  const nz = ndim === 3 ? n : 0;
  // Cell centers of an n-per-side grid, then the corners: a zone grazing a corner counts too.
  for (let k = 0; k <= nz; k++)
    for (let j = 0; j <= n; j++)
      for (let i = 0; i <= n; i++) {
        const p = [lo[0] + i * step, lo[1] + j * step, ndim === 3 ? lo[2] + k * step : 0];
        const t = pointTier(near, p);
        if (t < best) best = t;
      }
  for (let k = 0; k < Math.max(nz, 1); k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const p = [lo[0] + (i + 0.5) * step, lo[1] + (j + 0.5) * step, ndim === 3 ? lo[2] + (k + 0.5) * step : 0];
        const t = pointTier(near, p);
        if (t < best) best = t;
      }
  return best;
}
