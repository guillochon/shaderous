/**
 * Where the computed domain lies: the one definition of a cell's position, shared by the CPU
 * and (through Params.org, hydro/common.wgsl) the shaders. Cell (i, j, k) of size dx is centered
 * at (i + 0.5) dx + lo, on each axis.
 */

/** What decides the domain's placement: the box side, and the mirrored axes if any. */
export interface DomainShape {
  problem: { box: number; geometry?: 'cartesian' | 'cylindrical' | 'spherical'; sphere?: { rMin: number } };
  /** Axes mirrored about the origin (Part B): their domain is [0, box], not [-box/2, box/2]. */
  symmetry?: readonly boolean[];
}

/** The domain's lower corner, per axis. */
export function domainLo(c: DomainShape): [number, number, number] {
  const half = c.problem.box / 2;
  const s = c.symmetry;
  // An axisymmetric run's R starts at the axis.
  const cyl = c.problem.geometry === 'cylindrical';
  // A spherical run's r starts at its inner radius, its theta at the pole (code positions:
  // theta is (j + 1/2) pi / n, not this).
  if (c.problem.geometry === 'spherical') return [c.problem.sphere?.rMin ?? 0, 0, -half];
  return [0, 1, 2].map((a) => (s?.[a] || (a === 0 && cyl) ? 0 : -half)) as [number, number, number];
}

/**
 * The computed domain's side along `axis`: the box, or half of it along a halved mirrored axis
 * of a rectangular grid (gridDims; the shape's `n` and `ndim` say which).
 */
export function domainSide(c: DomainShape & Partial<GridShape>, axis: number): number {
  if (c.n === undefined || c.ndim === undefined || axis >= c.ndim) return c.problem.box;
  return (c.problem.box * gridDims(c as GridShape)[axis]) / c.n;
}
/**
 * A position as a fraction of the computed domain from its lower corner, along `axis`
 * (x / box + 0.5 for a centered cube). Fractions are what the textures, the probe, the brushes
 * and the API take: 0 to 1 over the cells along each axis.
 */
export function boxFrac(c: DomainShape & Partial<GridShape>, x: number, axis: number): number {
  const side = domainSide(c, axis);
  // (Along a side of the whole box, the arithmetic it always had, bit for bit.)
  if (side === c.problem.box) return x / c.problem.box - domainLo(c)[axis] / c.problem.box;
  return (x - domainLo(c)[axis]) / side;
}
/** The position at a fraction of the computed domain along `axis` ((f - 0.5) box for a centered cube). */
export function fracPos(c: DomainShape & Partial<GridShape>, f: number, axis: number): number {
  const side = domainSide(c, axis);
  if (side === c.problem.box) return (f + domainLo(c)[axis] / c.problem.box) * c.problem.box;
  return domainLo(c)[axis] + f * side;
}

/** The axes a mirror string ('xz') names. */
export function mirrorAxes(s: string | undefined): [boolean, boolean, boolean] {
  return [0, 1, 2].map((a) => !!s?.includes('xyz'[a])) as [boolean, boolean, boolean];
}

/** The mirror string of a set of axes: 'none' for none. */
export function mirrorName(axes: readonly boolean[] | undefined): string {
  const s = [0, 1, 2].filter((a) => axes?.[a]).map((a) => 'xyz'[a]).join('');
  return s || 'none';
}

/** What sets the grid's cells along each axis. */
export interface GridShape {
  n: number;
  ndim: 2 | 3;
  symmetry?: readonly boolean[];
  problem: { mhd?: unknown };
  /** levels of the adaptive mesh above its root grid (0 or absent: the uniform grid) */
  amrLevels?: number;
}
/** Cells per side of a block (amr/tree.ts BS). */
const BLOCK = 16;

/**
 * Cells along each axis of the uniform grid (plans/coordinate-systems-plan.md B6). A mirrored
 * axis has half of them, over [0, box / 2], where some axis in use is not mirrored: a half or a
 * 3D quadrant run then computes its part of the same cube a run without mirrors would, at the
 * same cell size, for half or a quarter of the cost. Where every axis in use is mirrored (an
 * octant, a 2D quadrant) the grid stays a cube of n over [0, box]. One layer along z in 2D.
 */
export function gridDims(c: GridShape): [number, number, number] {
  const axes = c.ndim === 2 ? [0, 1] : [0, 1, 2];
  const s = c.symmetry ?? [false, false, false];
  // (Magnetic fields keep the cube for now: their face passes are sized for one. On blocks a
  // halved axis needs an even number of root blocks.)
  const roots = (c.n >> (c.amrLevels ?? 0)) / BLOCK;
  const blocksOk = !c.amrLevels || (Number.isInteger(roots) && roots % 2 === 0);
  const rect = !c.problem.mhd && blocksOk && axes.some((a) => !s[a]);
  const d = [0, 1, 2].map((a) => (rect && s[a] && axes.includes(a) ? c.n / 2 : c.n)) as [number, number, number];
  if (c.ndim === 2) d[2] = 1;
  return d;
}
/** The grid is rectangular: some mirrored axis is halved (gridDims). */
export const rectangular = (c: GridShape): boolean => gridDims(c).some((m, a) => m !== (a === 2 && c.ndim === 2 ? 1 : c.n));

/** The domain's upper corner, per axis: the box, or half of it along a halved mirrored axis (gridDims). */
export function domainHi(c: DomainShape & Partial<GridShape>): [number, number, number] {
  const lo = domainLo(c);
  const d = c.n !== undefined && c.ndim !== undefined ? gridDims(c as GridShape) : undefined;
  return lo.map((l, a) => l + (d && a < c.ndim! ? (c.problem.box * d[a]) / c.n! : c.problem.box)) as [number, number, number];
}
