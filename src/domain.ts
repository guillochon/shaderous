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

/** A position as a fraction of the box from the domain's lower corner (x / box + 0.5 for a centered one), along `axis`. */
export function boxFrac(c: DomainShape, x: number, axis: number): number {
  return x / c.problem.box - domainLo(c)[axis] / c.problem.box;
}
/** The position at a fraction of the box along `axis` ((f - 0.5) box for a centered domain). */
export function fracPos(c: DomainShape, f: number, axis: number): number {
  return (f + domainLo(c)[axis] / c.problem.box) * c.problem.box;
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

/** The domain's upper corner, per axis. */
export function domainHi(c: DomainShape): [number, number, number] {
  const lo = domainLo(c);
  return lo.map((l) => l + c.problem.box) as [number, number, number];
}
