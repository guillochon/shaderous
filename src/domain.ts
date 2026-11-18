/**
 * Where the computed domain lies: the one definition of a cell's position, shared by the CPU
 * and (through Params.org, hydro/common.wgsl) the shaders. Cell (i, j, k) of size dx is centered
 * at (i + 0.5) dx + lo, on each axis.
 */

/** What decides the domain's placement: the box side, and the mirrored axes if any. */
export interface DomainShape {
  problem: { box: number };
  /** Axes mirrored about the origin (Part B): their domain is [0, box], not [-box/2, box/2]. */
  symmetry?: readonly boolean[];
}

/** The domain's lower corner, per axis. */
export function domainLo(c: DomainShape): [number, number, number] {
  const half = c.problem.box / 2;
  const s = c.symmetry;
  return [0, 1, 2].map((a) => (s?.[a] ? 0 : -half)) as [number, number, number];
}

/** The domain's upper corner, per axis. */
export function domainHi(c: DomainShape): [number, number, number] {
  const lo = domainLo(c);
  return lo.map((l) => l + c.problem.box) as [number, number, number];
}
