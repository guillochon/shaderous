/**
 * Spherical (r, theta) runs (Problem.geometry 'spherical', docs/numerics.md section 22): the
 * grid's geometry on the CPU, as hydro/common.wgsl (sph_cell, sph_volume) has it on the GPU.
 * Code x is r from rMin to rMin + box, uniform or logarithmic; y is theta from 0 to pi.
 */
export interface SphereShape {
  problem: { box: number; geometry?: string; sphere?: { rMin: number; log?: boolean } };
  n: number;
  ndim: number;
}

export interface SphGrid {
  n: number;
  /** the grid's reference length, box / n: the dx of Params */
  dx: number;
  rMin: number;
  rMax: number;
  /** the step in ln r, or 0 for uniform spacing */
  dlog: number;
  dtheta: number;
  /** radius of face i (0 to n) */
  rFace(i: number): number;
  /** radius of cell i's center (the geometric mean of its faces' on a logarithmic grid) */
  r(i: number): number;
  /** theta of cell j's center, and of face j */
  theta(j: number): number;
  thetaFace(j: number): number;
  /** the volume of cell (i, j), the azimuth's 2 pi included */
  volume(i: number, j: number): number;
  /** the volume of the shell of cells i */
  shell(i: number): number;
}

export const spherical = (c: SphereShape): boolean => c.problem.geometry === 'spherical' && c.ndim === 2;

export function sphGrid(c: SphereShape): SphGrid {
  const n = c.n;
  const rMin = c.problem.sphere?.rMin ?? 0;
  const rMax = rMin + c.problem.box;
  const log = !!c.problem.sphere?.log;
  const dlog = log ? Math.log(rMax / rMin) / n : 0;
  const dr = c.problem.box / n;
  const dtheta = Math.PI / n;
  const rFace = (i: number) => (log ? rMin * Math.exp(i * dlog) : rMin + i * dr);
  const r = (i: number) => (log ? rMin * Math.exp((i + 0.5) * dlog) : rMin + (i + 0.5) * dr);
  const thetaFace = (j: number) => j * dtheta;
  const theta = (j: number) => (j + 0.5) * dtheta;
  // (r_+^3 - r_-^3) / 3 and cos(theta_-) - cos(theta_+), in closed forms.
  const v3 = (i: number) => {
    const rc = r(i);
    return log ? (2 / 3) * rc ** 3 * Math.sinh(1.5 * dlog) : rc * rc * dr + dr ** 3 / 12;
  };
  const volume = (i: number, j: number) => 2 * Math.PI * v3(i) * 2 * Math.sin(theta(j)) * Math.sin(dtheta / 2);
  const shell = (i: number) => 4 * Math.PI * v3(i);
  return { n, dx: dr, rMin, rMax, dlog, dtheta, rFace, r, theta, thetaFace, volume, shell };
}

/** Params.sph and Params.sph2 (hydro/common.wgsl): zero unless the run is spherical. */
export function sphParams(c: SphereShape): number[] {
  if (!spherical(c)) return [0, 0, 0, 0, 0, 0, 0, 0];
  const g = sphGrid(c);
  const D = g.dlog;
  const consts = D > 0 ? [Math.exp(D / 2), 3 / (2 * Math.sinh(1.5 * D)), Math.sinh(D), 2 * Math.sinh(D / 2)] : [0, 0, 0, 0];
  // r_min / dx: the uniform grid's r_c / dx is this + i + 1/2, the logarithmic one's this e^((i + 1/2) D).
  return [g.rMin / g.dx, D, g.dtheta, 1, ...consts];
}

/**
 * A spherical run's setup at code positions, as the block grid places its cells (amr/init_cpu.ts):
 * code x runs from r_min over the box, uniform in the fraction u = (x - r_min) / box, which is r
 * itself on a uniform grid and r_min (r_max / r_min)^u on a logarithmic one; code y runs over the
 * box from 0, theta = pi y / box. At a cell's center these are sphGrid's r(i) and theta(j). Any
 * other run's setup is returned as it is.
 */
export function sphSetup<T>(c: SphereShape, fn: (x: number, y: number, z: number) => T): (x: number, y: number, z: number) => T {
  if (!spherical(c)) return fn;
  const g = sphGrid(c);
  const box = c.problem.box;
  const span = Math.log(g.rMax / g.rMin);
  const r = (x: number) => (g.dlog > 0 ? g.rMin * Math.exp(((x - g.rMin) / box) * span) : x);
  return (x, y, z) => fn(r(x), (Math.PI * y) / box, z);
}
