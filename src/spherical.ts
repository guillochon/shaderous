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

export const spherical = (c: SphereShape): boolean => c.problem.geometry === 'spherical' && c.ndim < 3;

export function sphGrid(c: SphereShape): SphGrid {
  const n = c.n;
  const rMin = c.problem.sphere?.rMin ?? 0;
  const rMax = rMin + c.problem.box;
  const log = !!c.problem.sphere?.log;
  const dlog = log ? Math.log(rMax / rMin) / n : 0;
  const dr = c.problem.box / n;
  // (A 1D run's one theta cell spans the sphere: plans/one-dimensional.md.)
  const dtheta = c.ndim === 1 ? Math.PI : Math.PI / n;
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

/*
 * The slice view of a spherical run draws the meridional half-plane, (R, z) = (r sin theta,
 * r cos theta): the view's square is R from 0 to 2 r_max and z from -r_max to r_max, so the
 * grid's half-disk (R up to r_max) fills its left half (the axis its left
 * edge, as R starts there in an (R, z) run). Positions in the square are fractions (a, b) of its
 * side from its lower left corner; the grid's are fractions of its cells (index fractions, so
 * logarithmic in r on a logarithmic grid). render/slice.wgsl sph_code() is sphFromView().
 */

/**
 * The view draws the grid as the half-disk (sphFromView): a spherical run, or a 1D axisymmetric
 * one, whose R the view repeats around the polar angle as rings (plans/one-dimensional.md). The
 * latter is the same map with no inner radius (sphGrid's r_min 0, as it has no Problem.sphere).
 */
export const curvedView = (c: SphereShape): boolean => spherical(c) || (c.problem.geometry === 'cylindrical' && c.ndim === 1);

/**
 * Camera mir.w (render/camera.wgsl): 0, or r_min / r_max of a spherical run, negative if r is
 * logarithmic. A 1D axisymmetric run, whose map has no inner radius, takes a value too small to
 * matter that is not 0 (0 says the view is not curved).
 */
export function sphViewParam(c: SphereShape): number {
  if (curvedView(c) && !spherical(c)) return 1e-30;
  if (!spherical(c)) return 0;
  const g = sphGrid(c);
  return (g.dlog > 0 ? -1 : 1) * (g.rMin / g.rMax);
}

/** The cell fractions (along r, along theta) under a point of the view's square, or null off the grid. */
export function sphFromView(c: SphereShape, a: number, b: number): [number, number] | null {
  const g = sphGrid(c);
  const R = 2 * a;
  const z = 2 * b - 1;
  const rho = Math.hypot(R, z);
  const f = g.rMin / g.rMax;
  if (R < 0 || rho < f || rho >= 1) return null;
  const fx = g.dlog > 0 ? Math.log(rho / f) / Math.log(1 / f) : (rho - f) / (1 - f);
  const fy = Math.atan2(R, z) / Math.PI;
  return [Math.min(fx, 1 - 1e-9), Math.min(fy, 1 - 1e-9)];
}

/**
 * The point of the view's square at cell fractions (along r, along theta): sphFromView's
 * inverse. Short of the grid (a point mass at the origin) r follows the box's fractions
 * linearly, as src/domain.ts boxFrac has them.
 */
export function sphToView(c: SphereShape, fx: number, fy: number): [number, number] {
  const g = sphGrid(c);
  const r = g.dlog > 0 && fx >= 0 ? g.rMin * Math.exp(fx * g.dlog * g.n) : g.rMin + fx * (g.rMax - g.rMin);
  const th = fy * Math.PI;
  const rho = Math.max(0, r) / g.rMax;
  return [(rho * Math.sin(th)) / 2, (1 + rho * Math.cos(th)) / 2];
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
