// The presets by the names the validation suite and the tests know them by, with the numbers
// of their setups those check against. The setups themselves are the files of this folder;
// what is here is read from them, or derived as the parts derive it.
import { laneEmden, laneEmdenCached, polytropeK, polytropeScales } from '../problems/lane_emden';
import type { Problem } from '../problems/types';
import { buildBodies, flybyOrbit } from '../setup/parts/bodies';
import { buildDisk } from '../setup/parts/disk';
import type { DiskPart, MassPart, RegionPart, StarPart } from '../setup/spec';

import { ASTRO } from '../units/constants';
import { gravityUnits, type UnitSystem } from '../units/units';
import { preset, presetSpec } from './index';

export { bubbleDensity, postShock } from '../setup/parts/flows/shock_bubble';
export { DETONATION_CJ_SPEED, zndOf } from '../setup/parts/flows/detonation';
export { RT } from '../setup/parts/flows/rayleigh_taylor';
export { JEANS } from '../setup/parts/flows/jeans';
export { TI, balanceTemperature } from '../setup/parts/flows/thermal_instability';
export { binaryMasses, collisionOrbit } from '../setup/parts/bodies';

const problem = (id: string): Problem => preset(id).problem;
const spec = (id: string) => presetSpec(id)!;

// ---- shock tubes ----
export interface TubeState {
  rho: number;
  u: number;
  p: number;
}
export type Tube = Problem & { dir: [number, number, number]; left: TubeState; right: TubeState };

/** A shock tube preset with its two states and the direction of its normal. */
function tube(id: string): Tube {
  const s = spec(id);
  const region = s.parts![0] as RegionPart;
  const shape = region.shape as { normal: [number, number, number] };
  const norm = Math.hypot(...shape.normal);
  return {
    ...problem(id),
    dir: shape.normal.map((c) => c / norm) as [number, number, number],
    left: { rho: region.rho!, u: 0, p: region.p! },
    right: { rho: s.background!.rho!, u: 0, p: s.background!.p! },
  };
}
export const sodX = tube('sod-x');
export const sodY = tube('sod-y');
export const sodZ = tube('sod-z');
export const sodDiag = tube('sod-diag');
export const strongShock = tube('strong-shock');

// ---- blasts and spheres ----
/** Sedov-Taylor similarity constant: R(t) = xi0 (E t^2 / rho)^(1/5). */
export const SEDOV_XI0: Record<string, number> = { '1.4': 1.033, '1.6667': 1.152 };
export const sedov = problem('sedov');
export const sedovRad = problem('sedov-rad');

const sphereRegion = spec('sphere').parts![0] as RegionPart;
/** The uniform sphere: its radius and density, G, and the density around it. */
export const SPHERE = { R: (sphereRegion.shape as { radius: number }).radius, rho: sphereRegion.rho!, G: problem('sphere').G, rhoAmb: spec('sphere').background!.rho! };
export function spherePotential(r: number): number {
  const M = (4 / 3) * Math.PI * SPHERE.R ** 3 * SPHERE.rho;
  return r < SPHERE.R ? (-SPHERE.G * M * (3 * SPHERE.R ** 2 - r * r)) / (2 * SPHERE.R ** 3) : (-SPHERE.G * M) / r;
}
export const sphere = problem('sphere');

// ---- the polytrope ----
const polyPart = spec('polytrope').parts![0] as StarPart;
/** The polytrope preset: radius, central density, G, the ambient density and where the n = 1 star is cut off. */
export const POLY = { R: polyPart.radius!, rhoc: polyPart.rhoc!, G: problem('polytrope').G, rhoAmb: spec('polytrope').background!.rho!, rhoCut: 1e-2 };
export const POLY_K = (2 * POLY.G * POLY.R ** 2) / Math.PI;

/** The star of index n in code units, as the polytrope preset builds it. */
export function polyStar(n = 1, R = POLY.R) {
  const st = buildBodies({ ...spec('polytrope'), parts: [{ ...polyPart, n, radius: R }] }, { G: POLY.G, dx: 0, ndim: 3, box: 1 }).stars[0];
  const g = 1 + 1 / n;
  const K = n === 1 ? (2 * POLY.G * R ** 2) / Math.PI : polytropeK(laneEmdenCached(n), POLY.rhoc, R, POLY.G);
  return { n, K, mass: st.M, rhoCut: POLY.rhoc * Math.pow(POLY.rhoAmb / POLY.rhoc, 1 / g), density: st.rho, pressure: st.pressure };
}


/** What one code unit is for a star of `m` solar masses and `r` solar radii with index `n` and radius `size` in code units. */
export function polytropeUnits(p: { m?: number; r?: number; n?: number; size?: number }): UnitSystem {
  return problemUnitsOf('polytrope', { m: p.m ?? 1, r: p.r ?? 1, n: p.n ?? 1, size: p.size ?? POLY.R });
}
function problemUnitsOf(id: string, params: Record<string, number>): UnitSystem {
  const u = problem(id).units;
  return typeof u === 'function' ? u(params) : u!;
}
export const polytrope = problem('polytrope');
export const polytropeDipole = problem('polytrope-dipole');
export const polytropeRandom = problem('polytrope-random');

// ---- the tidal disruption ----
const tdeSpec = spec('tidal-disruption');
const tdePart = tdeSpec.parts![0] as StarPart;
/** The tidal disruption: the star's index, mass and radius, G, where the mass starts (tidal radii), and the box its tests were calibrated in. */
export const TDE = { n: tdePart.n!, M: tdePart.mass!, R: tdePart.radius!, G: 1, start: 3, rhoAmb: tdeSpec.background!.rho!, rhoCut: tdeSpec.background!.cut!, box: 8 };
export const TDE_UNITS = problem('tidal-disruption').units as UnitSystem;
/** The gamma = 5/3 (n = 3/2) star: its central density and K, and its profile. */
export const TDE_STAR = (() => {
  const le = laneEmden(TDE.n);
  return { ...polytropeScales(le, TDE.M, TDE.R, TDE.G), theta: (s: number) => le.theta(s) };
})();
/** The perturber's orbit for a mass ratio and an impact parameter. */
export const tdeOrbit = (p: { q: number; beta: number }) => flybyOrbit(p.q, p.beta, TDE.start, TDE.M, TDE.R, TDE.G);
export const tidalDisruption = problem('tidal-disruption');
export const tidalDisruptionDipole = problem('tidal-disruption-dipole');

// ---- the magnetized disk ----
const mriSpec = spec('mri-disk');
const mriPart = mriSpec.parts![1] as DiskPart;
export const MRI = {
  cs: mriPart.cs, rIn: mriPart.rIn, rOut: mriPart.rOut, rho0: mriPart.sigma0, amb: mriPart.floor!, loop: mriPart.loops!.span, nLoop: mriPart.loops!.count,
  box: mriSpec.domain!.box!, rDamp: mriPart.damp!.from, damp: mriPart.damp!.rate, vaMax: mriSpec.physics!.fluff!.va!, refineDense: mriSpec.mesh!.refineDense!,
};
/** The disk's midplane density at cylindrical radius R. */
export const midplane = (R: number) => MRI.rho0 * R ** (mriPart.slope ?? -1.5) * Math.exp(-((MRI.rIn / R) ** 8) - (R / MRI.rOut) ** 8);
/** The disk in balance at (R, z) for a softening `eps`: density, the disk's share of it, and orbital speed. */
export function diskState(R: number, z: number, eps: number): { rho: number; disk: number; v: number } {
  const w = buildDisk(mriPart, { box: MRI.box, n: 1, ndim: 3, gm: (mriSpec.parts![0] as MassPart).gm, soft: eps }).state(R, 0, z);
  return { rho: w.rho, disk: w.rho - MRI.amb, v: Math.hypot(w.vx, w.vy) };
}
export const mriDisk = problem('mri-disk');

// ---- the empty box ----
export const EMPTY_UNITS = gravityUnits('solar', ASTRO.Rsun, ASTRO.Msun);
export const empty = problem('empty');
export const emptyMagnetic = problem('empty-mhd');


// ---- the rest, by name ----
export const shockBubble = problem('shock-bubble');
export const detonationStable = problem('detonation-stable');
export const advection = problem('advection');
export const kelvinHelmholtz = problem('kelvin-helmholtz');
export const rayleighTaylor = problem('rayleigh-taylor');
export const jeans = problem('jeans');
export const whiteDwarfStar = problem('white-dwarf');
export const evrard = problem('evrard');
export const binary = problem('binary');
export const thermalInstability = problem('thermal-instability');
export const alfvenWave = problem('alfven-wave');
export const brioWu = problem('brio-wu');
export const fieldLoop = problem('field-loop');
export const fieldLoopDense = problem('field-loop-dense');
export const mhdBlast = problem('mhd-blast');
export const mhdRotor = problem('mhd-rotor');
export const orszagTang = problem('orszag-tang');
