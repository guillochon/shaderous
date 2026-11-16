// The setup document: everything that describes a run before it starts, as plain JSON.
//
// A preset (src/presets/*.json) and a setup a user saves are the same kind of document. Every
// key but `format` and `version` is optional: what is absent takes the default of the option
// table (setup/options.ts). compileSetup (setup/compile.ts) turns one into the Problem and the
// SimConfig overrides the solvers already take. No code and no formulas: a file from a
// stranger is safe to open.
import type { CoolingKind, CoolingParams } from '../cooling/cooling';
import type { EosKind, EosParams } from '../eos/eos';
import type { Bc3, GravityMode } from '../problems/types';

export const SETUP_FORMAT = 'shaderous-setup';
/** Counts changes to the document's shape: setup/codec.ts keeps one migration per step. */
export const SETUP_VERSION = 1;

export type Vec3 = [number, number, number];

/** A physical value for one code unit: `{ "pc": 0.01 }`, optionally per `per` code units of a part's quantity. */
export interface UnitValue {
  /** the unit's name (setup/units.ts UNIT_SI) */
  unit: string;
  value: number;
  /**
   * The code value that `value` units stand for: a number, or a part's quantity as
   * "<part id>.mass" or "<part id>.radius" (default 1).
   */
  per?: number | string;
}

/** What one code unit is. Length and mass (the clock then makes G = 1), or length, density and velocity. */
export interface UnitsSpec {
  name?: string;
  length: UnitValue;
  mass?: UnitValue;
  density?: UnitValue;
  velocity?: UnitValue;
}

export interface DomainSpec {
  /** The grid the setup opens on. Absent: the grid already chosen is kept (Problem.grid). */
  ndim?: 2 | 3;
  n?: number;
  amr?: number;
  /** side of the box: the domain is [-box/2, box/2]^3 */
  box?: number;
  supports2d?: boolean;
  bc?: { lo: Bc3; hi: Bc3 };
  /** boundaries in 2D, where they differ */
  bc2d?: { lo: Bc3; hi: Bc3 };
  units?: UnitsSpec;
}

/** What the gas is made of: a named composition (units/composition.ts COMPOSITIONS) or mass fractions. */
export type CompositionSpec = string | { name: string; X: Record<string, number>; ionization?: 'full' | 'neutral' };

export type BurnSetup =
  | { network: 'aprox13'; inShocks?: boolean; limiter?: number }
  | { fuel: string; ash: string; q: number; ea: number; k: number; inShocks?: boolean };

export interface PhysicsSpec {
  eos?: EosKind;
  /** the equation of state's parameters, over those the units and composition give */
  eosParams?: Partial<EosParams>;
  gravity?: GravityMode;
  /** G in code units, or 'units': what the setup's units make it */
  G?: number | 'units';
  /** uniform acceleration: a vector, or `down`, a strength along the axis drawn downward (-y in 2D, -z in 3D) */
  gext?: Vec3 | { down: number };
  cooling?: CoolingKind;
  coolParams?: Partial<CoolingParams>;
  viscosity?: number;
  conductivity?: number;
  riemann?: 'hllc' | 'hllclm' | 'hll';
  limiter?: 'koren' | 'mc' | 'vanleer' | 'minmod';
  cfl?: number;
  floors?: { rho?: number; p?: number };
  fluff?: { rho?: number; u?: number; v?: number; va?: number };
  frame?: 'inertial' | 'gas' | 'masses';
  jeansFloor?: boolean;
  jeansCells?: number;
  vcycles?: number;
  gravEnergyFlux?: boolean;
  composition?: CompositionSpec;
  /** species carried with the gas, by name, or 'aprox13' for the burning network's thirteen */
  species?: string[] | 'aprox13';
  speciesGamma?: Record<string, number>;
  burn?: BurnSetup;
  toolScale?: { rho: number; v: number };
}

export interface MeshSpec {
  refineHi?: number;
  refineLo?: number;
  refineMasses?: boolean;
  refineDense?: number;
  refineRho?: number;
  timeBins?: number;
  jeansRefine?: number;
}

export interface BackgroundSpec {
  rho?: number;
  p?: number;
  v?: Vec3;
  /** passive scalar */
  s?: number;
  /** mass fractions by species name */
  X?: Record<string, number>;
  /** With stars: the density below which a cell is ambient gas, not star. */
  cut?: number;
  /** `rho` and `cut` are fractions of this star's central density. */
  relativeTo?: string;
}

export interface RunSpec {
  /** null runs forever */
  tEnd?: number | null;
  checkpointDt?: number;
  tDyn?: number;
  startPaused?: boolean;
}

export interface ViewSpec {
  field?: number;
  log?: boolean;
  mode?: 'volume' | 'slice' | 'projection';
  axis?: number;
  pitch?: number;
}

// ---- parts ----

export interface MassPart {
  kind: 'mass';
  id?: string;
  /** G M */
  gm: number;
  pos?: Vec3;
  vel?: Vec3;
  /** softening length (at least two cells; absent: two cells) */
  soft?: number;
  /** sink radius (absent: the softening length; 0: none) */
  sink?: number;
  sinkRate?: number;
  massless?: boolean;
  fixed?: boolean;
}

export interface OrbitPart {
  kind: 'orbit';
  id?: string;
  /** the two bodies, by id: point masses or stars */
  of: [string, string];
  /**
   * 'binary': both bodies about their center of mass, set at pericenter, speeds lowered for
   *   the softened pull (the binary accretion setup).
   * 'flyby': the second body on a parabola about the first, which stays at the origin (a
   *   tidal disruption in the star's rest frame).
   * 'collision': two stars on a parabola about their center of mass, started a few radii apart.
   */
  type: 'binary' | 'flyby' | 'collision';
  /** mass of the second body over the first: sets their masses ('binary': the total stays; 'flyby': the first stays) */
  q?: number;
  /** binary: semi-major axis and eccentricity */
  separation?: number;
  ecc?: number;
  /** binary: tilt of the orbit about the line through pericenter, degrees (3D) */
  incl?: number;
  /** binary: the orbit runs clockwise (2D) */
  retrograde?: boolean;
  /** flyby: tidal radius over pericenter distance */
  beta?: number;
  /** flyby: starting separation in tidal radii (never inside 1.5 pericenters) */
  start?: number;
  /** collision: pericenter over the sum of the two radii, and the starting separation in sums of radii */
  impact?: number;
  starts?: number;
  /** flyby: bin the debris by orbital energy about the second body (debris/debris.ts) */
  debris?: boolean;
}

export interface StarField {
  type: 'dipole' | 'random' | 'gm17-dipole' | 'gm17-random';
  /** plasma beta at the center (dipole, random) or Guillochon & McCourt's beta_M */
  beta: number;
  seed?: number;
}

export interface StarPart {
  kind: 'star';
  id?: string;
  /** a polytrope of index n, or a cold white dwarf */
  type: 'polytrope' | 'white-dwarf';
  /** polytrope: its index and radius, and either its mass or its central density */
  n?: number;
  radius?: number;
  mass?: number;
  rhoc?: number;
  pos?: Vec3;
  vel?: Vec3;
  /** passive scalar of its gas (default: spread over (0, 1] by its place among the stars) */
  dye?: number;
  /** mass fractions by species name */
  X?: Record<string, number>;
  field?: StarField;
}

export interface DiskPart {
  kind: 'disk';
  id?: string;
  /** the bodies it orbits, by id */
  about: string[];
  /**
   * 'thin': a surface density with a cavity inside rIn and an outer taper, a Gaussian vertical
   *   profile of scale height cs / Omega, on circular orbits (the binary's disk).
   * 'balanced': a midplane density with tapers at rIn and rOut, in vertical balance with the
   *   softened potential, rotating as the pressure gradient asks (the magnetized disk).
   */
  profile: 'thin' | 'balanced';
  rIn: number;
  rOut: number;
  /** surface density ('thin') or midplane density at r = 1 ('balanced'), and its slope in radius */
  sigma0: number;
  slope?: number;
  /** sound speed */
  cs: number;
  /** density of the gas outside the disk, as a fraction of sigma0 ('thin') or absolute ('balanced') */
  floor?: number;
  /** a damping zone: from this radius out, at this rate per local orbit at the box's edge */
  damp?: { from: number; rate: number };
  /** poloidal field loops: plasma beta where two meet, each loop's span in radius as a factor, and how many */
  loops?: { beta: number; span: number; count: number };
  /** report the disk's stresses and MRI quality (Problem.disk) */
  stresses?: boolean;
}

export type Shape =
  | { type: 'sphere'; center?: Vec3; radius?: number; /** radius in finest cells, instead */ radiusCells?: number }
  | { type: 'box'; min: Vec3; max: Vec3 }
  /** the gas where (x - at) . normal < 0 */
  | { type: 'half'; normal: Vec3; at?: number };

export interface RegionPart {
  kind: 'region';
  id?: string;
  shape: Shape;
  rho?: number;
  p?: number;
  /** instead of p: this much energy in all, spread evenly over the region's cells */
  energy?: number;
  v?: Vec3;
  s?: number;
  X?: Record<string, number>;
  /** cells the surface crosses take the share inside it, from this many samples a side */
  supersample?: number;
}

export interface ObstaclePart {
  kind: 'obstacle' | 'inflow';
  id?: string;
  shape: Shape;
}

export interface FieldPart {
  kind: 'field';
  id?: string;
  /** a uniform field: its vector, or a strength along the axis drawn upward (y in 2D, z in 3D) */
  b?: Vec3;
  up?: number;
}

export interface FlowPart {
  kind: 'flow';
  id?: string;
  /** a built-in initial condition (setup/parts/flows) */
  name: string;
  [param: string]: unknown;
}

export type Part = MassPart | OrbitPart | StarPart | DiskPart | RegionPart | ObstaclePart | FieldPart | FlowPart;
export type PartKind = Part['kind'];

/** An option shown with the preset: a path into the document, and how the side bar presents it. */
export interface Featured {
  /** the path, or several that take the same value */
  path: string | string[];
  /** the name old links know it by (`p.<id>`) */
  id?: string;
  label?: string;
  tip?: string;
  min?: number;
  max?: number;
  step?: number;
  factor?: number;
  control?: 'number' | 'slider' | 'checkbox';
  dims?: 2 | 3;
  /** shown in this unit with physical readouts: the code value times the featured value of id `times` */
  unit?: { label: string; times: string };
  unitChoices?: { label: string; value: number }[];
}

export interface SetupSpec {
  format: typeof SETUP_FORMAT;
  version: number;
  /** presets: the name links and tests use */
  id?: string;
  name?: string;
  /** its place in the preset list */
  group?: string;
  family?: string;
  familyName?: string;
  variant?: string;
  blurb?: string;
  /** a preset this document gives only its differences from (resolved on load) */
  extends?: string;
  domain?: DomainSpec;
  physics?: PhysicsSpec;
  mesh?: MeshSpec;
  background?: BackgroundSpec;
  parts?: Part[];
  run?: RunSpec;
  view?: ViewSpec;
  featured?: (string | Featured)[];
}

/** An empty setup: the defaults of everything. */
export const emptySetup = (): SetupSpec => ({ format: SETUP_FORMAT, version: SETUP_VERSION });

/** Read a value at a dotted path ('parts.2.ecc'); undefined where the path leads nowhere. */
export function getPath(o: unknown, path: string): unknown {
  let at = o as Record<string, unknown> | undefined;
  for (const key of path.split('.')) {
    if (at === null || typeof at !== 'object') return undefined;
    at = at[key] as Record<string, unknown> | undefined;
  }
  return at;
}

/** Set a value at a dotted path, making objects on the way (never through a prototype). */
export function setPath(o: object, path: string, value: unknown): void {
  const keys = path.split('.');
  let at = o as Record<string, unknown>;
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') return;
    if (i === keys.length - 1) {
      if (value === undefined) delete at[key];
      else at[key] = value;
      return;
    }
    if (at[key] === null || typeof at[key] !== 'object') {
      // Nothing to remove where the path leads nowhere: no objects are made on the way.
      if (value === undefined) return;
      at[key] = /^\d+$/.test(keys[i + 1]) ? [] : {};
    }
    at = at[key] as Record<string, unknown>;
  }
}

export const cloneSetup = <T>(s: T): T => JSON.parse(JSON.stringify(s)) as T;

/** A featured entry in full. */
export const featuredOf = (f: string | Featured): Featured => (typeof f === 'string' ? { path: f } : f);
export const featuredPaths = (f: Featured): string[] => (Array.isArray(f.path) ? f.path : [f.path]);
/** The id a featured option goes by: its own, or its path's last key. */
export const featuredId = (f: Featured): string => f.id ?? featuredPaths(f)[0].split('.').pop()!;
