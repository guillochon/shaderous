import type { Eos, EosKind, EosParams } from '../eos/eos';
import type { Settling, StarSpec } from './settle';
import type { CoolingKind, CoolingParams } from '../cooling/cooling';
import type { BurnSpec } from '../burn/burn';
import type { UnitSystem } from '../units/units';
import type { Composition } from '../units/composition';

export interface Prim {
  rho: number;
  vx: number;
  vy: number;
  vz: number;
  p: number;
  s: number; // passive scalar
  /** Mass fractions of the problem's species (Problem.species), in its order; summing to 1. */
  X?: readonly number[];
  /** The cell is part of an obstacle: a solid body the gas reflects off. */
  solid?: boolean;
  /** The cell is held at this state for the whole run (an inflow, say). */
  fixed?: boolean;
}

/** How a cell's kind is stored with its state: 1 solid, -1 fixed, 0 ordinary gas. */
export const cellMark = (w: Prim): number => (w.solid ? 1 : w.fixed ? -1 : 0);

export const PERIODIC = 0;
export const OUTFLOW = 1;
export const REFLECT = 2;
/** Zero-gradient, but the ghost's normal velocity may only point outward: nothing flows in. */
export const OUTFLOW_ONLY = 3;
export type Bc3 = [number, number, number];

/**
 * Gravity: none, a constant downward field of strength G (-y in 2D, -z in 3D, the axis drawn
 * upward), or self-gravity with periodic or isolated boundaries.
 */
export type GravityMode = 'off' | 'constant' | 'periodic' | 'isolated';
export const GRAVITY_MODES: readonly GravityMode[] = ['off', 'constant', 'periodic', 'isolated'];
/** The mode solves for the gas's own potential (3D only). */
export const isSelfGravity = (m: GravityMode): boolean => m === 'periodic' || m === 'isolated';

/** The quantities the view can show (render/field_value.wgsl computes them, in this order). */
export const FIELDS = [
  'Density', 'Temperature', 'Passive scalar', 'Pressure', 'Speed',
  'Velocity x', 'Velocity y', 'Velocity z', 'Mach number', 'Entropy',
  'Heating \u2212 cooling', 'Vorticity', 'Velocity divergence', 'Density gradient', 'Refinement level',
  'Gravitational potential', 'Kinetic energy', 'Jeans length (cells)', 'Timestep limit', 'Mass fraction',
  'Shock',
] as const;
/** The mass fraction field: one menu entry per species the run carries (ViewState.species). */
export const MASS_FRACTION_FIELD = 19;
/** The solver's shock flag (hydro/predictor.wgsl): 1 in the cells it treats as a shock. */
export const SHOCK_FIELD = 20;
/**
 * The field menu: FIELDS by index, in groups (the indices themselves are the shaders' order).
 * A run with species also gets a Composition group, one mass fraction per species.
 */
export const FIELD_MENU: { group: string; fields: number[] }[] = [
  { group: 'Gas', fields: [0, 3, 1, 9, 10, 2] },
  { group: 'Flow', fields: [4, 5, 6, 7, 8, 16] },
  { group: 'Derivatives', fields: [11, 12, 13, 20] },
  { group: 'Gravity', fields: [15, 17] },
  { group: 'Mesh and timestep', fields: [14, 18] },
];
/** What each field shows, for its tooltip. */
export const FIELD_TIPS = [
  'Mass per unit volume (per unit area in 2D)',
  'Temperature from the equation of state',
  'A dye carried with the gas: where the gas that started in a marked region has gone',
  'Gas pressure (plus radiation pressure with that equation of state)',
  'Magnitude of the velocity',
  'Velocity along x, signed',
  'Velocity along y, signed',
  'Velocity along z, signed',
  'Speed over the local sound speed',
  'The entropy function p / \u03c1^\u03b3: constant along the flow away from shocks, so it marks where gas has been shocked or heated',
  'Net heating rate per unit volume from the cooling function: positive where the gas heats, negative where it cools',
  'Curl of the velocity: its z component, signed, in 2D; its magnitude in 3D. Marks shear layers and eddies',
  'Divergence of the velocity: negative where the gas is compressed (shocks), positive where it expands',
  '|\u2207\u03c1| / \u03c1, a numerical schlieren image: picks out shocks, contacts and waves',
  'Level of the adaptive mesh block that holds each cell (0 is the coarsest)',
  'Gravitational potential: self-gravity, point masses and any uniform field together (with an accelerating frame, its own potential too)',
  'Kinetic energy per unit volume, \u03c1 v\u00b2 / 2',
  'Jeans length c_s (\u03c0 / G\u03c1)^\u00bd in cells. Below 4 a self-gravitating region can fragment artificially (Truelove et al. 1997)',
  'The step each cell would allow: CFL \u00d7 dx over its signal speed (|v| summed over the axes, the sound speed once per axis, diffusion), or over \u221a(dx / |g|) where gravity is stronger. The smallest sets the run\u2019s step (point-mass orbits can set it lower still)',
  'The share of the mass in one species, carried with the gas',
  'The solver’s shock detector, as of the last step: 1 where the flow converges and the pressure jumps by more than the shock threshold between neighbors (the cells it treats as a shock, and where burning stops when it is not allowed inside shocks), 0 elsewhere',
] as const;
/** Fields that take both signs: shown on a linear scale, centered on zero. */
export const SIGNED_FIELDS: ReadonlySet<number> = new Set([5, 6, 7, 10, 11, 12]);
/** Fields always shown on a linear scale: the signed ones, refinement levels, the potential and the shock flag. */
export const LINEAR_FIELDS: ReadonlySet<number> = new Set([...SIGNED_FIELDS, 14, 15, 20]);

export interface ProblemParam {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  /** Tooltip for the field. */
  title?: string;
  /** Change per click of the field's arrows. */
  step?: number;
  /** The arrows multiply or divide by this instead, for a quantity that spans decades. */
  factor?: number;
  /**
   * The value can change mid-run: it only sets the point masses' softening and sink radius,
   * which the run takes up where the masses are now. Without this flag the value is part of
   * the initial conditions, and changing it starts the run again.
   */
  live?: boolean;
  /**
   * The value only sets the physical scale (Problem.units): nothing in code units depends on
   * it, so with a barotropic gas a change relabels the readouts and the run carries on. Any
   * other equation of state takes physical constants from the units, and starts again.
   */
  scale?: boolean;
  /**
   * With the readouts in physical units (cgs or SI), the field shows the value in this unit:
   * the code value times `factor(params)`, with ` (label)` after the field's name.
   */
  unit?: { label: string; factor: (params: Record<string, number>) => number };
  /**
   * How the control looks: a number field (the default), a slider with a number field beside
   * it for typing a value, or a checkbox (value 0 or 1).
   */
  kind?: 'number' | 'slider' | 'checkbox';
  /** Shown, and used, only on a grid of this many dimensions. */
  dims?: 2 | 3;
}

export interface Problem {
  id: string;
  label: string;
  blurb: string;
  eos: EosKind;
  eosParams: Partial<EosParams>;
  bcLo: Bc3;
  bcHi: Bc3;
  gravity: GravityMode;
  G: number;
  /**
   * What one code unit of length, mass and time is in SI, for a setup that stands for
   * something real (units/units.ts). Without it the problem is scale free and shown in code
   * units. With it, `G` should be codeG(units). It may depend on the preset's parameters
   * (problemUnits below), for a scale-free setup whose physical scale is up to the user.
   */
  units?: UnitSystem | ((params: Record<string, number>) => UnitSystem);
  /** What the gas is made of (default: solar, fully ionized); sets the kelvin scale. */
  composition?: Composition;
  /**
   * Species whose mass fractions the gas carries, by name (e.g. 'he4', 'c12'): advected with
   * the mass and summing to 1 in every cell (species/species.ts). Any number; none by default.
   */
  species?: readonly string[];
  /**
   * Adiabatic index of each species by name, where the species set the equation of state
   * (species/mixture.ts); the others take the run's gamma.
   */
  speciesGamma?: Partial<Record<string, number>>;
  box: number; // side length; the domain is [-box/2, box/2]^3
  tEnd: number;
  /** Time between checkpoints, when the page is asked to save them (problem time units). */
  checkpointDt?: number;
  rhoFloor?: number;
  pFloor?: number;
  /** Enforce the Truelove pressure floor by default. */
  jeansFloor?: boolean;
  /** Cells per Jeans length for that floor (default 4). */
  jeansCells?: number;
  /** The setup also makes sense on a 2D (x, y) grid. */
  supports2d?: boolean;
  /** Boundaries to use in 2D, if they differ (z is always periodic there). */
  bc2d?: { lo: Bc3; hi: Bc3 };
  /**
   * Limits on tenuous gas: below density `rho`, cap the specific internal energy at `u` and
   * the speed at `v`. See hydro/fluff.wgsl.
   */
  fluff?: { rho: number; u: number; v: number };
  /** Extra controls shown when this preset is selected. Changing one restarts the run, unless it is `live`. */
  params?: ProblemParam[];
  /**
   * Point masses (at most MAX_MASSES): softened gravity on the gas, each with a sink that
   * takes gas in. They pull on one another and feel the gas. `dx` is the cell size the
   * masses sit in: on an adaptive mesh, the finest level's if `refineMasses` is set, otherwise
   * the coarsest level's.
   */
  masses?: (params: Record<string, number>, G: number, dx: number, ndim: 2 | 3) => PointMass[];
  /**
   * Debris energies (debris/debris.ts): bin the gas by its specific orbital energy about one
   * point mass, both what leaves the box and what is still in it, for the rate at which bound
   * debris returns (dM/dt). `mass` is the index of the point mass, `de` the energy scale the
   * bins are laid out in (the spread of a tidal disruption, G M_bh R_star / r_t^2), `tPeri` the
   * time of pericenter, and `refMass` a mass the fixed-point totals are scaled to. With
   * `scalar`, only the gas the passive scalar marks (the star's) is binned; the rest (the
   * ambient medium) is counted apart.
   */
  debris?: (params: Record<string, number>, G: number) => { mass: number; de: number; tPeri: number; refMass: number; scalar?: boolean };
  /**
   * On an adaptive mesh, keep the gas around every point mass at the finest level, so that
   * softening and sinks can be as small as the finest cells allow (amr/flag.wgsl).
   */
  refineMasses?: boolean;
  /**
   * With point masses: run in a frame that accelerates with the gas's center of mass ('gas')
   * or the point masses' ('masses') instead of an inertial one (gravity/bh_kick.wgsl).
   */
  frame?: 'gas' | 'masses';
  /** Density below which gas does not ask for refinement (default: the fluff density, else ten times a star's ambient density, else 0, off). */
  refineRho?: number;
  /** Adaptive mesh: density above which gas is always refined to the finest level, as the stars of a collision are (default 0, none). */
  refineDense?: number;
  /** Optically thin cooling switched on by default, and its parameters. */
  cooling?: CoolingKind;
  coolParams?: Partial<CoolingParams>;
  /**
   * Nuclear burning of one species into another (burn/burn.ts), from the preset's parameters:
   * a pass after the species pass on either grid. Needs the two species in `species`.
   */
  burn?: (params: Record<string, number>) => BurnSpec;
  /**
   * A damping zone (hydro/damping.wgsl). `wgsl` defines `const DAMPING = true;` and
   * `fn damping_target(x: vec3<f32>) -> DampTarget`, the state the gas at `x` (relative to the
   * box center) is relaxed toward and how fast; it reads the zone's inner radius and rate from
   * P.diff.z and P.diff.w. `radius` and `rate` are their defaults (SimConfig.dampRadius and
   * dampRate).
   */
  damping?: { wgsl: string; radius: number; rate: number };
  /** Kinematic viscosity and thermal conduction switched on by default (see SimConfig). */
  viscosity?: number;
  conductivity?: number;
  /** Uniform external gravitational acceleration, which may depend on the dimensionality. */
  gext?: (ndim: 2 | 3) => [number, number, number];
  /** A polytropic star at rest, settled into the grid's own hydrostatic balance before the run (problems/settle.ts). */
  star?: StarSpec;
  /** Stars of any cold equation of state, at rest or moving, to settle likewise (in place of `star`). */
  settle?: (params: Record<string, number>, eos: Eos) => Settling;
  /** The run waits, paused, for the user to start it. */
  startPaused?: boolean;
  /**
   * Least density and speed the editing tools measure their amounts against, for a box that
   * holds next to nothing to measure them by.
   */
  toolScale?: { rho: number; v: number };
  /** Default view; `axis` is the slice normal in 3D (0 x, 1 y, 2 z), `pitch` the camera elevation in radians. */
  view?: { field?: number; log?: boolean; mode?: 'volume' | 'slice' | 'projection'; axis?: number; pitch?: number };
  /** Returns the initial primitive state as a function of position. */
  init(n: number, eos: Eos, ndim: 2 | 3, params: Record<string, number>): (x: number, y: number, z: number) => Prim;
}

/** A point mass: position and velocity are relative to the box center. */
export interface PointMass {
  /** G M */
  gm: number;
  pos: number[];
  vel: number[];
  /** Plummer softening length of its gravity (default: two cells) */
  soft?: number;
  /** radius within which it takes gas in (default: the softening length; 0 for none) */
  sinkRadius?: number;
  /** removal rate at the sink's center, in units of sqrt(G M / radius^3) (default 1) */
  sinkRate?: number;
  /** gas mass accreted so far */
  accreted?: number;
  /** momentum that gas brought in */
  accretedMomentum?: number[];
  /** angular momentum that gas brought in, about the mass and in its rest frame */
  accretedSpin?: number[];
}

export const all = (b: number): Bc3 => [b, b, b];

/** A preset's physical units for these values of its parameters, or undefined if it is scale free. */
export const problemUnits = (p: Problem, params: Record<string, number>): UnitSystem | undefined =>
  typeof p.units === 'function' ? p.units(params) : p.units;
