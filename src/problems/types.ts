import type { Eos, EosKind, EosParams } from '../eos/eos';
import type { Settling, StarSpec } from './settle';
import type { CoolingKind, CoolingParams } from '../cooling/cooling';
import type { AnyBurnSpec } from '../burn/burn';
import type { UnitSystem } from '../units/units';
import type { Composition } from '../units/composition';
import type { SetupSpec } from '../setup/spec';
import type { ViewMode } from '../render/view_options';
import { cosmoUnits } from '../cosmo';

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
  /** Which body a solid cell belongs to (Problem.bodies, from 0): its forces are summed apart. */
  body?: number;
  /** The cell is held at this state for the whole run (an inflow, say). */
  fixed?: boolean;
  /**
   * With radiation (SimConfig.radiation): the radiation energy density and flux, in the gas's
   * frame. Without them E = a T^4 at the gas's temperature, and F its diffusion value
   * -c / (3 kappa_R rho) grad E from the grid's own differences (radiation/init.ts).
   */
  erad?: number;
  frad?: readonly [number, number, number];
}

/** Bodies whose forces are summed apart (hydro/forces.wgsl); bodies past the last share its sums. */
export const MAX_BODIES = 4;

/**
 * How a cell's kind is stored with its state: 1 + its body's index (at most MAX_BODIES - 1) for
 * a solid cell, -1 fixed, 0 ordinary gas. Every test of the mark is > 0.5 or < -0.5.
 */
export const cellMark = (w: Prim): number => (w.solid ? 1 + Math.min(Math.max(Math.floor(w.body ?? 0), 0), MAX_BODIES - 1) : w.fixed ? -1 : 0);

/** A body the gas exerts a force on: an obstacle of the setup, in its order (Problem.bodies). */
export interface BodyRef {
  id: string;
  /** the point torques are taken about, in box coordinates */
  center: [number, number, number];
}

/**
 * What the force coefficients are scaled by: C = F / (rho U^2 A / 2), a moment's by a further
 * length. `drag` is the axis the stream flows along, `lift` the one lift is measured along.
 */
export interface ForceRef {
  rho: number;
  speed: number;
  area: number;
  length: number;
  drag: 0 | 1 | 2;
  lift: 0 | 1 | 2;
  /** a rotor's frame: its angular velocity about `axis` (through the box's center) and the rotor's radius, for the power coefficient */
  rotor?: { omega: number; axis: 0 | 1 | 2; radius: number };
}

export { LINE_TIED, MOVING_WALL, OUTFLOW, OUTFLOW_ONLY, PERIODIC, REFLECT } from './boundaries';
export type Bc3 = [number, number, number];

/**
 * Gravity: none, a constant downward field of strength G (-y in 2D, -z in 3D, the axis drawn
 * upward), or self-gravity with periodic or isolated boundaries.
 */
export const GRAVITY_MODES = ['off', 'constant', 'periodic', 'isolated'] as const;
export type GravityMode = (typeof GRAVITY_MODES)[number];
/**
 * How the flux between two cells is worked out, and how steep the reconstruction in a cell may
 * be. Each list here is the one place its names are written: the types, the Setup dialog's
 * menus, the document codec and `describe()` all read it.
 */
export const RIEMANN_KINDS = ['hllc', 'hllclm', 'hll'] as const;
export type RiemannKind = (typeof RIEMANN_KINDS)[number];
export const LIMITER_KINDS = ['koren', 'mc', 'vanleer', 'minmod'] as const;
export type LimiterKind = (typeof LIMITER_KINDS)[number];
/** The mode solves for the gas's own potential (3D only). */
export const isSelfGravity = (m: GravityMode): boolean => m === 'periodic' || m === 'isolated';

/** The quantities the view can show (render/field_value.wgsl computes them, in this order). */
export const FIELDS = [
  'Density', 'Temperature', 'Passive scalar', 'Pressure', 'Speed',
  'Velocity x', 'Velocity y', 'Velocity z', 'Mach number', 'Entropy',
  'Heating \u2212 cooling', 'Vorticity', 'Velocity divergence', 'Density gradient', 'Refinement level',
  'Gravitational potential', 'Kinetic energy', 'Jeans length (cells)', 'Timestep limit', 'Mass fraction',
  'Shock', 'Time bin',
  'Magnetic field x', 'Magnetic field y', 'Magnetic field z', 'Magnetic field strength', 'Plasma beta', 'Magnetic divergence',
  'MRI quality Q_z', 'MRI quality Q_φ',
  'Nuclear heating',
  'Radiation energy', 'Radiation temperature', 'Radiation flux', 'Reduced flux', 'Cell optical depth', 'Speed of light factor r',
] as const;
/** The mass fraction field: one menu entry per species the run carries (ViewState.species). */
export const MASS_FRACTION_FIELD = 19;
/** The solver's shock flag (hydro/predictor.wgsl): 1 in the cells it treats as a shock. */
export const SHOCK_FIELD = 20;
/** Adaptive mesh with per-block timesteps: the time bin of each cell's block (amr/bins_on.wgsl). */
export const TIME_BIN_FIELD = 21;
/** The magnetic fields (mhd/mhd.ts): B_x, B_y, B_z, |B|, plasma beta, |div B| dx / |B|. Shown only for a run with a field. */
export const MAGNETIC_FIELDS: readonly number[] = [22, 23, 24, 25, 26, 27, 28, 29];
/** Burning's heating rate over the last step (burn/burn.ts). Shown only for a run that burns. */
export const NUCLEAR_HEATING_FIELD = 30;
/** Radiation transport (radiation/radiation.ts): E, (E / a)^(1/4), |F|, |F| / (c E), kappa_R rho dx, r. Shown only with radiation. */
export const RADIATION_FIELDS: readonly number[] = [31, 32, 33, 34, 35, 36];
/**
 * The field menu: FIELDS by index, in groups (the indices themselves are the shaders' order).
 * A run with species also gets a Composition group, one mass fraction per species.
 */
export const FIELD_MENU: { group: string; fields: number[] }[] = [
  { group: 'Gas', fields: [0, 3, 1, 9, 10, 30, 2] },
  { group: 'Flow', fields: [4, 5, 6, 7, 8, 16] },
  { group: 'Derivatives', fields: [11, 12, 13, 20] },
  { group: 'Gravity', fields: [15, 17] },
  { group: 'Mesh and timestep', fields: [14, 21, 18] },
  { group: 'Magnetic field', fields: [25, 22, 23, 24, 26, 27, 28, 29] },
  { group: 'Radiation', fields: [31, 32, 33, 34, 35, 36] },
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
  'With per-block timesteps on the adaptive mesh: the time bin of the block that holds each cell. A block in bin k steps 2^k times the smallest step (0 is the fastest)',
  'Magnetic field along x at the cell center, signed (in units where the magnetic pressure is B² / 2)',
  'Magnetic field along y at the cell center, signed',
  'Magnetic field along z at the cell center, signed',
  'Magnitude of the magnetic field at the cell center (in units where the magnetic pressure is B² / 2)',
  'Gas pressure over magnetic pressure, 2p / B²: below 1 the field dominates. Where the field is weaker than 1e-4 of its strongest (gas it has not reached, holding round-off) B² is taken as that much, so that gas does not set the color range',
  'The discrete divergence of the magnetic field from the cell’s six faces, |∇·B| dx, over the strongest |B| anywhere (as the ∇·B diagnostic): zero up to round-off, since constrained transport keeps it there',
  'With a point mass: cells across the fastest-growing wavelength of the magnetorotational instability for the field’s vertical part, Q_z = 2π |B_z| / (√ρ Ω dx), Ω the Keplerian angular velocity about the mass. The MRI grows at its proper rate where Q_z is above about 8 (Hawley, Guan & Krolik 2011)',
  'With a point mass: as Q_z, for the field’s part along the orbit, Q_φ = 2π |B_φ| / (√ρ Ω dx). Above about 20 the turbulence the MRI drives is resolved (Hawley, Guan & Krolik 2011)',
  'Energy nuclear burning released per unit volume and time over the last step: where the fuel is burning, and how fast',
  'Radiation energy per unit volume, E, in the gas’s frame',
  'Radiation temperature (E / a)^¼: where it differs from the gas’s, the two are out of equilibrium',
  'Magnitude of the radiation flux |F|, in the gas’s frame',
  'The reduced flux |F| / (c E): near 0 where the radiation is trapped and diffuses, 1 where it streams freely',
  'Optical depth of one cell, κ_R ρ dx: above 1 the radiation diffuses through the cell, below it streams',
  'The factor r on the speed of light in each cell: 1 where the true speed is used (opaque gas), chat / c where it is reduced',
] as const;
/** Fields that take both signs: shown on a linear scale, centered on zero. */
export const SIGNED_FIELDS: ReadonlySet<number> = new Set([5, 6, 7, 10, 11, 12, 22, 23, 24]);
/** Fields always shown on a linear scale: the signed ones, refinement levels, the potential and the shock flag. */
export const LINEAR_FIELDS: ReadonlySet<number> = new Set([...SIGNED_FIELDS, 14, 15, 20, 21, 34, 36]);

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
   * With the readouts in physical units, a menu beside the field picks the unit the value is
   * typed and shown in, the first by default: `value` is that unit in code units.
   */
  unitChoices?: { label: string; value: number }[];
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
  /** Obstacles whose forces are reported (Diagnostics.bodies), in the order of their marks. */
  bodies?: BodyRef[];
  /**
   * On an adaptive mesh, the obstacles' surfaces that ask for refinement (setup/parts/surfaces.ts,
   * amr/bodies.ts), for a grid of `n` finest cells across with lower corner `lo`.
   */
  surfaces?: (n: number, ndim: 2 | 3, lo: readonly number[], params?: Record<string, number>) => import('../setup/parts/surfaces').BodySurface[];
  /**
   * Every obstacle's signed distance (negative inside) where it starts, with its cells' mark: what
   * partly filled cells are cut from (hydro/cut.ts), for a grid of `n` finest cells across.
   */
  obstacleDistances?: (n: number, ndim: 2 | 3, params?: Record<string, number>) => { mark: number; distance(x: number, y: number, z: number): number }[];
  /** Obstacles that move (hydro/moving.ts), with their motions and chains: the uniform grid's stamp and exchange passes. */
  moving?: import('../hydro/moving').MovingBody[];
  /** The scales of the force coefficients (Diagnostics.bodies' coefficients), if there are any: fixed, or for the run's grid. */
  forceRef?: ForceRef | ((ndim: 2 | 3, n: number, params?: Record<string, number>) => ForceRef | undefined);
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
  /**
   * The parameter (by id) that is the box's side: `box` is then its default, and a run's
   * problem is this one with `box` set from its parameters (sizedProblem below). Changing it
   * builds the run again.
   */
  boxParam?: string;
  tEnd: number;
  /** Time between checkpoints, when the page is asked to save them (problem time units). */
  checkpointDt?: number;
  /**
   * The problem's dynamical time: the time over which its flow visibly changes (a star's
   * sqrt(R^3 / G M), a shear layer's roll-up, a binary's period). A movie takes a frame every
   * tDyn / 600 of it by default, so one dynamical time plays in 10 s at 60 fps. Without one, the
   * end time or ten checkpoint intervals stand in for it (dynamicalTime()).
   */
  tDyn?: number;
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
   * the speed at `v`. With a magnetic field, `va` caps the Alfven speed everywhere by adding
   * mass (hydro/fluff.wgsl alfven_floor). See hydro/fluff.wgsl.
   */
  fluff?: { rho: number; u: number; v: number; va?: number };
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
   * The grid the preset opens on: dimensions, cells per side (the finest level's on an
   * adaptive mesh) and levels of refinement. Picking the preset from the menu switches to it,
   * and a link that names the preset without a grid gets it. Without one, the menu keeps the
   * grid already chosen and a link gets the 3D default. Memory limits still apply.
   */
  grid?: { ndim?: 2 | 3; n?: number; amr?: number };
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
  /**
   * A frame turning at the constant angular velocity `omega` about an axis through the origin
   * (gravity/rot.wgsl): the Coriolis force, and the centrifugal force unless `centrifugal` is
   * false (an f-plane). In 2D only omega's z part is used. Velocities, in `init` and everywhere
   * after, are those relative to the turning frame (toRotating converts an inertial setup).
   * `track`: the frame follows a binary (about z): it turns faster or slower to keep the axis of
   * the mass's quadrupole in the plane where it started (that of the gas denser than `rhoMin`
   * and the point masses, about their center of mass), and every velocity feels the Euler force
   * (gravity/bh_kick.wgsl follow_binary).
   */
  rotation?: (params: Record<string, number>, ndim: 2 | 3) => { omega: [number, number, number]; centrifugal?: boolean; track?: { rhoMin: number } };
  /**
   * The mirrors the setup is symmetric under, each a plane through the origin normal to an axis:
   * a run may then compute only the part above them (SimConfig.symmetry). `mhd`: its magnetic
   * field is symmetric too (mhd/common.wgsl gives the planes the field's parity).
   */
  symmetry?: { axes: [boolean, boolean, boolean]; mhd?: boolean };
  /**
   * Comoving coordinates (src/cosmo.ts): positions comoving, densities and pressures comoving
   * (a^3 times the proper ones), velocities peculiar and proper, specific energies proper. Time
   * is in 1/H0 from t = 0 at a = aStart, and the mean comoving density should be 1.
   */
  cosmology?: import('../cosmo').Cosmology;
  /**
   * A shearing box (Part D): the frame corotates at `omega` (about z), the background flow is
   * v_y = -q omega x, and the x faces are periodic with the shift that flow gives.
   */
  shear?: { omega: number; q: number; stratified?: boolean };
  /**
   * 'cylindrical': a 2D run that is a body of revolution about the y axis, in (R, z): code x is
   * R from the axis (0 to box), y is z, and the third velocity component is v_phi (the layer is
   * the azimuth). Cells are rings, of volume 2 pi R dx^2: masses and energies are true ones.
   */
  geometry?: 'cartesian' | 'cylindrical' | 'spherical';
  /**
   * 'spherical' geometry (docs/numerics.md section 22): a 2D run in (r, theta), a body of
   * revolution about the z axis. Code x is r from `rMin` to rMin + box (uniform spacing, or
   * logarithmic with `log`), y is theta from 0 to pi (both poles axes), and the third velocity
   * component is v_phi. `init` is called with (r, theta, 0) at each cell's center and gives
   * (v_r, v_theta, v_phi) as (vx, vy, vz). Point masses must sit at the origin, at rest: their
   * pull joins the gas's own monopole field.
   */
  sphere?: { rMin: number; log?: boolean };
  /** Density below which gas does not ask for refinement (default: the fluff density, else ten times a star's ambient density, else 0, off). */
  refineRho?: number;
  /** Adaptive mesh: density above which gas is always refined to the finest level, as the stars of a collision are (default 0, none). */
  refineDense?: number;
  /** Optically thin cooling switched on by default, and its parameters. */
  cooling?: CoolingKind;
  coolParams?: Partial<CoolingParams>;
  /**
   * Nuclear burning, from the preset's parameters: a pass after the species pass on either
   * grid. One reaction of one species into another (burn/burn.ts; needs the two species in
   * `species`), or the trained network of the alpha chain (burn/network.ts; needs its 13
   * species, in its order, and physical units).
   */
  burn?: (params: Record<string, number>) => AnyBurnSpec;
  /**
   * A damping zone (hydro/damping.wgsl). `wgsl` defines `const DAMPING = true;` and
   * `fn damping_target(x: vec3<f32>) -> DampTarget`, the state the gas at `x` (relative to the
   * box center) is relaxed toward and how fast; it reads the zone's inner radius and rate from
   * P.diff.z and P.diff.w. `radius` and `rate` are their defaults (SimConfig.dampRadius and
   * dampRate).
   */
  damping?: { wgsl: string; radius: number; rate: number };
  /**
   * Radiation transport switched on by default (SimConfig.radiation), its opacity law and
   * constants (code units, over those the problem's units give), the speed of light (code
   * units; from the units without), and the reduced speed chat (default: c).
   */
  radiation?: import('../radiation/radiation').RadiationKind;
  opacity?: import('../opacity/opacity').OpacityKind;
  opacityParams?: Partial<import('../opacity/opacity').OpacityParams>;
  radC?: number;
  radChat?: number;
  /**
   * With radiation on, the equation of state keeps its own radiation term all the same: a
   * material whose heat capacity goes as T^3, as Su & Olson's (the Marshak wave test). Off by
   * default, when that term is the radiation's and E carries it.
   */
  radKeepEos?: boolean;
  /** Kinematic viscosity and thermal conduction switched on by default (see SimConfig). */
  viscosity?: number;
  conductivity?: number;
  /** Magnetic diffusivity eta switched on by default (see SimConfig.resistivity). */
  resistivity?: number;
  /** Velocities of the faces that are moving walls (SimConfig.wallVelocity). */
  wallVelocity?: Partial<import('./boundaries').WallVelocity>;
  /** Turbulence driving switched on by default (see SimConfig.driving). */
  driving?: Partial<import('../hydro/drive').DriveParams>;
  /** Uniform external gravitational acceleration, which may depend on the dimensionality. */
  gext?: (ndim: 2 | 3) => [number, number, number];
  /** A polytropic star at rest, settled into the grid's own hydrostatic balance before the run (problems/settle.ts). */
  star?: StarSpec;
  /** Stars of any cold equation of state, at rest or moving, to settle likewise (in place of `star`). */
  settle?: (params: Record<string, number>, eos: Eos) => Settling;
  /**
   * Heat put in once the stars are settled, which rewrites them as cold gas (a hot spot that
   * ignites a star): the internal energy density a cell at (x, y, z) of density rho should
   * have, or undefined to leave the cell as it is. Fresh starts only, not restarts; `init`
   * should put the same heat in, for runs that are not settled.
   */
  heat?: (params: Record<string, number>, eos: Eos) => (x: number, y: number, z: number, rho: number) => number | undefined;
  /** The run waits, paused, for the user to start it. */
  startPaused?: boolean;
  /**
   * Least density and speed the editing tools measure their amounts against, for a box that
   * holds next to nothing to measure them by.
   */
  toolScale?: { rho: number; v: number };
  /** Default view; `axis` is the slice normal in 3D (0 x, 1 y, 2 z), `pitch` the camera elevation in radians. */
  view?: { field?: number; log?: boolean; mode?: ViewMode; axis?: number; pitch?: number };
  /**
   * A magnetic field, which makes the run ideal MHD (mhd/mhd.ts). Units: the magnetic pressure
   * is B^2 / 2, so plasma beta is 2 p / B^2.
   */
  mhd?: MhdSpec;
  /**
   * A disk around the first point mass, in the x-y plane: the magnetic diagnostics report its
   * stresses and MRI quality factors (MagDiagnostics.disk), and the panel shows them.
   */
  disk?: boolean;
  /** The setup document this problem was compiled from (setup/compile.ts), if it was. */
  setup?: SetupSpec;
  /** Returns the initial primitive state as a function of position. */
  /** `levels`: on blocks, the refinement levels above the coarsest (n is the finest grid's): what a painter that must reach the coarsest cells (a tunnel's inlet) needs */
  init(n: number, eos: Eos, ndim: 2 | 3, params: Record<string, number>, levels?: number): (x: number, y: number, z: number) => Prim;
}

/**
 * The initial magnetic field. A vector potential A, whose discrete curl on the cell faces
 * is divergence-free to round-off; or the field on the faces directly, for a field that is
 * divergence-free by construction (one that varies only along the axis normal to it, say); or
 * both, which add (a uniform field, whose potential is not periodic, through a star's own).
 * Positions are relative to the box center; `axis` is the face normal (0 x, 1 y, 2 z).
 */
export interface MhdSpec {
  potential?: (params: Record<string, number>, ndim: 2 | 3) => (x: number, y: number, z: number) => [number, number, number];
  face?: (params: Record<string, number>, ndim: 2 | 3) => (axis: number, x: number, y: number, z: number) => number;
  /**
   * Or every face at once, as mhd/mhd.ts initialFaces lays them out for a grid of n cells per
   * side and nz along z: a run restarted from a file (io/capture.ts restartFrom).
   */
  faces?: (n: number, nz: number) => Float32Array;
  /**
   * Or, on blocks, a leaf's faces as amr/amr_mhd.ts blockFaces lays them out, from its level
   * and position in blocks: a run restarted from a file of its own mesh.
   */
  block?: (level: number, c: number[]) => Float32Array;
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
  /**
   * removal rate at the sink's center, in units of sqrt(G M / radius^3) (default 1); per unit
   * time for a massless particle
   */
  sinkRate?: number;
  /**
   * A particle without a mass of its own: `gm` is then a token amount, far too little to pull
   * on anything, it never grows, and what its sink takes is kept count of but changes neither
   * its mass nor its motion. It moves as the gas and the masses pull it.
   */
  massless?: boolean;
  /** Held where it is: nothing moves it (its velocity is zero), whatever pulls on it or it takes in. */
  fixed?: boolean;
  /**
   * Held on its orbit: the other point masses move it, the gas does not. Neither the gas's
   * pull nor what its sink takes changes its motion, and it does not grow, so a binary keeps the
   * orbit it was given. What the sink takes is still counted.
   */
  onOrbit?: boolean;
  /** gas mass accreted so far */
  accreted?: number;
  /** momentum that gas brought in */
  accretedMomentum?: number[];
  /** angular momentum that gas brought in, about the mass and in its rest frame */
  accretedSpin?: number[];
}

export const all = (b: number): Bc3 => [b, b, b];

/**
 * The preset as a run with these parameters has it: with the box side its `boxParam` asks for.
 * The preset itself, if it has no such parameter or the side is its default.
 */
export function sizedProblem(p: Problem, params: Record<string, number>): Problem {
  const side = p.boxParam ? params[p.boxParam] : undefined;
  return side !== undefined && side > 0 && side !== p.box ? { ...p, box: side } : p;
}

/** A preset's physical units for these values of its parameters, or undefined if it is scale free. */
export const problemUnits = (p: Problem, params: Record<string, number>): UnitSystem | undefined =>
  (typeof p.units === 'function' ? p.units(params) : p.units) ?? (p.cosmology ? cosmoUnits(p.cosmology, p.box) : undefined);

/** A problem's dynamical time (Problem.tDyn), or what stands in for it. */
export function dynamicalTime(p: Problem): number {
  if (p.tDyn !== undefined) return p.tDyn;
  const ten = 10 * (p.checkpointDt ?? 0.1);
  return Number.isFinite(p.tEnd) ? Math.min(p.tEnd, p.checkpointDt !== undefined ? ten : Infinity) : ten;
}

/** The default time between a movie's frames: a 600th of the dynamical time, to two figures. */
export const defaultFrameDt = (p: Problem): number => +(dynamicalTime(p) / 600).toPrecision(2);
