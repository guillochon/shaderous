import type { Eos, EosKind, EosParams } from '../eos/eos';
import type { StarSpec } from './settle';
import type { CoolingKind, CoolingParams } from '../cooling/cooling';

export interface Prim {
  rho: number;
  vx: number;
  vy: number;
  vz: number;
  p: number;
  s: number; // passive scalar
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

// Fields the renderer can show; order matches field.wgsl.
export const FIELDS = ['Density', 'Temperature', 'Passive scalar', 'Pressure', 'Speed'] as const;

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
  box: number; // side length; the domain is [-box/2, box/2]^3
  tEnd: number;
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
  masses?: (params: Record<string, number>, G: number, dx: number) => PointMass[];
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
  /** Optically thin cooling switched on by default, and its parameters. */
  cooling?: CoolingKind;
  coolParams?: Partial<CoolingParams>;
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
  /** The run waits, paused, for the user to start it. */
  startPaused?: boolean;
  /**
   * Least density and speed the editing tools measure their amounts against, for a box that
   * holds next to nothing to measure them by.
   */
  toolScale?: { rho: number; v: number };
  /** Default view; `axis` is the slice normal in 3D (0 x, 1 y, 2 z), `pitch` the camera elevation in radians. */
  view?: { field?: number; log?: boolean; mode?: 'volume' | 'slice'; axis?: number; pitch?: number };
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
}

export const all = (b: number): Bc3 => [b, b, b];
