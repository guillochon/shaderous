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

export type GravityMode = 'off' | 'periodic' | 'isolated';

// Fields the renderer can show; order matches field.wgsl.
export const FIELDS = ['Density', 'Temperature', 'Passive scalar', 'Pressure', 'Speed'] as const;

export interface ProblemParam {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
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
  /** Extra controls shown when this preset is selected. Changing one restarts the run. */
  params?: ProblemParam[];
  /**
   * Point masses (at most MAX_MASSES): softened gravity on the gas, each with a sink that
   * takes gas in. They pull on one another and feel the gas. `dx` is the cell size (the
   * coarsest level's on an adaptive mesh).
   */
  masses?: (params: Record<string, number>, G: number, dx: number) => PointMass[];
  /**
   * With point masses: run in the gas's rest frame, which accelerates with the gas's centre
   * of mass, instead of an inertial one.
   */
  gasFrame?: boolean;
  /** Optically thin cooling switched on by default, and its parameters. */
  cooling?: CoolingKind;
  coolParams?: Partial<CoolingParams>;
  /** Uniform external gravitational acceleration, which may depend on the dimensionality. */
  gext?: (ndim: 2 | 3) => [number, number, number];
  /** A polytropic star at rest, settled into the grid's own hydrostatic balance before the run (problems/settle.ts). */
  star?: StarSpec;
  /** Default view; `axis` is the slice normal in 3D (0 x, 1 y, 2 z), `pitch` the camera elevation in radians. */
  view?: { field?: number; log?: boolean; mode?: 'volume' | 'slice'; axis?: number; pitch?: number };
  /** Returns the initial primitive state as a function of position. */
  init(n: number, eos: Eos, ndim: 2 | 3, params: Record<string, number>): (x: number, y: number, z: number) => Prim;
}

/** A point mass: position and velocity are relative to the box centre. */
export interface PointMass {
  /** G M */
  gm: number;
  pos: number[];
  vel: number[];
  /** Plummer softening length of its gravity (default: two cells) */
  soft?: number;
  /** radius within which it takes gas in (default: the softening length; 0 for none) */
  sinkRadius?: number;
  /** removal rate at the sink's centre, in units of sqrt(G M / radius^3) (default 1) */
  sinkRate?: number;
  /** gas mass accreted so far */
  accreted?: number;
}

export const all = (b: number): Bc3 => [b, b, b];
