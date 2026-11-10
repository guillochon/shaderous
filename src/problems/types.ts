import type { Eos, EosKind, EosParams } from '../eos/eos';

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
   * A point mass moving past the gas. The run is then in the gas's rest frame: the frame
   * accelerates with the gas's centre of mass. Returns G M and the initial position and
   * velocity relative to the box centre.
   */
  perturber?: (params: Record<string, number>, G: number) => { gm: number; pos: number[]; vel: number[] };
  /** Uniform external gravitational acceleration, which may depend on the dimensionality. */
  gext?: (ndim: 2 | 3) => [number, number, number];
  /** Default view; `axis` is the slice normal in 3D (0 x, 1 y, 2 z), `pitch` the camera elevation in radians. */
  view?: { field?: number; log?: boolean; mode?: 'volume' | 'slice'; axis?: number; pitch?: number };
  /** Returns the initial primitive state as a function of position. */
  init(n: number, eos: Eos, ndim: 2 | 3, params: Record<string, number>): (x: number, y: number, z: number) => Prim;
}

export const all = (b: number): Bc3 => [b, b, b];
