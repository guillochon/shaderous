// The kinds of box face: the one table the interface, the setup format and links read. What each
// does to the gas is in hydro/boundary.wgsl, under the same codes (P.bc_lo, P.bc_hi). To add a
// kind: an entry here, its rules there (bc_ghost_cell, bc_vsign, bc_bsign, bc_side,
// bc_ghost_face, bc_wall_velocity), and any physics of its own (the line-tied and moving walls'
// electric field, mhd/corner.wgsl, mhd/emf.wgsl and mhd/amr_emf.wgsl). The CPU reference solvers (src/reference) know the first
// four only.

export interface BoundaryKind {
  code: number;
  label: string;
  tip: string;
  /** its faces come in pairs: the opposite face's cells are its ghosts */
  periodic: boolean;
  /** a solid wall: its ghosts are mirror images, and nothing crosses it */
  wall: boolean;
  /** its edges' color in the setup preview */
  color: string;
}

export const PERIODIC = 0;
export const OUTFLOW = 1;
export const REFLECT = 2;
/** Zero-gradient, but the ghost's normal velocity may only point outward: nothing flows in. */
export const OUTFLOW_ONLY = 3;
/** A no-slip wall that, with a magnetic field, holds the field lines crossing it: the flux through it never changes. */
export const LINE_TIED = 4;
/** A line-tied wall that moves along itself (SimConfig.wallVelocity): it drags the gas, and the footpoints of the field lines. */
export const MOVING_WALL = 5;

export const BOUNDARIES: readonly BoundaryKind[] = [
  { code: PERIODIC, label: 'Periodic', tip: 'Gas leaving through this face comes back in through the opposite one (both faces of an axis)', periodic: true, wall: false, color: '#6cb6ff' },
  { code: OUTFLOW, label: 'Outflow', tip: 'Open: the state at the face continues beyond it, so gas may leave or enter', periodic: false, wall: false, color: '#8792a8' },
  { code: REFLECT, label: 'Reflecting', tip: 'A solid wall the gas slides along; a magnetic field lies along it (its field lines do not cross it)', periodic: false, wall: true, color: '#ffb454' },
  { code: OUTFLOW_ONLY, label: 'Outflow only', tip: 'Open for gas leaving; nothing comes in', periodic: false, wall: false, color: '#5fd08a' },
  {
    code: LINE_TIED,
    label: 'Line-tied wall',
    tip: 'A solid wall the gas sticks to (no slip). A magnetic field may cross it: its field lines stay anchored where they cross, so the flux through the wall never changes',
    periodic: false,
    wall: true,
    color: '#e58fff',
  },
  {
    code: MOVING_WALL,
    label: 'Moving wall',
    tip: 'A line-tied wall that slides along itself at the velocity set for this face: it drags the gas next to it (no slip), and the footpoints of the field lines crossing it move with it',
    periodic: false,
    wall: true,
    color: '#ff8fa3',
  },
];

/** The faces in the order of the wall velocities (SimConfig.wallVelocity, Params.wallv). */
export const FACE_KEYS = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'] as const;
export type FaceKey = (typeof FACE_KEYS)[number];
export type WallVelocity = Record<FaceKey, [number, number, number]>;
export const noWallVelocity = (): WallVelocity => ({ xlo: [0, 0, 0], xhi: [0, 0, 0], ylo: [0, 0, 0], yhi: [0, 0, 0], zlo: [0, 0, 0], zhi: [0, 0, 0] });

/** Every face kind's code, in order. */
export const BOUNDARY_CODES: readonly number[] = BOUNDARIES.map((b) => b.code);

/** The kinds as a link's documentation lists them: "0 periodic, 1 outflow, ...". */
export const boundaryList = (): string => BOUNDARIES.map((b) => `${b.code} ${b.label.toLowerCase()}`).join(', ');
