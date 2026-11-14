// The simulation: owns the state textures and encodes hydro + gravity steps.
import { assemble } from './gpu/shaders';
import { RegionPass, type RegionGroups } from './hydro/region_pass';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture, type Gpu } from './gpu/device';
import { noteAllocation } from './gpu/memory';
import { MAX, MIN, Reducer, SUM } from './gpu/reduce';
import { beginPass } from './gpu/timing';
import { Multigrid } from './gravity/multigrid';
import { defaultEosParams, EOS_MODULES, eosConstants2, makeEos, physicalEosParams, type EosKind, type EosParams } from './eos/eos';
import { cellMark, isSelfGravity, problemUnits, type Bc3, type GravityMode, type PointMass, type Problem } from './problems/types';
import { BurnPass, type BurnSpec } from './burn/burn';
import { COOLING_MODULES, defaultCoolingParams, packCooling, type CoolingKind, type CoolingParams } from './cooling/cooling';
import type { Restart } from './io/snapshot';
import { Settler, settlingOf, SETTLE_ROUNDS, SETTLE_TOL, type SettleCells } from './problems/settle';
import { laneEmdenCached, polytropeK } from './problems/lane_emden';
import { packSpecies, speciesConst, speciesGroups, speciesSlot, SPECIES_PER_GROUP } from './species/species';
import { DebrisBuffers, debrisSlot, debrisSpec, type DebrisLedger, type DebrisReport } from './debris/debris';
import { hydrogenParticles, mixMoments, problemCoefficients, problemCompSlot, type Moments } from './species/mixture';

export type RiemannKind = 'hllc' | 'hllclm' | 'hll';
export type LimiterKind = 'koren' | 'mc' | 'vanleer' | 'minmod';

export interface SimConfig {
  /** Cells per side. The grid is n x n x n, or n x n x 1 (periodic in z) when ndim is 2. */
  n: number;
  ndim: 2 | 3;
  /** Uniform external gravitational acceleration. */
  gext: [number, number, number];
  /** Values of the preset's own parameters (Problem.params). */
  params: Record<string, number>;
  problem: Problem;
  eos: EosKind;
  eosParams: EosParams;
  riemann: RiemannKind;
  limiter: LimiterKind;
  gravity: GravityMode;
  G: number;
  bcLo: Bc3;
  bcHi: Bc3;
  cfl: number;
  tEnd: number; // Infinity runs forever
  rhoFloor: number;
  pFloor: number;
  etaDual: number;
  shockThresh: number;
  jeansFloor: boolean;
  /** Cells per Jeans length that the pressure floor enforces (Truelove's minimum is 4). */
  jeansCells: number;
  vcycles: number; // V-cycles per step (warm-started)
  /** Flux-form gravitational work (conserves total energy) instead of kick-consistent energy. */
  gravEnergyFlux: boolean;
  /**
   * Well-balanced pressure reconstruction: deviations from hydrostatic balance are what is
   * reconstructed, so a static stratified state holds to round-off (hydro/q.wgsl). Applies
   * with gravity and an equation of state that carries an energy. Always on, but for the
   * tests that compare against the plain reconstruction.
   */
  wellBalanced: boolean;
  /**
   * Adaptive mesh: number of refinement levels below the coarsest (0 = uniform grid). `n` is
   * then the finest resolution and the coarsest level has n >> amrLevels cells per side.
   */
  amrLevels: number;
  /** Atlas slots per side for the adaptive mesh (0 = choose from the root grid). */
  amrBudget: number;
  /**
   * Adaptive mesh: per-block timesteps. A block may step up to 2^timeBins times less often
   * than the fastest one, where its own signal speeds allow (0 = one timestep for all).
   */
  timeBins: number;
  /**
   * Keep per-block timesteps on even while they save too little work to pay for their extra
   * passes. Otherwise such stretches of a run are stepped with one timestep (tests set this).
   */
  timeBinsAlways: boolean;
  /** Optically thin cooling and heating (cooling/*.wgsl); needs an equation of state with an energy. */
  cooling: CoolingKind;
  coolParams: CoolingParams;
  /**
   * Kinematic viscosity nu: the stress is rho nu (grad v + its transpose - 2/3 div v), and the
   * walls of obstacles hold the gas (hydro/diffusion.wgsl). 0 is off.
   */
  viscosity: number;
  /**
   * Thermal conduction: the heat flux is -conductivity * rho * grad T, so an ideal gas's
   * temperature diffuses at (gamma - 1) * conductivity. Needs an equation of state with an
   * energy. 0 is off.
   */
  conductivity: number;
  /**
   * The preset's damping zone (Problem.damping): where it starts, as a distance from the box
   * center, and how fast it relaxes the gas at the box's edge (the preset defines the unit).
   * A rate of 0 switches it off.
   */
  dampRadius: number;
  dampRate: number;
  /** Start from a snapshot: its time, step count and point masses rather than the preset's. */
  restart?: Restart;
  /**
   * A time the clock stops at short of tEnd, as it does at tEnd: the page sets it to the next
   * checkpoint, and moves it on once the checkpoint is written. Not a setting of the run.
   */
  tStop?: number;
  /**
   * A block is split where its refinement indicator (a normalized second derivative, between 0
   * and 1: amr/flag.wgsl) exceeds refineHi, merged below refineLo.
   */
  refineHi: number;
  refineLo: number;
  /** Gas below this density does not ask for refinement (amr/flag.wgsl); 0 is off. */
  refineRho: number;
  /** adaptive mesh: density above which gas is always refined to the finest level (0: none) */
  refineDense: number;
  /**
   * Limits on tenuous gas (hydro/fluff.wgsl): below density `rho`, the specific internal
   * energy is capped at `u` and the speed at `v`. A zero density switches them off.
   */
  fluff: { rho: number; u: number; v: number };
  /** With self-gravity, refine where the Jeans length is under this many cells (0 = off). */
  jeansRefine: number;
}

/** One cell of the grid as the query tool reads it (Simulation.probe). */
export interface CellProbe {
  /** cell center, relative to the box center */
  pos: [number, number, number];
  /** index of the cell among the cells of its level across the whole box */
  index: number[];
  /** cell size */
  dx: number;
  /** conserved state: (rho, rho v) and (E, rho e, rho s, solid flag) */
  u1: number[];
  u2: number[];
  /** Adaptive mesh: the leaf block that holds the cell. */
  block?: { level: number; coords: number[]; slot: number; indicator: number | null };
}

/**
 * Read one texel of each texture, at the same coordinates, in one round trip. Each lands at its
 * own 256-byte offset of the buffer.
 */
export async function readTexels(d: GPUDevice, textures: GPUTexture[], at: [number, number, number]): Promise<Float32Array[]> {
  const buf = d.createBuffer({ size: 256 * textures.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = d.createCommandEncoder();
  textures.forEach((t, i) => enc.copyTextureToBuffer({ texture: t, origin: at }, { buffer: buf, offset: 256 * i }, [1, 1, 1]));
  d.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const all = new Float32Array(buf.getMappedRange().slice(0));
  buf.unmap();
  buf.destroy();
  return textures.map((_, i) => all.slice(64 * i, 64 * i + 4));
}

/** What the front end, renderer and tests need from a simulation, uniform or adaptive. */
export interface Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  /** finest cells per side, and along z (1 in 2D) */
  readonly n: number;
  readonly nz: number;
  /** parity of the state textures holding the current state */
  cur: number;
  readonly paramBuffer: GPUBuffer;
  /** Encode up to `count` steps; returns how many were encoded. */
  encodeSteps(enc: GPUCommandEncoder, count: number): number;
  /** Call after submitting an encoder that encodeSteps() wrote to. */
  submitted(): void;
  /**
   * When encodeSteps() took fewer steps than asked because it is waiting on a refinement check,
   * a promise that settles once the check is answered and stepping can go on; otherwise null.
   */
  stepsResume(): Promise<void> | null;
  readDiagnostics(): Promise<Diagnostics | null>;
  diagnostics(): Promise<Diagnostics>;
  runToEnd(batch?: number, maxSteps?: number): Promise<number>;
  reset(): void;
  /** A star in the initial conditions is still being settled, or a pass compiling: no steps are taken yet. */
  readonly preparing: boolean;
  /** A pass the steps need is still compiling (part of `preparing`). */
  readonly compiling?: boolean;
  writeParams(): void;
  buildHydro(): Promise<void>;
  refreshPotential(): void;
  /** Encode one application of an editing brush to the current state (running or paused). */
  encodeBrush(enc: GPUCommandEncoder, op: BrushOp): void;
  /**
   * Place a star (a `star` brush). On an adaptive mesh the star is first refined to the finest
   * level, and the blocks that does not need are merged again afterwards. Submits its own work;
   * nothing may be encoded meanwhile.
   */
  placeStar(op: BrushOp): Promise<void>;
  /** Call once gas has been added or removed, with the new total mass: self-gravity is re-solved. */
  massChanged(mass: number): void;
  /**
   * Add a point mass to the run (a reset removes it). Returns false if there is no room for
   * another, or this grid cannot hold point masses.
   */
  addMass(m: PointMass): boolean;
  /**
   * The point masses as they are now, or null if there are none or the last such read has not
   * come back yet. Cheap enough to call every frame.
   */
  readMasses(): Promise<Required<PointMass>[] | null>;
  /** Remove the point masses added with addMass(), keeping the preset's. */
  removeAddedMasses(): void;
  /**
   * Give the preset's point masses the softening and sink radii its current parameters ask
   * for, leaving them where they are and as they move.
   */
  retuneMasses(): void;
  /** The cell at `f` (in fractions of the box, each in [0, 1)) as it is now. */
  probe(f: [number, number, number]): Promise<CellProbe | null>;
  /** Replace the sources and sinks (at most MAX_SOURCES; a reset removes them all). */
  setSources(list: SourceSpec[]): void;
  destroy(): void;
  /** Size of the uniform volume the renderer displays, and the pass that fills it. */
  readonly displayDims: [number, number, number];
  readonly fieldShader: string;
  /** The display pass's resources with the state at `parity` and the species at `speciesParity`. */
  fieldResources(parity: number, speciesParity: number): Record<number, GPUBuffer | GPUTextureView>;
  /**
   * Species (species/species.ts): the parity of the textures holding rho X now, and the
   * texels along z from one group of four to the next (0 without species).
   */
  speciesCur: number;
  readonly speciesStride: number;
  /** Extra text for the status line. */
  status(): string;
  /**
   * Adaptive mesh: the leaf blocks, each as the corner nearest the origin and the side (four
   * numbers, in fractions of the box), and a version that changes whenever they do.
   */
  blockBoxes?(): { version: number; boxes: Float32Array };
  /**
   * A 3D adaptive mesh: what the volume view needs to ray-march the leaf blocks at their own
   * resolution (render/volume_atlas.wgsl) instead of the resampled display volume.
   */
  displayAtlas?(): DisplayAtlas | null;
  /**
   * Debris energies (debris/debris.ts): the gas binned by its orbital energy about the
   * problem's chosen point mass, what has left the box and what is in it, with the return rate
   * of the bound part. Null if the problem does not ask for them. Takes a census of the grid,
   * so call it between steps.
   */
  debris(): Promise<DebrisReport | null>;
  /** The mass that has left the box per energy bin, as a snapshot keeps it; null without debris. */
  debrisLedger(): Promise<DebrisLedger | null>;
}

/** See Simulation.displayAtlas. */
export interface DisplayAtlas {
  /** atlas texels along each side (the state atlas's) */
  side: number;
  slotsPerSide: number;
  /** root blocks per side, and the finest level */
  roots: number;
  levels: number;
  /** leaf slot per block of the finest level, and the list of leaf slots with how many there are */
  idx: GPUTexture;
  list: GPUBuffer;
  leaves: number;
}

export function configFor(problem: Problem, n: number, over: Partial<SimConfig> = {}): SimConfig {
  const ndim = over.ndim ?? 3;
  // 2D: one periodic layer in z, and no self-gravity (the multigrid solver is 3D only).
  const bc2 = problem.bc2d ?? {
    lo: [problem.bcLo[0], problem.bcLo[1], 0] as Bc3,
    hi: [problem.bcHi[0], problem.bcHi[1], 0] as Bc3,
  };
  const params = Object.fromEntries((problem.params ?? []).map((q) => [q.id, q.value]));
  return {
    n,
    ndim,
    gext: problem.gext?.(ndim) ?? [0, 0, 0],
    params,
    problem,
    eos: problem.eos,
    eosParams: { ...defaultEosParams(), ...physicalEosParams(problemUnits(problem, params), problem.composition), ...problem.eosParams },
    riemann: 'hllc',
    limiter: 'koren',
    gravity: ndim === 2 && isSelfGravity(problem.gravity) ? 'off' : problem.gravity,
    G: problem.G,
    bcLo: ndim === 2 ? bc2.lo : problem.bcLo,
    bcHi: ndim === 2 ? bc2.hi : problem.bcHi,
    cfl: 0.8,
    tEnd: problem.tEnd,
    rhoFloor: problem.rhoFloor ?? 1e-10,
    pFloor: problem.pFloor ?? 1e-14,
    etaDual: 1e-3,
    shockThresh: 0.5,
    jeansFloor: problem.jeansFloor ?? false,
    jeansCells: problem.jeansCells ?? 4,
    // Warm-started from the previous step. One cycle is enough for the validation tests, but
    // leaves total momentum drifting about ten times faster in long collapse runs. On the
    // adaptive mesh this is the most a step runs: cycles after the first run only while they
    // still cut the residual (amr/gravity/gate.wgsl), and three help where the potential
    // changes fast, as at the bounce of a stellar collision.
    vcycles: (over.amrLevels ?? 0) > 0 ? 3 : 2,
    gravEnergyFlux: true,
    wellBalanced: true,
    amrLevels: 0,
    amrBudget: 0,
    timeBins: 0,
    timeBinsAlways: false,
    // FLASH's defaults for the same indicator.
    refineHi: 0.8,
    refineLo: 0.2,
    refineRho: problem.refineRho ?? problem.fluff?.rho ?? (problem.star ? 10 * problem.star.rhoAmb : 0),
    refineDense: problem.refineDense ?? 0,
    fluff: { ...(problem.fluff ?? { rho: 0, u: 0, v: 0 }) },
    jeansRefine: 16,
    cooling: problem.cooling ?? 'off',
    coolParams: { ...defaultCoolingParams(), ...problem.coolParams },
    viscosity: problem.viscosity ?? 0,
    conductivity: problem.conductivity ?? 0,
    dampRadius: problem.damping?.radius ?? 0,
    dampRate: problem.damping?.rate ?? 0,
    ...over,
  };
}

/**
 * The uniform gravitational acceleration on the gas: the preset's own, plus the constant field
 * of strength G pointing down (-y in 2D, -z in 3D) when gravity is 'constant'.
 */
export function uniformField(c: SimConfig): [number, number, number] {
  const g = [...c.gext] as [number, number, number];
  if (c.gravity === 'constant') g[c.ndim === 2 ? 1 : 2] -= c.G;
  return g;
}

/** The run's burning of fuel to ash (burn/burn.ts), from the preset and its parameters, or null. */
export const burnSpec = (c: SimConfig): BurnSpec | null => c.problem.burn?.(c.params) ?? null;

/** The modules the hydro shaders are assembled with. */
export function hydroSlots(c: SimConfig): Record<string, string> {
  return {
    eos: EOS_MODULES[c.eos],
    riemann: `hydro/riemann/${c.riemann}.wgsl`,
    limiter: `hydro/limiters/${c.limiter}.wgsl`,
    cooling: COOLING_MODULES[c.cooling],
    // Which diffusive fluxes are compiled in (hydro/diffusion.wgsl); their strengths are in P.diff.
    diffusion: `const VISCOSITY = ${c.viscosity > 0};\nconst CONDUCTION = ${conducts(c)};\n`,
    // The preset's damping zone (hydro/damping.wgsl), or none.
    damping: c.problem.damping?.wgsl ?? NO_DAMPING,
    // Whether the update records face mass fluxes for the species pass (species/species.ts).
    species: speciesSlot(speciesGroups(c.problem)),
    // One composition everywhere, or the species' own in every cell (species/mixture.ts).
    comp: problemCompSlot(c.problem, c.eosParams.gamma, speciesGroups(c.problem)),
    // Whether the update tallies the gas leaving the box by its orbital energy (debris/debris.ts).
    debris: debrisSlot(c),
  };
}

const NO_DAMPING = 'const DAMPING = false;\nfn damping_target(x: vec3<f32>) -> DampTarget { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }\n';

/** Thermal conduction is on: it needs a coefficient and an equation of state with an energy. */
export const conducts = (c: SimConfig): boolean => c.conductivity > 0 && c.eos !== 'barotropic';

/** Which diffusive fluxes the shaders must be built with: a change here means rebuilding them. */
export const diffusionKey = (c: SimConfig): string => `${c.viscosity > 0}/${conducts(c)}`;

export const MAX_MASSES = 4;
/** Floats per mass in the Masses block (gravity/bh.wgsl). */
export const MASS_STRIDE = 40;
/** Floats in the Masses block: the frame's acceleration, then MASS_STRIDE per mass. */
export const MASS_FLOATS = 4 + MASS_STRIDE * MAX_MASSES;
/** Channels of the point masses' reduction (gravity/bh_cell.wgsl). */
export const MASS_RED = 1 + 10 * MAX_MASSES;

/**
 * The point masses a run starts with: the preset's, with what a snapshot recorded laid over
 * them. `dx` is the cell size (the coarsest level's on an adaptive mesh).
 */
export function initialMasses(c: SimConfig, dx: number): Required<PointMass>[] {
  const base = c.problem.masses?.(c.params, c.G, dx, c.ndim) ?? [];
  const saved = c.restart?.masses ?? [];
  const out = base.map((m, i) => completeMass({ ...m, ...saved[i] }, dx));
  // Masses placed by hand before the snapshot was taken come back whole.
  for (const m of saved.slice(base.length)) {
    if (m.gm !== undefined && m.pos && m.vel) out.push(completeMass(m as PointMass, dx));
  }
  return out.slice(0, MAX_MASSES);
}

/** A point mass with its defaults filled in: softening of two cells, a sink of that radius. */
export function completeMass(q: PointMass, dx: number): Required<PointMass> {
  const soft = q.soft ?? 2 * dx;
  return {
    gm: q.gm, pos: q.pos, vel: q.vel, soft, sinkRadius: q.sinkRadius ?? soft, sinkRate: q.sinkRate ?? 1,
    accreted: q.accreted ?? 0, accretedMomentum: q.accretedMomentum ?? [0, 0, 0], accretedSpin: q.accretedSpin ?? [0, 0, 0],
  };
}

/**
 * Write the softening and sink radii of the preset's point masses (from its current parameters)
 * into the Masses block, leaving their positions, velocities and masses as the run has them.
 */
export function writeMassRadii(d: GPUDevice, bh: GPUBuffer, c: SimConfig, dx: number, count: number) {
  const base = (c.problem.masses?.(c.params, c.G, dx, c.ndim) ?? []).slice(0, count);
  base.forEach((q, i) => {
    const m = completeMass(q, dx);
    const at = 16 + 4 * MASS_STRIDE * i;
    d.queue.writeBuffer(bh, at + 28, new Float32Array([m.soft]));
    d.queue.writeBuffer(bh, at + 44, new Float32Array([m.sinkRadius]));
  });
}

/** Pack the Masses block. The layout must match the struct in gravity/bh.wgsl. */
export function packMasses(list: Required<PointMass>[]): Float32Array<ArrayBuffer> {
  const f = new Float32Array(MASS_FLOATS);
  // A value summed with Kahan compensation goes in as its float32 rounding, with the
  // compensation set to what that rounding added, so that nothing is lost in the round trip.
  const c = (v: number) => Math.fround(v) - v;
  list.forEach((m, i) =>
    f.set(
      [
        ...m.pos, m.gm, ...m.vel, m.soft, 0, 0, 0, m.sinkRadius, m.sinkRate, m.accreted, 0, 0, ...m.accretedMomentum, 0, ...m.accretedSpin, 0,
        c(m.gm), c(m.accreted), 0, 0, ...m.vel.map(c), 0, ...m.accretedMomentum.map(c), 0, ...m.accretedSpin.map(c), 0,
      ],
      4 + MASS_STRIDE * i,
    ),
  );
  return f;
}

/** The point masses in a Masses block read back from the GPU. */
export function unpackMasses(f: Float32Array, count: number): Required<PointMass>[] {
  const out: Required<PointMass>[] = [];
  for (let i = 0; i < count; i++) {
    const o = 4 + MASS_STRIDE * i;
    // Kahan-summed values, less their compensation (gravity/bh.wgsl), in double precision.
    const v3 = (at: number, c: number) => [f[at] - f[c], f[at + 1] - f[c + 1], f[at + 2] - f[c + 2]];
    out.push({
      gm: f[o + 3] - f[o + 24], pos: [f[o], f[o + 1], f[o + 2]], vel: v3(o + 4, o + 28), soft: f[o + 7], sinkRadius: f[o + 11], sinkRate: f[o + 12],
      accreted: f[o + 13] - f[o + 25], accretedMomentum: v3(o + 16, o + 32), accretedSpin: v3(o + 20, o + 36),
    });
  }
  return out;
}

/** One application of an editing brush (edit/brush_core.wgsl). */
export interface BrushOp {
  /** `eos-out` and `eos-in` act on every cell: see convertEos() in ui/app.ts and edit/brush_core.wgsl. */
  kind: 'mass' | 'velocity' | 'temperature' | 'obstacle' | 'eos-out' | 'eos-in' | 'star';
  /** center and radius, in fractions of the box */
  pos: [number, number, number];
  radius: number;
  /** 0: hard edge; 1: falls off all the way from the center */
  softness: number;
  /**
   * mass: density added at the center (negative removes); temperature: log of the factor;
   * obstacle: positive makes the cells solid, negative turns them back into gas;
   * star: central density
   */
  amount: number;
  /** velocity tool: the velocity to set; star: its velocity */
  vel: [number, number, number];
  /** mass tool: new gas brings no momentum (otherwise it arrives with the local velocity) */
  atRest?: boolean;
  /** obstacle tool: the outline, drawn on the screen and cut through the box along the line of sight */
  outline?: Outline;
  /** star: see starBrush() */
  star?: StarProfile;
}

/** A polytrope for the star brush: p = K rho^gamma, and rho / rho_c at evenly spaced r / R from 0 to 1. */
export interface StarProfile {
  K: number;
  gamma: number;
  table: Float32Array;
}

/** Entries of StarProfile.table: as STAR_TABLE in edit/brush_core.wgsl. */
export const STAR_TABLE = 64;

/**
 * The star brush for an n polytrope of central density `rhoc` and radius `radius` (in units
 * of the box, which is `box` across), at `pos` (in fractions of the box) moving at `vel`.
 */
export function starBrush(n: number, rhoc: number, radius: number, box: number, G: number, pos: [number, number, number], vel: [number, number, number]): BrushOp {
  const poly = laneEmdenCached(n);
  const table = new Float32Array(STAR_TABLE);
  for (let i = 0; i < STAR_TABLE - 1; i++) table[i] = Math.pow(poly.theta(i / (STAR_TABLE - 1)), n);
  return {
    kind: 'star', pos, radius, softness: 0, amount: rhoc, vel,
    star: { K: polytropeK(poly, rhoc, radius * box, G), gamma: 1 + 1 / n, table },
  };
}

export interface Outline {
  /**
   * Takes a point of the box (in fractions of it, with a 1 appended) to homogeneous screen
   * coordinates, the units the outline is given in. Column-major, 16 numbers.
   */
  matrix: number[];
  /** an ellipse: center and half-axes ... */
  ellipse?: [number, number, number, number];
  /** ... or a polygon: x, y of each vertex, at most MAX_OUTLINE_VERTICES of them */
  polygon?: number[];
}

/** Whether a point of the box (in fractions of it) lies behind an outline: as in edit/brush_core.wgsl. */
export function outlineContains(o: Outline, x: number, y: number, z: number): boolean {
  const m = o.matrix;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w <= 0) return false;
  const sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
  const sy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
  if (o.ellipse) {
    const [cx, cy, hx, hy] = o.ellipse;
    return ((sx - cx) / hx) ** 2 + ((sy - cy) / hy) ** 2 <= 1;
  }
  const p = o.polygon ?? [];
  let inside = false;
  for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) {
    if (p[i + 1] > sy !== p[j + 1] > sy && sx < ((p[j] - p[i]) * (sy - p[i + 1])) / (p[j + 1] - p[i + 1]) + p[i]) inside = !inside;
  }
  return inside;
}

/** A source or sink of gas (hydro/sources.wgsl). */
export interface SourceSpec {
  /** center and radius, in fractions of the box */
  pos: [number, number, number];
  radius: number;
  softness: number;
  /** velocity of the gas a source adds; zero for one that spreads evenly */
  vel: [number, number, number];
  /** a source: density added per unit time at its center */
  rate: number;
  /** a sink: fraction of the gas removed per unit time at its center (0 for a source) */
  decay: number;
}

export const MAX_SOURCES = 8;
export const SOURCE_BYTES = 16 + 48 * MAX_SOURCES;

/** Pack the Sources block. The layout must match the struct in hydro/sources.wgsl. */
export function packSources(list: SourceSpec[]): ArrayBuffer {
  const buf = new ArrayBuffer(SOURCE_BYTES);
  const f = new Float32Array(buf);
  const use = list.slice(0, MAX_SOURCES);
  new Uint32Array(buf)[0] = use.length;
  use.forEach((q, i) => f.set([...q.pos, q.radius, ...q.vel, q.softness, q.rate, q.decay, 0, 0], 4 + 12 * i));
  return buf;
}

export const MAX_OUTLINE_VERTICES = 32;
export const BRUSH_BYTES = 384;

/** Pack the Brush block. The layout must match the struct in edit/brush_core.wgsl. */
export function packBrush(b: BrushOp): ArrayBuffer {
  const buf = new ArrayBuffer(BRUSH_BYTES);
  const f = new Float32Array(buf);
  f.set([...b.pos, b.radius, ...b.vel, b.softness], 0);
  const u = new Uint32Array(buf);
  u[8] = ['mass', 'velocity', 'temperature', 'obstacle', 'eos-out', 'eos-in', 'star'].indexOf(b.kind);
  f[9] = b.amount;
  u[10] = b.atRest ? 1 : 0;
  if (b.outline) {
    const poly = (b.outline.polygon ?? []).slice(0, 2 * MAX_OUTLINE_VERTICES);
    u[11] = poly.length / 2;
    f.set(b.outline.matrix, 12);
    f.set(b.outline.ellipse ?? [0, 0, 1, 1], 28);
    f.set(poly, 32);
  }
  if (b.star) {
    f.set([b.star.K, b.star.gamma], 28);
    f.set(b.star.table.subarray(0, STAR_TABLE), 32);
  }
  return buf;
}

/** Number of floats the diagnostics readback holds: clock (4), reduction, the Masses block. */
export const DIAG_FLOATS = 4 + 23 + MASS_FLOATS;

/** Decode a diagnostics readback: [clock (4), reduction channels, the Masses block]. */
export function diagnosticsFrom(a: Float32Array, nMasses: number): Diagnostics {
  const r = a.subarray(4);
  const masses = unpackMasses(a.subarray(4 + 23), nMasses);
  const ranges: [number, number][] = [];
  for (let i = 0; i < 5; i++) ranges.push([r[10 + 2 * i], r[11 + 2 * i]]);
  return {
    t: a[1] + a[0],
    dt: a[0],
    steps: a[3] + (a[0] > 0 ? 1 : 0),
    mass: r[0],
    momentum: [r[1], r[2], r[3]],
    eKin: r[4],
    eInt: r[5],
    eGrav: r[6],
    eTot: r[4] + r[5] + r[6],
    maxMach: r[7],
    floored: r[8],
    jeansUnresolved: r[9],
    com: [r[20] / r[0], r[21] / r[0], r[22] / r[0]],
    perturber: masses.length ? (masses[0].pos as [number, number, number]) : null,
    masses,
    ranges,
  };
}

/** Size of the Params uniform block (hydro/common.wgsl). */
export const PARAM_BYTES = 272;

/** Pack the Params block. The layout must match the struct in hydro/common.wgsl. */
export function packParams(
  c: SimConfig,
  x: { dims: number[]; dx: number; rhoMean: number; selfGravity: boolean; masses: number; amr?: number[]; refine?: number[]; solids?: boolean; sources?: boolean },
): ArrayBuffer {
  const buf = new ArrayBuffer(PARAM_BYTES);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u.set([...c.bcLo, 0], 0);
  u.set([...c.bcHi, 0], 4);
  f.set([c.eosParams.gamma, c.eosParams.K, c.eosParams.R, c.eosParams.arad], 8);
  u.set([...x.dims, 0], 12);
  // 2D quantities are per unit length along z.
  f.set([c.ndim === 2 ? x.dx ** 2 : x.dx ** 3, c.ndim, ...(x.refine ?? [0, 0])], 16);
  f.set([...uniformField(c), 0], 20);
  // Point masses: how many, whether the frame follows the gas, whether there are any, and
  // whether they are all the field texture holds (no Poisson potential).
  f.set([x.masses, c.problem.frame === 'gas' ? 1 : c.problem.frame === 'masses' ? 2 : 0, x.masses > 0 ? 1 : 0, x.selfGravity ? 0 : 1], 24);
  const fl = c.fluff;
  f.set(fl.rho > 0 ? [fl.rho, fl.u > 0 ? fl.u : 1e30, fl.v > 0 ? fl.v : 1e30, 0] : [0, 0, 0, 0], 28);
  u.set(x.amr ?? [0, 0, 0, 0], 32);
  // The potential of point masses alone is not periodic: its ghosts are extrapolated.
  u.set([c.n, x.selfGravity || x.masses > 0 ? 1 : 0, c.gravity === 'isolated' || !x.selfGravity ? 1 : 0, (c.gravEnergyFlux ? 1 : 0) | (x.solids ? 2 : 0) | (x.sources ? 4 : 0) | (c.wellBalanced && c.eos !== 'barotropic' ? 8 : 0) | (c.problem.refineMasses && c.amrLevels > 0 ? 16 : 0)], 36);
  f.set(
    [
      x.dx,
      c.cfl,
      c.G,
      x.rhoMean,
      c.rhoFloor,
      c.pFloor,
      c.etaDual,
      Math.min(Number.isFinite(c.tEnd) ? c.tEnd : 1e30, c.tStop ?? 1e30),
      1e30,
      c.shockThresh,
      c.jeansFloor ? (c.jeansCells / 4) ** 2 : 0, // scales the 4-cell floor pressure
      c.problem.box,
    ],
    40,
  );
  f.set(packCooling(c.cooling, c.coolParams), 52);
  f.set([Math.max(0, c.viscosity), Math.max(0, c.conductivity), Math.max(0, c.dampRadius), Math.max(0, c.dampRate)], 56);
  f.set([Math.max(0, c.refineRho), Math.max(0, c.refineDense), 0, 0], 60);
  f.set(eosConstants2(c.eos, c.eosParams), 64);
  return buf;
}

// Channel layout of the diagnostics reduction (hydro/diagnostics.wgsl).
export const DIAG_OPS = [SUM, SUM, SUM, SUM, SUM, SUM, SUM, MAX, SUM, SUM, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, SUM, SUM, SUM];

export interface Diagnostics {
  t: number;
  dt: number;
  steps: number;
  mass: number;
  momentum: [number, number, number];
  eKin: number;
  eInt: number;
  eGrav: number;
  eTot: number;
  maxMach: number;
  floored: number;
  jeansUnresolved: number;
  /** center of mass of the gas, relative to the box center */
  com: [number, number, number];
  /** position of the first point mass relative to the box center, if there is one */
  perturber: [number, number, number] | null;
  /** the point masses as they are now */
  masses: Required<PointMass>[];
  /** [min, max] per renderable field, in FIELDS order */
  ranges: [number, number][];
}


export interface RestartState {
  time: number;
  steps: number;
  masses?: Required<PointMass>[];
}

/** Read the clock and the point masses back, for a snapshot. */
export async function readRestart(d: GPUDevice, clock: GPUBuffer, bh: GPUBuffer, nMasses: number): Promise<RestartState> {
  const buf = d.createBuffer({ size: 16 + MASS_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(clock, 0, buf, 0, 16);
  enc.copyBufferToBuffer(bh, 0, buf, 16, MASS_FLOATS * 4);
  d.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0));
  buf.unmap();
  buf.destroy();
  return { time: f[1], steps: f[3], masses: nMasses ? unpackMasses(f.subarray(4), nMasses) : undefined };
}

/**
 * Reads the point masses back without ever waiting on the GPU, for drawing them every frame:
 * a call made while the last one is still in flight returns null.
 */
export class MassReader {
  private buf: GPUBuffer;
  private busy = false;
  constructor(private device: GPUDevice, private bh: GPUBuffer) {
    this.buf = device.createBuffer({ size: MASS_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  async read(count: number): Promise<Required<PointMass>[] | null> {
    if (this.busy || count === 0) return null;
    this.busy = true;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.bh, 0, this.buf, 0, MASS_FLOATS * 4);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.buf.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(this.buf.getMappedRange().slice(0));
      this.buf.unmap();
      return unpackMasses(f, count);
    } finally {
      this.busy = false;
    }
  }
}

export class Sim implements Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  readonly n: number;
  /** cells along z: n, or 1 in 2D */
  readonly nz: number;
  readonly dx: number;

  /** state[parity] = [U1, U2]; `cur` is the parity holding the current state */
  state: GPUTexture[][] = [];
  /** half-step predicted primitives and per-cell flags (hydro/predict.wgsl) */
  pred: GPUTexture[] = [];
  /** gravitational acceleration and half-step potential (gravity/field.wgsl) */
  gfield!: GPUTexture;
  cur = 0;
  /**
   * Species (species/species.ts): rho X in groups of four stacked along z, ping-pong with their
   * own parity `speciesCur` (the state's can return to where it was within a step), and the
   * face mass fluxes the update records for them. Empty without species.
   */
  species: GPUTexture[] = [];
  speciesCur = 0;
  readonly speciesGroups: number;
  /**
   * Where the species set the equation of state (species/mixture.ts): each species' share of
   * the four composition moments, and the moments of every cell, which the species pass writes
   * with rho X and the hydro passes read. Null and absent for a uniform composition.
   */
  get compCoeffs(): Moments[] | null {
    return problemCoefficients(this.cfg.problem, this.cfg.eosParams.gamma);
  }
  private comp?: GPUTexture;
  /** mass fractions half a step ahead (hydro/species_predict.wgsl) */
  private speciesHalf?: GPUTexture;
  private massFlux?: GPUBuffer;
  private pSpeciesPredict?: GPUComputePipeline;
  private pSpecies?: GPUComputePipeline;
  /** gSpecies[state parity][species parity], and the same for the predictor */
  private gSpecies: GPUBindGroup[][] = [];
  private gSpeciesPredict: GPUBindGroup[][] = [];
  /** Burning of fuel to ash (burn/burn.ts), after the species pass; none without. */
  private burn?: BurnPass;
  mg!: Multigrid;
  rhoMean = 0;

  private params: GPUBuffer;
  get paramBuffer(): GPUBuffer {
    return this.params;
  }
  private clock: GPUBuffer;
  /** A copy of the clock, for the timestep of a step not yet taken (previewDt). */
  private clockPreview: GPUBuffer;
  private gClockPreview!: GPUBindGroup;
  private pPrim!: GPUComputePipeline;
  private pUpdate!: RegionPass;
  private pClock!: GPUComputePipeline;
  private pCorrect!: GPUComputePipeline;
  private pPredict!: RegionPass;
  private pField!: GPUComputePipeline;
  private pBhDrift!: GPUComputePipeline;
  private pBhKick!: GPUComputePipeline;
  /** the kick pipeline that only evaluates the accelerations */
  private pBhEval!: GPUComputePipeline;
  private gBhDrift!: GPUBindGroup;
  private gBhKick!: GPUBindGroup;
  private gBhEval!: GPUBindGroup;
  private nMasses = 0;
  /** how many of them the preset (or a snapshot) brought, rather than addMass() */
  private nOwnMasses = 0;
  private redBh!: Reducer;
  /** the point masses (gravity/bh.wgsl) */
  private bh: GPUBuffer;
  private gPredict: RegionGroups[] = [];
  private gField!: GPUBindGroup;
  private gPrim: GPUBindGroup[] = [];
  private gUpdate: RegionGroups[] = [];
  private gCorrect: GPUBindGroup[] = [];
  private gClock!: GPUBindGroup;
  private redDt!: Reducer;
  private redDiag!: Reducer;
  private staging: GPUBuffer;
  private stagingBusy = false;
  private brush: GPUBuffer;
  private pBrush!: GPUComputePipeline;
  private gBrush: GPUBindGroup[] = [];
  /** the grid holds solid or fixed cells, so the passes must look for them */
  private solids = false;
  private sources: GPUBuffer;
  private nSources = 0;
  /** debris energies (debris/debris.ts), if the problem asks for them, and their census pass */
  private debrisBuf?: DebrisBuffers;
  private pCensus?: GPUComputePipeline;

  private constructor(gpu: Gpu, cfg: SimConfig) {
    this.device = gpu.device;
    this.cfg = cfg;
    this.n = cfg.n;
    this.nz = cfg.ndim === 2 ? 1 : cfg.n;
    this.speciesGroups = speciesGroups(cfg.problem);
    this.dx = cfg.problem.box / cfg.n;
    const d = this.device;
    this.params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clock = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.clockPreview = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.bh = d.createBuffer({ size: MASS_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.brush = d.createBuffer({ size: BRUSH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sources = d.createBuffer({ size: SOURCE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.staging = d.createBuffer({
      size: DIAG_FLOATS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * GPU bytes per cell of a uniform grid: seven rgba32float textures (two state pairs, predictor
   * pair, gravity field) and, in 3D, the multigrid solver's r32float levels (potential, source,
   * residual at each coarser level, about 8/7 of a grid each, and the previous potential).
   */
  static cellBytes(ndim: 2 | 3): number {
    return 7 * 16 + (ndim === 3 ? 4 + 3 * 4 * (8 / 7) : 0);
  }

  static async create(gpu: Gpu, cfg: SimConfig): Promise<Sim> {
    // Every grid, 2D included, lives in 3D textures, whose side WebGPU caps (2048 on most
    // adapters). Past it the textures are invalid and every step would silently do nothing.
    const side = gpu.device.limits.maxTextureDimension3D;
    if (cfg.n > side) {
      throw new Error(`a uniform grid of ${cfg.n} cells per side is larger than this GPU's 3D textures allow (${side}); use the adaptive mesh`);
    }
    noteAllocation(cfg.n ** cfg.ndim * Sim.cellBytes(cfg.ndim));
    const sim = new Sim(gpu, cfg);
    const d = sim.device;
    const n = cfg.n;
    const nz = sim.nz;
    const t = (label: string) => tex3d(d, n, 'rgba32float', label, nz);
    d.pushErrorScope('out-of-memory');
    for (let p = 0; p < 2; p++) sim.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    sim.pred = [t('PA'), t('PB')];
    sim.gfield = t('gravity field');
    const oom = await d.popErrorScope();
    if (oom) {
      sim.destroy();
      const gb = (cfg.n ** cfg.ndim * Sim.cellBytes(cfg.ndim)) / 1024 ** 3;
      throw new Error(`out of GPU memory for a uniform grid of ${cfg.n}^${cfg.ndim} cells (about ${gb.toFixed(1)} GB): ${oom.message}`);
    }
    const G = sim.speciesGroups;
    if (G > 0) {
      for (let p = 0; p < 2; p++) sim.species.push(tex3d(d, n, 'rgba32float', `species ${p}`, nz * G));
      sim.speciesHalf = tex3d(d, n, 'rgba32float', 'species half step', nz * G);
      sim.massFlux = d.createBuffer({ label: 'face mass fluxes', size: 3 * 4 * (n + 1) ** 2 * (nz + 1), usage: GPUBufferUsage.STORAGE });
      if (sim.compCoeffs) sim.comp = t('composition');
    }
    const spec = debrisSpec(cfg);
    if (spec) sim.debrisBuf = new DebrisBuffers(d, spec);
    sim.writeParams();

    // Every pipeline is asked for before any is awaited, so the browser compiles them all at
    // once (gpu/device.ts). The hydro passes wait only for the solver's potential and the
    // timestep's first records, which they bind.
    const v = (t: GPUTexture) => t.createView();
    const hydro = (async () => {
      // Self-gravity is 3D only. A 2D run keeps a 2^3 placeholder solver so the shared bindings
      // (potential, multipoles) still exist; it is never dispatched.
      [sim.mg, sim.redDt] = await Promise.all([
        Multigrid.create(d, cfg.ndim === 2 ? 2 : n, cfg.problem.box, sim.params, [sim.state[0][0], sim.state[1][0]]),
        Reducer.create(
          d,
          [n, n, nz],
          [MAX, MAX],
          '',
          {},
          [],
          'timestep',
          [n, n, nz].map((m) => Math.ceil(m / 4)), // the primitives pass does the first stage
        ),
      ]);
      await sim.buildHydro();
    })();
    [sim.pClock, sim.pField, sim.redBh, sim.pBhDrift, sim.pBhKick, sim.pBhEval, sim.pCensus] = await Promise.all([
      computePipeline(d, assemble('hydro/clock.wgsl'), 'clock'),
      computePipeline(d, assemble('gravity/field.wgsl'), 'gravity field'),
      // Point masses: their pull on the gas and what their sinks take are reduced each step, and
      // drive their orbits, their growth and (in the gas's frame) the frame's acceleration.
      Reducer.create(
        d,
        [n, n, nz],
        new Array(MASS_RED).fill(SUM),
        'gravity/bh_moment.wgsl',
        {},
        [0, 1].map((p) => ({ 0: sim.params, 1: v(sim.state[p][0]), 2: sim.bh, 3: v(sim.state[p][1]), 4: sim.clock })),
        'point masses',
      ),
      computePipeline(d, assemble('gravity/bh_drift.wgsl'), 'point mass drift'),
      computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass kick'),
      computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass accelerations', { LIVE: 0 }),
      spec ? computePipeline(d, assemble('debris/census.wgsl'), 'debris census') : undefined,
      hydro,
    ]);
    const phi = sim.mg.phi;
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock, 3: sim.bh });
    sim.gClockPreview = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clockPreview, 3: sim.bh });
    sim.gField = bindGroup(d, sim.pField, { 0: sim.params, 1: v(phi), 2: v(sim.mg.phiOld), 3: v(sim.gfield) });
    sim.gBhDrift = bindGroup(d, sim.pBhDrift, { 0: sim.params, 1: sim.clock, 2: sim.bh });
    const kick = { 0: sim.params, 1: sim.clock, 2: sim.redBh.result, 3: sim.bh };
    sim.gBhKick = bindGroup(d, sim.pBhKick, kick);
    sim.gBhEval = bindGroup(d, sim.pBhEval, kick);
    sim.reset();
    await sim.settled;
    return sim;
  }

  /** The composition moments for a pass that reads them (binding 27), if they vary. */
  private get compIn(): Record<number, GPUTextureView> {
    return this.comp ? { 27: this.comp.createView() } : {};
  }

  /** (Re)build every pipeline that depends on the EOS, Riemann solver or limiter. */
  async buildHydro() {
    const d = this.device;
    const c = this.cfg;
    const slots = hydroSlots(c);
    const v = (t: GPUTexture) => t.createView();
    const k = this.compIn;
    const phi = this.mg.phi;
    const dims = [this.n, this.n, this.nz];
    // The diagnostics' reduction compiles alongside the hydro passes.
    const diag = Reducer.create(
      d,
      [this.n, this.n, this.nz],
      DIAG_OPS,
      'hydro/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(phi), ...k })),
      'diagnostics',
    );
    diag.catch(() => {}); // a failure is reported where it is awaited, below
    this.pPredict?.destroy();
    this.pUpdate?.destroy();
    [this.pPrim, this.pPredict, this.pUpdate, this.pCorrect, this.pBrush] = await Promise.all([
      computePipeline(d, assemble('hydro/rates.wgsl', slots), 'timestep rates'),
      RegionPass.create(d, assemble('hydro/predict.wgsl', slots), 'predict', dims),
      RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update', dims, {}, true),
      computePipeline(d, assemble('gravity/correct.wgsl', slots), 'gravity correct'),
      computePipeline(d, assemble('edit/brush.wgsl', slots), 'brush'),
    ]);
    this.gBrush = [0, 1].map((p) =>
      bindGroup(d, this.pBrush, {
        0: this.params, 1: this.brush, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]), ...k,
      }),
    );
    this.gPrim = [];
    this.gUpdate = [];
    this.gCorrect = [];
    this.gPredict = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      this.gPrim.push(
        bindGroup(d, this.pPrim, {
          0: this.params,
          20: this.bh,
          1: v(this.state[p][0]),
          2: v(this.state[p][1]),
          5: v(this.gfield),
          6: this.redDt.first,
          ...k,
        }),
      );
      this.gPredict.push(
        this.pPredict.groups({
          0: this.params,
          20: this.bh,
          1: this.clock,
          2: v(this.state[p][0]),
          3: v(this.state[p][1]),
          4: v(this.gfield),
          5: v(this.pred[0]),
          6: v(this.pred[1]),
          ...k,
        }),
      );
      this.gUpdate.push(
        this.pUpdate.groups({
          0: this.params,
          20: this.bh,
          1: this.clock,
          2: v(this.state[p][0]),
          3: v(this.state[p][1]),
          6: v(this.gfield),
          7: v(this.state[o][0]),
          8: v(this.state[o][1]),
          9: v(this.pred[0]),
          10: v(this.pred[1]),
          11: this.sources,
          ...((this.massFlux ? { 13: this.massFlux } : {}) as Record<number, GPUBuffer>),
          ...(this.debrisBuf?.bindings(this.bh) ?? {}),
          ...k,
        }),
      );
      this.gCorrect.push(
        bindGroup(d, this.pCorrect, {
          0: this.params,
          1: this.clock,
          2: v(this.state[o][0]),
          3: v(this.state[o][1]),
          4: v(this.gfield),
          5: v(this.state[p][0]),
          6: v(this.state[p][1]),
          7: this.bh,
          ...k,
        }),
      );
    }
    if (this.speciesGroups > 0) {
      const ns = { species_n: speciesConst(this.speciesGroups), comp: slots.comp };
      [this.pSpeciesPredict, this.pSpecies] = await Promise.all([
        computePipeline(d, assemble('hydro/species_predict.wgsl', ns), 'species half step'),
        computePipeline(d, assemble('hydro/species.wgsl', ns), 'species'),
      ]);
      const both = (pipe: GPUComputePipeline, more: (p: number, q: number) => Record<number, GPUBindingResource | GPUBuffer>) =>
        [0, 1].map((p) =>
          [0, 1].map((q) =>
            bindGroup(d, pipe, { 0: this.params, 1: this.clock, 3: v(this.state[p][1]), 5: v(this.species[q]), ...more(p, q) }),
          ),
        );
      this.gSpeciesPredict = both(this.pSpeciesPredict, (p) => ({ 2: v(this.state[p][0]), 6: v(this.speciesHalf!) }));
      this.gSpecies = both(this.pSpecies, (p, q) => ({
        4: v(this.state[1 - p][0]),
        6: v(this.species[1 - q]),
        7: this.massFlux!,
        8: v(this.speciesHalf!),
        ...((this.comp ? { 26: v(this.comp), 24: v(this.pred[1]) } : {}) as Record<number, GPUTextureView>),
      }));
    }
    this.burn?.destroy();
    this.burn = undefined;
    const burn = burnSpec(c);
    if (burn && this.speciesGroups > 0) {
      if (this.comp) throw new Error('burning with species that set the equation of state is not supported');
      this.burn = await BurnPass.create(d, burn, c.problem.species, {
        dims: [this.n, this.n, this.nz], stride: this.nz, groups: this.speciesGroups, slots,
        params: this.params, clock: this.clock, state: this.state, species: this.species, pb: this.pred[1],
      });
    }
    this.redDiag = await diag;
  }

  /** Push the current configuration to the GPU parameter block. */
  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [c.n, c.n, this.nz],
      dx: this.dx,
      rhoMean: this.rhoMean,
      selfGravity: this.selfGravity,
      masses: this.nMasses,
      solids: this.solids,
      sources: this.nSources > 0,
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Load the problem's initial conditions and solve for the initial potential. */
  reset() {
    const c = this.cfg;
    const n = this.n;
    const nz = this.nz;
    const eos = makeEos(c.eos, c.eosParams);
    const fn = c.problem.init(n, eos, c.ndim, c.params);
    const masses = initialMasses(c, this.dx);
    this.nMasses = this.nOwnMasses = masses.length;
    this.device.queue.writeBuffer(this.bh, 0, packMasses(masses));
    this.debrisBuf?.reset(c.restart?.debris);
    const u1 = new Float32Array(n * n * nz * 4);
    const u2 = new Float32Array(n * n * nz * 4);
    const G = this.speciesGroups;
    const sp = new Float32Array(n * n * nz * 4 * G);
    const kc = this.compCoeffs;
    const km = kc ? new Float32Array(n * n * nz * 4) : undefined;
    const half = c.problem.box / 2;
    let mass = 0;
    let o = 0;
    this.solids = false;
    this.nSources = 0;
    this.device.queue.writeBuffer(this.sources, 0, packSources([]));
    for (let k = 0; k < nz; k++) {
      const z = nz === 1 ? 0 : (k + 0.5) * this.dx - half;
      for (let j = 0; j < n; j++) {
        const y = (j + 0.5) * this.dx - half;
        for (let i = 0; i < n; i++, o += 4) {
          const x = (i + 0.5) * this.dx - half;
          const w = fn(x, y, z);
          // Where the species set the equation of state, the cell's own.
          let ce = eos;
          if (km) {
            const m = mixMoments(kc!, w.X);
            km.set(m, o);
            ce = makeEos(c.eos, c.eosParams, m, hydrogenParticles(c.problem.composition));
          }
          const p = ce.hasEnergy ? w.p : ce.pressure(w.rho, 0);
          const eint = ce.eintFromP(w.rho, p);
          const mark = cellMark(w);
          if (mark) this.solids = true;
          const go = mark === 1 ? 0 : 1; // the gas kept inside an obstacle is at rest
          u1[o] = w.rho;
          u1[o + 1] = w.rho * w.vx * go;
          u1[o + 2] = w.rho * w.vy * go;
          u1[o + 3] = w.rho * w.vz * go;
          u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz) * go;
          u2[o + 1] = eint;
          u2[o + 2] = w.rho * w.s;
          u2[o + 3] = mark;
          if (G) packSpecies(sp, o, n * n * nz * 4, G, w.rho, w.X);
          mass += w.rho;
        }
      }
    }
    this.setState(u1, u2, mass / (n * n * nz));
    if (G) {
      this.speciesCur = 0;
      writeTexture(this.device, this.species[0], n, 4, sp, nz * G);
      if (km) writeTexture(this.device, this.comp!, n, 4, km, nz);
    }
    // The clock is (dt, t, Kahan compensation, steps).
    if (c.restart) this.device.queue.writeBuffer(this.clock, 0, new Float32Array([0, c.restart.time, 0, c.restart.steps]));
    this.startSettling(u1, u2);
  }

  /** Resolves once a star in the initial conditions has been settled (see problems/settle.ts). */
  settled: Promise<void> = Promise.resolve();
  private settleEpoch = 0;
  private settling = false;
  get preparing(): boolean {
    return this.settling;
  }
  /** largest relative change of density in each round of the last settling, and how long it took (inspection) */
  settleLog: number[] = [];
  settleMs = 0;

  /**
   * Settle a star into the grid's own hydrostatic balance: no steps are taken until it has.
   * A reset meanwhile abandons it.
   */
  private startSettling(u1: Float32Array, u2: Float32Array) {
    const epoch = ++this.settleEpoch;
    const c = this.cfg;
    this.settling = false;
    this.settled = Promise.resolve();
    if (c.restart || !c.wellBalanced || !this.selfGravity) return;
    const eos = makeEos(c.eos, c.eosParams);
    const set = c.problem.settle ? c.problem.settle(c.params, eos) : c.problem.star ? settlingOf(c.problem.star, c.G) : null;
    if (!set) return;
    this.settling = true;
    const vol = this.dx ** 3;
    const n = this.n;
    const half = c.problem.box / 2;
    const at = (i: number) => (i + 0.5) * this.dx - half;
    const cells: SettleCells = {
      count: u1.length / 4,
      pos: (q) => [at(q % n), at(Math.floor(q / n) % n), at(Math.floor(q / (n * n)))],
      vol: () => vol,
      u1,
      u2,
    };
    const t0 = performance.now();
    this.settled = (async () => {
      try {
        let peak = 0;
        for (let o = 0; o < u1.length; o += 4) peak = Math.max(peak, u1[o]);
        const settler = new Settler(set, cells);
        this.settleLog = [];
        for (let round = 0; round < SETTLE_ROUNDS; round++) {
          const phi = await this.readPotential();
          if (epoch !== this.settleEpoch) return;
          const change = settler.round(phi);
          this.settleLog.push(change / peak);
          writeTexture(this.device, this.state[this.cur][0], this.n, 4, u1, this.nz);
          this.refreshPotential();
          if (change < (set.tol ?? SETTLE_TOL) * peak) break;
        }
        settler.finish(eos.eintFromP);
        writeTexture(this.device, this.state[this.cur][0], this.n, 4, u1, this.nz);
        writeTexture(this.device, this.state[this.cur][1], this.n, 4, u2, this.nz);
        this.refreshPotential();
      } finally {
        if (epoch === this.settleEpoch) {
          this.settling = false;
          this.settleMs = performance.now() - t0;
        }
      }
    })();
  }

  /** Time, step count and point masses, for a snapshot (see io/snapshot.ts). */
  async readRestart(): Promise<RestartState> {
    return readRestart(this.device, this.clock, this.bh, this.nMasses);
  }

  async debrisLedger(): Promise<DebrisLedger | null> {
    return this.debrisBuf ? this.debrisBuf.readLedger() : null;
  }

  async debris(): Promise<DebrisReport | null> {
    const b = this.debrisBuf;
    if (!b) return null;
    const d = await this.diagnostics();
    const v0 = d.momentum.map((p) => p / d.mass);
    const group = bindGroup(this.device, this.pCensus!, {
      0: this.params, 1: this.state[this.cur][0].createView(), 2: this.state[this.cur][1].createView(), 3: this.mg.phi.createView(), 20: this.bh, 21: b.census,
    });
    const census = await b.takeCensus((enc) => {
      const pass = beginPass(enc);
      pass.setPipeline(this.pCensus!);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(this.n / 8), Math.ceil(this.n / 8), Math.ceil(this.nz / 4));
      pass.end();
    }, this.selfGravity, v0, d.ranges[0][1]);
    const r = await this.readRestart();
    return b.report(r.time, r.masses?.[b.spec.mass]?.gm ?? 0, await b.readLedger(), census);
  }

  /** Replace the state with raw conserved arrays (rho, m), (E, e_int, rho s, 0). */
  setState(u1: Float32Array, u2: Float32Array, rhoMean: number) {
    const d = this.device;
    const n = this.n;
    this.cur = 0;
    this.rhoMean = rhoMean;
    writeTexture(d, this.state[0][0], n, 4, u1, this.nz);
    writeTexture(d, this.state[0][1], n, 4, u2, this.nz);
    if (this.cfg.ndim === 3) writeTexture(d, this.mg.phi, n, 1, new Float32Array(n * n * n));
    d.queue.writeBuffer(this.clock, 0, new Float32Array(4));
    this.writeParams();
    this.refreshPotential();
  }

  /** Solve for the potential of the current state from the current guess. */
  solvePotential(cycles: number) {
    if (!this.selfGravity) return;
    const enc = this.device.createCommandEncoder();
    const pass = beginPass(enc);
    this.mg.encodeSource(pass, this.cur, this.cfg.gravity === 'isolated');
    for (let i = 0; i < cycles; i++) this.mg.encodeVCycle(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Call after changing the gravity mode or G at run time. */
  refreshPotential() {
    if (!this.fieldOn) return;
    this.solvePotential(16);
    const enc = this.device.createCommandEncoder();
    if (this.selfGravity) this.copyPhiToOld(enc);
    const pass = beginPass(enc);
    // No step is being taken: the masses' accelerations are evaluated, nothing more.
    if (this.hasMasses) this.encodeMassKick(pass, this.cur, false);
    this.encodeField(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private get selfGravity(): boolean {
    return isSelfGravity(this.cfg.gravity) && this.cfg.ndim === 3;
  }

  private get hasMasses(): boolean {
    return this.nMasses > 0;
  }

  /** The gravity field texture is in use: by self-gravity, point masses or both. */
  private get fieldOn(): boolean {
    return this.selfGravity || this.hasMasses;
  }

  /**
   * Reduce the masses' pull on the gas of the given parity and what their sinks take from it,
   * then finish their step (`live`) or only evaluate their accelerations.
   */
  private encodeMassKick(pass: GPUComputePassEncoder, parity: number, live: boolean) {
    this.redBh.encode(pass, parity);
    pass.setPipeline(live ? this.pBhKick : this.pBhEval);
    pass.setBindGroup(0, live ? this.gBhKick : this.gBhEval);
    pass.dispatchWorkgroups(1);
  }

  private dispatch(pass: GPUComputePassEncoder) {
    pass.dispatchWorkgroups(Math.ceil(this.n / 8), Math.ceil(this.n / 8), Math.ceil(this.nz / 4));
  }

  /** Acceleration and half-step potential from the current and previous potentials. */
  private encodeField(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pField);
    pass.setBindGroup(0, this.gField);
    this.dispatch(pass);
  }

  private copyPhiToOld(enc: GPUCommandEncoder) {
    const n = this.n;
    enc.copyTextureToTexture({ texture: this.mg.phi }, { texture: this.mg.phiOld }, [n, n, n]);
  }

  /** Encode `count` full steps: primitives, timestep, predictor, update, then gravity if enabled. */
  encodeSteps(enc: GPUCommandEncoder, count: number): number {
    if (this.settling) return 0; // a star is still being settled
    const selfGravity = this.selfGravity;
    const anyGravity = this.fieldOn || uniformField(this.cfg).some((g) => g !== 0);
    for (let i = 0; i < count; i++) {
      const p = this.cur;
      let pass = beginPass(enc);
      pass.setPipeline(this.pPrim);
      pass.setBindGroup(0, this.gPrim[p]);
      this.dispatch(pass);
      this.redDt.encode(pass);
      pass.setPipeline(this.pClock);
      pass.setBindGroup(0, this.gClock);
      pass.dispatchWorkgroups(1);
      this.pPredict.encode(pass, this.gPredict[p]);
      this.pUpdate.encode(pass, this.gUpdate[p]);
      if (this.pSpecies) {
        const q = this.speciesCur;
        for (const [pipe, g] of [[this.pSpeciesPredict!, this.gSpeciesPredict], [this.pSpecies, this.gSpecies]] as const) {
          pass.setPipeline(pipe);
          pass.setBindGroup(0, g[p][q]);
          pass.dispatchWorkgroups(Math.ceil(this.n / 8), Math.ceil(this.n / 8), Math.ceil(this.nz / 4));
        }
        this.speciesCur = 1 - q;
      }
      if (selfGravity) {
        // Keep the old potential and solve for that of the updated density (warm-started).
        pass.end();
        this.copyPhiToOld(enc);
        pass = beginPass(enc);
        this.mg.encodeSource(pass, 1 - p, this.cfg.gravity === 'isolated');
        for (let v = 0; v < this.cfg.vcycles; v++) this.mg.encodeVCycle(pass);
      }
      if (this.hasMasses) {
        // Move the masses to the new time, then their pull on the updated gas and what their
        // sinks take from it (removed in the correction pass below).
        pass.setPipeline(this.pBhDrift);
        pass.setBindGroup(0, this.gBhDrift);
        pass.dispatchWorkgroups(1);
        this.encodeMassKick(pass, 1 - p, true);
      }
      if (this.fieldOn) this.encodeField(pass);
      if (anyGravity) {
        // Second half kick, with the acceleration of the updated state.
        pass.setPipeline(this.pCorrect);
        pass.setBindGroup(0, this.gCorrect[p]);
        this.dispatch(pass);
      } else {
        this.cur = 1 - p;
      }
      if (this.burn) {
        // Burn the step's end state into the other parities (burn/burn.ts).
        this.burn.encode(pass, this.cur, this.speciesCur);
        this.cur = 1 - this.cur;
        this.speciesCur = 1 - this.speciesCur;
      }
      pass.end();
    }
    return count;
  }

  submitted() {}

  stepsResume(): Promise<void> | null {
    return null;
  }

  encodeBrush(enc: GPUCommandEncoder, op: BrushOp) {
    if (op.kind === 'obstacle' && !this.solids) {
      this.solids = true;
      this.writeParams();
    }
    this.device.queue.writeBuffer(this.brush, 0, packBrush(op));
    const pass = beginPass(enc);
    pass.setPipeline(this.pBrush);
    pass.setBindGroup(0, this.gBrush[this.cur]);
    this.dispatch(pass);
    pass.end();
    this.cur = 1 - this.cur;
  }

  async placeStar(op: BrushOp) {
    const enc = this.device.createCommandEncoder();
    this.encodeBrush(enc, op);
    this.device.queue.submit([enc.finish()]);
  }

  setSources(list: SourceSpec[]) {
    this.nSources = Math.min(list.length, MAX_SOURCES);
    this.device.queue.writeBuffer(this.sources, 0, packSources(list));
    this.writeParams();
  }

  private massReader?: MassReader;
  readMasses() {
    this.massReader ??= new MassReader(this.device, this.bh);
    return this.massReader.read(this.nMasses);
  }

  addMass(m: PointMass): boolean {
    if (this.nMasses >= MAX_MASSES) return false;
    // Only its own slot is written: the others are mid-orbit on the GPU.
    this.device.queue.writeBuffer(this.bh, 16 + 4 * MASS_STRIDE * this.nMasses, packMasses([completeMass(m, this.dx)]), 4, MASS_STRIDE);
    this.nMasses++;
    this.writeParams();
    this.refreshPotential();
    return true;
  }

  removeAddedMasses() {
    if (this.nMasses === this.nOwnMasses) return;
    this.nMasses = this.nOwnMasses;
    this.writeParams();
    // With no masses left the field texture goes out of use; otherwise rebuild it without them.
    this.refreshPotential();
  }

  async probe(f: [number, number, number]): Promise<CellProbe | null> {
    const cell = (x: number, n: number) => Math.min(n - 1, Math.max(0, Math.floor(x * n)));
    const index = [cell(f[0], this.n), cell(f[1], this.n), cell(f[2], this.nz)];
    const [u1, u2] = await readTexels(this.device, this.state[this.cur], index as [number, number, number]);
    const half = this.cfg.problem.box / 2;
    const pos = index.map((i) => (i + 0.5) * this.dx - half) as [number, number, number];
    if (this.nz === 1) pos[2] = 0;
    return { pos, index: this.nz === 1 ? index.slice(0, 2) : index, dx: this.dx, u1: [...u1], u2: [...u2] };
  }

  retuneMasses() {
    writeMassRadii(this.device, this.bh, this.cfg, this.dx, this.nOwnMasses);
    this.refreshPotential();
  }

  massChanged(mass: number) {
    // The periodic solver subtracts the mean density, so it has to follow the edit.
    this.rhoMean = mass / this.cfg.problem.box ** this.cfg.ndim;
    this.writeParams();
    this.refreshPotential();
  }

  get displayDims(): [number, number, number] {
    return [this.n, this.n, this.nz];
  }
  static readonly fieldShader = 'render/field.wgsl';
  readonly fieldShader = Sim.fieldShader;
  fieldResources(parity: number, speciesParity: number): Record<number, GPUBuffer | GPUTextureView> {
    const r: Record<number, GPUBuffer | GPUTextureView> = { 2: this.state[parity][0].createView(), 3: this.state[parity][1].createView(), 7: this.gfield.createView(), 10: this.pred[1].createView(), 20: this.bh };
    if (this.speciesGroups) r[9] = this.species[speciesParity].createView();
    return { ...r, ...this.compIn };
  }
  get speciesStride(): number {
    return this.speciesGroups ? this.nz : 0;
  }
  status(): string {
    return '';
  }

  /**
   * Time each pass in isolation: milliseconds per execution, measured with a readback so the
   * number is real GPU time in every browser. The state is scratch afterwards; call reset().
   */
  async profile(reps = 20): Promise<Record<string, number>> {
    const d = this.device;
    const n = this.n;
    const p = this.cur;
    const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const time = async (count: number, fn: (pass: GPUComputePassEncoder) => void) => {
      const run = async (k: number) => {
        const enc = d.createCommandEncoder();
        const pass = beginPass(enc);
        for (let i = 0; i < k; i++) fn(pass);
        pass.end();
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
        d.queue.submit([enc.finish()]);
        const t0 = performance.now();
        await buf.mapAsync(GPUMapMode.READ);
        buf.unmap();
        return performance.now() - t0;
      };
      // Difference of two batch sizes cancels the fixed readback latency.
      const a = await run(count);
      const b = await run(3 * count);
      return (b - a) / (2 * count);
    };
    const out: Record<string, number> = {};
    out.rates = await time(reps, (pass) => {
      pass.setPipeline(this.pPrim);
      pass.setBindGroup(0, this.gPrim[p]);
      this.dispatch(pass);
    });
    out.timestep = await time(reps, (pass) => this.redDt.encode(pass));
    out.predict = await time(reps, (pass) => {
      this.pPredict.encode(pass, this.gPredict[p]);
    });
    out.update = await time(reps, (pass) => {
      this.pUpdate.encode(pass, this.gUpdate[p]);
    });
    out.diagnostics = await time(reps, (pass) => this.redDiag.encode(pass, p));
    if (this.selfGravity) {
      out.gravity_source = await time(reps, (pass) => this.mg.encodeSource(pass, p, this.cfg.gravity === 'isolated'));
      out.vcycle = await time(reps, (pass) => this.mg.encodeVCycle(pass));
      out.gravity_field = await time(reps, (pass) => this.encodeField(pass));
      out.gravity_kick = await time(reps, (pass) => {
        pass.setPipeline(this.pCorrect);
        pass.setBindGroup(0, this.gCorrect[1 - p]);
        this.dispatch(pass);
      });
    }
    buf.destroy();
    for (const k of Object.keys(out)) out[k] = +out[k].toFixed(3);
    return out;
  }

  step(count = 1) {
    const enc = this.device.createCommandEncoder();
    this.encodeSteps(enc, count);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Reduce and read back the diagnostics. Returns null if a readback is already in flight,
   * so the frame loop can call this freely without ever waiting on the GPU.
   */
  async readDiagnostics(): Promise<Diagnostics | null> {
    if (this.stagingBusy) return null;
    this.stagingBusy = true;
    const enc = this.device.createCommandEncoder();
    const pass = beginPass(enc);
    this.redDiag.encode(pass, this.cur);
    pass.end();
    enc.copyBufferToBuffer(this.clock, 0, this.staging, 0, 16);
    enc.copyBufferToBuffer(this.redDiag.result, 0, this.staging, 16, DIAG_OPS.length * 4);
    enc.copyBufferToBuffer(this.bh, 0, this.staging, 16 + DIAG_OPS.length * 4, MASS_FLOATS * 4);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      const diag = diagnosticsFrom(a, this.nMasses);
      if (diag.steps === 0 && diag.dt === 0) diag.dt = await this.previewDt();
      return diag;
    } finally {
      this.stagingBusy = false;
    }
  }

  /**
   * The timestep the next step will take, before any has been: the clock holds a step's dt only
   * once the step starts, so a run loaded paused would show 0. The step's own timestep passes run
   * on a copy of the clock; the primitives they rewrite are rebuilt by every step anyway.
   * Called with the staging buffer held.
   */
  /**
   * Set by destroy(). A readback that was waiting when the run was replaced must not then
   * encode work on its destroyed textures (previewDt, after readDiagnostics' map).
   */
  private destroyed = false;

  private async previewDt(): Promise<number> {
    if (this.destroyed) return 0;
    if (this.settling) return 0;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.clock, 0, this.clockPreview, 0, 16);
    const pass = beginPass(enc);
    pass.setPipeline(this.pPrim);
    pass.setBindGroup(0, this.gPrim[this.cur]);
    this.dispatch(pass);
    this.redDt.encode(pass);
    pass.setPipeline(this.pClock);
    pass.setBindGroup(0, this.gClockPreview);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.clockPreview, 0, this.staging, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, 4);
    const dt = new Float32Array(this.staging.getMappedRange(0, 4))[0];
    this.staging.unmap();
    return dt;
  }

  /** Diagnostics, waiting for any in-flight readback first (tests). */
  async diagnostics(): Promise<Diagnostics> {
    for (;;) {
      const d = await this.readDiagnostics();
      if (d) return d;
      await new Promise((r) => setTimeout(r, 1));
    }
  }

  /**
   * Step until the clock reaches cfg.tEnd. Returns the number of steps taken.
   *
   * Steps are submitted in batches with one clock readback per batch. The batch grows while a
   * round trip is short (Firefox reports completion only every ~100 ms, so small batches would
   * be latency-bound) but never beyond the steps estimated to remain, since steps past tEnd
   * are no-ops that still cost a full pass over the grid.
   */
  async runToEnd(batch = 16, maxSteps = 1e6): Promise<number> {
    if (!Number.isFinite(this.cfg.tEnd)) throw new Error('runToEnd needs a finite tEnd');
    await this.settled;
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let steps = 0;
    let size = batch;
    try {
      for (;;) {
        const enc = this.device.createCommandEncoder();
        this.encodeSteps(enc, size);
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
        this.device.queue.submit([enc.finish()]);
        const t0 = performance.now();
        await buf.mapAsync(GPUMapMode.READ);
        const ms = performance.now() - t0;
        const c = new Float32Array(buf.getMappedRange().slice(0));
        buf.unmap();
        const [dt, t] = c;
        steps = c[3];
        if (!(dt === dt) || !(t === t)) throw new Error(`clock is NaN after ${steps} steps`);
        if (dt === 0 && t > 0) return steps;
        if (steps > maxSteps) throw new Error(`no convergence to tEnd after ${steps} steps (t = ${t})`);
        if (ms < 150) size = Math.min(size * 2, 512);
        else if (ms > 400) size = Math.max(batch, size >> 1);
        if (dt > 0) size = Math.max(1, Math.min(size, Math.ceil(((this.cfg.tEnd - t - dt) / dt) * 1.1) + 1));
      }
    } finally {
      buf.destroy();
    }
  }

  /**
   * Flux determinism check: evaluates one step's face mass fluxes and counts the
   * faces where the two cells sharing the face disagree in any bit. The state is not advanced.
   */
  async fluxMismatch(): Promise<{ faces: number; mismatched: number; maxDiff: number }> {
    const d = this.device;
    const n = this.n;
    const c = this.cfg;
    const slots = hydroSlots(c);
    // Split by region like the real pass, so faces between the two variants are checked too.
    const pipe = await RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update (flux debug)', [n, n, this.nz], { DEBUG_FLUX: 1 }, true);
    const p = this.cur;
    const o = 1 - p;
    const v = (t: GPUTexture) => t.createView();
    const group = pipe.groups({
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]),
      6: v(this.gfield), 20: this.bh,
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 11: this.sources,
      ...((this.massFlux ? { 13: this.massFlux } : {}) as Record<number, GPUBuffer>),
      ...(this.debrisBuf?.bindings(this.bh) ?? {}),
      ...this.compIn,
    });
    const enc = d.createCommandEncoder();
    const pass = beginPass(enc);
    pass.setPipeline(this.pPrim);
    pass.setBindGroup(0, this.gPrim[p]);
    this.dispatch(pass);
    this.pPredict.encode(pass, this.gPredict[p]);
    pipe.encode(pass, group);
    pass.end();
    d.queue.submit([enc.finish()]);
    const f1 = await readTexture(d, this.state[o][0], n, 4, this.nz);
    const f2 = await readTexture(d, this.state[o][1], n, 4, this.nz);
    const flux = (i: number, j: number, k: number, f: number) => {
      const cell = ((k * n + j) * n + i) * 4;
      return f < 4 ? f1[cell + f] : f2[cell + f - 4];
    };
    let faces = 0;
    let mismatched = 0;
    let maxDiff = 0;
    const dims = [n, n, this.nz];
    for (let k = 0; k < this.nz; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++)
          for (let a = 0; a < 3; a++) {
            const q = [i, j, k];
            q[a]++;
            if (q[a] >= dims[a]) {
              if (c.bcLo[a] !== 0) continue; // only periodic faces have a partner cell
              q[a] = 0;
            }
            const mine = flux(i, j, k, 2 * a + 1); // my upper face
            const theirs = flux(q[0], q[1], q[2], 2 * a); // neighbor's lower face
            faces++;
            if (mine !== theirs) {
              mismatched++;
              maxDiff = Math.max(maxDiff, Math.abs(mine - theirs));
            }
          }
    return { faces, mismatched, maxDiff };
  }

  /** Current conserved state as float arrays (validation only). */
  async readState(): Promise<{ u1: Float32Array; u2: Float32Array }> {
    const [a, b] = this.state[this.cur];
    return {
      u1: await readTexture(this.device, a, this.n, 4, this.nz),
      u2: await readTexture(this.device, b, this.n, 4, this.nz),
    };
  }

  /**
   * Mass fractions of every cell (validation): X[i][cell], the cells in the order of
   * readState(). Empty without species.
   */
  async readSpecies(): Promise<Float32Array[]> {
    const G = this.speciesGroups;
    if (!G) return [];
    const raw = await readTexture(this.device, this.species[this.speciesCur], this.n, 4, this.nz * G);
    const { u1 } = await this.readState();
    const cells = this.n * this.n * this.nz;
    const ns = this.cfg.problem.species!.length;
    const X = Array.from({ length: ns }, () => new Float32Array(cells));
    for (let i = 0; i < ns; i++) {
      const g = Math.floor(i / SPECIES_PER_GROUP);
      const k = i % SPECIES_PER_GROUP;
      for (let q = 0; q < cells; q++) X[i][q] = raw[(g * cells + q) * 4 + k] / u1[q * 4];
    }
    return X;
  }

  async readPotential(): Promise<Float32Array> {
    return readTexture(this.device, this.mg.phi, this.n, 1);
  }

  /** Max |residual| of the Poisson solve on the finest level. */
  async poissonResidual(): Promise<number> {
    const enc = this.device.createCommandEncoder();
    const pass = beginPass(enc);
    this.mg.encodeResidual(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const r = await readTexture(this.device, this.mg.levels[0].res, this.n, 1);
    let m = 0;
    for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i]));
    return m;
  }

  destroy() {
    this.destroyed = true;
    this.settleEpoch++; // abandon a settling in progress: its textures are going
    for (const s of this.state) for (const t of s) t.destroy();
    this.burn?.destroy();
    for (const t of [...this.pred, this.gfield, ...this.species]) t.destroy();
    this.comp?.destroy();
    this.speciesHalf?.destroy();
    this.massFlux?.destroy();
    this.debrisBuf?.destroy();
    this.mg?.destroy();
  }
}
