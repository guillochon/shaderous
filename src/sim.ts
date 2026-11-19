// The simulation: owns the state textures and encodes hydro + gravity steps.
import { FACE_KEYS, noWallVelocity, OUTFLOW, OUTFLOW_ONLY, PERIODIC, REFLECT, type WallVelocity } from './problems/boundaries';
import { defaultDrive, type DriveParams, writeDrive } from './hydro/drive';
import { prepareHelmholtz } from './eos/helmholtz';
import { assemble, shaderFiles } from './gpu/shaders';
import { RegionPass, type RegionGroups } from './hydro/region_pass';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture, type Bindable, type Gpu } from './gpu/device';
import { COMPILE_STAGE, INIT_STAGE, compileSetup, endLoadStage, setLoadStage, settleStage, showLoadStage } from './gpu/progress';
import { noteAllocation } from './gpu/memory';
import { MAX, MIN, Reducer, SUM } from './gpu/reduce';
import { beginPass } from './gpu/timing';
import { Multigrid } from './gravity/multigrid';
import { LumpReader, lumpGrid } from './gravity/lumps';
import { MassSums } from './gravity/mass_sums';
import type { Lump } from './gravity/preview';
import { comovingEosSlot, defaultEosParams, eosHasEnergy, EOS_MODULES, eosConstants1, eosConstants2, makeEos, physicalEosParams, type Eos, type EosKind, type EosParams } from './eos/eos';
import { PF_CLOSED, PF_OPEN, PF_PERIODIC, PressureSolver } from './pressure/multigrid';
import { cellMark, isSelfGravity, problemUnits, sizedProblem, type Bc3, type GravityMode, type PointMass, type Problem } from './problems/types';
import { createBurnPass, type AnyBurnSpec, type Burner } from './burn/burn';
import { COOLING_MODULES, defaultCoolingParams, packCooling, type CoolingKind, type CoolingParams } from './cooling/cooling';
import type { Restart } from './io/snapshot';
import { heatCells, placedSettling, Settler, settlingOf, SETTLE_ROUNDS, SETTLE_TOL, type PlacedStar, type SettleCells, type Settling } from './problems/settle';
import { laneEmdenCached, polytropeK } from './problems/lane_emden';
import { packSpecies, speciesConst, speciesGroups, speciesSlot, SPECIES_PER_GROUP } from './species/species';
import { DebrisBuffers, debrisSlot, debrisSpec, type DebrisLedger, type DebrisReport } from './debris/debris';
import { hydrogenParticles, mixMoments, problemCoefficients, problemCompSlot, type Moments } from './species/mixture';
import { cellField, initialFaces, MhdPasses, mhdOn, mhdUnsupported, type MagDiagnostics } from './mhd/mhd';
import { compiledFrom } from './setup/compile';
import { domainLo, gridDims, rectangular } from './domain';
export { gridDims, rectangular } from './domain';
import { spherical, sphGrid, sphParams } from './spherical';
import { hubble, omegaK, scaleFactor, type Cosmology } from './cosmo';
import { codeLightSpeed, defaultOpacityParams, OPACITY_MODULES, packOpacity, physicalOpacity, type OpacityKind, type OpacityParams } from './opacity/opacity';
import { packRadiation, radiationOn, radiationUnsupported, syncRadiationEos, type RadiationKind, type RadSolver } from './radiation/radiation';
import { COMPOSITIONS } from './units/composition';
import { RadPasses, type RadDiagnostics } from './radiation/passes';
import { diffusionFlux, splitPressure } from './radiation/init';

import type { LimiterKind, RiemannKind } from './problems/types';
export { LIMITER_KINDS, RIEMANN_KINDS, type LimiterKind, type RiemannKind } from './problems/types';

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
   * Magnetic diffusivity eta, with a field: the edge electric fields gain eta J, J = curl B, the
   * energy flux eta J x B, and the internal energy the Ohmic heat eta J^2 (mhd/ohm.wgsl). 0 is off.
   */
  resistivity: number;
  /** Velocity of each face that is a moving wall (hydro/boundary.wgsl), along the wall. */
  wallVelocity: WallVelocity;
  /**
   * Turbulence driving (hydro/drive.ts): a random acceleration in a band of large scales,
   * correlated over tau. An rms acceleration of 0 is off.
   */
  driving: DriveParams;
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
   * A soft stop: a movie's next frame time with a time cadence (ui/api.ts record()). The clock
   * holds at the step boundary nearest to it, and never shortens a step to land on it
   * (hydro/clock.wgsl). Not a setting of the run.
   */
  tFrame?: number;
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
   * energy is capped at `u` and the speed at `v`. A zero density switches them off. With a
   * magnetic field, `va` caps the Alfven speed everywhere by adding mass (0 or absent: none).
   */
  fluff: { rho: number; u: number; v: number; va?: number };
  /** With self-gravity, refine where the Jeans length is under this many cells (0 = off). */
  jeansRefine: number;
  /**
   * Mirror symmetry (Problem.symmetry): per axis, whether only the half above the plane through
   * the origin is computed. That axis's domain is then [0, box] (src/domain.ts), its lower face
   * is a plane of symmetry (reflecting for the gas, and marked in Params.sym so that gravity and
   * the well-balanced reconstruction treat it as one), and the totals are reported for the whole
   * configuration. configFor keeps only the axes the problem allows (none along z in 2D).
   */
  symmetry: [boolean, boolean, boolean];
  /**
   * How the pressure acts (plans/incompressible.md, stage 2). 'explicit': in the fluxes, the
   * timestep following the sound speed. 'implicit': advection is explicit and the pressure is
   * solved for each step (src/pressure), so the timestep follows the flow; for the liquid
   * equation of state on a uniform grid (implicitUnsupported).
   */
  pressure: 'explicit' | 'implicit';
  /**
   * Implicit pressure: the multigrid V-cycles a step takes (warm-started), and the residual,
   * relative to the largest right-hand side, above which a step's solve is counted as
   * unconverged (pressure/tally.wgsl).
   */
  pressureCycles: number;
  pressureTol: number;
  /**
   * Implicit pressure: the most acoustic steps one step may span. The sound speed enters the
   * timestep divided by this, which is what bounds the step of a liquid at rest.
   */
  acousticCfl: number;
  /**
   * Gray radiation transport with the M1 closure (radiation/radiation.ts): its settings. With it
   * on, the equation of state's own radiation term is off (eosParams.arad is 0; radA keeps it).
   */
  radiation: RadiationKind;
  opacity: OpacityKind;
  opacityParams: OpacityParams;
  radC: number;
  radChat: number;
  radA: number;
  radLocalC: boolean;
  radSolver: RadSolver;
  radSweeps: number;
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
/** Implicit pressure's solves since a run was built: how many, the last one's relative residual, and how many ended above the tolerance. */
export interface SolveStats {
  solves: number;
  residual: number;
  unconverged: number;
}

/** The status line's note on the implicit pressure solves: the last residual, and the solves that ended above the tolerance, if any. */
export function pressureNote(s: SolveStats): string {
  if (s.solves === 0) return '';
  const late = s.unconverged > 0 ? `, ${s.unconverged} of ${s.solves} above tolerance` : '';
  return `pressure solve ${s.residual.toExponential(1)}${late}`;
}

export interface Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  /** With implicit pressure: its solver, for the tally of its solves (pressureNote). */
  prs?: { stats(): Promise<SolveStats> };
  /** finest cells per side, and along z (1 in 2D) */
  readonly n: number;
  readonly nz: number;
  /** parity of the state textures holding the current state */
  cur: number;
  readonly paramBuffer: GPUBuffer;
  /** The clock on the GPU: (dt, t, Kahan term, steps), as float32. */
  readonly clockBuffer: GPUBuffer;
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
  /** Load the initial conditions (the adaptive mesh's may wait on workers: amr/init_pool.ts). */
  reset(): void | Promise<void>;
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
  /**
   * Settle a star just placed into the grid's own hydrostatic balance (problems/settle.ts), as
   * a preset's star is. Resolves with '' once it has, or with why it could not be.
   */
  settleStar(star: PlacedStar): Promise<string>;
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
  /** How many point masses the run holds: one added with addMass() is the last of them. */
  readonly massCount: number;
  /**
   * Remove these point masses, by their place among the run's (only ones added with addMass()):
   * those after them move up to take their places.
   */
  removeMasses(indices: number[]): void;
  /**
   * The gas as it is now, as lumps: each a region's mass at its center of mass, for previews of
   * a point mass's orbit (gravity/preview.ts). Null while the last such read is in flight.
   */
  readGasLumps(): Promise<Lump[] | null>;
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
  /**
   * The display pass's resources with the state at `parity`, the species at `speciesParity` and
   * the radiation (binding 16, uniform grid) at `radParity`.
   */
  fieldResources(parity: number, speciesParity: number, radParity?: number): Record<number, GPUBuffer | GPUTextureView>;
  /** Radiation (radiation/passes.ts): the parity of its texture holding the current radiation. */
  readonly radCur?: number;
  /**
   * With a field: reduce its magnetic diagnostics for the state at `parity`, which the display's
   * magnetic fields read (fieldResources' binding 14).
   */
  encodeMagnetic?(pass: GPUComputePassEncoder, parity: number): void;
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
  /** Called after the uniform block is written (writeParams): the session log watches the clock's stop time in it. */
  onParams?: () => void;
  /** An adaptive mesh: called as a re-mesh its refinement check decided is applied, with the steps the tree took (AmrSim.onRegrid). */
  onRegrid?: (steps: number[][]) => void;
  /** An adaptive mesh: whether it makes refinement checks of its own (AmrSim.autoRegrid). */
  autoRegrid?: boolean;
  /** An adaptive mesh: re-mesh by these steps, as a replay does (AmrSim.replayRegrid). */
  replayRegrid?(steps: number[][]): void;
  /** An adaptive mesh: re-mesh from these indicators, as a replay of a first-version log does (AmrSim.forceRegrid). */
  forceRegrid?(indicators: Float32Array): void;
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
  /** mask of the mirrored axes halved on a rectangular grid (src/domain.ts gridDims) */
  halved?: number;
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
  const cfg: Omit<SimConfig, 'symmetry'> & { symmetry?: [boolean, boolean, boolean] } = {
    n,
    ndim,
    gext: problem.gext?.(ndim) ?? [0, 0, 0],
    params,
    problem: sizedProblem(problem, params),
    eos: problem.eos,
    eosParams: {
      ...defaultEosParams(),
      ...physicalEosParams(problemUnits(problem, params), problem.composition),
      ...problem.eosParams,
      ...(problem.cosmology ? { comovingA3: problem.cosmology.aStart ** 3 } : {}),
    },
    riemann: 'hllc',
    limiter: 'koren',
    // (An axisymmetric run keeps it: it solves for its potential in (R, z).)
    gravity: ndim === 2 && isSelfGravity(problem.gravity) && problem.geometry !== 'cylindrical' && problem.geometry !== 'spherical' ? 'off' : problem.gravity,
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
    // (A spherical run's energy takes gravity's work in kick form: with the flux form and the
    // well-balanced reconstruction a settled polytrope's fundamental mode grew, 8% in 16 time
    // units; docs/numerics.md section 22.)
    gravEnergyFlux: problem.geometry !== 'spherical',
    wellBalanced: true,
    pressure: 'explicit',
    pressureCycles: 3,
    pressureTol: 1e-4,
    acousticCfl: 1000,
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
    resistivity: problem.resistivity ?? 0,
    driving: { ...defaultDrive(), ...problem.driving },
    wallVelocity: { ...noWallVelocity(), ...problem.wallVelocity },
    dampRadius: problem.damping?.radius ?? 0,
    dampRate: problem.damping?.rate ?? 0,
    ...radiationDefaults(problem, params),
    // A problem compiled from a setup document brings the solver's and the mesh's settings too.
    ...(problem.setup ? compiledFrom(problem.setup).overrides : {}),
    ...over,
  };
  const want = cfg.symmetry;
  return syncRadiationEos(withSymmetry({ ...cfg, symmetry: [false, false, false] } as SimConfig, want));
}

/**
 * Refresh what a run takes from its problem's units, for units that follow its parameters: the
 * equation of state's physical constants (the problem's own values still win), and with them the
 * speed of light, the radiation constant and the physical opacities. A reduced speed of light
 * that was the speed of light stays it. Changes `c` in place.
 */
export function unitConstants(c: SimConfig): void {
  const units = problemUnits(c.problem, c.params);
  const comp = c.problem.composition;
  Object.assign(c.eosParams, physicalEosParams(units, comp), c.problem.eosParams);
  const oldC = c.radC;
  c.radC = c.problem.radC ?? codeLightSpeed(units);
  c.radChat = c.radChat >= oldC ? c.radC : Math.min(c.radChat, c.radC);
  c.opacityParams = { ...c.opacityParams, ...(units ? physicalOpacity(units, comp ?? COMPOSITIONS.solar) : {}), ...c.problem.opacityParams };
  // The radiation constant is the equation of state's new one (moved to radA with radiation on).
  if (!(c.eosParams.arad > 0)) c.radA = 0;
  Object.assign(c, syncRadiationEos(c));
}

/**
 * A problem's radiation settings (SimConfig.radiation and the rest): its own, over the physical
 * opacities and speed of light of its units and composition, over the defaults. The radiation
 * constant is the equation of state's (radA is set from eosParams.arad by syncRadiationEos).
 */
function radiationDefaults(problem: Problem, params: Record<string, number>): Pick<SimConfig, 'radiation' | 'opacity' | 'opacityParams' | 'radC' | 'radChat' | 'radA' | 'radLocalC' | 'radSolver' | 'radSweeps'> {
  const units = problemUnits(problem, params);
  const radC = problem.radC ?? codeLightSpeed(units);
  return {
    radiation: problem.radiation ?? 'off',
    opacity: problem.opacity ?? 'constant',
    opacityParams: { ...defaultOpacityParams(), ...(units ? physicalOpacity(units, problem.composition ?? COMPOSITIONS.solar) : {}), ...problem.opacityParams },
    radC,
    radChat: Math.min(problem.radChat ?? radC, radC),
    radA: 0,
    radLocalC: true,
    radSolver: 'hll',
    radSweeps: 8,
  };
}

/**
 * The configuration with the mirrors `want` asks for, as far as the problem allows them: the
 * mirrored axes' lower faces become reflecting (their planes of symmetry).
 */
export function withSymmetry(cfg: SimConfig, want: readonly boolean[] | undefined): SimConfig {
  // (A magnetic field must be declared symmetric too, or no mirror is offered: the whole box
  // runs, as for a problem with no mirrors, rather than a link's mirror stopping the page.)
  const fieldOk = !cfg.problem.mhd || !!cfg.problem.symmetry?.mhd;
  const allowed = fieldOk ? (cfg.problem.symmetry?.axes ?? [false, false, false]) : [false, false, false];
  const sym = [0, 1, 2].map((a) => !!want?.[a] && allowed[a] && (a < 2 || cfg.ndim === 3)) as [boolean, boolean, boolean];
  // (An axisymmetric run's lower x face is its axis, which reflects as such a plane does.)
  const bcLo = cfg.bcLo.map((b, a) => (sym[a] || (a === 0 && cylindrical(cfg)) || (a === 1 && spherical(cfg)) ? REFLECT : b)) as Bc3;
  // (A spherical run's theta faces are both its axis: they reflect, v_phi odd across them.)
  const bcHi = cfg.bcHi.map((b, a) => (a === 1 && spherical(cfg) ? REFLECT : b)) as Bc3;
  return { ...cfg, symmetry: sym, bcLo, bcHi };
}

/** The run is axisymmetric, in (R, z) (Problem.geometry). */
export const cylindrical = (c: SimConfig): boolean => c.problem.geometry === 'cylindrical' && c.ndim === 2;

/**
 * Why an axisymmetric run cannot go as configured, or null. The first version: hydrodynamics on
 * a uniform 2D grid, with a uniform field along z (docs/numerics.md section 22).
 */
export function cylindricalUnsupported(c: SimConfig): string | null {
  if (c.problem.geometry !== 'cylindrical') return null;
  if (c.ndim !== 2) return 'an axisymmetric run is two-dimensional, in (R, z)';
  if (c.amrLevels > 0 && c.timeBins > 0) return 'per-block timesteps are not available for axisymmetric runs yet';
  if (c.problem.mhd && c.amrLevels > 0) return 'magnetic fields are not available for axisymmetric runs on blocks yet';
  if (c.problem.mhd && c.resistivity > 0) return 'resistivity is not available in axisymmetric runs yet';
  if (c.viscosity > 0) return 'viscosity is not available in axisymmetric runs yet (it needs the curvilinear stress)';
  for (const m of c.problem.masses?.(c.params, c.G, c.problem.box / c.n, c.ndim) ?? [])
    if ((m.pos[0] ?? 0) !== 0 || (m.pos[2] ?? 0) !== 0 || (m.vel[0] ?? 0) !== 0 || (m.vel[2] ?? 0) !== 0) return 'a point mass must sit on the axis (R = 0) and move along it';
  for (const p of c.problem.setup?.parts ?? []) {
    if (p.kind === 'orbit') return 'an orbit puts its bodies off the axis: an axisymmetric run takes bodies only on it';
    if (p.kind === 'star' && ((p.pos?.[0] ?? 0) !== 0 || (p.pos?.[2] ?? 0) !== 0 || (p.vel?.[0] ?? 0) !== 0 || (p.vel?.[2] ?? 0) !== 0)) return 'a star must sit on the axis (R = 0) and move along it';
  }
  if ((c.problem.species?.length ?? 0) > 0) return 'species are not available in axisymmetric runs yet';
  if (c.problem.rotation || c.problem.shear || c.problem.cosmology) return 'an axisymmetric run has no rotating frame, shear or expansion';
  if ((c.symmetry ?? []).some(Boolean)) return 'mirror symmetry is not available in axisymmetric runs yet';
  if (uniformField(c)[0] !== 0) return 'a uniform field must lie along the axis';
  if (c.driving.accel > 0) return 'turbulence driving is not available in axisymmetric runs';
  if (c.gravity === 'periodic') return 'an axisymmetric run’s self-gravity is isolated (its axis is no periodic face)';
  return null;
}

/**
 * Why a run cannot go as configured, or null: what either simulator refuses before it builds
 * anything (src/ui/api.ts check() and the Setup dialog ask the same, so a setup they pass loads).
 */
export function refusal(c: SimConfig): string | null {
  const why: [string, string | null][] = [
    ['MHD', mhdOn(c) ? mhdUnsupported(c) : null],
    ['Liquid', c.eos === 'liquid' ? liquidUnsupported(c) : null],
    ['Implicit pressure', implicitPressure(c) ? implicitUnsupported(c) : null],
    ['Mirror symmetry', symmetryUnsupported(c)],
    ['Comoving coordinates', cosmologyUnsupported(c)],
    ['Shearing box', shearUnsupported(c)],
    ['Axisymmetric run', cylindricalUnsupported(c)],
    ['Spherical run', sphericalUnsupported(c)],
    ['Radiation', radiationUnsupported(c)],
  ];
  const hit = why.find(([, w]) => w);
  return hit ? `${hit[0]}: ${hit[1]}` : null;
}

/**
 * Why a run of the weakly compressible liquid (eos/liquid.wgsl) cannot go as configured, or
 * null. Its pressure takes both signs, which the hydro passes allow for (p_signed) and the MHD
 * passes, the self-gravity closures and comoving coordinates do not.
 */
export function liquidUnsupported(c: SimConfig): string | null {
  if (c.problem.mhd) return 'a magnetic field is not available with the liquid equation of state';
  if (isSelfGravity(c.gravity)) return 'self-gravity is not available with the liquid equation of state';
  if (c.problem.cosmology) return 'comoving coordinates are not available with the liquid equation of state';
  if (!(c.eosParams.liqRho > 0 && c.eosParams.liqC > 0)) return 'the liquid needs a reference density and a sound speed above zero';
  return null;
}

/** The run solves for its pressure each step (SimConfig.pressure). */
export const implicitPressure = (c: SimConfig): boolean => c.pressure === 'implicit';
/** The clock's build constants for implicit pressure: a gas with an energy combines its two step limits (hydro/clock.wgsl PSPLIT_GAS). */
export const clockPressure = (c: SimConfig): Record<string, number> => (implicitPressure(c) && eosHasEnergy(c.eos) ? { PSPLIT_GAS: 1 } : {});

/**
 * Why a run cannot take implicit pressure as configured, or null. What is built is the liquid's
 * form (plans/incompressible.md 4.2.1) on a uniform Cartesian grid, with any of the box faces,
 * obstacles, inflows, viscosity and driving.
 */
export function implicitUnsupported(c: SimConfig): string | null {
  if (c.eos !== 'liquid' && c.eos !== 'ideal') return 'it is built for the liquid and the ideal gas only, so far';
  if (c.problem.geometry === 'cylindrical' || c.problem.geometry === 'spherical') return 'it is not available in axisymmetric or spherical runs';
  // (A uniform field is the pressure passes' own: pressure/correct.wgsl.)
  if (c.problem.masses) {
    // (Their pull kicks the faces, pressure/kick.wgsl. What a sink takes is removed by the
    // explicit scheme's second gravity kick, which this step does not run.)
    if (c.eos === 'liquid') return 'point masses are not available with it for a liquid yet';
    const sinks = c.problem.masses(c.params, c.G, c.problem.box / c.n, c.ndim).some((m) => (m.sinkRadius ?? m.soft ?? 1) > 0);
    if (sinks) return 'point masses that take gas in are not available with it yet (set their sink radius to 0)';
  }
  if (c.problem.rotation || c.problem.shear || c.problem.frame) return 'rotating frames and shearing boxes are not available with it';
  if ((c.symmetry ?? []).some(Boolean)) return 'mirror symmetry is not available with it yet';
  // (Species that only ride with the gas are moved after the correction, and scaled to its
  // density. Where they set the equation of state, or burn, the pressure passes would have to
  // know each cell's composition.)
  if (problemCoefficients(c.problem, c.eosParams.gamma) !== null) return 'species that set the equation of state are not available with it yet';
  if (c.problem.burn) return 'burning is not available with it';
  // (Its rates and the pressure's gradient take the same binding, and its transport the species' passes.)
  if (radiationOn(c)) return 'radiation transport is not available with it';
  if (c.cooling !== 'off' || c.problem.damping) return 'cooling and damping zones are not available with it';
  return null;
}

/**
 * Why a run on the block grid cannot take implicit pressure as configured, or null: what the
 * uniform grid's takes (implicitUnsupported), less per-block timesteps (one solve couples every cell at one time: plans/incompressible.md 5.5).
 */
export function implicitBlocksUnsupported(c: SimConfig): string | null {
  if (c.timeBins > 0) return 'per-block timesteps are not available with it: one solve couples every cell at one time';
  return null;
}

export { spherical } from './spherical';

/**
 * Why a spherical (r, theta) run cannot go as configured, or null: hydrodynamics in 2D, on the
 * uniform grid or on blocks (per-block timesteps included), with the monopole of its own gravity
 * and point masses at the origin (docs/numerics.md section 22).
 */
export function sphericalUnsupported(c: SimConfig): string | null {
  if (c.problem.geometry !== 'spherical') return null;
  if (c.ndim !== 2) return 'a spherical run is two-dimensional, in (r, theta)';
  const s = c.problem.sphere;
  if (!s || !(s.rMin > 0)) return 'a spherical run needs an inner radius above zero';
  if (c.problem.mhd) return 'magnetic fields are not available in spherical runs yet';
  if (c.viscosity > 0 || c.conductivity > 0) return 'viscosity and conduction are not available in spherical runs yet';
  for (const m of c.problem.masses?.(c.params, c.G, c.problem.box / c.n, c.ndim) ?? [])
    if (m.pos.some((x) => (x ?? 0) !== 0) || m.vel.some((v) => (v ?? 0) !== 0)) return 'a point mass must sit at the origin, at rest';
  if ((c.problem.species?.length ?? 0) > 0) return 'species are not available in spherical runs yet';
  if (c.problem.rotation || c.problem.shear || c.problem.cosmology || c.problem.frame) return 'a spherical run has no rotating or accelerating frame, shear or expansion';
  if ((c.symmetry ?? []).some(Boolean)) return 'mirror symmetry is not available in spherical runs (both poles are in the grid)';
  if (uniformField(c).some((g) => g !== 0)) return 'a uniform field is not available in spherical runs';
  if (c.driving.accel > 0 || c.problem.damping) return 'driving and damping zones are not available in spherical runs';
  if (c.gravity === 'periodic') return 'a spherical run\u2019s self-gravity is its monopole (isolated), not periodic';
  if (c.problem.debris) return 'debris tallies are not available in spherical runs';
  if (c.problem.burn) return 'burning is not available in spherical runs';
  if ((c.problem.setup?.parts ?? []).some((p) => p.kind === 'star')) return 'stars are not settled in spherical runs yet';
  if (c.bcLo[0] === PERIODIC || c.bcHi[0] === PERIODIC) return 'the radial faces of a spherical run cannot be periodic';
  if (c.problem.sphere?.log && c.n & (c.n - 1)) return 'a spherical run needs a power-of-two grid';
  return null;
}

/** The run's background expansion (Problem.cosmology), or null. */
export function cosmologyOf(c: SimConfig): Cosmology | null {
  return c.problem.cosmology ?? null;
}

/**
 * Why a run in comoving coordinates cannot go as configured, or null. The first version is a
 * periodic box of ideal gas, with or without self-gravity, on a uniform grid or on blocks that
 * share one timestep (docs/numerics.md section 22).
 */
export function cosmologyUnsupported(c: SimConfig): string | null {
  if (!cosmologyOf(c)) return null;
  if (c.problem.masses) return 'point masses and sinks are not available in comoving coordinates';
  if (![...c.bcLo, ...c.bcHi].every((b) => b === PERIODIC)) return 'comoving coordinates need a periodic box';
  if (c.gravity !== 'off' && c.gravity !== 'periodic') return 'comoving coordinates take periodic self-gravity or none';
  if (uniformField(c).some((g) => g !== 0)) return 'a uniform field is not available in comoving coordinates';
  if (c.eos === 'helmholtz') return 'the tabulated stellar equation of state is not available in comoving coordinates';
  if (c.viscosity > 0 || c.conductivity > 0) return 'viscosity and conduction are not available in comoving coordinates';
  if (c.problem.mhd) return 'magnetic fields are not available in comoving coordinates yet';
  if (c.amrLevels > 0 && c.timeBins > 0) return 'per-block timesteps are not available in comoving coordinates (each block would expand by its own a)';
  if (c.amrLevels > 0 && isSelfGravity(c.gravity) && c.jeansRefine > 0) return 'refining on the Jeans length is not available in comoving coordinates (set it to 0)';
  if (c.jeansFloor) return 'the Jeans pressure floor is not available in comoving coordinates';
  if (rotating(c) || c.problem.shear) return 'a rotating frame is not available in comoving coordinates';
  if (c.driving.accel > 0 || c.problem.damping) return 'driving and damping zones are not available in comoving coordinates';
  return null;
}

/**
 * Why a shearing box cannot go as configured, or null. The first version: hydrodynamics on a
 * uniform grid, periodic along y (and z), without self-gravity (docs/numerics.md section 22).
 */
export function shearUnsupported(c: SimConfig): string | null {
  const s = c.problem.shear;
  if (!s) return null;
  if (c.amrLevels > 0) return 'the adaptive mesh is not available for a shearing box (a block\u2019s neighbors across the boundary change with time)';
  if (c.problem.mhd && (c.resistivity > 0 || c.viscosity > 0)) return 'resistivity and viscosity are not available in a magnetized shearing box yet';
  if (c.bcLo[0] !== PERIODIC || c.bcHi[0] !== PERIODIC || c.bcLo[1] !== PERIODIC || c.bcHi[1] !== PERIODIC) return 'a shearing box is periodic along x (with the shear) and y';
  if (c.ndim === 3 && !s.stratified && (c.bcLo[2] !== PERIODIC || c.bcHi[2] !== PERIODIC)) return 'an unstratified shearing box is periodic along z';
  if (isSelfGravity(c.gravity) || c.problem.masses) return 'self-gravity and point masses are not available in a shearing box';
  if (c.problem.rotation || cosmologyOf(c)) return 'a shearing box has its own rotation and no expansion';
  if (c.viscosity > 0 || c.conductivity > 0) return 'viscosity and conduction are not available in a shearing box yet';
  if ((c.symmetry ?? []).some(Boolean)) return 'mirror symmetry is not available in a shearing box';
  return null;
}

/** The shearing box's offset of its x faces along y at `time` (Part D), or 0. */
export function shearOffset(c: SimConfig, time: number): number {
  const s = c.problem.shear;
  if (!s) return 0;
  const L = c.problem.box;
  const d = (s.q * s.omega * L * time) % L;
  return d < 0 ? d + L : d;
}

/** How many axes are mirrored: the computed part is 2^-k of the whole configuration. */
export const mirrors = (c: SimConfig): number => (c.symmetry ?? []).filter(Boolean).length;

/**
 * Why a run with mirror symmetry cannot go as configured, or null. What the mirrors would get
 * wrong is refused rather than run: anything that is not itself symmetric under them.
 */
export function symmetryUnsupported(c: SimConfig): string | null {
  const sym = c.symmetry ?? [false, false, false];
  if (!sym.some(Boolean)) return null;
  const axes = [0, 1, 2].filter((a) => sym[a]);
  const name = (a: number) => 'xyz'[a];
  for (const a of axes) {
    if (c.bcHi[a] === PERIODIC) return `the upper ${name(a)} face is periodic, which a plane of symmetry at the lower one cannot be`;
    if (uniformField(c)[a] !== 0) return `the uniform field has a part along ${name(a)}, across a plane of symmetry`;
  }
  if (c.gravity === 'periodic') return 'periodic self-gravity needs a periodic box';
  if (c.problem.mhd && !c.problem.symmetry?.mhd) return 'the magnetic field is not symmetric under the mirrors';
  if (c.driving.accel > 0) return 'turbulence driving is not symmetric under the mirrors';
  const w = rotationOf(c);
  for (const a of axes)
    for (let b = 0; b < 3; b++) if (b !== a && w[b] !== 0) return `a frame turning about an axis other than ${name(a)} is not symmetric under the ${name(a)} mirror`;
  for (const m of c.problem.masses?.(c.params, c.G, c.problem.box / c.n, c.ndim) ?? [])
    for (const a of axes) if ((m.pos[a] ?? 0) !== 0 || (m.vel[a] ?? 0) !== 0) return `a point mass lies or moves off the ${name(a)} plane of symmetry`;
  const stars = (c.problem.setup?.parts ?? []).filter((p) => p.kind === 'star');
  if (stars.length > 1) return 'more than one star: settling handles one, on the planes';
  for (const s of stars) for (const a of axes) if (((s as { pos?: number[] }).pos?.[a] ?? 0) !== 0) return `a star lies off the ${name(a)} plane of symmetry`;
  return null;
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
export const burnSpec = (c: SimConfig): AnyBurnSpec | null => c.problem.burn?.(c.params) ?? null;

/** The modules the hydro shaders are assembled with. */
export function hydroSlots(c: SimConfig): Record<string, string> {
  return {
    // Comoving coordinates: the module behind wrappers that convert to proper densities.
    eos: cosmologyOf(c) ? comovingEosSlot(c.eos, shaderFiles) : EOS_MODULES[c.eos],
    // With implicit pressure the flux is advection's alone (hydro/riemann/advect.wgsl), and the
    // predictor takes the last solve's pressure gradient (hydro/predictor.wgsl).
    riemann: implicitPressure(c) ? 'hydro/riemann/advect.wgsl' : `hydro/riemann/${c.riemann}.wgsl`,
    psplit: `const PSPLIT = ${implicitPressure(c)};\nconst PSPLIT_LAG = ${implicitPressure(c)};\n`,
    psplit_grad: implicitPressure(c) ? 'pressure/predict_on.wgsl' : 'pressure/predict_off.wgsl',
    // A gas with an energy: the update keeps the face velocities that carried its fluxes.
    psplit_store: implicitPressure(c) && eosHasEnergy(c.eos) ? 'pressure/store_on.wgsl' : 'pressure/store_off.wgsl',
    limiter: `hydro/limiters/${c.limiter}.wgsl`,
    cooling: COOLING_MODULES[c.cooling],
    // Which diffusive fluxes are compiled in (hydro/diffusion.wgsl); their strengths are in P.diff.
    diffusion: `const VISCOSITY = ${c.viscosity > 0};\nconst CONDUCTION = ${conducts(c)};\nconst RESISTIVITY = ${resists(c)};\n`,
    // The preset's damping zone (hydro/damping.wgsl), or none.
    damping: c.problem.damping?.wgsl ?? NO_DAMPING,
    // Whether the update records face mass fluxes for the species pass (species/species.ts) and
    // the radiation's advection (radiation/advect.wgsl).
    species: speciesSlot(speciesGroups(c.problem) + (radiationOn(c) ? 1 : 0)),
    // Radiation's push on the timestep (radiation/rate_on.wgsl), and the opacity law.
    rad_rate: radiationOn(c) ? 'radiation/rate_on.wgsl' : 'radiation/rate_off.wgsl',
    opacity: OPACITY_MODULES[c.opacity],
    // One composition everywhere, or the species' own in every cell (species/mixture.ts).
    comp: problemCompSlot(c.problem, c.eosParams.gamma, speciesGroups(c.problem)),
    // Whether the update tallies the gas leaving the box by its orbital energy (debris/debris.ts).
    debris: debrisSlot(c),
    // Forces of a non-inertial frame that depend on the velocity (gravity/frame_off.wgsl): a
    // rotating frame's Coriolis force, or the expansion drag of comoving coordinates.
    frame: cosmologyOf(c) ? 'gravity/frame_cosmo.wgsl' : rotating(c) ? 'gravity/frame_rot.wgsl' : 'gravity/frame_off.wgsl',
    // A spherical run (hydro/common.wgsl spherical()).
    geometry: `const SPHERICAL = ${spherical(c)};\n`,
  };
}

/**
 * A rotating frame (Problem.rotation) as Params.rot holds it: the angular velocity, and 1 for
 * the centrifugal force. In 2D only the part along z can turn the plane into itself.
 */
export function rotationOf(c: SimConfig): [number, number, number, number] {
  // A shearing box turns about z: the Coriolis force alone, its tidal force taking the place of
  // the centrifugal one (gravity/rot.wgsl).
  if (c.problem.shear) return [0, 0, c.problem.shear.omega, 0];
  const r = c.problem.rotation?.(c.params, c.ndim);
  if (!r) return [0, 0, 0, 0];
  const w: [number, number, number] = c.ndim === 2 ? [0, 0, r.omega[2]] : [...r.omega];
  if (w.every((x) => x === 0)) return [0, 0, 0, 0];
  return [...w, r.centrifugal === false ? 0 : 1];
}

/** The run is in a rotating frame. */
export const rotating = (c: SimConfig): boolean => rotationOf(c).some((x) => x !== 0);

const NO_DAMPING = 'const DAMPING = false;\nfn damping_target(x: vec3<f32>) -> DampTarget { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }\n';

/** Thermal conduction is on: it needs a coefficient and an equation of state with an energy. */
export const conducts = (c: SimConfig): boolean => c.conductivity > 0 && eosHasEnergy(c.eos);

/** Which diffusive fluxes the shaders must be built with: a change here means rebuilding them. */
export const diffusionKey = (c: SimConfig): string => `${c.viscosity > 0}/${conducts(c)}/${resists(c)}`;

/** Resistivity is on: it needs a coefficient and a magnetic field. */
export const resists = (c: SimConfig): boolean => c.resistivity > 0 && !!c.problem.mhd;

export const MAX_MASSES = 32;
/** Floats per mass in the Masses block (gravity/bh.wgsl). */
export const MASS_STRIDE = 40;
/** Floats in the Masses block: the frame's acceleration, then MASS_STRIDE per mass. */
export const MASS_FLOATS = 4 + MASS_STRIDE * MAX_MASSES;

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
    massless: q.massless ?? false, fixed: q.fixed ?? false, onOrbit: q.onOrbit ?? false,
  };
}

/**
 * Write the softening and sink radii of the preset's point masses, and their flags (from its
 * current parameters), into the Masses block, leaving their positions, velocities and masses as
 * the run has them.
 */
export function writeMassRadii(d: GPUDevice, bh: GPUBuffer, c: SimConfig, dx: number, count: number) {
  const base = (c.problem.masses?.(c.params, c.G, dx, c.ndim) ?? []).slice(0, count);
  base.forEach((q, i) => {
    const m = completeMass(q, dx);
    const at = 16 + 4 * MASS_STRIDE * i;
    d.queue.writeBuffer(bh, at + 28, new Float32Array([m.soft]));
    d.queue.writeBuffer(bh, at + 44, new Float32Array([m.sinkRadius]));
    // Whether it is held on its orbit can change mid-run too.
    d.queue.writeBuffer(bh, at + 60, new Float32Array([massFlags(m)]));
  });
}

/** A mass's flags as the Masses block holds them (gravity/bh.wgsl). */
const massFlags = (m: Required<PointMass>): number => (m.massless ? 1 : 0) + (m.fixed ? 2 : 0) + (m.onOrbit ? 4 : 0);

/** Pack the Masses block. The layout must match the struct in gravity/bh.wgsl. */
export function packMasses(list: Required<PointMass>[]): Float32Array<ArrayBuffer> {
  const f = new Float32Array(MASS_FLOATS);
  // A value summed with Kahan compensation goes in as its float32 rounding, with the
  // compensation set to what that rounding added, so that nothing is lost in the round trip.
  const c = (v: number) => Math.fround(v) - v;
  list.forEach((m, i) =>
    f.set(
      [
        ...m.pos, m.gm, ...m.vel, m.soft, 0, 0, 0, m.sinkRadius, m.sinkRate, m.accreted, 0, massFlags(m), ...m.accretedMomentum, 0, ...m.accretedSpin, 0,
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
      massless: (f[o + 15] & 1) !== 0, fixed: (f[o + 15] & 2) !== 0, onOrbit: (f[o + 15] & 4) !== 0,
    });
  }
  return out;
}

/** One application of an editing brush (edit/brush_core.wgsl). */
export interface BrushOp {
  /**
   * `eos-out` and `eos-in` act on every cell: see convertEos() in ui/app.ts and edit/brush_core.wgsl.
   * `field` and `pull` change the magnetic field (edit/field_brush_core.wgsl): scale it by
   * exp(amount) at full weight, or carry it by `vel` (fractions of the box).
   */
  kind: 'mass' | 'velocity' | 'temperature' | 'obstacle' | 'eos-out' | 'eos-in' | 'star' | 'field' | 'pull';
  /** center and radius, in fractions of the box */
  pos: [number, number, number];
  radius: number;
  /** 0: hard edge; 1: falls off all the way from the center */
  softness: number;
  /**
   * mass: density added at the center (negative removes); temperature: log of the factor;
   * obstacle: positive makes the cells solid, negative turns them back into gas;
   * star: central density; field: log of the factor on the field; pull: the grip, the share of
   * the drag the field at the brush's center moves by (below 1)
   */
  amount: number;
  /** velocity tool: the velocity to set; star: its velocity; pull: the drag, in fractions of the box */
  vel: [number, number, number];
  /** mass tool: new gas brings no momentum (otherwise it arrives with the local velocity) */
  atRest?: boolean;
  /** obstacle tool: the outline, drawn on the screen and cut through the box along the line of sight */
  outline?: Outline;
  /** star: see starBrush() */
  star?: StarProfile;
}

/** Whether a brush changes the magnetic field (edit/field_brush_core.wgsl) rather than the gas. */
export const isFieldBrush = (b: BrushOp): boolean => b.kind === 'field' || b.kind === 'pull';

/** Most substeps one pull is cut into: a frame's drag the tool allows needs about 20. */
const MAX_PULL_SUBSTEPS = 64;

/**
 * A field brush as the passes apply it, and how many times: a pull is cut into substeps that
 * move the field by at most half a cell of the finest level (`n` across the box), as the upwind
 * induction step it is needs (edit/field_brush_core.wgsl); the strength brush is one step.
 */
export function fieldBrushSteps(op: BrushOp, n: number): { op: BrushOp; steps: number } {
  if (op.kind !== 'pull') return { op, steps: 1 };
  const move = Math.min(Math.max(op.amount, 0), 0.95) * Math.hypot(...op.vel) * n;
  const steps = Math.min(MAX_PULL_SUBSTEPS, Math.max(1, Math.ceil(move / 0.5)));
  return { op: { ...op, vel: op.vel.map((v) => v / steps) as [number, number, number] }, steps };
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
  /**
   * The point mass it rides, by its place among the run's: the source is then where that mass
   * is, and the gas it adds moves as the mass does (`pos` and `vel` are not used).
   */
  carrier?: number;
}

export const MAX_SOURCES = 8;
export const SOURCE_BYTES = 16 + 48 * MAX_SOURCES;

/** Pack the Sources block. The layout must match the struct in hydro/sources.wgsl. */
export function packSources(list: SourceSpec[]): ArrayBuffer {
  const buf = new ArrayBuffer(SOURCE_BYTES);
  const f = new Float32Array(buf);
  const use = list.slice(0, MAX_SOURCES);
  new Uint32Array(buf)[0] = use.length;
  use.forEach((q, i) => f.set([...q.pos, q.radius, ...q.vel, q.softness, q.rate, q.decay, (q.carrier ?? -1) + 1, 0], 4 + 12 * i));
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
  u[8] = ['mass', 'velocity', 'temperature', 'obstacle', 'eos-out', 'eos-in', 'star', 'field', 'pull'].indexOf(b.kind);
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

/** Channels of the diagnostics reduction (DIAG_OPS). */
const DIAG_CHANNELS = 24;
/** Number of floats the diagnostics readback holds: clock (4), reduction, the Masses block. */
/** Floats and bytes of the clock (hydro/common.wgsl Clock). */
export const CLOCK_FLOATS = 10;
export const CLOCK_BYTES = 4 * CLOCK_FLOATS;

/**
 * The clock as a run starts it, at `time` after `steps` steps: (dt, t, Kahan compensation,
 * steps), then the scale factor at t three times and the expansion rate (Part C; 1, 1, 1, 0
 * without comoving coordinates), then the shearing box's offset and its compensation.
 */
export function clockState(c: SimConfig, time = 0, steps = 0): Float32Array<ArrayBuffer> {
  const k = cosmologyOf(c);
  const a = k ? scaleFactor(k, time) : 1;
  const shear = shearOffset(c, time);
  return new Float32Array([0, time, 0, steps, a, a, a, k ? hubble(k, a) : 0, shear, 0]);
}

export const DIAG_FLOATS = CLOCK_FLOATS + DIAG_CHANNELS + MASS_FLOATS;

/** Burning's heating rate texture where a run does not burn: one texel of nothing. */
export function noHeat(d: GPUDevice): GPUTexture {
  return d.createTexture({ label: 'no nuclear heating', size: [1, 1, 1], dimension: '3d', format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
}

/** Where Params.cosmo.w (the state's scale factor, read by cosmo_a3 in hydro/common.wgsl) lies. */
export const PARAM_SCALE_BYTE = (108 + 3) * 4;

/**
 * Comoving coordinates: copy the scale factor at the end of the last step (clock.a1) into
 * Params.cosmo.w, so that passes without the clock (the timestep's rates, the diagnostics, the
 * display) convert the equation of state at the state's own a. Encoded between passes.
 */
export function syncScale(enc: GPUCommandEncoder, c: SimConfig, clock: GPUBuffer, params: GPUBuffer): void {
  if (cosmologyOf(c)) enc.copyBufferToBuffer(clock, 6 * 4, params, PARAM_SCALE_BYTE, 4);
}

/** Comoving coordinates: the state's scale factor from the clock (where the last step ended, a1). */
export function withScale(d: Diagnostics, clock: Float32Array): Diagnostics {
  d.scale = clock[6];
  d.redshift = 1 / clock[6] - 1;
  return d;
}

/** Decode a diagnostics readback: [clock (CLOCK_FLOATS), reduction channels, the Masses block]. */
export function diagnosticsFrom(a: Float32Array, nMasses: number): Diagnostics {
  const r = a.subarray(CLOCK_FLOATS);
  const masses = unpackMasses(a.subarray(CLOCK_FLOATS + DIAG_CHANNELS), nMasses);
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
    nuclearHeating: r[23],
    com: [r[20] / r[0], r[21] / r[0], r[22] / r[0]],
    perturber: masses.length ? (masses[0].pos as [number, number, number]) : null,
    masses,
    ranges,
  };
}

/**
 * Diagnostics of a run with mirror symmetry, for the whole configuration: the grid holds one
 * 2^k-th of it. Extensive totals (mass, energies, burning's heating, the field's energy) are
 * times 2^k; momentum and the center of mass are zero along the mirrored axes and times 2^k
 * (momentum) or unchanged (center) along the others. Cell counts stay the grid's.
 */
export function unfoldDiagnostics(d: Diagnostics, symmetry: readonly boolean[] | undefined, axisymmetric = false): Diagnostics {
  // An axisymmetric run's rings have no net radial or azimuthal momentum, and their center of
  // mass is on the axis: what is left is along z (code y).
  if (axisymmetric) {
    d.momentum = [0, d.momentum[1], 0];
    d.com = [0, d.com[1], 0];
  }
  d.symmetry = [0, 1, 2].map((a) => !!symmetry?.[a]) as [boolean, boolean, boolean];
  const k = d.symmetry.filter(Boolean).length;
  if (!k) return d;
  const f = 2 ** k;
  d.mass *= f;
  d.eKin *= f;
  d.eInt *= f;
  d.eGrav *= f;
  d.eTot *= f;
  d.nuclearHeating *= f;
  d.momentum = d.momentum.map((p, a) => (d.symmetry![a] ? 0 : p * f)) as [number, number, number];
  d.com = d.com.map((x, a) => (d.symmetry![a] ? 0 : x)) as [number, number, number];
  if (d.magnetic) d.magnetic.eMag *= f;
  return d;
}

/** Size of the Params uniform block (hydro/common.wgsl). */
export const PARAM_BYTES = 544;

/** Pack the Params block. The layout must match the struct in hydro/common.wgsl. */
export function packParams(
  c: SimConfig,
  x: { dims: number[]; rect?: boolean; dx: number; rhoMean: number; selfGravity: boolean; masses: number; amr?: number[]; refine?: number[]; solids?: boolean; sources?: boolean; scale?: number; vref?: number; pref?: number },
): ArrayBuffer {
  const buf = new ArrayBuffer(PARAM_BYTES);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u.set([...c.bcLo, 0], 0);
  u.set([...c.bcHi, 0], 4);
  f.set(eosConstants1(c.eos, c.eosParams), 8);
  // The fourth: 1 on a rectangular uniform grid (gridDims; hydro/common.wgsl grid_n).
  u.set([...x.dims, x.rect ? 1 : 0], 12);
  // 2D quantities are per unit length along z.
  f.set([c.ndim === 2 ? x.dx ** 2 : x.dx ** 3, c.ndim, ...(x.refine ?? [0, 0])], 16);
  f.set([...uniformField(c), 0], 20);
  // Point masses: how many, whether the frame follows the gas, whether there are any, and
  // whether they are all the field texture holds (no Poisson potential).
  f.set([x.masses, c.problem.frame === 'gas' ? 1 : c.problem.frame === 'masses' ? 2 : 0, x.masses > 0 ? 1 : 0, x.selfGravity ? 0 : 1], 24);
  const fl = c.fluff;
  f.set(fl.rho > 0 ? [fl.rho, fl.u > 0 ? fl.u : 1e30, fl.v > 0 ? fl.v : 1e30] : [0, 0, 0], 28);
  f[31] = c.problem.mhd && (fl.va ?? 0) > 0 ? fl.va! : 0;
  u.set(x.amr ?? [0, 0, 0, 0], 32);
  // The potential of point masses alone is not periodic: its ghosts are extrapolated.
  // (Flux-form gravitational work, bit 0, is off in comoving coordinates: it charges the energy
  // with the potential's differences, while the kick takes the proper field grad(phi) / a, so
  // there the two disagreed by a factor a, and in cold gas E - KE was that disagreement. The kick
  // form adds the kinetic energy the kick adds, exactly.)
  u.set([c.n, x.selfGravity || x.masses > 0 ? 1 : 0, c.gravity === 'isolated' || !x.selfGravity || spherical(c) ? 1 : 0, (c.gravEnergyFlux && !cosmologyOf(c) ? 1 : 0) | (x.solids ? 2 : 0) | (x.sources ? 4 : 0) | (c.wellBalanced && eosHasEnergy(c.eos) && !cosmologyOf(c) ? 8 : 0) | (c.problem.refineMasses && c.amrLevels > 0 ? 16 : 0) | (c.problem.mhd ? 32 : 0)], 36);
  f.set(
    [
      x.dx,
      c.cfl,
      c.G,
      x.rhoMean,
      c.rhoFloor,
      // A liquid's pressure takes both signs: a negative floor says so (hydro/common.wgsl p_signed).
      c.eos === 'liquid' ? -1e30 : c.pFloor,
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
  f.set([Math.max(0, c.refineRho), Math.max(0, c.refineDense), c.tFrame ?? 1e30, 0], 60);
  f.set(eosConstants2(c.eos, c.eosParams), 64);
  // (The second: one over the acoustic steps a step may span with implicit pressure, hydro/rates.wgsl.)
  // (The third: the speed a liquid's velocity refinement indicator measures ripples against, amr/flag.wgsl.
  // The fourth: the reference pressure of a gas's implicit solve, pressure/source_gas.wgsl.)
  f.set([resists(c) ? c.resistivity : 0, implicitPressure(c) ? 1 / Math.max(c.acousticCfl, 1) : 0, x.vref ?? 0, x.pref ?? 0], 68);
  FACE_KEYS.forEach((k, i) => f.set([...c.wallVelocity[k], 0], 72 + 4 * i));
  f.set([...domainLo(c), 0], 96);
  f.set(rotationOf(c), 100);
  // The fourth: a mask of the mirrored axes halved on a rectangular grid (src/domain.ts
  // gridDims), for the block grid's fractions (amr/grid.wgsl cube_frac).
  const gd = gridDims(c);
  const halved = [0, 1, 2].reduce((m, a) => m | (gd[a] * 2 === c.n ? 1 << a : 0), 0);
  u.set([...[0, 1, 2].map((a) => (a === 0 && cylindrical(c) ? 2 : c.symmetry?.[a] ? 1 : 0)), halved], 104);
  const k = cosmologyOf(c);
  // The fourth is the scale factor of the state (cosmo_a3, hydro/common.wgsl): the CPU's latest
  // here, and the clock's own copied in by syncScale after every step.
  f.set(k ? [k.omegaM, omegaK(k), k.omegaL, x.scale ?? k.aStart] : [0, 0, 0, 0], 108);
  const sh = c.problem.shear;
  f.set(sh ? [sh.omega, sh.q, sh.stratified ? 1 : 0, 1] : [0, 0, 0, 0], 112);
  f.set(sphParams(c), 116);
  // Radiation (radiation/radiation.ts): P.rad, P.rad2, and the opacity law's constants P.kap.
  f.set(packRadiation(c), 124);
  f.set(packOpacity(c.opacity, c.opacityParams), 132);
  return buf;
}

// Channel layout of the diagnostics reduction (hydro/diagnostics.wgsl).
export const DIAG_OPS = [SUM, SUM, SUM, SUM, SUM, SUM, SUM, MAX, SUM, SUM, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, SUM, SUM, SUM, SUM];

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
  /** burning: the energy it released per unit time over the last step, summed over the gas (0 without) */
  nuclearHeating: number;
  /** center of mass of the gas, relative to the box center */
  com: [number, number, number];
  /** position of the first point mass relative to the box center, if there is one */
  perturber: [number, number, number] | null;
  /** the point masses as they are now */
  masses: Required<PointMass>[];
  /**
   * The mirrored axes (SimConfig.symmetry): the totals above are then those of the whole
   * configuration, of which the grid holds one 2^k-th (unfoldDiagnostics).
   */
  symmetry?: [boolean, boolean, boolean];
  /** Comoving coordinates: the scale factor of the state, and its redshift 1 / a - 1. */
  scale?: number;
  redshift?: number;
  /** [min, max] per renderable field, in FIELDS order */
  ranges: [number, number][];
  /**
   * With a magnetic field: its energy, which eTot includes and
   * eInt then does not, and the largest |div B| dx relative to the largest |B|.
   */
  magnetic?: MagDiagnostics;
  /** With radiation transport: its totals, luminosity and warnings (radiation/passes.ts). */
  radiation?: RadDiagnostics;
}


export interface RestartState {
  time: number;
  steps: number;
  masses?: Required<PointMass>[];
}

/** Read the clock and the point masses back, for a snapshot. */
export async function readRestart(d: GPUDevice, clock: GPUBuffer, bh: GPUBuffer, nMasses: number): Promise<RestartState> {
  const buf = d.createBuffer({ size: CLOCK_BYTES + MASS_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(clock, 0, buf, 0, CLOCK_BYTES);
  enc.copyBufferToBuffer(bh, 0, buf, CLOCK_BYTES, MASS_FLOATS * 4);
  d.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0));
  buf.unmap();
  buf.destroy();
  // The clock holds the start of the step last taken and its dt, banked as the next step
  // begins: the state is at their sum, as diagnosticsFrom has it.
  return { time: f[1] + f[0], steps: f[3] + (f[0] > 0 ? 1 : 0), masses: nMasses ? unpackMasses(f.subarray(CLOCK_FLOATS), nMasses) : undefined };
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
  /** Called after the uniform block is written (Simulation.onParams). */
  onParams?: () => void;
  readonly device: GPUDevice;
  cfg: SimConfig;
  /** cells along the longest axis: dx = box / n */
  readonly n: number;
  /** cells along x and y: n, or n / 2 along a halved mirrored axis (gridDims) */
  readonly nx: number;
  readonly ny: number;
  /** cells along z: n (or n / 2), or 1 in 2D */
  readonly nz: number;
  readonly dx: number;

  /** state[parity] = [U1, U2]; `cur` is the parity holding the current state */
  state: GPUTexture[][] = [];
  /** half-step predicted primitives and per-cell flags (hydro/predict.wgsl) */
  pred: GPUTexture[] = [];
  /** Burning: each cell's heating rate over the last step (r32float; one texel without burning). */
  heat!: GPUTexture;
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
  /** With implicit pressure: the species pass scaling to the corrected state's density (parity p). */
  private gSpeciesFinal: GPUBindGroup[][] = [];
  private gSpeciesPredict: GPUBindGroup[][] = [];
  /** Nuclear burning (burn/burn.ts, burn/network.ts), after the species pass; none without. */
  private burn?: Burner;
  /** Radiation transport (radiation/passes.ts); none without. */
  rad?: RadPasses;
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
  /** The hydro step's passes; null with a magnetic field, whose step is its own (mhd/mhd.ts). */
  private pUpdate: RegionPass | null = null;
  private pClock!: GPUComputePipeline;
  /** The clock for previewDt: leaves the driving alone (hydro/clock.wgsl ADVANCE_DRIVE). */
  private pClockPreview!: GPUComputePipeline;
  private pCorrect!: GPUComputePipeline;
  /** Implicit pressure (SimConfig.pressure): its solver, and the source and correction passes. */
  prs?: PressureSolver;
  private pPsSource?: GPUComputePipeline;
  private pPsCorrect?: GPUComputePipeline;
  private gPsSource: GPUBindGroup[] = [];
  private gPsCorrect: GPUBindGroup[] = [];
  /** The solved pressure is still to be started from the state's (pressure/init.wgsl): after a reset. */
  private prsFresh = true;
  private pPsInit?: GPUComputePipeline;
  private gPsInit: GPUBindGroup[] = [];
  /** A gas's reference pressure for the implicit solve: its mean when the state was set (0 for a liquid). */
  private pRef = 0;
  /** The face velocities the update kept, for a gas's correction (pressure/store_on.wgsl). */
  private faceVel?: GPUTexture;
  /** A gas's implicit solve is for the pressure's change: the pressure at the start of the step, and the new one (for the predictor), both less the reference. */
  private pLag?: GPUTexture;
  private pOld?: GPUTexture;
  private pFull?: GPUTexture;
  private pPredict: RegionPass | null = null;
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
  /** a spherical run: the G M of its point masses, all at the origin (gravity/sph_monopole.wgsl) */
  private centralGM = 0;
  /** a spherical run's monopole gravity: its passes, their bind groups by parity, and its buffer */
  private sphMono: { shells: GPUComputePipeline; enclose: GPUComputePipeline; field: GPUComputePipeline; g: GPUBindGroup[][]; buf: GPUBuffer } | null = null;
  private redBh!: MassSums;
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
  /** Ideal MHD (mhd/mhd.ts): the face field and the passes that take a step with it; none without. */
  mhd?: MhdPasses;

  private constructor(gpu: Gpu, cfg: SimConfig) {
    this.device = gpu.device;
    this.cfg = cfg;
    this.n = cfg.n;
    [this.nx, this.ny, this.nz] = gridDims(cfg);
    this.speciesGroups = speciesGroups(cfg.problem);
    this.dx = cfg.problem.box / cfg.n;
    const d = this.device;
    this.params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Before any pass is bound: the driving buffer is shared, bound where a pass declares it.
    writeDrive(d, cfg.driving, cfg.ndim, cfg.problem.box);
    this.clock = d.createBuffer({
      size: CLOCK_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.clockPreview = d.createBuffer({ size: CLOCK_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
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

  /**
   * GPU bytes per cell that implicit pressure adds (src/pressure): the solver's pressure,
   * right-hand side, residual (4 each) and coefficients (16) on every multigrid level, the old
   * density (4), and for a gas with an energy the face velocities (16).
   */
  static implicitBytes(cfg: SimConfig): number {
    if (!implicitPressure(cfg)) return 0;
    return 28 * (cfg.ndim === 3 ? 8 / 7 : 4 / 3) + 4 + (eosHasEnergy(cfg.eos) ? 24 : 0);
  }

  static async create(gpu: Gpu, cfg: SimConfig): Promise<Sim> {
    // The tabulated white dwarf equation of state needs its table on the device (eos/helmholtz.ts).
    if (cfg.eos === 'helmholtz') await prepareHelmholtz(gpu.device);
    // Every grid, 2D included, lives in 3D textures, whose side WebGPU caps (2048 on most
    // adapters). Past it the textures are invalid and every step would silently do nothing.
    const side = gpu.device.limits.maxTextureDimension3D;
    if (cfg.n > side) {
      throw new Error(`a uniform grid of ${cfg.n} cells per side is larger than this GPU's 3D textures allow (${side}); use the adaptive mesh`);
    }
    const why = refusal(cfg);
    if (why) throw new Error(why);
    const cells = gridDims(cfg).reduce((a, b) => a * b, 1);
    noteAllocation(cells * (Sim.cellBytes(cfg.ndim) + (mhdOn(cfg) ? MhdPasses.cellBytes : 0) + Sim.implicitBytes(cfg) + (radiationOn(cfg) ? RadPasses.cellBytes : 0)));
    setLoadStage('Allocating the grid');
    const sim = new Sim(gpu, cfg);
    const d = sim.device;
    const n = cfg.n;
    const { nx, ny, nz } = sim;
    const t = (label: string) => tex3d(d, nx, 'rgba32float', label, nz, ny);
    d.pushErrorScope('out-of-memory');
    for (let p = 0; p < 2; p++) sim.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    sim.pred = [t('PA'), t('PB')];
    sim.gfield = t('gravity field');
    const oom = await d.popErrorScope();
    if (oom) {
      sim.destroy();
      const gb = (cells * Sim.cellBytes(cfg.ndim)) / 1024 ** 3;
      throw new Error(`out of GPU memory for a uniform grid of ${cfg.n}^${cfg.ndim} cells (about ${gb.toFixed(1)} GB): ${oom.message}`);
    }
    const G = sim.speciesGroups;
    // The face mass fluxes the update records, for the species and the radiation's advection.
    if (G > 0 || radiationOn(cfg)) {
      sim.massFlux = d.createBuffer({ label: 'face mass fluxes', size: 3 * 4 * (nx + 1) * (ny + 1) * (nz + 1), usage: GPUBufferUsage.STORAGE });
    }
    if (G > 0) {
      for (let p = 0; p < 2; p++) sim.species.push(tex3d(d, nx, 'rgba32float', `species ${p}`, nz * G, ny));
      sim.speciesHalf = tex3d(d, nx, 'rgba32float', 'species half step', nz * G, ny);
      if (sim.compCoeffs) sim.comp = t('composition');
    }
    // Burning: each cell's heating rate over the last step (burn/burn.ts); else one texel of nothing.
    sim.heat = burnSpec(cfg) && G > 0 ? tex3d(d, nx, 'r32float', 'nuclear heating', nz, ny) : noHeat(d);
    const spec = debrisSpec(cfg);
    if (spec) sim.debrisBuf = new DebrisBuffers(d, spec);
    sim.writeParams();

    // Every pipeline is asked for before any is awaited, so the browser compiles them all at
    // once (gpu/device.ts). The hydro passes wait only for the solver's potential and the
    // timestep's first records, which they bind.
    const v = (t: GPUTexture) => t.createView();
    setLoadStage(COMPILE_STAGE);
    compileSetup(['uniform', cfg]);
    const hydro = (async () => {
      // Self-gravity is 3D only. A 2D run keeps a 2^3 placeholder solver so the shared bindings
      // (potential, multipoles) still exist; it is never dispatched.
      [sim.mg, sim.redDt] = await Promise.all([
        // (A spherical run's monopole writes the solver's potential textures in its place.)
        Multigrid.create(d, cfg.ndim === 2 && !cylindrical(cfg) && !spherical(cfg) ? 2 : [nx, ny, nz], cfg.problem.box, sim.params, [sim.state[0][0], sim.state[1][0]], cfg.symmetry, sim.clock, cylindrical(cfg) || spherical(cfg)),
        Reducer.create(
          d,
          [nx, ny, nz],
          [MAX, MAX],
          '',
          {},
          [],
          'timestep',
          [nx, ny, nz].map((m) => Math.ceil(m / 4)), // the primitives pass does the first stage
        ),
      ]);
      await sim.buildHydro();
    })();
    [sim.pClock, sim.pClockPreview, sim.pField, sim.redBh, sim.pBhDrift, sim.pBhKick, sim.pBhEval, sim.pCensus] = await Promise.all([
      computePipeline(d, assemble('hydro/clock.wgsl'), 'clock', clockPressure(cfg)),
      computePipeline(d, assemble('hydro/clock.wgsl'), 'clock preview', { ADVANCE_DRIVE: 0, ...clockPressure(cfg) }),
      computePipeline(d, assemble('gravity/field.wgsl'), 'gravity field'),
      // Point masses: their pull on the gas and what their sinks take are reduced each step, and
      // drive their orbits, their growth and (in the gas's frame) the frame's acceleration.
      MassSums.create(
        d,
        MAX_MASSES,
        [nx, ny, nz],
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
    sim.gClockPreview = bindGroup(d, sim.pClockPreview, { 0: sim.params, 1: sim.redDt.result, 2: sim.clockPreview, 3: sim.bh });
    sim.gField = bindGroup(d, sim.pField, { 0: sim.params, 1: v(phi), 2: v(sim.mg.phiOld), 3: v(sim.gfield), 4: sim.clock });
    if (spherical(cfg)) {
      const code = assemble('gravity/sph_monopole.wgsl');
      const [shells, enclose, field] = await Promise.all(['shells', 'enclose', 'field'].map((e) => computePipeline(d, code, `spherical monopole ${e}`, undefined, e)));
      const buf = d.createBuffer({ label: 'spherical monopole', size: 4 * (4 + 4 * n), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const g = [0, 1].map((p) => [
        bindGroup(d, shells, { 0: sim.params, 1: v(sim.state[p][0]), 2: buf }),
        bindGroup(d, enclose, { 0: sim.params, 2: buf }),
        bindGroup(d, field, { 0: sim.params, 2: buf, 3: v(phi), 4: v(sim.gfield) }),
      ]);
      sim.sphMono = { shells, enclose, field, g, buf };
    }
    sim.gBhDrift = bindGroup(d, sim.pBhDrift, { 0: sim.params, 1: sim.clock, 2: sim.bh });
    const kick = { 0: sim.params, 1: sim.clock, 2: sim.redBh.result, 3: sim.bh };
    sim.gBhKick = bindGroup(d, sim.pBhKick, kick);
    sim.gBhEval = bindGroup(d, sim.pBhEval, kick);
    await showLoadStage(INIT_STAGE);
    sim.reset();
    endLoadStage(INIT_STAGE);
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
    const dims = [this.nx, this.ny, this.nz];
    // The diagnostics' reduction compiles alongside the hydro passes.
    const diag = Reducer.create(
      d,
      [this.nx, this.ny, this.nz],
      DIAG_OPS,
      'hydro/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(phi), 4: v(this.heat), ...k })),
      'diagnostics',
    );
    diag.catch(() => {}); // a failure is reported where it is awaited, below
    this.pPredict?.destroy();
    this.pUpdate?.destroy();
    // Implicit pressure: the solver (kept across rebuilds, with its warm start) and its passes.
    if (implicitPressure(c)) {
      // (A gas's open faces carry no pressure gradient, pressure/source_gas.wgsl: closed, to the solve.)
      const kind = (b: number) => (b === PERIODIC ? PF_PERIODIC : (b === OUTFLOW || b === OUTFLOW_ONLY) && !eosHasEnergy(c.eos) ? PF_OPEN : PF_CLOSED);
      this.prs ??= await PressureSolver.create(d, [this.nx, this.ny, this.nz], this.dx, c.bcLo.map(kind), c.bcHi.map(kind));
      const gas = eosHasEnergy(c.eos);
      [this.pPsSource, this.pPsCorrect, this.pPsInit] = await Promise.all([
        computePipeline(d, assemble(gas ? 'pressure/source_gas.wgsl' : 'pressure/source.wgsl', slots), 'pressure source'),
        computePipeline(d, assemble(gas ? 'pressure/correct_gas.wgsl' : 'pressure/correct.wgsl', slots), 'pressure correct'),
        computePipeline(d, assemble('pressure/init.wgsl', slots), 'pressure start'),
      ]);
      const prs = this.prs;
      prs.setTolerance(c.pressureTol);
      if (gas) {
        this.faceVel ??= tex3d(d, this.nx, 'rgba32float', 'face velocities', this.nz, this.ny);
        this.pOld ??= tex3d(d, this.nx, 'r32float', 'pressure at the start of the step', this.nz, this.ny);
        this.pFull ??= tex3d(d, this.nx, 'r32float', 'pressure', this.nz, this.ny);
      }
      // The pressure the predictor and the timestep take the gradient of.
      this.pLag = gas ? this.pFull! : prs.p;
      this.gPsInit = [0, 1].map((p) => bindGroup(d, this.pPsInit!, { 0: this.params, 1: v(this.state[p][0]), 2: v(this.pLag!), 3: v(this.state[p][1]), ...k }));
      this.gPsSource = [0, 1].map((p) =>
        bindGroup(d, this.pPsSource!, {
          0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[1 - p][0]), 4: v(this.state[1 - p][1]),
          5: v(prs.levels[0].rhs), 6: v(prs.levels[0].coef), 8: v(prs.p),
          ...((gas ? { 9: v(this.state[p][1]), 10: v(this.pOld!), 11: v(this.pFull!), 12: v(this.gfield), 20: this.bh } : { 7: v(prs.rho0) }) as Record<number, Bindable>), ...k,
        }),
      );
      this.gPsCorrect = [0, 1].map((p) =>
        bindGroup(d, this.pPsCorrect!, {
          0: this.params, 1: this.clock, 2: v(this.state[1 - p][0]), 3: v(this.state[1 - p][1]), 4: v(prs.p),
          6: v(this.state[p][0]), 7: v(this.state[p][1]),
          ...((gas ? { 9: v(this.faceVel!), 10: v(this.pOld!), 11: v(this.pFull!), 12: v(this.gfield), 20: this.bh } : { 5: v(prs.rho0) }) as Record<number, Bindable>), ...k,
        }),
      );
    } else {
      this.prs?.destroy();
      this.prs = undefined;
    }
    // A magnetized run steps with the MHD passes alone: the hydro update (the slowest pass of all
    // to compile) and prediction are not built for it.
    const hydro = !c.problem.mhd;
    [this.pPrim, this.pPredict, this.pUpdate, this.pCorrect, this.pBrush] = await Promise.all([
      computePipeline(d, assemble('hydro/rates.wgsl', slots), 'timestep rates'),
      hydro ? RegionPass.create(d, assemble('hydro/predict.wgsl', slots), 'predict', dims) : null,
      hydro ? RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update', dims, {}, true) : null,
      // With a field the correction keeps its energy out of what it rebuilds (MHD).
      computePipeline(d, assemble('gravity/correct.wgsl', slots), 'gravity correct', { MHD: mhdOn(c) ? 1 : 0 }),
      computePipeline(d, assemble('edit/brush.wgsl', slots), 'brush'),
    ]);
    if (radiationOn(c)) {
      if (this.rad) await this.rad.rebuild(slots, k);
      else {
        this.rad = await RadPasses.create({
          device: d, cfg: c, dims: [this.nx, this.ny, this.nz], params: this.params, clock: this.clock, state: this.state, massFlux: this.massFlux!, slots, comp: k,
        });
      }
    }
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
      // (With radiation, one for each parity of its texture too: the rates read it, binding 30.)
      for (let q = 0; q < 2; q++)
        this.gPrim.push(
          bindGroup(d, this.pPrim, {
            0: this.params,
            20: this.bh,
            1: v(this.state[p][0]),
            2: v(this.state[p][1]),
            5: v(this.gfield),
            6: this.redDt.first,
            // (Or, with implicit pressure, the pressure whose gradient limits the step: the two are not combined.)
            ...((this.rad ? { 30: v(this.rad.R[q]) } : this.prs ? { 30: v(this.pLag!) } : {}) as Record<number, GPUTextureView>),
            ...k,
          }),
        );
      if (!this.pPredict || !this.pUpdate) continue;
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
          ...((this.prs ? { 30: v(this.pLag!) } : {}) as Record<number, GPUTextureView>),
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
          ...((this.faceVel && implicitPressure(c) && eosHasEnergy(c.eos) ? { 32: v(this.faceVel) } : {}) as Record<number, GPUTextureView>),
          11: this.sources,
          ...((this.massFlux ? { 13: this.massFlux } : {}) as Record<number, GPUBuffer>),
          ...(this.debrisBuf?.bindings(this.bh) ?? {}),
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
      // (Where the species set the equation of state: the moments out, and the predictor's flag.)
      const kio = (this.comp ? { 26: v(this.comp), 24: v(this.pred[1]) } : {}) as Record<number, GPUTextureView>;
      this.gSpecies = both(this.pSpecies, (p, q) => ({
        4: v(this.state[1 - p][0]),
        6: v(this.species[1 - q]),
        7: this.massFlux!,
        8: v(this.speciesHalf!),
        ...kio,
      }));
      // With implicit pressure the density is final only after the correction, which writes it
      // back into parity p: the species are moved then, and scaled to that density.
      // (Built only then, and with the composition's bindings like the other: without them every
      // uniform run whose species set the equation of state reported a WebGPU validation error.)
      this.gSpeciesFinal = implicitPressure(c) ? both(this.pSpecies, (p, q) => ({ 4: v(this.state[p][0]), 6: v(this.species[1 - q]), 7: this.massFlux!, 8: v(this.speciesHalf!), ...kio })) : [];
    }
    this.burn?.destroy();
    this.burn = undefined;
    const burn = burnSpec(c);
    if (burn && this.speciesGroups > 0) {
      this.burn = await createBurnPass(d, burn, { ...c.problem, units: problemUnits(c.problem, c.params) }, {
        dims: [this.nx, this.ny, this.nz], stride: this.nz, groups: this.speciesGroups, slots,
        params: this.params, clock: this.clock, state: this.state, species: this.species, pb: this.pred[1], heat: this.heat, comp: this.comp,
      });
    }
    this.redDiag = await diag;
    if (mhdOn(c)) {
      if (this.mhd) {
        await this.mhd.rebuild(this.redDt.first, slots, k);
      } else {
        this.mhd = await MhdPasses.create(
          {
            device: d, cfg: c, n: this.n, nz: this.nz, params: this.params, clock: this.clock, state: this.state, half: this.pred, slots, comp: k,
            gfield: this.gfield, bh: this.bh, massFlux: this.massFlux, brush: this.brush,
          },
          this.redDt.first,
        );
      }
    }
    // The second half kick, from the updated state (parity o) back into p; with a field, it reads
    // the updated faces too (a placeholder texel without one).
    this.gCorrect = [0, 1].map((p) => {
      const o = 1 - p;
      return bindGroup(d, this.pCorrect, {
        0: this.params, 1: this.clock, 2: v(this.state[o][0]), 3: v(this.state[o][1]), 4: v(this.gfield),
        5: v(this.state[p][0]), 6: v(this.state[p][1]), 7: this.bh, 8: v(this.mhd?.faces[o] ?? this.noFaces()), ...k,
      });
    });
  }

  /** Push the current configuration to the GPU parameter block. */
  /** Comoving coordinates: the scale factor the diagnostics last read from the clock. */
  private lastScale: number | undefined;

  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [this.nx, this.ny, this.nz],
      rect: rectangular(c),
      dx: this.dx,
      rhoMean: this.rhoMean,
      // Comoving coordinates: the state's scale factor as last read (syncScale keeps it current).
      scale: this.lastScale,
      selfGravity: this.selfGravity,
      masses: this.nMasses,
      solids: this.solids,
      sources: this.nSources > 0,
      pref: this.pRef,
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
    this.prs?.setTolerance(c.pressureTol);
    this.onParams?.();
  }

  /** Load the problem's initial conditions and solve for the initial potential. */
  reset() {
    this.prsFresh = true;
    const c = this.cfg;
    const n = this.n;
    const { nx, ny, nz } = this;
    const cells = nx * ny * nz;
    const eos = makeEos(c.eos, c.eosParams);
    const fn = c.problem.init(n, eos, c.ndim, c.params);
    const sph = spherical(c) ? sphGrid(c) : null;
    // A spherical run's point masses sit at the origin: their pull joins the monopole field
    // (gravity/sph_monopole.wgsl), and they are not kept as particles.
    const masses = sph ? [] : initialMasses(c, this.dx);
    this.centralGM = sph ? initialMasses(c, this.dx).reduce((a, m) => a + m.gm, 0) : 0;
    this.nMasses = this.nOwnMasses = masses.length;
    this.device.queue.writeBuffer(this.bh, 0, packMasses(masses));
    this.debrisBuf?.reset(c.restart?.debris);
    const u1 = new Float32Array(cells * 4);
    const u2 = new Float32Array(cells * 4);
    const G = this.speciesGroups;
    const sp = new Float32Array(cells * 4 * G);
    const kc = this.compCoeffs;
    const km = kc ? new Float32Array(cells * 4) : undefined;
    const lo = domainLo(c);
    // The magnetic field on the faces, and its energy in each cell's total.
    const faces = mhdOn(c) ? initialFaces(c, n, nz, this.dx) : null;
    // Each cell's B^2 / 2, which a star's settling (it rebuilds E) adds back.
    const emag = faces ? new Float32Array(cells) : null;
    let mass = 0;
    let o = 0;
    // Radiation (radiation/init.ts): (E, F) per cell, and what the flux's diffusion value needs.
    const radOn = radiationOn(c);
    const rad = radOn ? new Float32Array(cells * 4) : null;
    const radRho = radOn ? new Float32Array(cells) : null;
    const radT = radOn ? new Float32Array(cells) : null;
    const radGiven = radOn ? new Uint8Array(cells) : null;
    this.solids = false;
    this.nSources = 0;
    this.device.queue.writeBuffer(this.sources, 0, packSources([]));
    for (let k = 0; k < nz; k++) {
      const z = nz === 1 ? 0 : (k + 0.5) * this.dx + lo[2];
      for (let j = 0; j < ny; j++) {
        // (A spherical run's init takes (r, theta).)
        const y = sph ? sph.theta(j) : (j + 0.5) * this.dx + lo[1];
        for (let i = 0; i < nx; i++, o += 4) {
          const x = sph ? sph.r(i) : (i + 0.5) * this.dx + lo[0];
          const w = fn(x, y, z);
          // Where the species set the equation of state, the cell's own.
          let ce = eos;
          if (km) {
            const m = mixMoments(kc!, w.X);
            km.set(m, o);
            ce = makeEos(c.eos, c.eosParams, m, hydrogenParticles(c.problem.composition));
          }
          const p = ce.hasEnergy ? w.p : ce.pressure(w.rho, 0);
          let eint = ce.eintFromP(w.rho, p);
          if (rad) {
            // The problem's pressure is the gas's and the radiation's: split where the equation
            // of state had a radiation term of its own; E = a T^4 unless given.
            const q = o / 4;
            // (A cell that gives its radiation, as a snapshot's do, gives the gas's pressure alone.)
            const sp = w.erad === undefined ? splitPressure(c, ce, w.rho, p, km ? [...km.subarray(o, o + 4)] : undefined, km ? hydrogenParticles(c.problem.composition) : undefined) : { eint, T: ce.temperature(w.rho, eint) };
            eint = sp.eint;
            radRho![q] = w.rho;
            radT![q] = sp.T;
            rad[o] = w.erad ?? c.radA * sp.T ** 4;
            if (w.frad) {
              rad.set(w.frad, o + 1);
              radGiven![q] = 1;
            }
          }
          const mark = cellMark(w);
          if (mark) this.solids = true;
          const go = mark === 1 ? 0 : 1; // the gas kept inside an obstacle is at rest
          u1[o] = w.rho;
          u1[o + 1] = w.rho * w.vx * go;
          u1[o + 2] = w.rho * w.vy * go;
          u1[o + 3] = w.rho * w.vz * go;
          u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz) * go;
          if (faces) {
            const B = cellField(c, faces, n, nz, i, j, k);
            emag![o / 4] = 0.5 * (B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
            u2[o] += emag![o / 4];
          }
          u2[o + 1] = eint;
          u2[o + 2] = w.rho * w.s;
          u2[o + 3] = mark;
          if (G) packSpecies(sp, o, cells * 4, G, w.rho, w.X);
          mass += w.rho;
        }
      }
    }
    this.setState(u1, u2, mass / cells);
    if (faces) this.mhd!.setFaces(faces);
    if (rad) {
      diffusionFlux(c, [nx, ny, nz], this.dx, rad, radRho!, radT!, radGiven!);
      this.rad!.set(rad);
      this.rad!.resetEscaped();
    }
    if (G) {
      this.speciesCur = 0;
      writeTexture(this.device, this.species[0], nx, 4, sp, nz * G, ny);
      if (km) writeTexture(this.device, this.comp!, nx, 4, km, nz, ny);
    }
    // The clock (clockState), at the snapshot's time for a restart.
    this.device.queue.writeBuffer(this.clock, 0, clockState(c, c.restart?.time ?? 0, c.restart?.steps ?? 0));
    this.startSettling(u1, u2, emag);
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
  private startSettling(u1: Float32Array, u2: Float32Array, emag: Float32Array | null = null) {
    const epoch = ++this.settleEpoch;
    const c = this.cfg;
    this.settling = false;
    this.settled = Promise.resolve();
    if (c.restart || !c.wellBalanced || !this.selfGravity) return;
    const eos = makeEos(c.eos, c.eosParams);
    const set = c.problem.settle ? c.problem.settle(c.params, eos) : c.problem.star ? settlingOf(c.problem.star, c.G) : null;
    if (!set) return;
    this.settleCells(set, u1, u2, eos, emag ? (_s, q) => emag[q] : undefined, epoch, true);
  }

  /**
   * Settle the star `star` just placed with the star tool (placeStar) among the gas that is
   * there. Returns why not, if it cannot be settled; '' once it has been.
   */
  async settleStar(star: PlacedStar): Promise<string> {
    const c = this.cfg;
    if (!c.wellBalanced || !this.selfGravity) return 'it needs self-gravity and the well-balanced scheme';
    const epoch = ++this.settleEpoch;
    const { u1, u2 } = await this.readState();
    if (epoch !== this.settleEpoch) return '';
    const eos = makeEos(c.eos, c.eosParams);
    const set = placedSettling(star, c.G, this.cellsOf(u1, u2), eos.pressure, c.problem.box);
    if (typeof set === 'string') return set;
    // With a magnetic field, E holds its energy too: what is left of E in a cell goes back in.
    const rest = this.mhd ? new Float32Array(u1.length / 4) : null;
    if (rest) {
      for (let q = 0, o = 0; q < rest.length; q++, o += 4) {
        const ke = (0.5 * (u1[o + 1] ** 2 + u1[o + 2] ** 2 + u1[o + 3] ** 2)) / Math.max(u1[o], 1e-30);
        rest[q] = Math.max(u2[o] - ke - u2[o + 1], 0);
      }
    }
    this.settleCells(set, u1, u2, eos, rest ? (st, q) => (st.rewrote[q] ? rest[q] : 0) : undefined, epoch);
    await this.settled;
    return '';
  }

  /** The grid's cells and this state of them, for settling. */
  private cellsOf(u1: Float32Array, u2: Float32Array): SettleCells {
    const vol = this.dx ** 3;
    const { nx, ny } = this;
    const lo = domainLo(this.cfg);
    const at = (i: number, a: number) => (i + 0.5) * this.dx + lo[a];
    // In (R, z), (R, z, 0), and a cell is a ring of 2 pi R dx^2: the settling's sums are of
    // volumes, so a star on the axis settles as a body of revolution (plan E5).
    if (this.cfg.problem.geometry === 'cylindrical') {
      const area = 2 * Math.PI * this.dx * this.dx;
      return { count: u1.length / 4, pos: (q) => [at(q % nx, 0), at(Math.floor(q / nx) % ny, 1), 0], vol: (q) => area * at(q % nx, 0), u1, u2 };
    }
    return {
      count: u1.length / 4,
      pos: (q) => [at(q % nx, 0), at(Math.floor(q / nx) % ny, 1), at(Math.floor(q / (nx * ny)), 2)],
      vol: () => vol,
      u1,
      u2,
    };
  }

  /** The rounds of a settling on the state (u1, u2), which is the grid's. `emag`: magnetic energy to add to a cell's E afterward. */
  private settleCells(set: Settling, u1: Float32Array, u2: Float32Array, eos: Eos, emag: ((s: Settler, q: number) => number) | undefined, epoch: number, fresh = false) {
    this.settling = true;
    const cells = this.cellsOf(u1, u2);
    const t0 = performance.now();
    this.settled = (async () => {
      try {
        let peak = 0;
        for (let o = 0; o < u1.length; o += 4) peak = Math.max(peak, u1[o]);
        const settler = new Settler(set, cells);
        this.settleLog = [];
        for (let round = 0; round < SETTLE_ROUNDS; round++) {
          setLoadStage(settleStage(round));
          const phi = await this.readPotential();
          if (epoch !== this.settleEpoch) return;
          const change = settler.round(phi);
          this.settleLog.push(change / peak);
          writeTexture(this.device, this.state[this.cur][0], this.nx, 4, u1, this.nz, this.ny);
          this.refreshPotential();
          if (change < (set.tol ?? SETTLE_TOL) * peak) break;
        }
        // (With radiation, the settled pressure is the gas's and the radiation's: split as the
        // initial conditions split it, radiation/init.ts.)
        settler.finish(this.rad ? (rho, p) => splitPressure(this.cfg, eos, rho, p).eint : eos.eintFromP);
        // The settled E is the gas's own: a magnetic field's energy goes back in.
        if (emag) for (let q = 0; q < cells.count; q++) u2[4 * q] += emag(settler, q);
        if (fresh && this.cfg.problem.heat) heatCells(this.cfg.problem.heat(this.cfg.params, eos), cells);
        writeTexture(this.device, this.state[this.cur][0], this.nx, 4, u1, this.nz, this.ny);
        writeTexture(this.device, this.state[this.cur][1], this.nx, 4, u2, this.nz, this.ny);
        if (this.rad) await this.settledRadiation(u1, u2, eos, fresh ? null : settler.rewrote);
        this.refreshPotential();
      } finally {
        if (epoch === this.settleEpoch) {
          this.settling = false;
          setLoadStage('');
          this.settleMs = performance.now() - t0;
        }
      }
    })();
  }

  /**
   * With radiation, after a settling: E = a T^4 at the settled gas's temperature and F its
   * diffusion value (radiation/init.ts), in every cell, or only in those the settling rewrote.
   */
  private async settledRadiation(u1: Float32Array, u2: Float32Array, eos: Eos, rewrote: ArrayLike<boolean | number> | null) {
    const rad = this.rad!;
    const c = this.cfg;
    const cells = u1.length / 4;
    const R = rewrote ? await rad.read() : new Float32Array(cells * 4);
    const rho = new Float32Array(cells);
    const T = new Float32Array(cells);
    const given = new Uint8Array(cells);
    for (let q = 0; q < cells; q++) {
      rho[q] = u1[4 * q];
      T[q] = eos.temperature(Math.max(rho[q], c.rhoFloor), u2[4 * q + 1]);
      if (rewrote && !rewrote[q]) given[q] = 1;
      else R[4 * q] = c.radA * T[q] ** 4;
    }
    diffusionFlux(c, [this.nx, this.ny, this.nz], this.dx, R, rho, T, given);
    rad.set(R);
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
      pass.dispatchWorkgroups(Math.ceil(this.nx / 8), Math.ceil(this.ny / 8), Math.ceil(this.nz / 4));
      pass.end();
    }, this.selfGravity, v0, d.ranges[0][1]);
    const r = await this.readRestart();
    return b.report(r.time, r.masses?.[b.spec.mass]?.gm ?? 0, await b.readLedger(), census);
  }

  /** Replace the state with raw conserved arrays (rho, m), (E, e_int, rho s, 0). */
  setState(u1: Float32Array, u2: Float32Array, rhoMean: number) {
    const d = this.device;
    const { nx, ny, nz } = this;
    this.cur = 0;
    this.rhoMean = rhoMean;
    this.prsFresh = true;
    this.pRef = 0;
    if (implicitPressure(this.cfg) && eosHasEnergy(this.cfg.eos)) {
      // The mean pressure of the gas: what the solve's pressure is measured from.
      const eos = makeEos(this.cfg.eos, this.cfg.eosParams);
      let sum = 0;
      let count = 0;
      let lo = Infinity;
      let hi = 0;
      for (let o = 0; o < u1.length; o += 4) {
        if (u2[o + 3] !== 0) continue;
        const p = eos.pressure(u1[o], u2[o + 1]);
        sum += p;
        lo = Math.min(lo, p);
        hi = Math.max(hi, p);
        count++;
      }
      // Where the pressure spans decades (a stratified gas) no one reference serves, and none
      // is taken: single precision then holds each cell's pressure to its own size.
      this.pRef = count && hi < 4 * lo ? sum / count : 0;
    }
    writeTexture(d, this.state[0][0], nx, 4, u1, nz, ny);
    writeTexture(d, this.state[0][1], nx, 4, u2, nz, ny);
    if (this.selfGravity) writeTexture(d, this.mg.phi, nx, 1, new Float32Array(nx * ny * nz), nz, ny);
    d.queue.writeBuffer(this.clock, 0, clockState(this.cfg));
    writeDrive(d, this.cfg.driving, this.cfg.ndim, this.cfg.problem.box);
    this.writeParams();
    this.refreshPotential();
  }

  /** Solve for the potential of the current state from the current guess. */
  solvePotential(cycles: number) {
    if (!this.selfGravity) return;
    const enc = this.device.createCommandEncoder();
    const pass = beginPass(enc);
    if (this.sphMono) {
      this.device.queue.writeBuffer(this.sphMono.buf, 0, new Float32Array([this.centralGM, isSelfGravity(this.cfg.gravity) ? 1 : 0, 0, 0]));
      this.encodeMonopole(pass, this.cur);
      this.encodeMonopole(pass, this.cur);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      return;
    }
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
    if (this.selfGravity && !this.sphMono) this.copyPhiToOld(enc);
    const pass = beginPass(enc);
    // No step is being taken: the masses' accelerations are evaluated, nothing more.
    if (this.hasMasses) this.encodeMassKick(pass, this.cur, false);
    // (A spherical run's monopole has written the field already.)
    if (!this.sphMono) this.encodeField(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private get selfGravity(): boolean {
    // (A spherical run's monopole holds its point masses' pull too.)
    if (spherical(this.cfg)) return isSelfGravity(this.cfg.gravity) || this.centralGM !== 0;
    // (An axisymmetric run solves for its potential in (R, z): gravity/stencil.wgsl.)
    return isSelfGravity(this.cfg.gravity) && (this.cfg.ndim === 3 || cylindrical(this.cfg));
  }

  /**
   * A spherical run's gravity from the state at `parity`: the shells' masses, their prefix sum,
   * the field and potential textures (gravity/sph_monopole.wgsl). The previous potential moves
   * to its place first: encoding it twice leaves the two the same.
   */
  private encodeMonopole(pass: GPUComputePassEncoder, parity: number) {
    const m = this.sphMono!;
    const n = this.n;
    pass.setPipeline(m.shells);
    pass.setBindGroup(0, m.g[parity][0]);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.setPipeline(m.enclose);
    pass.setBindGroup(0, m.g[parity][1]);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(m.field);
    pass.setBindGroup(0, m.g[parity][2]);
    pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), 1);
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
    this.redBh.encode(pass, parity, this.nMasses);
    pass.setPipeline(live ? this.pBhKick : this.pBhEval);
    pass.setBindGroup(0, live ? this.gBhKick : this.gBhEval);
    pass.dispatchWorkgroups(1);
  }

  /** First stage of the timestep reduction: the signal rates of the state at `parity`. */
  private encodeRates(pass: GPUComputePassEncoder, parity: number) {
    if (this.mhd) {
      this.mhd.encodeRates(pass, parity);
      return;
    }
    pass.setPipeline(this.pPrim);
    pass.setBindGroup(0, this.primGroup(parity));
    this.dispatch(pass);
  }

  /** The timestep rates' bind group for the state at `parity` (and the radiation as it is now). */
  private primGroup(parity: number): GPUBindGroup {
    return this.gPrim[2 * parity + (this.rad?.cur ?? 0)];
  }

  private dispatch(pass: GPUComputePassEncoder) {
    pass.dispatchWorkgroups(Math.ceil(this.nx / 8), Math.ceil(this.ny / 8), Math.ceil(this.nz / 4));
  }

  /** Acceleration and half-step potential from the current and previous potentials. */
  private encodeField(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pField);
    pass.setBindGroup(0, this.gField);
    this.dispatch(pass);
  }

  private copyPhiToOld(enc: GPUCommandEncoder) {
    enc.copyTextureToTexture({ texture: this.mg.phi }, { texture: this.mg.phiOld }, [this.nx, this.ny, this.nz]);
  }

  /** Encode `count` full steps: primitives, timestep, predictor, update, then gravity if enabled. */
  encodeSteps(enc: GPUCommandEncoder, count: number): number {
    if (this.settling) return 0; // a star is still being settled
    const selfGravity = this.selfGravity;
    // (A rotating frame needs the correction pass for its centrifugal kick and closing turn, and
    // comoving coordinates for the second half of the expansion drag.)
    const anyGravity = this.fieldOn || uniformField(this.cfg).some((g) => g !== 0) || rotating(this.cfg) || !!cosmologyOf(this.cfg);
    syncScale(enc, this.cfg, this.clock, this.params);
    for (let i = 0; i < count; i++) {
      const p = this.cur;
      let pass = beginPass(enc);
      if (this.prs && this.prsFresh) {
        pass.setPipeline(this.pPsInit!);
        pass.setBindGroup(0, this.gPsInit[p]);
        this.dispatch(pass);
        this.prsFresh = false;
      }
      this.encodeRates(pass, p);
      this.redDt.encode(pass);
      pass.setPipeline(this.pClock);
      pass.setBindGroup(0, this.gClock);
      pass.dispatchWorkgroups(1);
      if (this.mhd) {
        // Ideal MHD: its own two-stage step, with gravity's first half kick and cooling in it
        // (mhd/update.wgsl); the second half kick follows below as for the gas alone.
        this.mhd.encode(pass, p);
      } else {
        this.pPredict!.encode(pass, this.gPredict[p]);
        this.pUpdate!.encode(pass, this.gUpdate[p]);
      }
      // (With implicit pressure the second of the two species passes waits for the correction.)
      const speciesPass = (pipe: GPUComputePipeline, g: GPUBindGroup) => {
        pass.setPipeline(pipe);
        pass.setBindGroup(0, g);
        pass.dispatchWorkgroups(Math.ceil(this.nx / 8), Math.ceil(this.ny / 8), Math.ceil(this.nz / 4));
      };
      if (this.pSpecies) {
        const q = this.speciesCur;
        speciesPass(this.pSpeciesPredict!, this.gSpeciesPredict[p][q]);
        if (!this.prs) {
          speciesPass(this.pSpecies, this.gSpecies[p][q]);
          this.speciesCur = 1 - q;
        }
      }
      // Implicit pressure with self-gravity: the potential of the advected density, before the
      // pressure passes, whose face kicks take it as it is (gravity/field.wgsl). The gas the kicks
      // act on is then the gas the potential is of. With the potential of the step before
      // extrapolated half a step instead, a star crossing the grid at 0.8 of a cell a step was
      // pulled back toward where its field lagged it, and lost 10% of its momentum in a
      // dynamical time.
      const early = !!this.prs && selfGravity && !this.sphMono;
      if (early) {
        pass.end();
        this.copyPhiToOld(enc);
        pass = beginPass(enc);
        this.mg.encodeSource(pass, 1 - p, this.cfg.gravity === 'isolated');
        for (let v = 0; v < this.cfg.vcycles; v++) this.mg.encodeVCycle(pass);
        this.encodeField(pass);
      }
      if (this.prs) {
        // Implicit pressure: the equation from the old state (parity p) and the advected one
        // (parity 1 - p), the solve, and the correction back into parity p.
        pass.setPipeline(this.pPsSource!);
        pass.setBindGroup(0, this.gPsSource[p]);
        this.dispatch(pass);
        this.prs.encodeCoarsen(pass);
        this.prs.encodeSolve(pass, this.cfg.pressureCycles);
        pass.setPipeline(this.pPsCorrect!);
        pass.setBindGroup(0, this.gPsCorrect[p]);
        this.dispatch(pass);
        if (this.pSpecies) {
          speciesPass(this.pSpecies, this.gSpeciesFinal[p][this.speciesCur]);
          this.speciesCur = 1 - this.speciesCur;
        }
      }
      // (With implicit pressure the final state is in parity p already, and the potential is its.)
      const gp = this.prs ? p : 1 - p;
      // Radiation carried with the gas, by the update's mass fluxes (radiation/advect.wgsl).
      this.rad?.encodeAdvect(pass, p);
      if (selfGravity && this.sphMono) {
        this.encodeMonopole(pass, 1 - p);
      } else if (selfGravity && !early) {
        // Keep the old potential and solve for that of the updated density (warm-started).
        pass.end();
        this.copyPhiToOld(enc);
        pass = beginPass(enc);
        this.mg.encodeSource(pass, gp, this.cfg.gravity === 'isolated');
        for (let v = 0; v < this.cfg.vcycles; v++) this.mg.encodeVCycle(pass);
      }
      if (this.hasMasses) {
        // Move the masses to the new time, then their pull on the updated gas and what their
        // sinks take from it (removed in the correction pass below).
        // (With implicit pressure they are given what the correction gave the advected gas, and
        // drift within the kick's pass: gravity/bh_kick.wgsl.)
        if (!this.prs) {
          pass.setPipeline(this.pBhDrift);
          pass.setBindGroup(0, this.gBhDrift);
          pass.dispatchWorkgroups(1);
        }
        this.encodeMassKick(pass, 1 - p, true);
      }
      if (this.fieldOn && !this.sphMono) this.encodeField(pass);
      if (this.prs) {
        // (The state is back in parity p: the pressure passes ran before the potential's solve, above.)
      } else if (anyGravity) {
        // Second half kick, with the acceleration of the updated state.
        pass.setPipeline(this.pCorrect);
        pass.setBindGroup(0, this.gCorrect[p]);
        this.dispatch(pass);
        if (this.mhd) {
          // The state is back in parity p; its faces follow it.
          pass.end();
          this.mhd.copyFaces(enc, 1 - p, p);
          pass = beginPass(enc);
        }
      } else {
        this.cur = 1 - p;
      }
      if (this.rad) {
        // The radiation sweeps, then the gas's side of them folded into the other parity.
        this.rad.encodeSweeps(pass, this.cur);
        this.cur = 1 - this.cur;
      }
      if (this.burn) {
        // Burn the step's end state into the other parities (burn/burn.ts).
        this.burn.encode(pass, this.cur, this.speciesCur);
        if (this.mhd) {
          // The burnt state is in the other parity; its faces follow it.
          pass.end();
          this.mhd.copyFaces(enc, this.cur, 1 - this.cur);
          pass = beginPass(enc);
        }
        this.cur = 1 - this.cur;
        this.speciesCur = 1 - this.speciesCur;
      }
      pass.end();
      syncScale(enc, this.cfg, this.clock, this.params);
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
    if (isFieldBrush(op)) {
      if (!this.mhd) return;
      const { op: one, steps } = fieldBrushSteps(op, this.n);
      this.device.queue.writeBuffer(this.brush, 0, packBrush(one));
      const pass = beginPass(enc);
      for (let i = 0; i < steps; i++) {
        this.mhd.encodeFieldBrush(pass, this.cur);
        this.cur = 1 - this.cur;
      }
      pass.end();
      return;
    }
    this.device.queue.writeBuffer(this.brush, 0, packBrush(op));
    const pass = beginPass(enc);
    pass.setPipeline(this.pBrush);
    pass.setBindGroup(0, this.gBrush[this.cur]);
    this.dispatch(pass);
    pass.end();
    // A brush leaves the field as it is: the faces go along to the brushed state's parity.
    this.mhd?.copyFaces(enc, this.cur, 1 - this.cur);
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

  private lumpReader?: LumpReader;
  readGasLumps() {
    this.lumpReader ??= new LumpReader(this.device, 'gravity/lumps.wgsl');
    const k = lumpGrid(this.n, this.nz);
    this.device.queue.writeBuffer((this.lumpGrid ??= this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })), 0, new Uint32Array([...k, 0]));
    const [u1, u2] = this.state[this.cur];
    return this.lumpReader.read(k[0] * k[1] * k[2], { 0: this.params, 1: u1.createView(), 2: u2.createView(), 3: this.lumpGrid }, (pass) =>
      pass.dispatchWorkgroups(Math.ceil(k[0] / 4), Math.ceil(k[1] / 4), Math.ceil(k[2] / 4)),
    );
  }
  private lumpGrid?: GPUBuffer;

  addMass(m: PointMass): boolean {
    if (this.nMasses >= MAX_MASSES) return false;
    // Only its own slot is written: the others are mid-orbit on the GPU.
    this.device.queue.writeBuffer(this.bh, 16 + 4 * MASS_STRIDE * this.nMasses, packMasses([completeMass(m, this.dx)]), 4, MASS_STRIDE);
    this.nMasses++;
    this.writeParams();
    this.refreshPotential();
    return true;
  }

  get massCount(): number {
    return this.nMasses;
  }

  private pRemove?: GPUComputePipeline;
  removeMasses(indices: number[]) {
    const gone = new Set(indices.filter((i) => i >= this.nOwnMasses && i < this.nMasses));
    if (!gone.size) return;
    let mask = 0;
    for (let i = 0; i < this.nMasses; i++) if (!gone.has(i)) mask |= 1 << i;
    const d = this.device;
    const keep = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(keep, 0, new Uint32Array([mask >>> 0, 0, 0, 0]));
    // Compiled where it is first needed, at once (a few lines): the masses' places among the
    // run's change as this returns, and the pass is queued behind whatever steps are in flight.
    this.pRemove ??= d.createComputePipeline({
      label: 'point mass removal',
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: assemble('gravity/bh_remove.wgsl') }), entryPoint: 'main' },
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass({ label: 'point mass removal' });
    pass.setPipeline(this.pRemove);
    pass.setBindGroup(0, bindGroup(d, this.pRemove, { 0: this.bh, 1: keep }));
    pass.dispatchWorkgroups(1);
    pass.end();
    d.queue.submit([enc.finish()]);
    keep.destroy();
    this.nMasses -= gone.size;
    this.writeParams();
    this.refreshPotential();
  }

  removeAddedMasses() {
    if (this.nMasses === this.nOwnMasses) return;
    this.nMasses = this.nOwnMasses;
    this.writeParams();
    // With no masses left the field texture goes out of use; otherwise rebuild it without them.
    this.refreshPotential();
  }

  async probe(f: [number, number, number]): Promise<CellProbe | null> {
    // Fractions of the computed box along each axis (domain.ts boxFrac): of half the box along
    // a halved mirrored axis (gridDims).
    const dims = this.displayDims;
    const cell = (x: number, a: number) => Math.min(dims[a] - 1, Math.max(0, Math.floor(x * dims[a])));
    const index = [cell(f[0], 0), cell(f[1], 1), cell(f[2], 2)];
    const [u1, u2] = await readTexels(this.device, this.state[this.cur], index as [number, number, number]);
    const lo = domainLo(this.cfg);
    const pos = index.map((i, a) => (i + 0.5) * this.dx + lo[a]) as [number, number, number];
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
    return [this.nx, this.ny, this.nz];
  }
  static readonly fieldShader = 'render/field.wgsl';
  readonly fieldShader = Sim.fieldShader;
  get clockBuffer(): GPUBuffer {
    return this.clock;
  }
  fieldResources(parity: number, speciesParity: number, radParity = 0): Record<number, GPUBuffer | GPUTextureView> {
    const r: Record<number, GPUBuffer | GPUTextureView> = {
      2: this.state[parity][0].createView(), 3: this.state[parity][1].createView(), 7: this.gfield.createView(), 10: this.pred[1].createView(), 20: this.bh,
      13: (this.mhd?.faces[parity] ?? this.noFaces()).createView(),
      14: this.mhd?.magResult ?? this.noMag(), 15: this.heat.createView(),
      16: (this.rad?.R[radParity] ?? this.noFaces()).createView(),
    };
    if (this.speciesGroups) r[9] = this.species[speciesParity].createView();
    return { ...r, ...this.compIn };
  }
  get speciesStride(): number {
    return this.speciesGroups ? this.nz : 0;
  }
  get radCur(): number {
    return this.rad?.cur ?? 0;
  }
  /** The display pass's face field without a magnetic field: one texel of nothing. */
  private placeholderFaces?: GPUTexture;
  private noFaces(): GPUTexture {
    this.placeholderFaces ??= tex3d(this.device, 1, 'rgba32float', 'no face field', 1);
    return this.placeholderFaces;
  }
  private placeholderMag?: GPUBuffer;
  /** Without a field, the display's magnetic reduction binding: four words nothing reads. */
  private noMag(): GPUBuffer {
    this.placeholderMag ??= this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return this.placeholderMag;
  }
  encodeMagnetic(pass: GPUComputePassEncoder, parity: number) {
    this.mhd?.encodeMagnetic(pass, parity);
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
    const p = this.cur;
    const buf = d.createBuffer({ size: CLOCK_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const time = async (count: number, fn: (pass: GPUComputePassEncoder) => void) => {
      const run = async (k: number) => {
        const enc = d.createCommandEncoder();
        const pass = beginPass(enc);
        for (let i = 0; i < k; i++) fn(pass);
        pass.end();
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, CLOCK_BYTES);
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
      pass.setBindGroup(0, this.primGroup(p));
      this.dispatch(pass);
    });
    out.timestep = await time(reps, (pass) => this.redDt.encode(pass));
    out.predict = await time(reps, (pass) => {
      this.pPredict?.encode(pass, this.gPredict[p]);
    });
    out.update = await time(reps, (pass) => {
      this.pUpdate?.encode(pass, this.gUpdate[p]);
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
    enc.copyBufferToBuffer(this.clock, 0, this.staging, 0, CLOCK_BYTES);
    enc.copyBufferToBuffer(this.redDiag.result, 0, this.staging, CLOCK_BYTES, DIAG_OPS.length * 4);
    enc.copyBufferToBuffer(this.bh, 0, this.staging, CLOCK_BYTES + DIAG_OPS.length * 4, MASS_FLOATS * 4);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      const diag = diagnosticsFrom(a, this.nMasses);
      if (cosmologyOf(this.cfg)) this.lastScale = withScale(diag, a).scale;
      if (diag.steps === 0 && diag.dt === 0) diag.dt = await this.previewDt();
      // A magnetic field: its diagnostics, and its energy taken out of the reduction's internal
      // energy (E - KE there includes B^2 / 2).
      if (this.mhd && !this.destroyed) {
        diag.magnetic = await this.mhd.diagnostics(this.cur);
        diag.eInt -= diag.magnetic.eMag;
      }
      if (this.rad && !this.destroyed) diag.radiation = (await this.rad.diagnostics(this.cur)) ?? undefined;
      return unfoldDiagnostics(diag, this.cfg.symmetry, cylindrical(this.cfg) || spherical(this.cfg));
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
   * encode work on its destroyed textures (previewDt and the magnetic diagnostics, after
   * readDiagnostics' map).
   */
  private destroyed = false;

  private async previewDt(): Promise<number> {
    if (this.destroyed) return 0;
    if (this.settling) return 0;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.clock, 0, this.clockPreview, 0, CLOCK_BYTES);
    const pass = beginPass(enc);
    this.encodeRates(pass, this.cur);
    this.redDt.encode(pass);
    pass.setPipeline(this.pClockPreview);
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
    const buf = this.device.createBuffer({ size: CLOCK_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let steps = 0;
    let size = batch;
    try {
      for (;;) {
        const enc = this.device.createCommandEncoder();
        this.encodeSteps(enc, size);
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, CLOCK_BYTES);
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
  async fluxMismatch(): Promise<{ faces: number; mismatched: number; maxDiff: number; mirror: number }> {
    const d = this.device;
    const { nx, ny } = this;
    const c = this.cfg;
    const slots = hydroSlots(c);
    // Split by region like the real pass, so faces between the two variants are checked too.
    const pipe = await RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update (flux debug)', [nx, ny, this.nz], { DEBUG_FLUX: 1 }, true);
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
    pass.setBindGroup(0, this.primGroup(p));
    this.dispatch(pass);
    if (!this.pPredict) throw new Error('fluxMismatch checks the hydro update, which a magnetized run does not have');
    this.pPredict.encode(pass, this.gPredict[p]);
    pipe.encode(pass, group);
    pass.end();
    d.queue.submit([enc.finish()]);
    const f1 = await readTexture(d, this.state[o][0], nx, 4, this.nz, ny);
    const f2 = await readTexture(d, this.state[o][1], nx, 4, this.nz, ny);
    const flux = (i: number, j: number, k: number, f: number) => {
      const cell = ((k * ny + j) * nx + i) * 4;
      return f < 4 ? f1[cell + f] : f2[cell + f - 4];
    };
    let faces = 0;
    let mismatched = 0;
    let maxDiff = 0;
    // The largest mass flux through a plane of mirror symmetry (SimConfig.symmetry): zero.
    let mirror = 0;
    for (let k = 0; k < this.nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++)
          for (let a = 0; a < 3; a++) if (c.symmetry[a] && [i, j, k][a] === 0) mirror = Math.max(mirror, Math.abs(flux(i, j, k, 2 * a)));
    const dims = [nx, ny, this.nz];
    for (let k = 0; k < this.nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++)
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
    return { faces, mismatched, maxDiff, mirror };
  }

  /** Current conserved state as float arrays (validation only). */
  async readState(): Promise<{ u1: Float32Array; u2: Float32Array }> {
    const [a, b] = this.state[this.cur];
    return {
      u1: await readTexture(this.device, a, this.nx, 4, this.nz, this.ny),
      u2: await readTexture(this.device, b, this.nx, 4, this.nz, this.ny),
    };
  }

  /**
   * Mass fractions of every cell (validation): X[i][cell], the cells in the order of
   * readState(). Empty without species.
   */
  async readSpecies(): Promise<Float32Array[]> {
    const G = this.speciesGroups;
    if (!G) return [];
    const raw = await readTexture(this.device, this.species[this.speciesCur], this.nx, 4, this.nz * G, this.ny);
    const { u1 } = await this.readState();
    const cells = this.nx * this.ny * this.nz;
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
    return readTexture(this.device, this.mg.phi, this.nx, 1, this.nz, this.ny);
  }

  /** Max |residual| of the Poisson solve on the finest level. */
  async poissonResidual(): Promise<number> {
    const enc = this.device.createCommandEncoder();
    const pass = beginPass(enc);
    this.mg.encodeResidual(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const r = await readTexture(this.device, this.mg.levels[0].res, this.nx, 1, this.nz, this.ny);
    let m = 0;
    for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i]));
    return m;
  }

  destroy() {
    this.destroyed = true;
    this.lumpReader?.destroy();
    this.lumpGrid?.destroy();
    this.settleEpoch++; // abandon a settling in progress: its textures are going
    for (const s of this.state) for (const t of s) t.destroy();
    this.burn?.destroy();
    for (const t of [...this.pred, this.gfield, ...this.species]) t.destroy();
    this.heat?.destroy();
    this.comp?.destroy();
    this.speciesHalf?.destroy();
    this.massFlux?.destroy();
    this.prs?.destroy();
    this.faceVel?.destroy();
    this.pOld?.destroy();
    this.pFull?.destroy();
    this.debrisBuf?.destroy();
    this.mg?.destroy();
    this.mhd?.destroy();
    this.rad?.destroy();
    this.placeholderFaces?.destroy();
  }

  /** Magnetic energy and divergence of the current state; null without a magnetic field. */
  async magnetic(): Promise<MagDiagnostics | null> {
    return this.mhd ? this.mhd.diagnostics(this.cur) : null;
  }
}
