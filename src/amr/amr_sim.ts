// Block-structured adaptive mesh refinement on the GPU.
//
// The grid is a tree of 16^3-cell blocks (src/amr/tree.ts) packed into one 3D texture atlas
// per field, with two ghost layers stored around each block. All blocks take the same
// timestep, unless per-block timesteps are on (cfg.timeBins, see encodeCycle()). A step is:
//
//   assemble   primitives for every block, ghosts included, straight from the conserved state
//   timestep   reduce the signal rate over leaf cells; set dt
//   predict    half-step states for every block and its first ghost layer
//   update     leaf cells only, with fluxes matched at refinement boundaries
//
// Re-meshing is driven from the CPU. Every few steps a pass reduces a refinement indicator per
// block; the result is read back without blocking, the tree is adapted, and one pass refills
// the new leaves (copy, prolong or restrict) into the other state texture.
//
// With self-gravity (3D, or 2D in (R, z)), the potential is solved on every tree level after the update
// (src/amr/gravity.ts), and the second half of the gravity kick runs as its own pass.
import { domainLo } from '../domain';
import { prepareHelmholtz } from '../eos/helmholtz';
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, rendersTo3d, tex3d, type Bindable, type Gpu } from '../gpu/device';
import { COMPILE_STAGE, INIT_STAGE, compileSetup, endLoadStage, setLoadStage, settleStage, showLoadStage } from '../gpu/progress';
import { MAX, Reducer } from '../gpu/reduce';
import { beginPass } from '../gpu/timing';
import { memoryBudget, noteAllocation } from '../gpu/memory';
import { AmrGravity, FIELD_SLOTS, FIELD_SLOTS_BINS } from './gravity';
import { AmrPressure } from './pressure';
import { PF_CLOSED, PF_OPEN, PF_PERIODIC } from '../pressure/multigrid';
import { Multipoles } from '../gravity/multipole';
import { LumpReader } from '../gravity/lumps';
import { MassSums } from '../gravity/mass_sums';
import { makeEos, type Eos } from '../eos/eos';
import { createBurnPass, flagBurnSlot, isNetworkBurn, type Burner } from '../burn/burn';
import { isSelfGravity, PERIODIC, problemUnits } from '../problems/types';
import { heatCells, placedSettling, Settler, settlingOf, SETTLE_ROUNDS, SETTLE_TOL, type PlacedStar, type SettleCells, type Settling } from '../problems/settle';

/** Most cells of a mesh the star tool's settling reads back (about 100 bytes each). */
const SETTLE_CPU_CELLS = 4e7;
import { writeDrive } from '../hydro/drive';
import { gridDims } from '../domain';
import { sphSetup, spherical } from '../spherical';
import { implicitPressure, implicitBlocksUnsupported, refusal, CLOCK_BYTES, clockState, syncScale, cosmologyOf, withScale, cosmologyUnsupported, shearUnsupported, cylindrical, cylindricalUnsupported, sphericalUnsupported, symmetryUnsupported, unfoldDiagnostics, type CellProbe, type DisplayAtlas, readTexels, completeMass, MassReader, MAX_MASSES, BRUSH_BYTES, DIAG_FLOATS, DIAG_OPS, diagnosticsFrom, hydroSlots, initialMasses, MASS_FLOATS, MASS_STRIDE, MAX_SOURCES, outlineContains, packBrush, packMasses, packParams, packSources, PARAM_BYTES, readRestart, SOURCE_BYTES, type RestartState, type BrushOp, type Diagnostics, type Outline, type SimConfig, type Simulation, type SourceSpec, writeMassRadii, burnSpec, noHeat, isFieldBrush, fieldBrushSteps } from '../sim';
import { BlockTree, BS, BT_STRIDE, NG, PAD, RG_STRIDE, type Block, gridBox, type GridBox, type RegridOp, type AdaptStep } from './tree';
import { indicator, leafState, samplesPerCell, surfaceOf, type LeafState } from './init_cpu';
import { InitRun } from './init_pool';
import { FLAT_ROWS, interior3d, padded3d, rows, SLAB_BLOCKS, SLAB_QUADS } from './dispatch';
import { flagSpeciesSlot, speciesAtlasSize, speciesGroupOffset, speciesGroups, SPECIES_PER_GROUP } from '../species/species';
import { DebrisBuffers, debrisSpec, type DebrisLedger, type DebrisReport } from '../debris/debris';
import { compSrcSlot, problemCoefficients, type Moments } from '../species/mixture';
import { AmrMhd, maxBinding, MHD_BLOCK_BYTES, registerBytes } from '../mhd/amr_mhd';
import { mhdUnsupported } from '../mhd/mhd';

/**
 * Steps between refinement checks, and how far the flow may run past an unanswered check.
 * Signals cross at most about a third of a cell per step, and a block is 16 cells with a
 * buffer block kept around flagged ones, so 24 steps cannot carry a feature out of its refinement.
 */
const REGRID_EVERY = 8;
const MAX_STEPS_AHEAD = 24;
/** Most bytes read back from the GPU in one staging buffer (AmrSim.readBoxes). */
const READ_CHUNK_BYTES = 256 * 1024 * 1024;
/** Bytes per block of the species' face mass fluxes (amr/species.wgsl): (BS + 1)^d faces of three axes. */
const speciesFaceBytes = (ndim: 2 | 3) => (BS + 1) ** 2 * (ndim === 3 ? BS + 1 : 1) * 3 * 4;
/** The GPU's settling (amr/settle.wgsl): its uniform's size, and the most stars it takes. */
const SETTLE_BYTES = 560;
const SETTLE_MAX_STARS = 4;
/** The first window of the face mass flux records: a binding's largest size, aligned for the second's offset. */
const massFluxWindowBytes = (d: GPUDevice) => Math.floor(d.limits.maxStorageBufferBindingSize / d.limits.minStorageBufferOffsetAlignment) * d.limits.minStorageBufferOffsetAlignment;
/** Most bytes of face mass flux records: two windows, in a buffer no larger than a buffer may be. */
const massFluxMaxBytes = (d: GPUDevice) => Math.min(d.limits.maxBufferSize, massFluxWindowBytes(d) + d.limits.maxStorageBufferBindingSize);
/** Most bins of per-block timesteps: a cycle of 2^3 substeps is one refinement interval. */
const MAX_TIME_BINS = 3;
/** The dispatch-size table of per-block timesteps (amr/bins_args.wgsl): 64 words, triple i at 12 i bytes. */
const BIN_ARGS_BYTES = 256;
const ARG = (i: number) => 12 * i;
/** Workgroups per block of a flat dispatch (amr/list.wgsl, inv_flat) out to `reach` ghost layers. */
const flatGroups = (ndim: number, reach: number) => Math.ceil((BS + 2 * reach) ** ndim / 64);
const BIN_TOTALS = 60;
/** The first line of a (GPU error) message. */
const firstLine = (s: string) => s.split(/\r?\n/)[0];
/**
 * Per-block timesteps pay for their extra passes only when enough blocks sit out enough
 * substeps. Every few cycles the share of block updates actually made is read back; above
 * these shares (measured: about 0.8 without a gravity field, 0.65 with one, where every
 * substep also relaxes the potential) the run goes back to one timestep for a while, a
 * longer while each time the answer is the same.
 */
const BIN_POLL_CYCLES = 4;
const BIN_SHARE_MAX = { hydro: 0.8, field: 0.65 };
const BIN_PAUSE_STEPS = 192;
/**
 * A run starts with this many single steps before its first cycle. Initial conditions are
 * often out of balance (gas at rest in a deep potential falls at once), and the timestep can
 * fall 30-fold within the first few steps; a cycle commits 2^K substeps to the timestep of its
 * start, and one begun at t = 0 took them all at the step the gas at rest allowed (an (r,
 * theta) run with a heavy point mass, found by tools/random-test). Single steps measure the
 * timestep every step.
 */
const BIN_START_STEPS = 16;
const BIN_PAUSE_MAX = 1536;

interface RegridPlan {
  ops: RegridOp[];
  /** For a re-mesh the refinement check decided: the steps the tree took (BlockTree.adapt, see onRegrid). */
  steps?: AdaptStep[];
}

export interface LeafCells {
  /** per leaf cell: position of the center, cell size, and the conserved state */
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
  dx: Float32Array;
  u1: Float32Array; // 4 per cell
  u2: Float32Array;
  count: number;
}

export class AmrSim implements Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  /** finest cells per side */
  readonly n: number;
  readonly nz: number;
  readonly levels: number;
  /** cells per side at level 0, and its cell size */
  readonly n0: number;
  readonly dx0: number;
  /**
   * Atlas slots per side of a layer, layers of slots along z (in 3D as many as slotsPerSide, a
   * cube; in 2D one cell deep each, as many as the budget needs), and slots in total.
   */
  readonly slotsPerSide: number;
  readonly layers: number;
  readonly cap: number;
  tree!: BlockTree;
  cur = 0;

  state: GPUTexture[][] = [];
  private prim: GPUTexture[] = [];
  private pred: GPUTexture[] = [];
  /** Burning: each texel's heating rate over the last step (r32float; one texel without burning). */
  private heat!: GPUTexture;
  /**
   * Species (species/species.ts), in atlases laid out like the state's with the groups of four
   * stacked along z, `speciesDepth` texels apart: rho X ping-pong with its own parity
   * `speciesCur`, the fractions with ghosts (SX) and half a step ahead (SH), and the face mass
   * fluxes the update records for them. Empty without species, which also turn per-block
   * timesteps off.
   */
  species: GPUTexture[] = [];
  speciesCur = 0;
  readonly speciesGroups: number;
  /**
   * Where the species set the equation of state (species/mixture.ts): each species' share of
   * the composition moments, and the moments of every cell in an atlas like the primitives',
   * ghosts included (amr/assemble.wgsl writes them all; the species passes write the leaves'
   * cells with rho X). Null and absent for a uniform composition.
   */
  get compCoeffs(): Moments[] | null {
    return problemCoefficients(this.cfg.problem, this.cfg.eosParams.gamma);
  }
  private comp?: GPUTexture;
  private speciesDepth = 0;
  private speciesX?: GPUTexture;
  private speciesHalf?: GPUTexture;
  private massFlux?: GPUBuffer;
  /**
   * The face mass flux records as the shaders bind them (amr/species_index.wgsl): MF (binding 7
   * in the species pass, 12 in the updates) the first window, MF_HI (32) the rest, and the
   * index of the first record past the first window. A mesh whose records fit in one binding
   * has the whole buffer as MF and no MF_HI: its shaders bind no more than they ever did (one
   * more storage buffer took the MHD update with self-gravity and time bins past the 8 a stage
   * may have on most devices).
   */
  massFluxWindows: { lo: Bindable; hi: Bindable | null; split: number } | null = null;
  /** The mf_load and mf_store slots (amr/species.wgsl, amr/species_on.wgsl, mhd/amr_update.wgsl). */
  private get mfSlots(): Record<string, string> {
    const w = this.massFluxWindows;
    if (!w?.hi) return {};
    const decl = (access: string) => `@group(0) @binding(32) var<storage, ${access}> MF_HI: array<f32>;\nconst MF_SPLIT: u32 = ${w.split}u;\n`;
    return {
      mf_load: decl('read') + 'fn mf_load(i: u32) -> f32 {\n  if (i < MF_SPLIT) { return MF[i]; }\n  return MF_HI[i - MF_SPLIT];\n}\n',
      mf_store: decl('read_write') + 'fn mf_store(i: u32, v: f32) {\n  if (i < MF_SPLIT) { MF[i] = v; } else { MF_HI[i - MF_SPLIT] = v; }\n}\n',
    };
  }
  private pSpecies: GPUComputePipeline[] = [];
  /** per pipeline (assemble, predict, move, regrid): [state parity][species parity] */
  private gSpecies: GPUBindGroup[][][] = [];
  /** Burning of fuel to ash (burn/burn.ts), over the whole atlas after the species pass; none without. */
  private burn?: Burner;
  private idx: GPUTexture;
  private params: GPUBuffer;
  private clock: GPUBuffer;
  /** A copy of the clock, for the timestep of a step not yet taken (previewDt). */
  private clockPreview: GPUBuffer;
  private gClockPreview!: GPUBindGroup;
  /** The clock for previewDt: leaves the driving alone (hydro/clock.wgsl ADVANCE_DRIVE). */
  private pClockPreview!: GPUComputePipeline;
  private bt: GPUBuffer;
  private listAll: GPUBuffer;
  private listLeaf: GPUBuffer;
  private flags: GPUBuffer;
  private flagStaging: GPUBuffer;
  private rg: GPUBuffer;
  /** one list per tree level, for the per-level gravity solve */
  private listLevel: GPUBuffer[] = [];
  private nLevel: number[] = [];
  /** gravity field atlas: acceleration and half-step potential */
  private gfield: GPUTexture;
  /** the point masses (gravity/bh.wgsl) */
  private bh: GPUBuffer;
  /** debris energies (debris/debris.ts), if the problem asks for them, and their census pass */
  private debrisBuf?: DebrisBuffers;
  private pCensus?: GPUComputePipeline;
  private nMasses = 0;
  private nOwnMasses = 0;
  private grav?: AmrGravity;
  /**
   * A spherical run's gravity (amr/sph_monopole.wgsl): the monopole of its gas and its point
   * masses at the origin, which are not particles. Its passes, their bind groups by parity (and
   * the field's over all blocks or, with per-block timesteps, those starting a step), its buffer
   * and its potential atlas.
   */
  private sphMono?: {
    rows: GPUComputePipeline; shells: GPUComputePipeline; enclose: GPUComputePipeline; field: GPUComputePipeline; fieldNow: GPUComputePipeline;
    gRows: GPUBindGroup[]; gShells: GPUBindGroup; gEnclose: GPUBindGroup; gField: GPUBindGroup; gFieldNow: GPUBindGroup[];
    buf: GPUBuffer; phi: GPUTexture;
  };
  /** a spherical run: the G M of its point masses, all at the origin */
  private centralGM = 0;
  private dummyPhi: GPUTexture;
  private redMp?: Multipoles;
  private redBh?: MassSums;
  private pBhDrift?: GPUComputePipeline;
  private pBhKick?: GPUComputePipeline;
  private pBhEval?: GPUComputePipeline;
  private gBhDrift?: GPUBindGroup;
  private gBhKick?: GPUBindGroup;
  private gBhEval?: GPUBindGroup;
  /** 2D: the field pass for point masses alone (a 3D run uses its gravity solver's) */
  private pMassField?: GPUComputePipeline;
  private gMassField?: GPUBindGroup;
  private pCorrect!: GPUComputePipeline;
  /** The largest speed of the initial conditions: the scale of the liquid's velocity refinement indicator (amr/flag.wgsl). */
  private vRef = 0;
  private gSpeciesFinal: GPUBindGroup[][] = [];
  /** Implicit pressure (SimConfig.pressure): the solve and its passes on the leaves (amr/pressure.ts). */
  prs?: AmrPressure;
  private gCorrect: GPUBindGroup[] = [];
  private brush: GPUBuffer;
  private pBrush!: GPUComputePipeline;
  private gBrush: GPUBindGroup[] = [];
  /** the grid holds solid or fixed cells, so the passes must look for them */
  private solids = false;
  private sources: GPUBuffer;
  private nSources = 0;
  /** counts re-meshes made outside the usual cycle; a refinement check from before one is dropped */
  private meshEpoch = 0;
  private rhoMean = 0;
  private staging: GPUBuffer;
  private stagingBusy = false;

  // Per-block timesteps (amr/bins_on.wgsl): bins and their scratch copy, the per-substep
  // block lists with their counters and dispatch sizes, and the passes that maintain them.
  private bins: GPUBuffer;
  private binsB: GPUBuffer;
  private listAct: GPUBuffer;
  private listUpd: GPUBuffer;
  private listEnd: GPUBuffer;
  /** Per-block timesteps: the leaves that only take a deposit this substep (amr/bins_lists.wgsl, LD). */
  private listDep: GPUBuffer;
  private binCnt: GPUBuffer;
  private binArgs: GPUBuffer;
  private pTick?: GPUComputePipeline;
  private pAdvance?: GPUComputePipeline;
  private gAdvance?: GPUBindGroup;
  private gCorrectB: GPUBindGroup[] = [];
  private gCommitL: GPUBindGroup[] = [];
  /** the point masses' reduction, in which a sink takes gas from the blocks finishing a step */
  private redBhB?: MassSums;
  private gBhKickB?: GPUBindGroup;
  /** 2D: the point masses' field over the blocks starting a step, and over all of them */
  private pMassFieldB?: GPUComputePipeline;
  private gMassFieldB: GPUBindGroup[] = [];
  private pSetDt?: GPUComputePipeline;
  private pBinsInit?: GPUComputePipeline;
  private pBinsDilate?: GPUComputePipeline;
  private pBinsRelax?: GPUComputePipeline;
  private pBinsLists?: GPUComputePipeline;
  private pBinsArgs?: GPUComputePipeline;
  private pCommit?: GPUComputePipeline;
  private gTick?: GPUBindGroup;
  private gSetDt?: GPUBindGroup;
  private gBinsInit?: GPUBindGroup;
  private gBinsDilate?: GPUBindGroup;
  private gBinsRelax: GPUBindGroup[] = [];
  private gBinsLists?: GPUBindGroup;
  private gBinsArgs?: GPUBindGroup;
  /** bind groups over the per-substep lists; [parity] where the state is bound */
  private gAssembleB: GPUBindGroup[] = [];
  private gPredictB?: GPUBindGroup;
  private gUpdateB: GPUBindGroup[] = [];
  private gCommit: GPUBindGroup[] = [];
  private gCommitB: GPUBindGroup[] = [];
  /** the bins the pipelines were built for, and whether the bin buffer is set up for cycles */
  private builtBins = 0;
  private binsPrimed: boolean | null = null;
  /** block updates made, and what a single timestep would have made (per-block timesteps) */
  private binCycles = 0;
  private binFullWork = 0;
  private binFirstWork = 0;
  // The automatic choice between cycles and single steps (see BIN_SHARE_MAX).
  private binStaging: GPUBuffer;
  private binPollBusy = false;
  private wantBinPoll: { first: number; full: number } | null = null;
  private binSeen = { updates: 0, single: 0 };
  private binEpoch = 0;
  private cyclesSincePoll = 0;
  /** single steps left before cycles are tried again, and how many the next pause will be */
  private binPause = BIN_START_STEPS;
  private binPauseNext = BIN_PAUSE_STEPS;
  /** share of block updates made in the last cycles measured (1 = no saving), or null */
  binShare: number | null = null;

  private pAssemble!: GPUComputePipeline;
  /** The hydro prediction and update: not built with a magnetic field, whose step is its own (mhd/amr_mhd.ts). */
  private pPredict!: GPUComputePipeline;
  private pUpdate!: GPUComputePipeline;
  private pClock!: GPUComputePipeline;
  private pFlag!: GPUComputePipeline;
  private pRegrid!: GPUComputePipeline;
  private gAssemble: GPUBindGroup[] = [];
  /** With a varying composition: the assembly also reads rho X, so [state parity][species parity]. */
  private gAssembleQ: GPUBindGroup[][] = [];
  private gPredict!: GPUBindGroup;
  private gUpdate: GPUBindGroup[] = [];
  private gRegrid: GPUBindGroup[] = [];
  private gClock!: GPUBindGroup;
  private gFlag!: GPUBindGroup;
  private redDt!: Reducer;
  /** the block list changed since the timestep records were last cleared */
  private dtStale = true;
  private redDiag!: Reducer;

  private nAll = 0;
  private nLeaf = 0;
  private stepsSinceFlag = 0;
  private wantMap = false;
  private flagInFlight = false;
  private flagWait: Promise<void> | null = null;
  private pending: RegridPlan | null = null;
  /**
   * Called as a re-mesh that a refinement check decided is applied (at the start of the batch
   * of steps it is encoded in), with the steps the tree took: a session log keeps them
   * (ui/session.ts), since the step it lands on depends on when the check was read back.
   */
  onRegrid?: (steps: AdaptStep[]) => void;
  /** Called after the uniform block is written (Simulation.onParams). */
  onParams?: () => void;
  /**
   * Off, the run makes no refinement checks of its own: a replay re-meshes where the log says
   * (replayRegrid()).
   */
  autoRegrid = true;
  /** total re-meshes applied, for the status line and tests */
  regrids = 0;
  /** the most recent refinement indicators, by slot (inspection only) */
  lastIndicators: Float32Array | null = null;
  /** Each leaf's indicator at the last re-mesh, by block (slots are reused as the mesh changes). */
  private indicatorOf = new WeakMap<Block, number>();

  get paramBuffer(): GPUBuffer {
    return this.params;
  }

  private constructor(gpu: Gpu, cfg: SimConfig, slots = 0) {
    const d = (this.device = gpu.device);
    this.cfg = cfg;
    // Before any pass is bound: the driving buffer is shared, bound where a pass declares it.
    writeDrive(d, cfg.driving, cfg.ndim, cfg.problem.box);
    this.speciesGroups = speciesGroups(cfg.problem);
    this.n = cfg.n;
    this.nz = cfg.ndim === 2 ? 1 : cfg.n;
    this.levels = cfg.amrLevels;
    this.n0 = cfg.n >> this.levels;
    if (this.n0 % BS !== 0 || this.n0 < BS) throw new Error(`the coarsest level needs a multiple of ${BS} cells per side (got ${this.n0})`);
    this.dx0 = cfg.problem.box / this.n0;
    const nr = this.n0 / BS;
    const roots = nr ** cfg.ndim;
    // Atlas size: room for the roots plus refinement, within what a texture can hold.
    const limit = Math.floor(Math.min(gpu.limits.maxTextureDimension3D, 2048) / PAD);
    // Default size: what refining everywhere would need (leaves plus their parents), within the
    // device's GPU memory budget (gpu/memory.ts: set by the user, or guessed: 2.5 GB on a
    // desktop, 13^3 blocks in 3D, 1 GB on a phone, less after a crash), the atlas being nearly
    // all of a run's memory. create() falls back to a smaller atlas if the GPU cannot hold that.
    const full = Math.ceil(roots * (1 << (cfg.ndim * this.levels)) * (cfg.ndim === 3 ? 8 / 7 : 4 / 3));
    const explicit = slots || cfg.amrBudget;
    let fit = Math.floor(memoryBudget() / AmrSim.blockBytesFor(cfg));
    // A magnetic field's flux register is one buffer of a few kB per block: no more blocks than it may hold.
    if (cfg.problem.mhd) fit = Math.min(fit, Math.floor(maxBinding(d) / registerBytes(cfg.ndim)));
    // With species, the face mass fluxes are one buffer of some 59 kB per block in 3D, bound as
    // two windows of at most a binding's size each (massFluxWindows).
    if (this.speciesGroups > 0) fit = Math.min(fit, Math.floor(massFluxMaxBytes(d) / speciesFaceBytes(cfg.ndim)));
    // Where the atlas textures cannot be render attachments, WebGPU zero-fills each on first use
    // through one staging buffer of its size, which may be no larger than a buffer may be (4 GiB
    // on NVIDIA; see tex3d in gpu/device.ts): a 35^3-block atlas's 5.5 GB textures lost the
    // device. Each atlas texture holds PAD^d texels of 16 bytes per block; the species' one
    // stacks its groups along z.
    if (!rendersTo3d(d)) fit = Math.min(fit, Math.floor(d.limits.maxBufferSize / (PAD ** cfg.ndim * 16 * Math.max(1, this.speciesGroups))));
    if (cfg.ndim === 3) {
      const auto = Math.max(6, Math.ceil(Math.cbrt(Math.max(full, roots * 2)) - 1e-9));
      // 3D passes over the blocks, self-gravity's multigrid's too, run in slabs along z
      // (amr/dispatch.ts), so the dispatch limit does not bound the blocks.
      this.slotsPerSide = Math.min(limit, explicit ? Math.ceil(Math.cbrt(explicit) - 1e-9) : auto, Math.max(1, Math.floor(Math.cbrt(fit) + 1e-9)));
      // With species, their atlas's stacks of groups must fit a texture's sides too (thirteen
      // species, four groups: 51^3 blocks).
      if (this.speciesGroups > 0) {
        const fits = (s: number) => Math.max(...speciesAtlasSize(s * PAD, s * PAD, s * PAD, this.speciesGroups).slice(1)) <= gpu.limits.maxTextureDimension3D;
        while (this.slotsPerSide > 1 && !fits(this.slotsPerSide)) this.slotsPerSide--;
      }
      this.layers = this.slotsPerSide;
    } else {
      const want = Math.min(explicit || Math.max(full, roots * 2, 144), Math.max(1, fit));
      this.layers = Math.ceil(want / limit ** 2);
      this.slotsPerSide = Math.min(limit, Math.floor(Math.sqrt(fit / this.layers) + 1e-9), Math.ceil(Math.sqrt(want / this.layers) - 1e-9));
    }
    this.cap = this.slotsPerSide ** 2 * this.layers;
    if (roots > this.cap) {
      throw new Error(
        fit < Infinity && roots > fit
          ? `${roots} root blocks need more GPU memory than this device allows (${Math.round(memoryBudget() / 2 ** 20)} MB): pick a lower resolution or more refinement levels`
          : `${roots} root blocks do not fit in an atlas of ${this.cap}`,
      );
    }
    noteAllocation(this.cap * AmrSim.blockBytesFor(cfg));

    const side = this.slotsPerSide * PAD;
    const t = (label: string) => tex3d(d, side, 'rgba32float', label, this.layers * (cfg.ndim === 2 ? 1 : PAD));
    for (let p = 0; p < 2; p++) this.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    this.prim = [t('W1'), t('W2')];
    this.pred = [t('PA'), t('PB')];
    const G = this.speciesGroups;
    if (G > 0) {
      this.speciesDepth = this.layers * (cfg.ndim === 2 ? 1 : PAD);
      // The groups stack along z, and past 2048 texels in further stacks along y (species/species.ts).
      const [, sy, sz] = speciesAtlasSize(side, side, this.speciesDepth, G);
      if (Math.max(sy, sz) > d.limits.maxTextureDimension3D) {
        throw new Error(`${cfg.problem.species!.length} species need an atlas ${sy} x ${sz} texels, more than this GPU allows (${d.limits.maxTextureDimension3D}): pick a lower resolution`);
      }
      const ts = (label: string) => tex3d(d, side, 'rgba32float', label, sz, sy);
      this.species = [ts('species 0'), ts('species 1')];
      this.speciesX = ts('species fractions');
      this.speciesHalf = ts('species half step');
      const bytes = this.cap * speciesFaceBytes(cfg.ndim);
      this.massFlux = d.createBuffer({ label: 'face mass fluxes', size: bytes, usage: GPUBufferUsage.STORAGE });
      const window = massFluxWindowBytes(d);
      this.massFluxWindows =
        bytes <= window
          ? { lo: this.massFlux, hi: null, split: 0xffffffff }
          : { lo: { buffer: this.massFlux, offset: 0, size: window }, hi: { buffer: this.massFlux, offset: window, size: bytes - window }, split: window / 4 };
      if (this.compCoeffs) this.comp = t('composition');
    }
    // Burning: each texel's heating rate over the last step (burn/burn.ts); else one texel of nothing.
    this.heat = burnSpec(cfg) ? tex3d(d, side, 'r32float', 'nuclear heating', this.layers * (cfg.ndim === 2 ? 1 : PAD)) : noHeat(d);
    const m = nr << this.levels;
    this.idx = d.createTexture({
      label: 'leaf index',
      size: [m, m, cfg.ndim === 3 ? m : 1],
      dimension: '3d',
      format: 'r32uint',
      // COPY_SRC: re-meshing a magnetic field reads the old index (mhd/amr_regrid.wgsl).
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clock = d.createBuffer({ size: CLOCK_BYTES, usage: storage });
    this.clockPreview = d.createBuffer({ size: CLOCK_BYTES, usage: storage });
    this.brush = d.createBuffer({ size: BRUSH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sources = d.createBuffer({ size: SOURCE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bt = d.createBuffer({ size: this.cap * BT_STRIDE * 4, usage: storage });
    this.listAll = d.createBuffer({ size: (this.cap + 1) * 4, usage: storage });
    this.listLeaf = d.createBuffer({ size: (this.cap + 1) * 4, usage: storage });
    for (let l = 0; l <= this.levels; l++) this.listLevel.push(d.createBuffer({ size: (this.cap + 1) * 4, usage: storage }));
    this.gfield = t('gravity field');
    this.bh = d.createBuffer({ size: MASS_FLOATS * 4, usage: storage });
    this.dummyPhi = d.createTexture({ size: [1, 1, 1], dimension: '3d', format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    this.flags = d.createBuffer({ size: this.cap * 4, usage: storage });
    this.rg = d.createBuffer({ size: this.cap * RG_STRIDE * 4, usage: storage });
    this.flagStaging = d.createBuffer({ size: this.cap * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const list = () => d.createBuffer({ size: (this.cap + 1) * 4, usage: storage });
    this.bins = list();
    this.binsB = list();
    this.listAct = list();
    this.listUpd = list();
    this.listEnd = list();
    this.listDep = list();
    this.binCnt = d.createBuffer({ size: 16, usage: storage });
    this.binArgs = d.createBuffer({ size: BIN_ARGS_BYTES, usage: storage | GPUBufferUsage.INDIRECT });
    this.binStaging = d.createBuffer({ size: BIN_ARGS_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.staging = d.createBuffer({ size: DIAG_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  /**
   * GPU bytes per atlas block: nine rgba32float textures (two state pairs, primitives,
   * predictor, gravity field) and two r32float ones (gravity's previous potential and its rate).
   */
  static blockBytes(ndim: 2 | 3, mhd = false, speciesGroups = 0, comp = false): number {
    // A magnetic field adds its own atlas textures and its flux register (mhd/amr_mhd.ts).
    // Species add four atlas textures of their groups (the fractions at both parities, at the
    // half step and as fractions), the composition's moments where they set the equation of
    // state, and their face mass fluxes: for the 13 network species, more than the rest of a
    // block together.
    // (Burning, which needs species, adds its heating rate: 4 bytes a texel, counted with them.)
    const texel = 9 * 16 + 2 * 4 + (mhd ? MHD_BLOCK_BYTES : 0) + speciesGroups * 4 * 16 + (speciesGroups > 0 ? 4 : 0) + (speciesGroups > 0 && comp ? 16 : 0);
    return PAD ** ndim * texel + (mhd ? registerBytes(ndim) : 0) + (speciesGroups > 0 ? speciesFaceBytes(ndim) : 0);
  }

  /** Bytes of one block of the atlas for a run's problem (blockBytes with its field and species). */
  static blockBytesFor(cfg: SimConfig): number {
    const groups = speciesGroups(cfg.problem);
    // Implicit pressure (amr/pressure.ts): six one-channel atlases.
    const implicit = implicitPressure(cfg) ? 6 * 4 * PAD ** cfg.ndim : 0;
    return AmrSim.blockBytes(cfg.ndim, cfg.problem.mhd !== undefined, groups, groups > 0 && problemCoefficients(cfg.problem, cfg.eosParams.gamma) !== null) + implicit;
  }

  /**
   * Build the simulation; if the GPU runs out of memory, try again with a smaller atlas.
   * @param opts.deferUpdate return before the update pass has compiled (the slowest by far:
   *   2.2 s of a 3.4 s build on a cold cache, measured for the 2D default). The run can be
   *   shown meanwhile; it is `preparing`, and takes no steps, until `compiled` resolves.
   */
  static async create(gpu: Gpu, cfg: SimConfig, opts: { deferUpdate?: boolean } = {}): Promise<AmrSim> {
    // The tabulated white dwarf equation of state needs its table on the device (eos/helmholtz.ts).
    if (cfg.eos === 'helmholtz') await prepareHelmholtz(gpu.device);
    const why = refusal(cfg) ?? (implicitPressure(cfg) && implicitBlocksUnsupported(cfg) ? `Implicit pressure: ${implicitBlocksUnsupported(cfg)}` : null);
    if (why) throw new Error(why);
    let lastTried: number | null = null;
    for (let slots = 0; ; ) {
      // A resource the GPU cannot make (out of memory, or larger than it allows one to be, which
      // is a validation error) fails the attempt, which is made again with fewer blocks. The
      // validation scope also keeps the errors that follow from it (views of a texture that was
      // never made) out of the page's error list.
      gpu.device.pushErrorScope('out-of-memory');
      gpu.device.pushErrorScope('validation');
      let sim: AmrSim | null = null;
      let failure: unknown = null;
      // The errors of making the resources, asked for once they are all made: an atlas the GPU
      // could not make is given up then, not after the initial conditions are set up and a star
      // settled on it (with gravity broken, all 30 rounds), which made each retry take minutes.
      let early: GPUError | null = null;
      const allocated = async () => {
        const inv = await gpu.device.popErrorScope();
        early = (await gpu.device.popErrorScope()) ?? inv;
        gpu.device.pushErrorScope('out-of-memory');
        gpu.device.pushErrorScope('validation');
        if (early) throw early;
      };
      try {
        sim = await AmrSim.build(gpu, cfg, slots, !!opts.deferUpdate, (s) => ((sim = s), allocated()));
        await gpu.device.queue.onSubmittedWorkDone();
      } catch (e) {
        if (e !== early) failure = e;
      }
      const later = await gpu.device.popErrorScope();
      const invalid: GPUError | null = early ?? later;
      const oom = early ?? (await gpu.device.popErrorScope()) ?? later;
      if (early) await gpu.device.popErrorScope(); // the out-of-memory scope, still open
      if (!oom && sim && !failure) return sim;
      const tried = sim?.cap ?? slots;
      sim?.destroy();
      const next = Math.floor(tried * 0.85 ** cfg.ndim);
      const roots = ((cfg.n >> cfg.amrLevels) / BS) ** cfg.ndim;
      // An atlas no smaller than the last one tried (a 3D atlas rounds a request up to a cube)
      // fails again the same way: give up rather than try it forever.
      const stuck = lastTried !== null && tried >= lastTried;
      lastTried = tried;
      if (!oom || !tried || next < roots * 2 || stuck) {
        throw failure ?? new Error(`not enough GPU memory for the adaptive mesh at this resolution${invalid ? ` (${firstLine(invalid.message)})` : ''}`);
      }
      console.warn(`adaptive mesh: ${invalid ? `could not make its resources (${firstLine(invalid.message)})` : 'out of GPU memory'} with ${tried} blocks, retrying with ${next}`);
      slots = next;
    }
  }

  /** `allocated`: called with the simulation once its resources are made, before it is set up; throws if they failed. */
  private static async build(gpu: Gpu, cfg: SimConfig, slots: number, deferUpdate: boolean, allocated: (sim: AmrSim) => Promise<void>): Promise<AmrSim> {
    setLoadStage('Allocating the mesh');
    const sim = new AmrSim(gpu, cfg, slots);
    setLoadStage(COMPILE_STAGE);
    compileSetup(['adaptive', slots, deferUpdate, cfg]);
    const d = sim.device;
    const v = (t: GPUTexture) => t.createView();
    const depth = cfg.ndim === 2 ? 1 : BS;
    // Every pipeline is asked for before any is awaited, so the browser compiles them all at
    // once (gpu/device.ts). The hydro passes, the slowest, start first; buildHydro below finds
    // them compiled or compiling.
    sim.hydroPipelines();
    // With species the indicator also reads their fractions, ghosts included (amr/species_assemble.wgsl).
    // With burning, also how fast it burns (burn/burn.ts).
    const burn = burnSpec(cfg);
    const flagSlots: Record<string, string> = { flag_species: flagSpeciesSlot(sim.speciesGroups, sim.speciesDepth) };
    // (The network's fronts are refined on the jumps in its species alone.)
    if (burn && !isNetworkBurn(burn)) Object.assign(flagSlots, { flag_burn: flagBurnSlot(burn, cfg.problem.species), eos: hydroSlots(cfg).eos });
    const spec = debrisSpec(cfg);
    if (spec) sim.debrisBuf = new DebrisBuffers(d, spec);
    // 3D: self-gravity machinery is built for every run, so gravity can be switched on later.
    // 2D: for an axisymmetric (R, z) run with self-gravity (its solver's passes in their 2D form).
    const flatGravity = cylindrical(cfg) && isSelfGravity(cfg.gravity);
    const gravity = async () => {
      const u1 = [0, 1].map((p) => v(sim.state[p][0]));
      sim.redMp = await Multipoles.create(d, [sim.cap * BS, BS, depth], flatGravity ? 'amr/multipole_cyl.wgsl' : 'amr/multipole.wgsl', u1.map((t) => ({ 0: sim.params, 1: t, 30: sim.bt })), 'amr multipole', cfg.symmetry);
      sim.grav = await AmrGravity.create({
        device: d, slotsPerSide: sim.slotsPerSide, flat: cfg.ndim === 2, layers: sim.layers, levels: sim.levels, params: sim.params, bt: sim.bt,
        listAll: sim.listAll, listLeaf: sim.listLeaf, listLevel: sim.listLevel, rg: sim.rg, clock: sim.clock, bh: sim.bh,
        multipole: sim.redMp.result, stateU1: [sim.state[0][0], sim.state[1][0]], gfield: sim.gfield,
        bins: sim.bins, listAct: sim.listAct, listUpd: sim.listUpd, binArgs: sim.binArgs,
      });
    };
    [sim.redDt, sim.pClock, sim.pFlag, sim.redBh, sim.pBhDrift, sim.pBhKick, sim.pBhEval, sim.pMassField, sim.pCensus] = await Promise.all([
      Reducer.create(
        d,
        [sim.cap * BS, BS, depth],
        [MAX, MAX],
        '',
        {},
        [],
        'amr timestep',
        [flatGroups(cfg.ndim, NG), sim.cap, 1], // one record per workgroup of the assembly pass
      ),
      computePipeline(d, assemble('hydro/clock.wgsl'), 'clock', { PER_DX: 1 }),
      computePipeline(d, assemble('amr/flag.wgsl', flagSlots), 'amr flag'),
      // Point masses, in 2D and 3D: their reduction over the leaf cells and their orbit passes.
      MassSums.create(
        d, MAX_MASSES, [sim.cap * BS, BS, depth], 'amr/bh_moment.wgsl', { bins: 'amr/bins_off.wgsl' },
        [0, 1].map((p) => ({ 0: sim.params, 1: v(sim.state[p][0]), 2: sim.bh, 3: v(sim.state[p][1]), 4: sim.clock, 30: sim.bt })), 'amr point masses',
      ),
      computePipeline(d, assemble('gravity/bh_drift.wgsl'), 'point mass drift'),
      computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass kick'),
      computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass accelerations', { LIVE: 0 }),
      // 2D without self-gravity, so no solver: the field atlas holds the point masses alone.
      cfg.ndim === 2 && !flatGravity ? computePipeline(d, assemble('amr/gravity/field.wgsl', FIELD_SLOTS), 'amr point mass field') : undefined,
      spec ? computePipeline(d, assemble('debris/census_amr.wgsl'), 'amr debris census') : undefined,
      cfg.ndim === 3 || flatGravity ? gravity() : undefined,
    ]);
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock, 3: sim.bh });
    sim.pClockPreview = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock preview', { PER_DX: 1, ADVANCE_DRIVE: 0 });
    sim.gClockPreview = bindGroup(d, sim.pClockPreview, { 0: sim.params, 1: sim.redDt.result, 2: sim.clockPreview, 3: sim.bh });
    if (cfg.problem.mhd) {
      const side = sim.slotsPerSide * PAD;
      sim.mhd = await AmrMhd.create({
        device: d, cfg, cap: sim.cap,
        texture: (label, format = 'rgba32float') => tex3d(d, side, format, label, sim.layers * (cfg.ndim === 2 ? 1 : PAD)),
        prim: sim.prim,
        params: sim.params, clock: sim.clock, bt: sim.bt, listLeaf: sim.listLeaf, idx: sim.idx, rg: sim.rg, brush: sim.brush,
        state: sim.state, half: sim.pred, rates: sim.redDt.first, slots: sim.slots, gfield: sim.gfield, bh: sim.bh, massFlux: sim.massFluxWindows ?? undefined,
        bins: { A: sim.bins, LL: sim.listEnd, LU: sim.listUpd, LD: sim.listDep, args: sim.binArgs },
      });
    }
    // With a magnetic field the refinement indicator reads its assembly's primitives (the same layout).
    const flagW = sim.mhd ? sim.mhd.primitives : sim.prim;
    sim.gFlag = bindGroup(d, sim.pFlag, {
      0: sim.params, 1: v(flagW[0]), 2: v(flagW[1]), 3: sim.flags, 4: sim.bh, 30: sim.bt, 31: sim.listLeaf,
      ...((sim.speciesX ? { 5: v(sim.speciesX) } : {}) as Record<number, GPUTextureView>),
    });
    sim.gBhDrift = bindGroup(d, sim.pBhDrift, { 0: sim.params, 1: sim.clock, 2: sim.bh });
    const kick = { 0: sim.params, 1: sim.clock, 2: sim.redBh.result, 3: sim.bh };
    sim.gBhKick = bindGroup(d, sim.pBhKick, kick);
    sim.gBhEval = bindGroup(d, sim.pBhEval, kick);
    if (spherical(cfg) && (isSelfGravity(cfg.gravity) || (cfg.problem.masses?.(cfg.params, cfg.G, cfg.problem.box / cfg.n, cfg.ndim).length ?? 0) > 0)) await sim.buildMonopole();
    if (sim.pMassField) {
      // The potential bindings are never read (there is none), but the pass declares them.
      sim.gMassField = bindGroup(d, sim.pMassField, {
        0: sim.params, 1: v(sim.dummyPhi), 2: v(sim.dummyPhi), 3: v(sim.gfield), 5: sim.clock, 30: sim.bt, 31: sim.listAll,
      });
    }
    await sim.buildHydro(deferUpdate);
    await allocated(sim);
    await showLoadStage(INIT_STAGE);
    await sim.reset();
    endLoadStage(INIT_STAGE);
    await sim.settled;
    return sim;
  }

  /** Bins of per-block timesteps in use (0 = off). */
  private get timeBins(): number {
    // Species are moved with one timestep for all blocks (amr/species.wgsl).
    if (this.speciesGroups > 0) return 0;
    return Math.max(0, Math.min(MAX_TIME_BINS, Math.floor(this.cfg.timeBins || 0)));
  }
  /** Per-block timesteps are stepping the run (they are on, and not paused for saving too little). */
  private get binsLive(): boolean {
    return this.builtBins > 0 && this.binPause === 0;
  }

  private get slots() {
    const on = this.timeBins > 0;
    return {
      ...hydroSlots(this.cfg),
      bins: on ? 'amr/bins_on.wgsl' : 'amr/bins_off.wgsl',
      mid: on ? 'amr/mid_on.wgsl' : 'amr/mid_off.wgsl',
      species: this.speciesGroups > 0 ? 'amr/species_on.wgsl' : 'amr/species_off.wgsl',
      comp_src: compSrcSlot(this.compCoeffs, this.speciesDepth),
      ...this.mfSlots,
    };
  }

  /** The composition moments for a pass that reads them (binding 27), if they vary. */
  private get compIn(): Record<number, GPUTextureView> {
    return this.comp ? { 27: this.comp.createView() } : {};
  }

  /** The assembly's bind group for the state of parity p (and the species as they are now). */
  private assembleGroup(p: number): GPUBindGroup {
    return this.comp ? this.gAssembleQ[p][this.speciesCur] : this.gAssemble[p];
  }

  /**
   * The update's face-flux tile (amr/tile.wgsl): on if it fits in the device's workgroup
   * memory. 576 faces in 2D, 896 in 3D, of 32 bytes each plus 16 for the viscous part; 3D with
   * viscosity (43 KB) gathers instead.
   */
  private tileConstants(): Record<string, number> {
    const flat = this.cfg.ndim === 2;
    const bytes = (flat ? 576 : 896) * (this.cfg.viscosity > 0 ? 48 : 32);
    const fits = bytes <= this.device.limits.maxComputeWorkgroupStorageSize;
    return { TILE: fits ? 1 : 0, TILE_2D: flat ? 1 : 0 };
  }

  /**
   * The hydro passes' pipelines for the current settings (compiled once, see gpu/device.ts). A
   * magnetized run steps with the MHD passes alone: no prediction or update (the update, the
   * slowest pass of all to compile, would only hold its load up).
   */
  private hydroPipelines() {
    const d = this.device;
    const slots = this.slots;
    const hydro = !this.cfg.problem.mhd;
    return [
      computePipeline(d, assemble('amr/assemble.wgsl', slots), 'amr assemble'),
      hydro ? computePipeline(d, assemble('amr/predict.wgsl', slots), 'amr predict') : null,
      hydro ? computePipeline(d, assemble('amr/update.wgsl', slots), 'amr update', this.tileConstants()) : null,
      computePipeline(d, assemble('amr/regrid.wgsl', slots), 'amr regrid'),
      // With a field the correction keeps its energy out of what it rebuilds (MHD).
      computePipeline(d, assemble('amr/correct.wgsl', slots), 'amr gravity correct', { MHD: this.cfg.problem.mhd ? 1 : 0 }),
      computePipeline(d, assemble('amr/brush.wgsl', slots), 'amr brush'),
    ] as const;
  }

  /**
   * (Re)build every pipeline that depends on the EOS, Riemann solver or limiter. Deferred, the
   * update pass is bound once it has compiled, and steps wait for it (see `create`).
   */
  async buildHydro(deferUpdate = false) {
    const epoch = ++this.hydroEpoch;
    const d = this.device;
    const slots = this.slots;
    const v = (t: GPUTexture) => t.createView();
    // The diagnostics' reduction compiles alongside the hydro passes.
    const diag = Reducer.create(
      d,
      [this.cap * BS, BS, this.cfg.ndim === 2 ? 1 : BS],
      DIAG_OPS,
      'amr/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.grav?.phi ?? this.sphMono?.phi ?? this.dummyPhi), 4: v(this.heat), 30: this.bt, ...this.compIn })),
      'amr diagnostics',
    );
    diag.catch(() => {}); // a failure is reported where it is awaited, below
    const [assembleP, predictP, updateP, regridP, correctP, brushP] = this.hydroPipelines();
    // Not deferred, the update pass changes over together with the others.
    let pUpdate: GPUComputePipeline | null | undefined;
    let pPredict: GPUComputePipeline | null;
    [this.pAssemble, pPredict, this.pRegrid, this.pCorrect, this.pBrush, pUpdate] = await Promise.all([
      assembleP, predictP, regridP, correctP, brushP, deferUpdate ? undefined : updateP,
    ]);
    if (pPredict) this.pPredict = pPredict;
    if (implicitPressure(this.cfg)) {
      const c = this.cfg;
      const kind = (b: number) => (b === PERIODIC ? PF_PERIODIC : b === 1 || b === 3 ? PF_OPEN : PF_CLOSED);
      this.prs ??= await AmrPressure.create({
        device: d, params: this.params, clock: this.clock, bt: this.bt, listLeaf: this.listLeaf, listLevel: this.listLevel, levels: this.levels,
        state: this.state, prim: this.prim[0], slots,
        rootDims: [this.n0, this.n0, c.ndim === 2 ? 1 : this.n0], dx0: this.dx0, lo: c.bcLo.map(kind), hi: c.bcHi.map(kind),
        interior: (pass, count) => this.interior(pass, count),
      });
      this.prs.setTolerance(c.pressureTol);
    } else {
      this.prs?.destroy();
      this.prs = undefined;
    }
    this.gBrush = [0, 1].map((p) =>
      bindGroup(d, this.pBrush, {
        0: this.params, 1: this.brush, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]),
        30: this.bt, 31: this.listLeaf, ...this.compIn,
      }),
    );
    const K = (this.builtBins = this.timeBins);
    await this.mhd?.setBins(K > 0, this.cfg, slots);
    if (K > 0) await this.buildBins(K);
    // Off, every block is in bin 0, as the display's time-bin field shows it (amr/field.wgsl).
    else d.queue.writeBuffer(this.bins, 0, new Uint32Array(this.cap + 1));
    // With per-block timesteps the passes also read the bins, and the assembly the half-step
    // predictions of blocks part way through a step.
    const bin: Record<number, Bindable> = K > 0 ? { 29: this.bins! } : {};
    const mid: Record<number, Bindable> = K > 0 ? { 7: v(this.pred[0]), 8: v(this.pred[1]) } : {};
    const predict = (list: GPUBuffer) =>
      bindGroup(d, this.pPredict, {
        0: this.params, 1: this.clock, 2: v(this.prim[0]), 3: v(this.prim[1]), 4: v(this.gfield), 5: v(this.pred[0]), 6: v(this.pred[1]), 20: this.bh, 30: this.bt, 31: list, ...bin,
        ...this.compIn,
      });
    const assembly = (p: number, list: GPUBuffer, q = -1) =>
      bindGroup(d, this.pAssemble, {
        0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]),
        5: v(this.gfield), 6: this.redDt.first, 20: this.bh, 30: this.bt, 31: list, ...bin, ...mid,
        ...((q >= 0 ? { 25: v(this.species[q]), 26: v(this.comp!) } : {}) as Record<number, GPUTextureView>),
      });
    if (pPredict) this.gPredict = predict(this.listLeaf);
    this.gAssemble = [];
    this.gRegrid = [];
    this.gCorrect = [];
    this.gAssembleB = [];
    this.gCommit = [];
    this.gCommitB = [];
    this.gCorrectB = [];
    this.gCommitL = [];
    if (K > 0 && pPredict) this.gPredictB = predict(this.listEnd);
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // Second half kick: from the updated state (the other parity) back into this one.
      const correct = (list: GPUBuffer) =>
        bindGroup(d, this.pCorrect, {
          0: this.params, 1: this.clock, 2: v(this.state[o][0]), 3: v(this.state[o][1]), 4: v(this.gfield),
          5: v(this.state[p][0]), 6: v(this.state[p][1]), 7: this.bh, 30: this.bt, 31: list, ...bin, ...this.compIn,
          8: v(this.mhd?.faces[o] ?? this.noFaces()),
        });
      if (K > 0) {
        this.gAssembleB.push(assembly(p, this.listAct));
        // Back from the other parity into this one.
        const commit = (list: GPUBuffer) =>
          bindGroup(d, this.pCommit!, { 0: this.params, 1: v(this.state[o][0]), 2: v(this.state[o][1]), 3: v(this.state[p][0]), 4: v(this.state[p][1]), 31: list });
        this.gCommit.push(commit(this.listLeaf));
        this.gCommitB.push(commit(this.listUpd));
        this.gCommitL.push(commit(this.listEnd));
        this.gCorrectB.push(correct(this.listEnd));
      }
      this.gCorrect.push(correct(this.listLeaf));
      if (this.comp) this.gAssembleQ[p] = [0, 1].map((q) => assembly(p, this.listAll, q));
      else this.gAssemble.push(assembly(p, this.listAll));
      this.gRegrid.push(
        bindGroup(d, this.pRegrid, {
          0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]),
          3: v((this.mhd?.primitives ?? this.prim)[0]), 4: v((this.mhd?.primitives ?? this.prim)[1]),
          5: v(this.state[o][0]), 6: v(this.state[o][1]), 7: this.rg, 30: this.bt, 31: this.listLeaf,
        }),
      );
    }
    const bindUpdate = (pipe: GPUComputePipeline) => {
      if (epoch !== this.hydroEpoch) return; // rebuilt since
      this.pUpdate = pipe;
      this.gUpdate = [0, 1].map((p) => this.updateGroup(pipe, p));
      this.gUpdateB = K > 0 ? [0, 1].map((p) => this.updateGroup(pipe, p, this.listUpd)) : [];
      this.compiling = false;
    };
    if (pUpdate) bindUpdate(pUpdate);
    if (this.speciesGroups > 0) await this.buildSpecies();
    this.redDiag = await diag;
    if (!deferUpdate || !updateP) {
      this.compiled = Promise.resolve();
      this.compiling = false;
      return;
    }
    this.compiling = true;
    this.compiled = updateP.then(bindUpdate, (e) => {
      // Nothing awaits a deferred build: say what went wrong where errors are shown.
      if (epoch === this.hydroEpoch) globalThis.dispatchEvent?.(new CustomEvent('shaderous-gpu-error', { detail: `The update pass failed to compile: ${(e as Error)?.message ?? e}` }));
      throw e;
    });
    this.compiled.catch(() => {});
  }

  /** Pipelines and bind groups of the species passes (amr/species_*.wgsl). */
  private async buildSpecies() {
    const d = this.device;
    const v = (t: GPUTexture) => t.createView();
    const ns = { species_n: `const NG: u32 = ${this.speciesGroups}u;\nconst SP_DEPTH: i32 = ${this.speciesDepth};\n`, comp: hydroSlots(this.cfg).comp, ...this.mfSlots };
    const kout = (this.comp ? { 26: v(this.comp) } : {}) as Record<number, GPUTextureView>;
    const kflag = (this.comp ? { 24: v(this.pred[1]) } : {}) as Record<number, GPUTextureView>;
    this.pSpecies = await Promise.all(
      ['assemble', 'predict', '', 'regrid'].map((k) =>
        computePipeline(d, assemble(k ? `amr/species_${k}.wgsl` : 'amr/species.wgsl', ns), `amr species ${k || 'move'}`),
      ),
    );
    const [pA, pP, pM, pR] = this.pSpecies;
    const sx = v(this.speciesX!);
    const sh = v(this.speciesHalf!);
    const per = (f: (p: number, q: number) => GPUBindGroup) => [0, 1].map((p) => [0, 1].map((q) => f(p, q)));
    this.gSpecies = [
      per((_p, q) => bindGroup(d, pA, { 0: this.params, 3: v(this.species[q]), 4: v(this.speciesX!), 30: this.bt, 31: this.listLeaf })),
      // The velocities with their ghosts: the hydro assembly's, or with a field the MHD
      // assembly's (the same layout; as the species pass runs after the step, the half step's).
      per(() => bindGroup(d, pP, { 0: this.params, 1: this.clock, 2: v(this.mhd?.prim ?? this.prim[0]), 4: sx, 5: v(this.speciesHalf!), 30: this.bt, 31: this.listLeaf })),
      per((p, q) =>
        bindGroup(d, pM, {
          0: this.params, 1: this.clock, 3: v(this.state[p][1]), 4: v(this.state[1 - p][0]), 5: v(this.species[q]), 6: v(this.species[1 - q]),
          7: this.massFluxWindows!.lo, ...((this.massFluxWindows!.hi ? { 32: this.massFluxWindows!.hi } : {}) as Record<number, Bindable>), 8: sx, 9: sh, 30: this.bt, 31: this.listLeaf, ...kout, ...kflag,
        }),
      ),
      // Re-meshing goes from the state of parity p to the other.
      per((p, q) => bindGroup(d, pR, { 0: this.params, 2: v(this.species[q]), 4: v(this.state[1 - p][0]), 6: v(this.species[1 - q]), 7: this.rg, 31: this.listLeaf, ...kout })),
    ];
    // With implicit pressure the density is final only after the correction, back in parity p:
    // the species are moved then, and scaled to that density (as on the uniform grid).
    this.gSpeciesFinal = per((p, q) =>
      bindGroup(d, pM, {
        0: this.params, 1: this.clock, 3: v(this.state[p][1]), 4: v(this.state[p][0]), 5: v(this.species[q]), 6: v(this.species[1 - q]),
        7: this.massFluxWindows!.lo, ...((this.massFluxWindows!.hi ? { 32: this.massFluxWindows!.hi } : {}) as Record<number, Bindable>), 8: sx, 9: sh, 30: this.bt, 31: this.listLeaf, ...kout, ...kflag,
      }),
    );
    this.burn?.destroy();
    this.burn = undefined;
    const burn = burnSpec(this.cfg);
    if (burn) {
      const t = this.state[0][0];
      // Pointwise, copying what it does not burn: the network's pass over the leaves' interior
      // cells (burnDispatch), the one-reaction pass over every texel of the atlas.
      this.burn = await createBurnPass(d, burn, { ...this.cfg.problem, units: problemUnits(this.cfg.problem, this.cfg.params) }, {
        dims: [t.width, t.height, t.depthOrArrayLayers], stride: this.speciesDepth, groups: this.speciesGroups, slots: hydroSlots(this.cfg),
        params: this.params, clock: this.clock, state: this.state, species: this.species, pb: this.pred[1], heat: this.heat, comp: this.comp, blockTable: this.bt,
        leaves: this.listLeaf,
      });
    }
  }

  /** Burn the state as it is now into the other parities (burn/burn.ts), in a pass of its own. */
  private encodeBurn(enc: GPUCommandEncoder) {
    if (!this.burn) return;
    const pass = beginPass(enc);
    this.burn.encode(pass, this.cur, this.speciesCur, (p) => this.burnDispatch(p, this.nLeaf));
    // With a field: the burnt state is in the other parity, and the leaves' faces follow it.
    this.mhd?.encodeCommitFaces(pass, 1 - this.cur);
    pass.end();
    this.cur = 1 - this.cur;
    this.speciesCur = 1 - this.speciesCur;
  }

  /**
   * The interior cells of the first `count` blocks of the leaf list, in workgroups of 4 x 4 x 4
   * (hydro/burn_net.wgsl), laid out as interior() lays them out (amr/list.wgsl inv_interior).
   */
  private burnDispatch(pass: GPUComputePassEncoder, count: number) {
    const w = BS / 4;
    if (this.cfg.ndim === 2) {
      const q = Math.ceil(count / 4);
      pass.dispatchWorkgroups(Math.min(q, SLAB_QUADS) * w, w, Math.ceil(q / SLAB_QUADS));
    } else pass.dispatchWorkgroups(Math.min(count, SLAB_BLOCKS) * w, w, w * Math.ceil(count / SLAB_BLOCKS));
  }

  /** Move the species through the step just updated from parity p (after the update's dispatch). */
  private encodeSpecies(pass: GPUComputePassEncoder, p: number) {
    if (!this.speciesGroups) return;
    const q = this.speciesCur;
    const [pA, pP, pM] = this.pSpecies;
    pass.setPipeline(pA);
    pass.setBindGroup(0, this.gSpecies[0][p][q]);
    this.flat(pass, this.nLeaf, NG);
    pass.setPipeline(pP);
    pass.setBindGroup(0, this.gSpecies[1][p][q]);
    this.flat(pass, this.nLeaf, 1);
    // (With implicit pressure the move waits for the correction: encodeSpeciesMove.)
    if (this.prs) return;
    pass.setPipeline(pM);
    pass.setBindGroup(0, this.gSpecies[2][p][q]);
    this.interior(pass, this.nLeaf);
    this.speciesCur = 1 - q;
  }

  /** Implicit pressure: the species' move, after the correction has put the final state in parity p. */
  private encodeSpeciesMove(pass: GPUComputePassEncoder, p: number) {
    if (!this.speciesGroups) return;
    const q = this.speciesCur;
    pass.setPipeline(this.pSpecies[2]);
    pass.setBindGroup(0, this.gSpeciesFinal[p][q]);
    this.interior(pass, this.nLeaf);
    this.speciesCur = 1 - q;
  }

  private updateGroup(pipe: GPUComputePipeline, p: number, list = this.listLeaf): GPUBindGroup {
    const v = (t: GPUTexture) => t.createView();
    const o = 1 - p;
    const bin: Record<number, Bindable> = this.builtBins > 0 ? { 29: this.bins! } : {};
    return bindGroup(this.device, pipe, {
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 4: v(this.prim[0]), 5: v(this.prim[1]), 6: v(this.gfield),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 11: this.sources, 20: this.bh, 30: this.bt, 31: list,
      ...bin,
      ...this.compIn,
      ...((this.massFluxWindows ? { 12: this.massFluxWindows.lo, ...(this.massFluxWindows.hi ? { 32: this.massFluxWindows.hi } : {}) } : {}) as Record<number, Bindable>),
      ...(this.debrisBuf?.bindings(this.bh) ?? {}),
    });
  }

  /** Buffers, pipelines and bind groups of per-block timesteps, for cycles of 2^K substeps. */
  private async buildBins(K: number) {
    const d = this.device;
    const cycle = 1 << K;
    [this.pTick, this.pAdvance, this.pSetDt, this.pBinsInit, this.pBinsDilate, this.pBinsRelax, this.pBinsLists, this.pBinsArgs, this.pCommit] = await Promise.all([
      computePipeline(d, assemble('amr/bins_tick.wgsl'), 'bins clock', { MODE: 0, CYCLE: cycle }),
      computePipeline(d, assemble('amr/bins_tick.wgsl'), 'bins substep', { MODE: 1, CYCLE: cycle }),
      computePipeline(d, assemble('hydro/clock.wgsl'), 'bins timestep', { PER_DX: 1, BANK: 0, CYCLE: cycle }),
      computePipeline(d, assemble('amr/bins_init.wgsl'), 'bins init', { KMAX: K, ASM_WG: flatGroups(this.cfg.ndim, NG) }),
      // With a magnetic field, blocks sharing only an edge are neighbors too (amr/bins_edges.wgsl).
      computePipeline(d, assemble('amr/bins_relax.wgsl'), 'bins dilate', { KMAX: K, DILATE: 1, EDGES: this.mhd ? 1 : 0 }),
      computePipeline(d, assemble('amr/bins_relax.wgsl'), 'bins relax', { KMAX: K, EDGES: this.mhd ? 1 : 0 }),
      computePipeline(d, assemble('amr/bins_lists.wgsl'), 'bins lists', { EDGES: this.mhd ? 1 : 0 }),
      computePipeline(d, assemble('amr/bins_args.wgsl'), 'bins dispatch sizes'),
      computePipeline(d, assemble('amr/commit.wgsl'), 'amr commit'),
    ]);
    const A = this.bins;
    const B = this.binsB;
    // 0: Kahan summation of the clock reads P (opaque() in hydro/common.wgsl), in both modes.
    this.gTick = bindGroup(d, this.pTick, { 0: this.params, 1: this.clock, 2: A, 3: this.binCnt });
    this.gAdvance = bindGroup(d, this.pAdvance, { 0: this.params, 1: this.clock, 2: A, 3: this.binCnt });
    await this.grav?.setBins();
    const v = (t: GPUTexture) => t.createView();
    this.redBhB ??= await MassSums.create(
      d, MAX_MASSES, [this.cap * BS, BS, this.cfg.ndim === 2 ? 1 : BS], 'amr/bh_moment.wgsl', { bins: 'amr/bins_on.wgsl' },
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: this.bh, 3: v(this.state[p][1]), 4: this.clock, 29: A, 30: this.bt })), 'amr point masses (bins)',
    );
    this.gBhKickB = bindGroup(d, this.pBhKick!, { 0: this.params, 1: this.clock, 2: this.redBhB.result, 3: this.bh });
    if (!this.grav && !this.pMassFieldB) {
      // 2D: no potential (the bindings are declared but never read).
      this.pMassFieldB = await computePipeline(d, assemble('amr/gravity/field.wgsl', FIELD_SLOTS_BINS), 'amr point mass field (bins)');
      this.gMassFieldB = [this.listAct, this.listAll].map((list) =>
        bindGroup(d, this.pMassFieldB!, { 0: this.params, 1: v(this.dummyPhi), 3: v(this.gfield), 5: this.clock, 6: v(this.dummyPhi), 29: A, 30: this.bt, 31: list }),
      );
    }
    this.gSetDt = bindGroup(d, this.pSetDt, { 0: this.params, 1: this.redDt.result, 2: this.clock, 3: this.bh });
    // The rates come one row per block of the assembly's list: with a field, the leaves (parents
    // are set from their children by the sweeps that follow).
    this.gBinsInit = bindGroup(d, this.pBinsInit, { 0: this.params, 1: this.clock, 2: this.redDt.first, 3: A, 30: this.bt, 31: this.mhd ? this.listLeaf : this.listAll });
    const idx = v(this.idx);
    this.gBinsDilate = bindGroup(d, this.pBinsDilate, { 0: this.params, 1: A, 2: B, 28: idx, 30: this.bt, 31: this.listAll });
    this.gBinsRelax = [
      bindGroup(d, this.pBinsRelax, { 0: this.params, 1: B, 2: A, 28: idx, 30: this.bt, 31: this.listAll }),
      bindGroup(d, this.pBinsRelax, { 0: this.params, 1: A, 2: B, 28: idx, 30: this.bt, 31: this.listAll }),
    ];
    this.gBinsLists = bindGroup(d, this.pBinsLists, { 0: this.params, 1: this.listAct, 2: this.listUpd, 3: this.binCnt, 4: this.listEnd, 5: this.listDep, 28: idx, 29: A, 30: this.bt, 31: this.listAll });
    this.gBinsArgs = bindGroup(d, this.pBinsArgs, { 0: this.params, 1: this.listAct, 2: this.listUpd, 3: this.binCnt, 4: this.binArgs, 5: this.listEnd, 6: this.listDep });
    this.binsPrimed = null;
    // What was measured with other bins says nothing about these.
    this.binPause = BIN_START_STEPS;
    this.binPauseNext = BIN_PAUSE_STEPS;
    this.binShare = null;
  }

  /**
   * Whenever the run changes between cycles and single steps (point masses come or go), or
   * starts: clear the bins, which makes every block step every time with the clock's dt until
   * a cycle assigns them, and drop the history of the potential.
   */
  private primeBins(enc: GPUCommandEncoder) {
    if (this.builtBins === 0) return;
    const live = this.binsLive;
    if (this.binsPrimed === live) return;
    this.device.queue.writeBuffer(this.bins, 0, new Uint32Array(this.cap + 1));
    if (this.selfGravity) {
      const pass = beginPass(enc);
      this.grav!.encodeForget(pass, this.nAll);
      pass.end();
    }
    this.binsPrimed = live;
  }

  /**
   * One cycle of per-block timesteps: 2^K substeps of the smallest step, after which every
   * block is at the same time again. On the first substep every block steps, with the passes
   * and lists of a single timestep; in between come the timestep and the bins, from the rates
   * the assembly has just measured. On the others the blocks that start a step, and those
   * owed the second half of a faster neighbor's fluxes, are gathered into lists on the GPU
   * and the same passes run over those. The state stays in one parity: the update writes the
   * other, and a copy brings the blocks that stepped back.
   */
  private encodeCycle(enc: GPUCommandEncoder) {
    const p = this.cur;
    const o = 1 - p;
    const K = this.builtBins;
    const n = 1 << K;
    const blocks = Math.ceil(this.nAll / 64);
    const g = this.selfGravity ? this.grav! : null;
    const field = this.fieldOn;
    const run = (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, group: GPUBindGroup) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
    };
    // The field atlas for every block (count >= 0) or for the blocks starting a step (-1).
    const fieldB = (pass: GPUComputePassEncoder, count: number) => {
      if (this.grav) return this.grav.encodeFieldB(pass, count);
      if (this.sphMono) {
        // A spherical run: the monopole of the leaves as they stand, the potential now (the
        // blocks' half steps differ).
        this.encodeMonopole(pass, p);
        run(pass, this.sphMono.fieldNow, this.sphMono.gFieldNow[count < 0 ? 0 : 1]);
        if (count < 0) pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(0));
        else this.padded(pass, count);
        return;
      }
      run(pass, this.pMassFieldB!, this.gMassFieldB[count < 0 ? 0 : 1]);
      if (count < 0) pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(0));
      else this.padded(pass, count);
    };
    // The point masses move with the smallest step, every substep.
    const stepMasses = (pass: GPUComputePassEncoder) => {
      run(pass, this.pBhDrift!, this.gBhDrift!);
      pass.dispatchWorkgroups(1);
      this.redBhB!.encode(pass, p, this.nMasses);
      run(pass, this.pBhKick!, this.gBhKickB!);
      pass.dispatchWorkgroups(1);
    };
    // The timestep from the rates the assembly measured, then the bins.
    const setBins = (pass: GPUComputePassEncoder) => {
      this.redDt.encode(pass);
      run(pass, this.pSetDt!, this.gSetDt!);
      pass.dispatchWorkgroups(1);
      run(pass, this.pBinsInit!, this.gBinsInit!);
      pass.dispatchWorkgroups(blocks);
      run(pass, this.pBinsDilate!, this.gBinsDilate!);
      pass.dispatchWorkgroups(blocks);
      // An odd number of sweeps, so the result lands back in the buffer the passes read.
      // Each rule moves a bin one block per sweep, and no chain is longer than the number
      // of bins plus the depth of the tree.
      const sweeps = (K + this.levels + 2) | 1;
      for (let i = 0; i < sweeps; i++) {
        run(pass, this.pBinsRelax!, this.gBinsRelax[i % 2]);
        pass.dispatchWorkgroups(blocks);
      }
    };
    const mhd = this.mhd;
    for (let s = 0; s < n; s++) {
      const pass = beginPass(enc);
      run(pass, this.pTick!, this.gTick!);
      pass.dispatchWorkgroups(1);
      if (mhd) {
        // A magnetic field (mhd/amr_mhd.ts): every leaf steps on the first substep, the leaves
        // starting a step and those owed a deposit on the others; then cells and faces back.
        // Gravity as for the gas alone: each block's own field once the bins are known, the
        // source of the potential from the cells just committed, and the closing kick below
        // (which reads the faces in this parity, where the commit has left them, for the
        // field's energy, and changes none of them).
        if (s === 0) {
          mhd.encodeFull(pass, p, () => this.flat(pass, this.nLeaf, NG), () => {
            setBins(pass);
            if (field) fieldB(pass, this.nAll);
          });
          run(pass, this.pCommit!, this.gCommit[p]);
          this.interior(pass, this.nLeaf);
          g?.encodeSource(pass, p, this.nLevel, this.nLeaf);
        } else {
          mhd.encodeSubstep(pass, p);
          run(pass, this.pCommit!, this.gCommitB[p]);
          pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(1));
          g?.encodeSourceB(pass, p);
        }
      } else if (s === 0) {
        run(pass, this.pAssemble, this.assembleGroup(p));
        this.flat(pass, this.nAll, NG);
        setBins(pass);
        // The half-step potential is each block's own: again, now that the bins are known.
        if (field) fieldB(pass, this.nAll);
        run(pass, this.pPredict, this.gPredict);
        this.flat(pass, this.nLeaf, 1);
        run(pass, this.pUpdate, this.gUpdate[p]);
        this.interior(pass, this.nLeaf);
        run(pass, this.pCommit!, this.gCommit[p]);
        this.interior(pass, this.nLeaf);
        g?.encodeSource(pass, p, this.nLevel, this.nLeaf);
      } else {
        run(pass, this.pAssemble, this.gAssembleB[p]);
        pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(14));
        run(pass, this.pPredict, this.gPredictB!);
        pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(13));
        run(pass, this.pUpdate, this.gUpdateB[p]);
        pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(1));
        run(pass, this.pCommit!, this.gCommitB[p]);
        pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(1));
        g?.encodeSourceB(pass, p);
      }
      // On to the next substep: the blocks that start a step then are the ones that have
      // just finished one. At the end of the cycle that is every block.
      run(pass, this.pAdvance!, this.gAdvance!);
      pass.dispatchWorkgroups(1);
      const last = s === n - 1;
      if (!last) {
        run(pass, this.pBinsLists!, this.gBinsLists!);
        pass.dispatchWorkgroups(blocks);
        run(pass, this.pBinsArgs!, this.gBinsArgs!);
        pass.dispatchWorkgroups(1);
      }
      if (field) {
        // The field for the time now reached, where it is needed: the blocks finishing a step
        // get its closing kick from it, and start their next step in it.
        if (g && last) {
          // Isolated boundaries: the multipole moments of the gas, once per cycle. They set
          // the potential at the edge of the box, which changes slowly. (Every substep was
          // tried: it costs a pass over every cell and changes the energy error by nothing.)
          if (this.cfg.gravity === 'isolated') this.redMp!.encode(pass, p);
          g.encodeSolve(pass, this.nLevel, this.nAll, this.cfg.vcycles, false, true);
          g.encodeSync(pass, this.nAll);
        } else if (g) {
          g.encodeExtrap(pass, this.nAll);
          g.encodeSolveB(pass, this.cfg.vcycles);
          g.encodeSync(pass, -1);
        }
        if (this.hasMasses) stepMasses(pass);
        fieldB(pass, last ? this.nAll : -1);
        // The kick goes through the other state texture and back.
        run(pass, this.pCorrect, last ? this.gCorrect[o] : this.gCorrectB[o]);
        if (last) this.interior(pass, this.nLeaf);
        else pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(2));
        run(pass, this.pCommit!, last ? this.gCommit[p] : this.gCommitL[p]);
        if (last) this.interior(pass, this.nLeaf);
        else pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(2));
      }
      pass.end();
    }
    this.binCycles++;
    this.binFirstWork += this.nLeaf;
    this.binFullWork += this.nLeaf << K;
  }

  /**
   * Work done with per-block timesteps since the last reset: block updates made (deposits of
   * a neighbor's fluxes included), and what a single timestep would have made.
   */
  async binWork(): Promise<{ cycles: number; updates: number; single: number }> {
    const buf = this.device.createBuffer({ size: BIN_ARGS_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.binArgs, 0, buf, 0, BIN_ARGS_BYTES);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const later = new Uint32Array(buf.getMappedRange().slice(0))[BIN_TOTALS];
    buf.destroy();
    return { cycles: this.binCycles, updates: this.binFirstWork + later, single: this.binFullWork };
  }

  /** Comoving coordinates: the scale factor the diagnostics last read from the clock. */
  private lastScale: number | undefined;

  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [this.n0, this.n0, c.ndim === 2 ? 1 : this.n0],
      dx: this.dx0,
      rhoMean: this.rhoMean,
      // Comoving coordinates: the state's scale factor as last read (syncScale keeps it current).
      scale: this.lastScale,
      // (A spherical run's monopole is the potential the diagnostics read.)
      selfGravity: this.selfGravity || !!this.sphMono,
      masses: this.hasMasses ? this.nMasses : 0,
      amr: [this.slotsPerSide, this.n0 / BS, this.levels, this.layers],
      refine: [c.refineHi, c.jeansRefine],
      solids: this.solids,
      vref: this.vRef,
      sources: this.nSources > 0,
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
    this.onParams?.();
    // The block table records what lies beyond each face of the domain: follow a change of
    // boundary. (A face may not become periodic this way: the blocks on either side of it
    // would have to be balanced first. The front end restarts the run for that.)
    const tree = this.tree;
    if (tree && (tree.bcLo.some((b, a) => b !== c.bcLo[a]) || tree.bcHi.some((b, a) => b !== c.bcHi[a]))) {
      this.flush();
      tree.bcLo = c.bcLo;
      tree.bcHi = c.bcHi;
      this.device.queue.writeBuffer(this.bt, 0, tree.table() as Int32Array<ArrayBuffer>);
    }
  }

  private get selfGravity(): boolean {
    return isSelfGravity(this.cfg.gravity) && !!this.grav;
  }

  /** A spherical run's monopole gravity (amr/sph_monopole.wgsl): its passes and buffers. */
  private async buildMonopole() {
    const d = this.device;
    const code = assemble('amr/sph_monopole.wgsl');
    const [rows, shells, enclose, field, fieldNow] = await Promise.all([
      computePipeline(d, code, 'spherical monopole rows', undefined, 'rows'),
      computePipeline(d, code, 'spherical monopole shells', undefined, 'shells'),
      computePipeline(d, code, 'spherical monopole enclose', undefined, 'enclose'),
      computePipeline(d, code, 'spherical monopole field', undefined, 'field'),
      computePipeline(d, code, 'spherical monopole field (now)', { HALF: 0 }, 'field'),
    ]);
    const n = this.n;
    const buf = d.createBuffer({ label: 'spherical monopole', size: 4 * (4 + 9 * n + 4 + this.cap * BS), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const phi = tex3d(d, this.slotsPerSide * PAD, 'r32float', 'spherical potential', this.layers);
    const v = (t: GPUTexture) => t.createView();
    this.sphMono = {
      rows, shells, enclose, field, fieldNow, buf, phi,
      gRows: [0, 1].map((p) => bindGroup(d, rows, { 0: this.params, 1: v(this.state[p][0]), 2: buf, 30: this.bt, 31: this.listLeaf })),
      gShells: bindGroup(d, shells, { 0: this.params, 2: buf, 30: this.bt, 31: this.listLeaf }),
      gEnclose: bindGroup(d, enclose, { 0: this.params, 2: buf }),
      gField: bindGroup(d, field, { 0: this.params, 2: buf, 3: v(phi), 4: v(this.gfield), 30: this.bt, 31: this.listAll }),
      gFieldNow: [this.listAct, this.listAll].map((list) => bindGroup(d, fieldNow, { 0: this.params, 2: buf, 3: v(phi), 4: v(this.gfield), 30: this.bt, 31: list })),
    };
  }

  /**
   * A spherical run's shells from the state at `parity`: the leaves' rows, the finest shells'
   * masses and the sums inside and outside each (the last ones kept: encoding it twice leaves
   * no history).
   */
  private encodeMonopole(pass: GPUComputePassEncoder, parity: number) {
    const m = this.sphMono!;
    pass.setPipeline(m.rows);
    pass.setBindGroup(0, m.gRows[parity]);
    pass.dispatchWorkgroups(Math.ceil((this.nLeaf * BS) / 64));
    pass.setPipeline(m.shells);
    pass.setBindGroup(0, m.gShells);
    pass.dispatchWorkgroups(Math.ceil(this.n / 64));
    pass.setPipeline(m.enclose);
    pass.setBindGroup(0, m.gEnclose);
    pass.dispatchWorkgroups(1);
  }
  private get hasMasses(): boolean {
    return this.nMasses > 0;
  }

  /** Acceleration and half-step potential for every block, into the field atlas. */
  private encodeField(pass: GPUComputePassEncoder) {
    if (this.sphMono) {
      pass.setPipeline(this.sphMono.field);
      pass.setBindGroup(0, this.sphMono.gField);
      this.padded(pass, this.nAll);
    } else if (this.grav) {
      this.grav.encodeField(pass, this.nAll);
    } else {
      pass.setPipeline(this.pMassField!);
      pass.setBindGroup(0, this.gMassField!);
      this.padded(pass, this.nAll);
    }
  }
  /** The gravity field atlas is in use: by self-gravity, point masses or both. */
  private get fieldOn(): boolean {
    return this.selfGravity || this.hasMasses || !!this.sphMono;
  }

  /**
   * Encode a full gravity solve for the state of the given parity into an open pass: source,
   * multipoles, every level's solve (with self-gravity), and optionally the point masses:
   * 'kick' evaluates their accelerations, 'step' moves them through a step. The field pass
   * is separate because the "previous potential" copy may have to happen in between.
   */
  private encodeSolve(pass: GPUComputePassEncoder, parity: number, cycles: number, perturber: 'none' | 'kick' | 'step', cold = false, together = false) {
    if (this.selfGravity) {
      const g = this.grav!;
      g.encodeSource(pass, parity, this.nLevel, this.nLeaf);
      if (this.cfg.gravity === 'isolated') this.redMp!.encode(pass, parity);
      g.encodeSolve(pass, this.nLevel, this.nAll, cycles, cold, together);
    }
    if (this.sphMono) this.encodeMonopole(pass, parity);
    if (this.hasMasses && perturber !== 'none') {
      const live = perturber === 'step';
      if (live) {
        pass.setPipeline(this.pBhDrift!);
        pass.setBindGroup(0, this.gBhDrift!);
        pass.dispatchWorkgroups(1);
      }
      this.redBh!.encode(pass, parity, this.nMasses);
      pass.setPipeline(live ? this.pBhKick! : this.pBhEval!);
      pass.setBindGroup(0, live ? this.gBhKick! : this.gBhEval!);
      pass.dispatchWorkgroups(1);
    }
  }

  /** Solve for the potential of the current state from scratch-ish, and rebuild the field. */
  private encodeFreshPotential(enc: GPUCommandEncoder, cycles: number, perturber: 'none' | 'kick', newBlocks: boolean) {
    let pass = beginPass(enc);
    if (newBlocks && this.selfGravity) this.grav!.encodeNewBlocks(pass, this.nLeaf);
    // Without new blocks to seed this is a cold start: each level begins from the coarser one.
    this.encodeSolve(pass, this.cur, cycles, perturber, !newBlocks);
    // (A spherical run's monopole twice: no history either.)
    if (this.sphMono) this.encodeMonopole(pass, this.cur);
    pass.end();
    // No history on this mesh: the half-step extrapolation starts from the potential itself.
    if (this.selfGravity) this.grav!.copyPhiToOld(enc, this.tree.highSlot);
    pass = beginPass(enc);
    if (this.selfGravity) this.grav!.encodeForget(pass, this.nAll);
    this.encodeField(pass);
    pass.end();
  }

  /** Call after changing the gravity mode or G at run time. */
  refreshPotential() {
    if (!this.fieldOn) return;
    const enc = this.device.createCommandEncoder();
    this.encodeFreshPotential(enc, 12, 'kick', false);
    this.device.queue.submit([enc.finish()]);
  }

  // ---------------------------------------------------------------- initial conditions

  private finerInSnapshot(b: Block, levelAt: (x: number, y: number, z: number) => number): boolean {
    const three = this.cfg.ndim === 3;
    const size = (BS * this.dx0) / (1 << b.level);
    const lo = domainLo(this.cfg);
    for (let o = 0; o < (three ? 8 : 4); o++) {
      const at = (a: number) => (b.c[a] + (((o >> a) & 1) + 0.5) / 2) * size + lo[a];
      if (levelAt(at(0), at(1), three ? at(2) : 0) > b.level) return true;
    }
    return false;
  }

  /** Time, step count and point masses, for a snapshot (see io/snapshot.ts). */
  async readRestart(): Promise<RestartState> {
    return readRestart(this.device, this.clock, this.bh, this.hasMasses ? this.nMasses : 0);
  }

  async debrisLedger(): Promise<DebrisLedger | null> {
    return this.debrisBuf ? this.debrisBuf.readLedger() : null;
  }

  async debris(): Promise<DebrisReport | null> {
    const b = this.debrisBuf;
    if (!b) return null;
    const d = await this.diagnostics();
    const v0 = d.momentum.map((p) => p / d.mass);
    const v = (t: GPUTexture) => t.createView();
    const group = bindGroup(this.device, this.pCensus!, {
      0: this.params, 1: v(this.state[this.cur][0]), 2: v(this.state[this.cur][1]), 3: v(this.grav?.phi ?? this.dummyPhi), 20: this.bh, 21: b.census, 30: this.bt, 31: this.listLeaf,
    });
    const census = await b.takeCensus((enc) => {
      const pass = beginPass(enc);
      pass.setPipeline(this.pCensus!);
      pass.setBindGroup(0, group);
      this.interior(pass, this.nLeaf);
      pass.end();
    }, this.selfGravity, v0, d.ranges[0][1]);
    const r = await this.readRestart();
    return b.report(r.time, r.masses?.[b.spec.mass]?.gm ?? 0, await b.readLedger(), census);
  }

  /**
   * Load the problem's initial conditions on a mesh refined to fit them. The CPU work (sampling
   * the setup for the mesh, and each leaf's state) is spread over workers where it can be
   * (amr/init_pool.ts), and done on the page otherwise, with the same result.
   */
  async reset() {
    this.resetting = true;
    try {
      await this.resetState();
    } finally {
      this.resetting = false;
    }
  }

  private async resetState() {
    const c = this.cfg;
    const d = this.device;
    const eos = makeEos(c.eos, c.eosParams);
    // (A spherical run's setup takes (r, theta): src/spherical.ts sphSetup.)
    const fn = sphSetup(c, c.problem.init(this.n, eos, c.ndim, c.params));
    const surface = surfaceOf(fn, eos);
    let run = await InitRun.start(c, this.n, this.dx0, this.levels);
    // With self-gravity a level that only partly fits the block budget is left out altogether:
    // a refinement boundary that cuts a steep hydrostatic gradient at an arbitrary place
    // disturbs the balance far more than resolving all of it one level coarser.
    const strict = this.selfGravity;
    const levelAt = c.restart?.levelAt;
    const sampled = new Map<Block, { sub: number; ind: number }>();
    for (let limit = this.levels; ; limit--) {
      // A rectangular grid (src/domain.ts gridDims): half the root blocks along a halved axis.
      const gd = gridDims(c);
      const lim = [0, 1, 2].map((a) => (gd[a] * 2 === c.n ? this.n0 / BS / 2 : this.n0 / BS));
      this.tree = new BlockTree(c.ndim, this.n0 / BS, this.levels, this.cap, c.bcLo, c.bcHi, c.problem.mhd !== undefined, lim);
      this.tree.levelLimit = limit;
      for (let pass = 0; pass < limit; pass++) {
        // From a snapshot: the mesh it was saved on. A block is refined wherever the snapshot
        // is finer than it at the center of one of its children.
        if (levelAt) {
          if (!this.tree.adapt((b) => (this.finerInSnapshot(b, levelAt) ? 1 : 0), 0.5, -1)) break;
          continue;
        }
        // A block kept from the last pass was sampled at least as finely then (the budget is
        // shared by more leaves now): only new blocks are sampled. The indicators of all the
        // leaves are found first (adapt asks for exactly these), on the workers if there are any.
        const leaves = this.tree.leaves();
        const todo: { b: Block; sub: number }[] = [];
        for (const b of leaves) {
          const sub = samplesPerCell(c, this.levels, b.level, leaves.length);
          const seen = sampled.get(b);
          if (!seen || seen.sub < sub) todo.push({ b, sub });
        }
        let values: Float64Array | null = null;
        if (run && todo.length > 1) values = await run.indicators(todo.map(({ b, sub }) => ({ level: b.level, c: b.c, sub }))).catch(() => ((run = null), null));
        todo.forEach(({ b, sub }, i) => sampled.set(b, { sub, ind: values ? values[i] : indicator(c, this.dx0, b, surface, sub) }));
        if (!this.tree.adapt((b) => sampled.get(b)!.ind, c.refineHi, c.refineLo)) break;
      }
      if (levelAt || !strict || limit === 0 || this.tree.freeSlots >= this.tree.nchild) break;
    }

    const three = c.ndim === 3;
    const depth = three ? BS : 1;
    const G = this.speciesGroups;
    const blockCells = BS * BS * depth;
    const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
    const leaves = this.tree.leaves();
    // Each leaf's mass, summed in leaf order at the end, so the total is the same whoever made them.
    const leafMass = new Float64Array(leaves.length);
    this.solids = false;
    this.vRef = 0;
    this.nSources = 0;
    d.queue.writeBuffer(this.sources, 0, packSources([]));
    const write = (i: number, s: LeafState) => {
      const origin = this.origin(leaves[i].slot);
      if (s.faces) this.mhd!.writeFaces(origin, s.faces);
      d.queue.writeTexture({ texture: this.state[0][0], origin }, s.u1, layout, [BS, BS, depth]);
      d.queue.writeTexture({ texture: this.state[0][1], origin }, s.u2, layout, [BS, BS, depth]);
      if (s.km) d.queue.writeTexture({ texture: this.comp!, origin }, s.km, layout, [BS, BS, depth]);
      for (let g = 0; g < G; g++) {
        const go = speciesGroupOffset(g, this.speciesDepth);
        const og = [origin[0] + go[0], origin[1] + go[1], origin[2] + go[2]];
        d.queue.writeTexture({ texture: this.species[0], origin: og }, s.sp!.subarray(g * blockCells * 4, (g + 1) * blockCells * 4), layout, [BS, BS, depth]);
      }
      leafMass[i] = s.mass;
      if (s.solids) this.solids = true;
      this.vRef = Math.max(this.vRef, s.vmax ?? 0);
    };
    let filled = false;
    if (run && leaves.length > 1) filled = await run.leaves(leaves, write).then(() => true, () => false);
    if (!filled) {
      const ctx = { cfg: c, dx0: this.dx0, levels: this.levels, fn, eos, kc: this.compCoeffs, groups: G, mhd: !!this.mhd };
      leaves.forEach((b, i) => write(i, leafState(ctx, b)));
    }
    let mass = 0;
    for (const m of leafMass) mass += m;
    this.cur = 0;
    this.speciesCur = 0;
    this.pending = null;
    this.stepsSinceFlag = 0;
    this.regrids = 0;
    this.binsPrimed = null;
    this.binCycles = this.binFirstWork = this.binFullWork = 0;
    this.binEpoch++;
    this.binSeen = { updates: 0, single: 0 };
    this.wantBinPoll = null;
    this.cyclesSincePoll = 0;
    this.binPause = BIN_START_STEPS;
    this.binPauseNext = BIN_PAUSE_STEPS;
    this.binShare = null;
    d.queue.writeBuffer(this.binArgs, 0, new Uint32Array(BIN_ARGS_BYTES / 4));
    this.rhoMean = mass / c.problem.box ** c.ndim;
    // The clock (clockState), at the snapshot's time for a restart.
    d.queue.writeBuffer(this.clock, 0, clockState(c, c.restart?.time ?? 0, c.restart?.steps ?? 0));
    // A spherical run's point masses sit at the origin: their pull joins the monopole field
    // (amr/sph_monopole.wgsl), and they are not kept as particles.
    const sph = spherical(c);
    const masses = sph ? [] : initialMasses(c, this.massDx);
    this.centralGM = sph ? initialMasses(c, this.massDx).reduce((a, m) => a + m.gm, 0) : 0;
    if (this.sphMono) d.queue.writeBuffer(this.sphMono.buf, 0, new Float32Array([this.centralGM, isSelfGravity(c.gravity) ? 1 : 0, 0, 0]));
    this.nMasses = this.nOwnMasses = masses.length;
    d.queue.writeBuffer(this.bh, 0, packMasses(masses));
    this.debrisBuf?.reset(c.restart?.debris);
    writeDrive(d, c.driving, c.ndim, c.problem.box);
    this.writeParams();
    this.uploadTree();
    if (this.selfGravity) {
      // The root level's potential from zero, not from the last run's (AmrGravity.clear).
      // (In 2D the corner's z is the slot's layer, which has no ghosts.)
      const flat = this.cfg.ndim === 2;
      this.grav!.clear([...this.tree.lists().byLevel[0]].map((slot) => this.origin(slot).map((o, a) => (flat && a === 2 ? o : o - NG))));
    }
    if (this.fieldOn) {
      // Cold start: enough cycles per level to converge from zero. The clock's dt is zero, so
      // the point masses' accelerations are only evaluated.
      const enc = d.createCommandEncoder();
      this.encodeFreshPotential(enc, 12, 'kick', false);
      d.queue.submit([enc.finish()]);
    }
    this.startSettling();
  }

  /** Resolves once a star in the initial conditions has been settled (see problems/settle.ts). */
  settled: Promise<void> = Promise.resolve();
  private settleEpoch = 0;
  private settling = false;
  get preparing(): boolean {
    return this.settling || this.compiling || this.resetting;
  }
  /** reset() is building the initial state (it waits on workers): no steps are taken yet. */
  private resetting = false;
  /** The update pass is still compiling (a deferred build, see `create`): no steps are taken yet. */
  compiling = false;
  /** Resolves once the update pass is compiled and bound. */
  compiled: Promise<void> = Promise.resolve();
  private hydroEpoch = 0;
  /** largest relative change of density in each round of the last settling, and how long it took (inspection) */
  settleLog: number[] = [];
  settleMs = 0;

  /**
   * Settle a star into the mesh's own hydrostatic balance: no steps are taken until it has.
   * A reset meanwhile abandons it. The mesh stays as the analytic star refined it.
   */
  private startSettling() {
    const epoch = ++this.settleEpoch;
    const c = this.cfg;
    this.settling = false;
    this.settled = Promise.resolve();
    if (c.restart || !c.wellBalanced || !this.selfGravity) return;
    const eos = makeEos(c.eos, c.eosParams);
    const set = c.problem.settle ? c.problem.settle(c.params, eos) : c.problem.star ? settlingOf(c.problem.star, c.G) : null;
    if (!set) return;
    this.settling = true;
    const tree = this.tree;
    const t0 = performance.now();
    // On the GPU where every star's gas is one the GPU knows; the CPU's settling reads the whole
    // mesh back, some 100 bytes a cell, which a mesh of tens of GB does not fit in a page.
    // (The GPU's sums are of cubes: an axisymmetric run's rings settle on the CPU.)
    const onGpu = c.problem.geometry !== 'cylindrical' && !set.keep && set.stars.length <= SETTLE_MAX_STARS && set.stars.every((st) => st.barotrope.gpu) && !(globalThis as { shaderousCpuSettling?: boolean }).shaderousCpuSettling;
    this.settled = (async () => {
      try {
        setLoadStage(settleStage(0));
        const stale = () => epoch !== this.settleEpoch || tree !== this.tree;
        if (onGpu) await this.settleOnGpu(set, stale);
        else {
          const leaf = await this.readLeaves();
          if (epoch !== this.settleEpoch) return;
          await this.settleOnCpu(set, eos, leaf, stale);
        }
        if (!stale() && c.problem.heat) await this.heatSettled(c.problem.heat(c.params, eos), stale);
      } finally {
        if (epoch === this.settleEpoch) {
          this.settling = false;
          setLoadStage('');
          this.settleMs = performance.now() - t0;
        }
      }
    })();
  }

  /** Put the problem's heat (Problem.heat) into the settled mesh: every leaf is read, and those it changes written back. */
  private async heatSettled(heat: (x: number, y: number, z: number, rho: number) => number | undefined, stale: () => boolean) {
    const leaf = await this.readLeaves();
    if (stale()) return;
    const per = BS * BS * (this.cfg.ndim === 3 ? BS : 1);
    const cells = AmrSim.cellsOf(leaf, this.cfg.problem.geometry === 'cylindrical');
    const leaves = this.tree.leaves();
    const touched = new Uint8Array(cells.count);
    if (!heatCells(heat, cells, touched)) return;
    const changed = leaves.filter((_, i) => touched.subarray(i * per, (i + 1) * per).some((t) => t > 0));
    const d = this.device;
    const depth = this.cfg.ndim === 3 ? BS : 1;
    const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
    const index = new Map(leaves.map((b, i) => [b, i]));
    for (const b of changed) {
      const i = index.get(b)!;
      d.queue.writeTexture({ texture: this.state[this.cur][1], origin: this.origin(b.slot) }, leaf.u2.subarray(i * per * 4, (i + 1) * per * 4) as Float32Array<ArrayBuffer>, layout, [BS, BS, depth]);
    }
  }

  /** The leaves' cells as readLeaves() returned them, for settling. */
  private static cellsOf(leaf: LeafCells, cyl = false): SettleCells {
    // In (R, z) a cell is a ring, 2 pi R dx^2: the settling's sums are of its volume.
    const vol = cyl ? (q: number) => 2 * Math.PI * leaf.x[q] * leaf.dx[q] ** 2 : (q: number) => leaf.dx[q] ** 3;
    return { count: leaf.count, pos: (q) => [leaf.x[q], leaf.y[q], leaf.z[q]], vol, u1: leaf.u1, u2: leaf.u2 };
  }

  /**
   * The rounds of a settling on the CPU, on the leaves' cells as read back. `emag`: magnetic
   * energy to add to a cell's E afterward. `stale`: the run was reset or the mesh changed.
   */
  private async settleOnCpu(set: Settling, eos: Eos, leaf: LeafCells, stale: () => boolean, emag?: (s: Settler, q: number) => number) {
    const d = this.device;
    const tree = this.tree;
    const cells = AmrSim.cellsOf(leaf, this.cfg.problem.geometry === 'cylindrical');
    const { u1, u2 } = cells;
    let peak = 0;
    for (let q = 0; q < cells.count; q++) peak = Math.max(peak, u1[q * 4]);
    const settler = new Settler(set, cells);
    // A block's cells: one layer of them in 2D (an axisymmetric run's).
    const depth = this.cfg.ndim === 3 ? BS : 1;
    const per = BS * BS * depth;
    // Only the blocks a star may reach are ever rewritten.
    const leaves = tree.leaves().filter((_, i) => settler.owner.subarray(i * per, (i + 1) * per).some((o) => o >= 0));
    const index = new Map(tree.leaves().map((b, i) => [b, i]));
    const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
    const upload = (tex: GPUTexture, u: Float32Array) =>
      leaves.forEach((b) => {
        const i = index.get(b)!;
        d.queue.writeTexture({ texture: tex, origin: this.origin(b.slot) }, u.subarray(i * per * 4, (i + 1) * per * 4) as Float32Array<ArrayBuffer>, layout, [BS, BS, depth]);
      });
    this.settleLog = [];
    for (let round = 0; round < SETTLE_ROUNDS; round++) {
      setLoadStage(settleStage(round));
      const phi = await this.readLeafPotential();
      if (stale()) return;
      const change = settler.round(phi);
      this.settleLog.push(change / peak);
      upload(this.state[this.cur][0], u1);
      this.refreshPotential();
      if (change < (set.tol ?? SETTLE_TOL) * peak) break;
    }
    settler.finish(eos.eintFromP);
    if (emag) for (let q = 0; q < cells.count; q++) u2[4 * q] += emag(settler, q);
    upload(this.state[this.cur][0], u1);
    upload(this.state[this.cur][1], u2);
    this.refreshPotential();
  }

  /**
   * Settle the star `star` just placed with the star tool (placeStar) among the gas that is
   * there, on the CPU: the whole mesh is read back, so a mesh of more than SETTLE_CPU_CELLS
   * cells is left as the tool placed it. Returns why not, if it is not settled; '' once it is.
   */
  async settleStar(star: PlacedStar): Promise<string> {
    const c = this.cfg;
    if (!c.wellBalanced || !this.selfGravity) return 'it needs self-gravity and the well-balanced scheme';
    if (this.tree.leaves().length * BS ** 3 > SETTLE_CPU_CELLS) return 'the mesh is too large to read back';
    const epoch = ++this.settleEpoch;
    const tree = this.tree;
    const stale = () => epoch !== this.settleEpoch || tree !== this.tree;
    const leaf = await this.readLeaves();
    if (stale()) return '';
    const eos = makeEos(c.eos, c.eosParams);
    const { u1, u2 } = leaf;
    const set = placedSettling(star, c.G, AmrSim.cellsOf(leaf, this.cfg.problem.geometry === 'cylindrical'), eos.pressure, c.problem.box);
    if (typeof set === 'string') return set;
    // With a magnetic field, E holds its energy too: what is left of E in a cell goes back in.
    const rest = this.mhd ? new Float32Array(leaf.count) : null;
    if (rest) {
      for (let q = 0, o = 0; q < rest.length; q++, o += 4) {
        const ke = (0.5 * (u1[o + 1] ** 2 + u1[o + 2] ** 2 + u1[o + 3] ** 2)) / Math.max(u1[o], 1e-30);
        rest[q] = Math.max(u2[o] - ke - u2[o + 1], 0);
      }
    }
    this.settling = true;
    const t0 = performance.now();
    this.settled = (async () => {
      try {
        await this.settleOnCpu(set, eos, leaf, stale, rest ? (st, q) => (st.rewrote[q] ? rest[q] : 0) : undefined);
      } finally {
        if (epoch === this.settleEpoch) {
          this.settling = false;
          setLoadStage('');
          this.settleMs = performance.now() - t0;
        }
      }
    })();
    await this.settled;
    return '';
  }

  /**
   * startSettling() on the GPU (amr/settle.wgsl): the same rounds, with every cell's density
   * worked out where it lies. Each round finds each star's constant C from per-block sums of its
   * mass at trial values of C (16 a pass: first spread out from the bottom of the well, then
   * across the bracket found), which the host adds in double precision; only those sums come
   * back. `stale`: the run was reset or the mesh changed, and the settling is to be dropped.
   */
  private async settleOnGpu(set: Settling, stale: () => boolean) {
    const d = this.device;
    const code = assemble('amr/settle.wgsl', this.slots);
    const [pMeasure0, pMeasure1, pMeasure2, pMeasure3, pApply, pFinish, pCopy] = await Promise.all([
      computePipeline(d, code, 'settle masses', { MODE: 0 }, 'measure'),
      computePipeline(d, code, 'settle least potential', { MODE: 1 }, 'measure'),
      computePipeline(d, code, 'settle trial masses', { MODE: 2 }, 'measure'),
      computePipeline(d, code, 'settle centers of mass', { MODE: 3 }, 'measure'),
      computePipeline(d, code, 'settle apply', undefined, 'apply'),
      computePipeline(d, code, 'settle finish', undefined, 'finish'),
      computePipeline(d, code, 'settle copy back', undefined, 'copy_back'),
    ]);
    if (stale()) return;
    this.flush();
    const nRows = this.nLeaf;
    const NC = 16;
    const bytes = Math.max(1, nRows) * NC * 4;
    const uni = d.createBuffer({ label: 'settle', size: SETTLE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const out = d.createBuffer({ label: 'settle sums', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const back = d.createBuffer({ label: 'settle sums readback', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const v = (t: GPUTexture) => t.createView();
    const phi = v(this.grav!.phi);
    const groups = rows(1, nRows);
    const f = new Float32Array(SETTLE_BYTES / 4);
    const u = new Uint32Array(f.buffer);
    const stars = set.stars;
    const mass = stars.map(() => 0);
    const C: number[] = stars.map(() => 0);
    // Where each star's mass is: the others' pull on it is taken from there.
    const at = stars.map((st) => [st.center[0], st.center[1], st.center[2] ?? 0]);
    const relax = set.relax ?? 1;
    // The uniform (amr/settle.wgsl, Settle): star S of the measures, and their trial values of C.
    const pack = (star: number, cands: number[] = []) => {
      f.fill(0);
      u[0] = stars.length;
      u[1] = star;
      f.set([set.rhoCut, set.rhoAmb, set.pAmb, relax, set.G], 4);
      stars.forEach((st, i) => {
        const o = 12 + 24 * i;
        const g = st.barotrope.gpu!;
        f.set([st.center[0], st.center[1], st.center[2] ?? 0, st.R, st.reach ?? 3e38, mass[i], st.scalar, g.kind === 'polytrope' ? 0 : 1], o);
        f.set(g.kind === 'polytrope' ? [g.n, g.K, (g.n + 1) * g.K, 0] : [g.degRho, g.degP, g.c1 ?? 0, g.rho0 ?? 1], o + 8);
        f.set([st.vel[0], st.vel[1], st.vel[2] ?? 0, 0], o + 12);
        const hi = Math.fround(C[i]);
        f.set([hi, C[i] - hi], o + 16);
        f.set(at[i], o + 20);
      });
      cands.forEach((c, k) => {
        const hi = Math.fround(c);
        f.set([hi, c - hi], 108 + 2 * k);
      });
      d.queue.writeBuffer(uni, 0, f);
    };
    // Bind groups for the current parity: the measures read it; apply and finish write the other.
    const groupsFor = (p: number) => ({
      measure: [pMeasure0, pMeasure1, pMeasure2, pMeasure3].map((pipe) =>
        bindGroup(d, pipe, { 0: this.params, 10: uni, 2: v(this.state[p][0]), 4: phi, 11: out, 30: this.bt, 31: this.listLeaf }),
      ),
      apply: bindGroup(d, pApply, {
        0: this.params, 10: uni, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 4: phi, 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]),
        11: out, 30: this.bt, 31: this.listLeaf,
      }),
      finish: bindGroup(d, pFinish, {
        0: this.params, 10: uni, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]),
        30: this.bt, 31: this.listLeaf, ...this.compIn,
      }),
      copy: bindGroup(d, pCopy, { 0: this.params, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]), 31: this.listLeaf }),
    });
    // The state stays in its parity: what apply and finish write into the other is copied back.
    const bg = groupsFor(this.cur);
    const back2 = groupsFor(1 - this.cur).copy;
    const encode = (pipe: GPUComputePipeline, group: GPUBindGroup) => {
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
      if (nRows > 0) pass.dispatchWorkgroups(...groups);
      pass.end();
      return enc;
    };
    // Runs a pass and returns its per-block results, NC floats a block.
    const measure = async (pipe: GPUComputePipeline, group: GPUBindGroup): Promise<Float32Array> => {
      const enc = encode(pipe, group);
      enc.copyBufferToBuffer(out, 0, back, 0, nRows * NC * 4);
      d.queue.submit([enc.finish()]);
      await back.mapAsync(GPUMapMode.READ);
      const r = new Float32Array(back.getMappedRange().slice(0));
      back.unmap();
      return r;
    };
    const swap = () => d.queue.submit([encode(pCopy, back2).finish()]);
    try {
      // The stars' masses as the initial conditions placed them, and the densest cell.
      pack(0);
      const m0 = await measure(pMeasure0, bg.measure[0]);
      let peak = 0;
      for (let r = 0; r < nRows; r++) {
        for (let i = 0; i < stars.length; i++) mass[i] += m0[r * NC + i];
        peak = Math.max(peak, m0[r * NC + 4]);
      }
      // Star i's mass at trial values of C.
      const trial = async (i: number, cands: number[]): Promise<number[]> => {
        pack(i, cands);
        const m = await measure(pMeasure2, bg.measure[2]);
        const sums = cands.map(() => 0);
        for (let r = 0; r < nRows; r++) for (let k = 0; k < NC; k++) sums[k] += m[r * NC + k];
        return sums;
      };
      const solveC = async (i: number) => {
        pack(i);
        const least = await measure(pMeasure1, bg.measure[1]);
        let phiMin = Infinity;
        for (let r = 0; r < nRows; r++) phiMin = Math.min(phiMin, least[r * NC]);
        if (!(phiMin < 1e38)) return;
        const scale = Math.abs(phiMin) + 1e-6;
        // Out from the bottom of the well, doubling, until the mass is reached...
        let lo = phiMin;
        let mLo = 0;
        let hi = NaN;
        let mHi = NaN;
        for (let start = -12; Number.isNaN(hi) && start < 60; start += NC) {
          const cands = Array.from({ length: NC }, (_, k) => phiMin + scale * 2 ** (start + k));
          const m = await trial(i, cands);
          const k = m.findIndex((x) => x >= mass[i]);
          if (k < 0) {
            lo = cands[NC - 1];
            mLo = m[NC - 1];
            continue;
          }
          if (k > 0) {
            lo = cands[k - 1];
            mLo = m[k - 1];
          }
          hi = cands[k];
          mHi = m[k];
        }
        if (Number.isNaN(hi)) throw new Error('settling: no constant gives the star its mass');
        // ...then across the bracket, cut into 17 a pass, and the mass interpolated in the last.
        for (let pass = 0; pass < 12 && hi - lo > 1e-11 * scale; pass++) {
          const cands = Array.from({ length: NC }, (_, k) => lo + ((hi - lo) * (k + 1)) / (NC + 1));
          const m = await trial(i, cands);
          const k = m.findIndex((x) => x >= mass[i]);
          if (k < 0) {
            lo = cands[NC - 1];
            mLo = m[NC - 1];
            continue;
          }
          hi = cands[k];
          mHi = m[k];
          if (k > 0) {
            lo = cands[k - 1];
            mLo = m[k - 1];
          }
        }
        C[i] = mHi > mLo ? lo + ((hi - lo) * (mass[i] - mLo)) / (mHi - mLo) : hi;
      };
      this.settleLog = [];
      for (let round = 0; round < SETTLE_ROUNDS; round++) {
        setLoadStage(settleStage(round));
        if (stale()) return;
        if (stars.length > 1) {
          pack(0);
          const mom = await measure(pMeasure3, bg.measure[3]);
          stars.forEach((_, i) => {
            const s4 = [0, 0, 0, 0];
            for (let r = 0; r < nRows; r++) for (let a = 0; a < 4; a++) s4[a] += mom[r * NC + 4 * i + a];
            if (s4[0] > 0) at[i] = [s4[1] / s4[0], s4[2] / s4[0], s4[3] / s4[0]];
          });
        }
        for (let i = 0; i < stars.length; i++) await solveC(i);
        pack(0);
        const ch = await measure(pApply, bg.apply);
        let change = 0;
        for (let r = 0; r < nRows; r++) change = Math.max(change, ch[r * NC]);
        swap();
        this.refreshPotential();
        this.settleLog.push(change / peak);
        if (change < (set.tol ?? SETTLE_TOL) * peak) break;
      }
      if (stale()) return;
      pack(0);
      d.queue.submit([encode(pFinish, bg.finish).finish()]);
      swap();
      this.refreshPotential();
    } finally {
      uni.destroy();
      out.destroy();
      back.destroy();
    }
  }

  /** Atlas texel of a slot's first interior cell. */
  private origin(slot: number): [number, number, number] {
    const a = this.slotsPerSide;
    if (this.cfg.ndim === 2) return [(slot % a) * PAD + NG, (Math.floor(slot / a) % a) * PAD + NG, Math.floor(slot / (a * a))];
    return [(slot % a) * PAD + NG, (Math.floor(slot / a) % a) * PAD + NG, Math.floor(slot / (a * a)) * PAD + NG];
  }

  private meshVersion = 0;
  private boxes: { version: number; boxes: Float32Array } | null = null;
  blockBoxes() {
    if (this.boxes?.version !== this.meshVersion) {
      const leaves = this.tree.leaves();
      const out = new Float32Array(4 * leaves.length);
      leaves.forEach((b, i) => {
        const size = (BS * this.dx0) / (1 << b.level) / this.cfg.problem.box;
        out.set([b.c[0] * size, b.c[1] * size, b.c[2] * size, size], 4 * i);
      });
      this.boxes = { version: this.meshVersion, boxes: out };
    }
    return this.boxes;
  }

  private uploadTree() {
    this.meshVersion++;
    const d = this.device;
    const { all, leaves, byLevel } = this.tree.lists();
    this.nAll = all.length;
    this.dtStale = true;
    this.nLeaf = leaves.length;
    this.mhd?.setLeaves(this.nLeaf);
    this.nLevel = byLevel.map((l) => l.length);
    // Only the count and the listed slots: what lies past them is never read.
    const pack = (list: Uint32Array) => {
      const out = new Uint32Array(list.length + 1);
      out[0] = list.length;
      out.set(list, 1);
      return out;
    };
    d.queue.writeBuffer(this.bt, 0, this.tree.table() as Int32Array<ArrayBuffer>);
    d.queue.writeBuffer(this.listAll, 0, pack(all));
    d.queue.writeBuffer(this.listLeaf, 0, pack(leaves));
    byLevel.forEach((l, i) => d.queue.writeBuffer(this.listLevel[i], 0, pack(l)));
    // Only what changed: a re-mesh at a fine level touches a few slabs of the grid.
    const m = (this.n0 / BS) << this.levels;
    const boxes: GridBox[] = [];
    const grid = this.tree.indexGrid(boxes) as Uint32Array<ArrayBuffer>;
    // Each box is copied out on its own and written from its start. Writing it straight from
    // the whole grid, at an offset with the grid's row and image strides, is valid WebGPU, but
    // Firefox drew stale blocks that way (the index kept slots the re-mesh had reused), while
    // Chrome drew the same state correctly.
    for (const { o, s } of boxes) {
      const data = gridBox(grid, m, o, s) as Uint32Array<ArrayBuffer>;
      d.queue.writeTexture({ texture: this.idx, origin: o }, data, { offset: 0, bytesPerRow: s[0] * 4, rowsPerImage: s[1] }, s);
    }
  }

  // ---------------------------------------------------------------- stepping

  // See amr/list.wgsl: in 2D each workgroup covers four blocks, and a long list is cut into
  // slabs along z, to stay within the workgroups a dispatch may have along an axis.
  private padded(pass: GPUComputePassEncoder, count: number) {
    if (this.cfg.ndim === 2) {
      const q = Math.ceil(count / 4);
      pass.dispatchWorkgroups(Math.min(q, SLAB_QUADS) * 3, 3, Math.ceil(q / SLAB_QUADS));
    } else pass.dispatchWorkgroups(...padded3d(count));
  }
  /** Dispatch for inv_flat() (amr/list.wgsl): cells and `reach` ghost layers of `count` blocks. */
  private flat(pass: GPUComputePassEncoder, count: number, reach: number) {
    if (count > 0) pass.dispatchWorkgroups(flatGroups(this.cfg.ndim, reach), Math.min(count, FLAT_ROWS), Math.ceil(count / FLAT_ROWS));
  }
  private interior(pass: GPUComputePassEncoder, count: number) {
    if (this.cfg.ndim === 2) {
      const q = Math.ceil(count / 4);
      pass.dispatchWorkgroups(Math.min(q, SLAB_QUADS) * 2, 2, Math.ceil(q / SLAB_QUADS));
    } else pass.dispatchWorkgroups(...interior3d(count));
  }

  private applyRegrid(enc: GPUCommandEncoder) {
    const plan = this.pending!;
    this.pending = null;
    if (plan.steps) this.onRegrid?.(plan.steps);
    if (this.mhd) {
      // Before the new tree replaces it: the primitives of the current state with their ghosts
      // (the slopes of prolongation), and a copy of the old block table and leaf index, which
      // the field's re-meshing looks up. Submitted now, ahead of the new tree's upload.
      const pre = this.device.createCommandEncoder();
      const pp = beginPass(pre);
      this.mhd.encodeAssembly(pp, this.cur, () => this.flat(pp, this.nLeaf, NG));
      pp.end();
      this.mhd.snapshotTree(pre);
      this.device.queue.submit([pre.finish()]);
    }
    this.uploadTree();
    this.device.queue.writeBuffer(this.rg, 0, BlockTree.regridTable(plan.ops, this.cap) as Int32Array<ArrayBuffer>);
    const pass = beginPass(enc);
    pass.setPipeline(this.pRegrid);
    pass.setBindGroup(0, this.gRegrid[this.cur]);
    this.interior(pass, this.nLeaf);
    // The field's faces, then the energy rebuilt and everything back in this parity.
    this.mhd?.encodeRegrid(pass, this.cur);
    if (this.speciesGroups) {
      pass.setPipeline(this.pSpecies[3]);
      pass.setBindGroup(0, this.gSpecies[3][this.cur][this.speciesCur]);
      this.interior(pass, this.nLeaf);
      this.speciesCur = 1 - this.speciesCur;
    }
    pass.end();
    if (!this.mhd) this.cur = 1 - this.cur;
    this.regrids++;
    // The potential and field belong to the old mesh: solve again on the new one, starting
    // new blocks from their parents' values.
    if (this.fieldOn) this.encodeFreshPotential(enc, Math.max(2, this.cfg.vcycles), 'kick', true);
  }

  /**
   * Encode up to `count` steps and return how many were encoded. Fewer are taken while a
   * refinement check is overdue, so the flow never runs far ahead of the mesh.
   *
   * The encoder must be submitted, and submitted() called, before the next call: a re-mesh
   * replaces the tables that earlier passes in an unsubmitted encoder would still expect.
   */
  encodeSteps(enc: GPUCommandEncoder, count: number): number {
    if (this.settling || this.compiling) return 0; // a star is still being settled, or the update pass compiling
    if (this.pending) this.applyRegrid(enc);
    if (this.dtStale) {
      // The block list changed: drop timestep records of workgroups it no longer covers.
      enc.clearBuffer(this.redDt.first);
      this.dtStale = false;
    }
    this.primeBins(enc);
    syncScale(enc, this.cfg, this.clock, this.params);
    const cycle = this.binsLive ? 1 << this.builtBins : 0;
    let done = 0;
    for (; done < count; done += cycle || 1) {
      if (this.levels > 0 && (this.flagInFlight || this.wantMap) && this.stepsSinceFlag >= MAX_STEPS_AHEAD) break;
      if (cycle) {
        // Whole cycles only: the mesh, the display and the diagnostics need every block at
        // the same time. May run past `count`.
        this.encodeCycle(enc);
        this.cyclesSincePoll++;
        this.stepsSinceFlag += cycle;
        if (this.encodeFlags(enc)) {
          done += cycle;
          break;
        }
        continue;
      }
      if (this.binPause > 0) this.binPause--;
      const p = this.cur;
      if (this.mhd) {
        const mp = beginPass(enc);
        this.mhd.encode(
          mp,
          p,
          () => this.flat(mp, this.nLeaf, NG),
          () => {
            this.redDt.encode(mp);
            mp.setPipeline(this.pClock);
            mp.setBindGroup(0, this.gClock);
            mp.dispatchWorkgroups(1);
          },
        );
        // The species move with the mass fluxes the update recorded (amr/species.wgsl).
        this.encodeSpecies(mp, p);
        mp.end();
        if (this.fieldOn) {
          // Self-gravity or point masses: as for the gas alone below, the second half kick
          // brings the cells back into this parity, and the faces follow them.
          if (this.selfGravity) this.grav!.copyPhiToOld(enc, this.tree.highSlot);
          const gp = beginPass(enc);
          this.encodeSolve(gp, 1 - p, this.cfg.vcycles, 'step', false, true);
          this.encodeField(gp);
          gp.setPipeline(this.pCorrect);
          gp.setBindGroup(0, this.gCorrect[p]);
          this.interior(gp, this.nLeaf);
          this.mhd.encodeCommitFaces(gp, p);
          gp.end();
        } else {
          this.cur = 1 - p;
        }
        this.encodeBurn(enc);
        this.stepsSinceFlag++;
        if (this.encodeFlags(enc)) {
          done++;
          break;
        }
        continue;
      }
      const pass = beginPass(enc);
      pass.setPipeline(this.pAssemble);
      pass.setBindGroup(0, this.assembleGroup(p));
      this.flat(pass, this.nAll, NG);
      this.redDt.encode(pass);
      pass.setPipeline(this.pClock);
      pass.setBindGroup(0, this.gClock);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pPredict);
      pass.setBindGroup(0, this.gPredict);
      this.flat(pass, this.nLeaf, 1);
      pass.setPipeline(this.pUpdate);
      pass.setBindGroup(0, this.gUpdate[p]);
      this.interior(pass, this.nLeaf);
      this.encodeSpecies(pass, p);
      if (this.prs) {
        // Implicit pressure: the advected state's primitives, ghosts and all, then the equation,
        // the solve and the correction back into this parity (amr/pressure.ts).
        pass.setPipeline(this.pAssemble);
        pass.setBindGroup(0, this.assembleGroup(1 - p));
        this.flat(pass, this.nAll, NG);
        this.prs.encode(pass, p, this.nLevel, this.nLeaf, this.cfg.pressureCycles);
        this.encodeSpeciesMove(pass, p);
      }
      pass.end();
      if (this.prs) {
        // (The state is back in parity p.)
      } else if (this.fieldOn) {
        // Keep the old potential, solve for that of the updated state (warm-started), move
        // the point masses, rebuild the field, and apply the second half kick (and the
        // masses' sinks) back into this parity.
        if (this.selfGravity) this.grav!.copyPhiToOld(enc, this.tree.highSlot);
        const gp = beginPass(enc);
        this.encodeSolve(gp, 1 - p, this.cfg.vcycles, 'step', false, true);
        this.encodeField(gp);
        gp.setPipeline(this.pCorrect);
        gp.setBindGroup(0, this.gCorrect[p]);
        this.interior(gp, this.nLeaf);
        gp.end();
      } else {
        this.cur = 1 - p;
      }
      syncScale(enc, this.cfg, this.clock, this.params);
      this.encodeBurn(enc);
      this.stepsSinceFlag++;
      if (this.encodeFlags(enc)) {
        done++;
        break;
      }
    }
    if (cycle && this.cyclesSincePoll >= BIN_POLL_CYCLES && !this.binPollBusy && !this.wantBinPoll) {
      // How much work the last cycles saved: read back without blocking (see submitted()).
      enc.copyBufferToBuffer(this.binArgs, 0, this.binStaging, 0, BIN_ARGS_BYTES);
      this.wantBinPoll = { first: this.binFirstWork, full: this.binFullWork };
      this.cyclesSincePoll = 0;
    }
    return done;
  }

  /** Read the block-update totals copied by encodeSteps(), and pause the cycles if they save too little. */
  private pollBins() {
    const at = this.wantBinPoll;
    if (!at) return;
    this.wantBinPoll = null;
    this.binPollBusy = true;
    const epoch = this.binEpoch;
    this.binStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const later = new Uint32Array(this.binStaging.getMappedRange().slice(0))[BIN_TOTALS];
        this.binStaging.unmap();
        if (epoch !== this.binEpoch) return; // the run was reset meanwhile
        const updates = at.first + later;
        const du = updates - this.binSeen.updates;
        const ds = at.full - this.binSeen.single;
        this.binSeen = { updates, single: at.full };
        if (ds <= 0) return;
        this.binShare = du / ds;
        if (this.cfg.timeBinsAlways) return;
        if (this.binShare > (this.fieldOn ? BIN_SHARE_MAX.field : BIN_SHARE_MAX.hydro)) {
          this.binPause = this.binPauseNext;
          this.binPauseNext = Math.min(BIN_PAUSE_MAX, 2 * this.binPauseNext);
        } else {
          this.binPauseNext = BIN_PAUSE_STEPS;
        }
      })
      .catch(() => {})
      .finally(() => {
        this.binPollBusy = false;
      });
  }

  /**
   * Every few steps, measure the refinement indicator from the latest primitives. Returns
   * whether a check was encoded: encodeSteps() then ends
   * the batch, so the check can be read back as soon as it is done rather than after steps
   * queued behind it in the same submission: the flow may only run MAX_STEPS_AHEAD past it.
   */
  private encodeFlags(enc: GPUCommandEncoder): boolean {
    if (this.levels === 0 || !this.autoRegrid || this.flagInFlight || this.wantMap || this.stepsSinceFlag < REGRID_EVERY) return false;
    enc.clearBuffer(this.flags);
    const fp = beginPass(enc);
    // With a magnetic field: the indicator reads the primitives of the state as it is now.
    if (this.mhd) this.mhd.encodeAssembly(fp, this.cur, () => this.flat(fp, this.nLeaf, NG));
    fp.setPipeline(this.pFlag);
    fp.setBindGroup(0, this.gFlag);
    this.interior(fp, this.nLeaf);
    fp.end();
    enc.copyBufferToBuffer(this.flags, 0, this.flagStaging, 0, this.cap * 4);
    this.wantMap = true;
    this.stepsSinceFlag = 0;
    return true;
  }

  /** Whether the edge of an outline passes through a block: some of the block inside, some not. */
  private crossesOutline(b: Block, outline: Outline): boolean {
    const size = (BS * this.dx0) / (1 << b.level) / this.cfg.problem.box; // as a fraction of the box
    const n = 8; // samples every two cells, faces included
    const nz = this.cfg.ndim === 3 ? n : 0;
    let first: boolean | undefined;
    for (let k = 0; k <= nz; k++)
      for (let j = 0; j <= n; j++)
        for (let i = 0; i <= n; i++) {
          const inside = outlineContains(outline, (b.c[0] + i / n) * size, (b.c[1] + j / n) * size, nz ? (b.c[2] + k / n) * size : 0.5);
          if (first === undefined) first = inside;
          else if (inside !== first) return true;
        }
    return false;
  }

  /**
   * Refine the mesh to its finest level along the edge of an outline, so that an obstacle
   * drawn with it gets the sharpest boundary the grid can hold. Each level is its own re-mesh,
   * submitted here.
   */
  private refineOutline(outline: Outline) {
    this.flush();
    for (let level = 0; level < this.levels; level++) {
      const edge = this.tree.leaves().filter((b) => b.level < this.tree.levelLimit && this.crossesOutline(b, outline));
      if (!edge.length) break;
      // A flow can hold the whole block budget (a turbulent shear layer does): the obstacle's
      // edge comes first, so merge blocks elsewhere to make room for it.
      this.makeRoom(edge, this.tree.costToRefine(edge) - this.tree.freeSlots);
      // Refine where the edge passes; merge nothing.
      const want = new Set(edge);
      const ops = this.tree.adapt((b) => (want.has(b) ? 1 : 0), 0.5, -1);
      if (!ops) break;
      this.pending = { ops };
      this.flush();
    }
    this.meshEpoch++;
  }

  /**
   * Free at least `need` slots by merging refined blocks away from `keep`: those whose gas
   * asked least for refinement at the last re-mesh first. Blocks on the surface of an obstacle
   * (indicator 1) are never merged. Submitted here, as its own re-mesh.
   */
  private makeRoom(keep: Block[], need: number) {
    if (need <= 0) return;
    const tree = this.tree;
    // Keep the edge and its face neighbors, so refining the edge does not split them again.
    const near = new Set<Block>();
    for (const b of keep) {
      near.add(b);
      for (let f = 0; f < tree.nface; f++) {
        const nb = tree.neighbor(b, f);
        if (nb.kind !== 'bc') near.add(nb.block);
      }
    }
    const ind = this.lastIndicators;
    const score = (b: Block) => (ind && ind[b.slot] > 0 ? ind[b.slot] : 0);
    const parents = new Set<Block>();
    for (const b of tree.leaves()) if (b.parent) parents.add(b.parent);
    const candidates = [...parents]
      .filter((p) => tree.canCoarsen(p) && p.children!.every((ch) => !near.has(ch) && score(ch) < 0.999))
      .map((p) => ({ p, s: Math.max(...p.children!.map(score)) }))
      .sort((a, b) => a.s - b.s);
    const merge = new Set<Block>();
    let freed = 0;
    for (const { p } of candidates) {
      if (freed >= need) break;
      for (const ch of p.children!) merge.add(ch);
      freed += tree.nchild;
    }
    if (!merge.size) return;
    // Merge the chosen parents and nothing else; refine nothing.
    const ops = tree.adapt((b) => (merge.has(b) ? -1 : 1), 2, 0);
    if (!ops) return;
    this.pending = { ops };
    this.flush();
  }

  /** Whether a block reaches into a ball (center and radius in fractions of the box). */
  private touchesBall(b: Block, pos: [number, number, number], r: number): boolean {
    const size = (BS * this.dx0) / (1 << b.level) / this.cfg.problem.box;
    let d2 = 0;
    for (let a = 0; a < this.cfg.ndim; a++) {
      const lo = b.c[a] * size;
      // The nearest point of the block to the center, or to its periodic image.
      const shifts = this.cfg.bcLo[a] === PERIODIC ? [-1, 0, 1] : [0];
      d2 += Math.min(...shifts.map((k) => (Math.min(Math.max(pos[a] + k, lo), lo + size) - pos[a] - k) ** 2));
    }
    return d2 < r * r;
  }

  /** Refine the mesh to its finest level throughout a ball, one level per re-mesh, submitted here. */
  private refineBall(pos: [number, number, number], r: number) {
    this.flush();
    for (let level = 0; level < this.levels; level++) {
      const want = this.tree.leaves().filter((b) => b.level < this.tree.levelLimit && this.touchesBall(b, pos, r));
      if (!want.length) break;
      this.makeRoom(want, this.tree.costToRefine(want) - this.tree.freeSlots);
      const set = new Set(want);
      const ops = this.tree.adapt((b) => (set.has(b) ? 1 : 0), 0.5, -1);
      if (!ops) break;
      this.pending = { ops };
      this.flush();
    }
    this.meshEpoch++;
  }

  /** The refinement indicator of every leaf (by slot) for the current state, read back now. */
  private async measureIndicators(): Promise<Float32Array> {
    // The stepping's own read uses the same staging buffer: let it finish (and be ignored).
    this.meshEpoch++;
    this.wantMap = false;
    if (this.flagWait) await this.flagWait.catch(() => {});
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.flags);
    const pass = beginPass(enc);
    pass.setPipeline(this.pAssemble);
    pass.setBindGroup(0, this.assembleGroup(this.cur));
    this.flat(pass, this.nAll, NG);
    if (this.speciesGroups) {
      // The species' fractions with ghosts, for their part of the indicator.
      pass.setPipeline(this.pSpecies[0]);
      pass.setBindGroup(0, this.gSpecies[0][this.cur][this.speciesCur]);
      this.flat(pass, this.nLeaf, NG);
    }
    pass.setPipeline(this.pFlag);
    pass.setBindGroup(0, this.gFlag);
    this.interior(pass, this.nLeaf);
    pass.end();
    enc.copyBufferToBuffer(this.flags, 0, this.flagStaging, 0, this.cap * 4);
    this.device.queue.submit([enc.finish()]);
    this.flagInFlight = true;
    const done = this.flagStaging.mapAsync(GPUMapMode.READ).then(() => {
      const ind = new Float32Array(this.flagStaging.getMappedRange().slice(0));
      this.flagStaging.unmap();
      return ind;
    });
    this.flagWait = done.then(
      () => {},
      () => {},
    );
    try {
      return await done;
    } finally {
      this.flagInFlight = false;
      this.stepsSinceFlag = 0;
    }
  }

  private submitBrush(op: BrushOp) {
    const enc = this.device.createCommandEncoder();
    this.encodeBrush(enc, op);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Place a star: refine all of it to the finest level, write it, then merge the new blocks
   * that the refinement indicator of the gas as placed does not want (never below the mesh as
   * it was), and write it again on whatever was merged.
   */
  async placeStar(op: BrushOp) {
    if (this.levels === 0) return this.submitBrush(op);
    const tree = this.tree;
    const before = new Set(tree.leaves());
    this.refineBall(op.pos, op.radius);
    this.submitBrush(op);
    let merged = false;
    const between = 0.5 * (this.cfg.refineHi + this.cfg.refineLo);
    for (let level = 0; level < this.levels; level++) {
      const ind = await this.measureIndicators();
      if (tree !== this.tree) return; // the run was reset meanwhile
      // Only blocks the star made are judged, and only for merging: the rest neither refine nor merge.
      const ops = tree.adapt(
        (b) => (before.has(b) ? between : ind[b.slot] === 0 ? -1 : Math.min(ind[b.slot], this.cfg.refineHi)),
        this.cfg.refineHi,
        this.cfg.refineLo,
      );
      if (!ops) break;
      merged = true;
      this.pending = { ops };
      this.flush();
    }
    // Merging averaged the star's profile over the coarser cells: sample it there instead.
    if (merged) this.submitBrush(op);
    this.meshEpoch++;
  }

  /**
   * Apply an editing brush to the leaf cells. Like encodeSteps(), submit before the next call.
   * An obstacle first refines the mesh along its edge, in command buffers of its own, so `enc`
   * must hold nothing that depends on the mesh as it was.
   */
  encodeBrush(enc: GPUCommandEncoder, op: BrushOp) {
    if (op.kind === 'obstacle' && op.outline && this.levels > 0) this.refineOutline(op.outline);
    // A waiting re-mesh goes first: it replaces the tables this pass reads.
    if (this.pending) this.applyRegrid(enc);
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
        this.mhd.encodeFieldBrush(pass, this.cur, (p) => this.interior(p, this.nLeaf));
        this.cur = 1 - this.cur;
      }
      pass.end();
      return;
    }
    this.device.queue.writeBuffer(this.brush, 0, packBrush(op));
    const pass = beginPass(enc);
    pass.setPipeline(this.pBrush);
    pass.setBindGroup(0, this.gBrush[this.cur]);
    this.interior(pass, this.nLeaf);
    // A brush leaves the field as it is: the leaves' faces go along to the brushed state's parity.
    this.mhd?.encodeCommitFaces(pass, 1 - this.cur);
    pass.end();
    this.cur = 1 - this.cur;
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
    this.lumpReader ??= new LumpReader(this.device, 'amr/lumps.wgsl');
    const [u1, u2] = this.state[this.cur];
    const n = this.nLeaf;
    return this.lumpReader.read(n, { 0: this.params, 1: u1.createView(), 2: u2.createView(), 30: this.bt, 31: this.listLeaf }, (pass) =>
      pass.dispatchWorkgroups(Math.ceil(n / 64)),
    );
  }

  addMass(m: Parameters<Simulation['addMass']>[0]): boolean {
    if (this.nMasses >= MAX_MASSES) return false;
    this.device.queue.writeBuffer(this.bh, 16 + 4 * MASS_STRIDE * this.nMasses, packMasses([completeMass(m, this.dx0)]), 4, MASS_STRIDE);
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
    this.refreshPotential();
  }

  /**
   * The cell size the preset's point masses are sized by (their default softening and sink
   * radius): the finest level's where the mesh is kept fully refined around them, otherwise
   * the coarsest level's, which is all they can count on.
   */
  get massDx(): number {
    return this.cfg.problem.refineMasses ? this.dx0 / (1 << this.levels) : this.dx0;
  }

  async probe(f: [number, number, number]): Promise<CellProbe | null> {
    this.flush();
    const three = this.cfg.ndim === 3;
    const axes = three ? 3 : 2;
    for (const b of this.tree.leaves()) {
      // Cells across the computed box at this level (half along a halved axis: f is a fraction of it).
      const h = this.halvedAxes();
      const index = f.slice(0, axes).map((x, a) => {
        const across = (this.n0 << b.level) >> (h[a] ? 1 : 0);
        return Math.min(across - 1, Math.max(0, Math.floor(x * across)));
      });
      if (!index.every((i, a) => Math.floor(i / BS) === b.c[a])) continue;
      const o = this.origin(b.slot);
      const at = [o[0] + index[0] - b.c[0] * BS, o[1] + index[1] - b.c[1] * BS, three ? o[2] + index[2] - b.c[2] * BS : o[2]] as [number, number, number];
      const tree = this.tree;
      const [u1, u2] = await readTexels(this.device, this.state[this.cur], at);
      const dx = this.dx0 / (1 << b.level);
      const lo = domainLo(this.cfg);
      const pos = [0, 1, 2].map((a) => (a < axes ? (index[a] + 0.5) * dx + lo[a] : 0)) as [number, number, number];
      return {
        pos, index, dx, u1: [...u1], u2: [...u2],
        // A block that has been split or merged since the read would show another's state.
        block: tree === this.tree ? { level: b.level, coords: b.c.slice(0, axes), slot: b.slot, indicator: this.indicatorOf.get(b) ?? null } : undefined,
      };
    }
    return null;
  }

  retuneMasses() {
    writeMassRadii(this.device, this.bh, this.cfg, this.massDx, this.nOwnMasses);
    this.refreshPotential();
  }

  massChanged(mass: number) {
    this.rhoMean = mass / this.cfg.problem.box ** this.cfg.ndim;
    this.writeParams();
    this.refreshPotential();
  }

  /** Call after submitting an encoder that encodeSteps() wrote to. */
  submitted() {
    this.pollBins();
    if (!this.wantMap) return;
    this.wantMap = false;
    this.flagInFlight = true;
    const tree = this.tree;
    const epoch = this.meshEpoch;
    this.flagWait = this.flagStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const ind = new Float32Array(this.flagStaging.getMappedRange().slice(0));
        this.flagStaging.unmap();
        if (tree !== this.tree || epoch !== this.meshEpoch) return; // the run was reset, or re-meshed by hand, meanwhile
        this.lastIndicators = ind;
        const leaves = tree.leaves();
        for (const b of leaves) this.indicatorOf.set(b, ind[b.slot]);
        if (!this.autoRegrid) return; // a check already in flight when a replay took over
        // Exactly 0: no gas in the block, only the inside of an obstacle (see amr/flag.wgsl).
        const ops = tree.adapt((b) => (ind[b.slot] === 0 ? -1 : ind[b.slot]), this.cfg.refineHi, this.cfg.refineLo);
        if (ops) this.pending = { ops, steps: tree.lastSteps };
      })
      .finally(() => {
        this.flagInFlight = false;
      });
  }

  /**
   * Re-mesh now, taking the steps a refinement check's re-mesh took (BlockTree.lastSteps). For a
   * replay (ui/session.ts), with autoRegrid off: now, so that it keeps its place among the other
   * edits logged at the same step.
   */
  replayRegrid(steps: AdaptStep[]) {
    this.applyNow(this.tree.replayAdapt(steps));
  }

  /**
   * Re-mesh now, as a refinement check with these indicators (one per leaf, in the order
   * BlockTree.leaves() lists them) would: a replay of a log of the first version, which kept the
   * indicators rather than the steps.
   */
  forceRegrid(indicators: Float32Array) {
    const leaves = this.tree.leaves();
    if (leaves.length !== indicators.length) throw new Error(`the mesh has ${leaves.length} leaves where the log's re-mesh had ${indicators.length}: the replay has gone another way`);
    const at = new Map(leaves.map((b, i) => [b, indicators[i]]));
    for (const b of leaves) this.indicatorOf.set(b, at.get(b)!);
    this.applyNow(this.tree.adapt((b) => (at.get(b) === 0 ? -1 : at.get(b)!), this.cfg.refineHi, this.cfg.refineLo));
  }

  /** Apply a re-mesh the tree has just taken, now (after any re-mesh by hand that waits). */
  private applyNow(ops: RegridOp[] | null) {
    if (!ops) return;
    if (this.pending) {
      // A re-mesh by hand that waits (a star's, an obstacle's) goes first, as it would have.
      const first = this.device.createCommandEncoder();
      this.applyRegrid(first);
      this.device.queue.submit([first.finish()]);
    }
    this.pending = { ops, steps: this.tree.lastSteps };
    const enc = this.device.createCommandEncoder();
    this.applyRegrid(enc);
    this.device.queue.submit([enc.finish()]);
  }

  stepsResume(): Promise<void> | null {
    if (this.compiling) return this.compiled.catch(() => {});
    return this.flagInFlight && this.flagWait ? this.flagWait.catch(() => {}) : null;
  }

  /**
   * Bring the GPU's mesh up to date with the tree. The tree is adapted as soon as indicators
   * arrive, but the GPU tables only change at the start of the next batch of steps; anything
   * that interprets GPU data through the tree must call this first.
   */
  flush() {
    if (!this.pending) return;
    const enc = this.device.createCommandEncoder();
    this.applyRegrid(enc);
    this.device.queue.submit([enc.finish()]);
  }

  step(count = 1): number {
    const enc = this.device.createCommandEncoder();
    const n = this.encodeSteps(enc, count);
    this.device.queue.submit([enc.finish()]);
    this.submitted();
    return n;
  }

  /** Take exactly `count` steps, re-meshing as it goes (tests). */
  async advance(count: number) {
    await this.compiled;
    await this.settled;
    let left = count;
    while (left > 0) {
      left -= this.step(Math.min(left, REGRID_EVERY));
      if (this.flagWait) await this.flagWait;
    }
  }

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
      const diag = diagnosticsFrom(a, this.hasMasses ? this.nMasses : 0);
      if (cosmologyOf(this.cfg)) this.lastScale = withScale(diag, a).scale;
      if (diag.steps === 0 && diag.dt === 0) diag.dt = await this.previewDt();
      // A magnetic field: its diagnostics, and its energy taken out of the reduction's internal
      // energy (E - KE there includes B^2 / 2).
      if (this.mhd && !this.destroyed) {
        diag.magnetic = await this.mhd.diagnostics(this.cur);
        diag.eInt -= diag.magnetic.eMag;
      }
      return unfoldDiagnostics(diag, this.cfg.symmetry);
    } finally {
      this.stagingBusy = false;
    }
  }

  /**
   * The timestep the next step will take, before any has been: the clock holds a step's dt only
   * once the step starts, so a run loaded paused would show 0. The step's own timestep passes
   * (assembly, reduction, clock) run on a copy of the clock; what the assembly rewrites is
   * rebuilt by every step anyway. With per-block timesteps this is the smallest step, the one
   * the fastest blocks take. Called with the staging buffer held.
   */
  /**
   * Set by destroy(). A readback that was waiting when the run was replaced must not then
   * encode work on its destroyed textures (previewDt and the magnetic diagnostics, after
   * readDiagnostics' map).
   */
  private destroyed = false;

  private async previewDt(): Promise<number> {
    if (this.destroyed) return 0;
    // A re-mesh waiting to be applied, or a star being settled: the next step will tell.
    if (this.settling || this.pending) return 0;
    const enc = this.device.createCommandEncoder();
    if (this.dtStale) {
      // As encodeSteps does: drop timestep records of workgroups the block list no longer covers.
      enc.clearBuffer(this.redDt.first);
      this.dtStale = false;
    }
    enc.copyBufferToBuffer(this.clock, 0, this.clockPreview, 0, CLOCK_BYTES);
    const pass = beginPass(enc);
    if (this.mhd) {
      // A magnetic field: its own assembly measures the rates, with the fast magnetosonic speed.
      this.mhd.encodeAssembly(pass, this.cur, () => this.flat(pass, this.nLeaf, NG));
    } else {
      pass.setPipeline(this.pAssemble);
      pass.setBindGroup(0, this.assembleGroup(this.cur));
      this.flat(pass, this.nAll, NG);
    }
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

  async diagnostics(): Promise<Diagnostics> {
    for (;;) {
      const d = await this.readDiagnostics();
      if (d) return d;
      await new Promise((r) => setTimeout(r, 1));
    }
  }

  /** Step until the clock reaches cfg.tEnd, re-meshing as it goes. Returns the steps taken. */
  async runToEnd(_batch = 16, maxSteps = 1e6): Promise<number> {
    if (!Number.isFinite(this.cfg.tEnd)) throw new Error('runToEnd needs a finite tEnd');
    await this.compiled;
    await this.settled;
    const buf = this.device.createBuffer({ size: CLOCK_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      for (;;) {
        const enc = this.device.createCommandEncoder();
        this.encodeSteps(enc, this.levels > 0 ? REGRID_EVERY : 16);
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, CLOCK_BYTES);
        this.device.queue.submit([enc.finish()]);
        this.submitted();
        await buf.mapAsync(GPUMapMode.READ);
        const c = new Float32Array(buf.getMappedRange().slice(0));
        buf.unmap();
        if (this.flagWait) await this.flagWait;
        const [dt, t] = c;
        const steps = c[3];
        if (!(dt === dt) || !(t === t)) throw new Error(`clock is NaN after ${steps} steps`);
        if (dt === 0 && t > 0) return steps;
        if (steps > maxSteps) throw new Error(`no convergence to tEnd after ${steps} steps (t = ${t})`);
      }
    } finally {
      buf.destroy();
    }
  }

  // ---------------------------------------------------------------- display and inspection

  /** Display volume: the finest resolution, capped so it stays a reasonable texture. */
  get displayDims(): [number, number, number] {
    const nd = Math.min(this.n, this.cfg.ndim === 2 ? 2048 : 256);
    // Half along a halved axis of a rectangular grid (src/domain.ts gridDims).
    const h = this.halvedAxes();
    return [h[0] ? nd / 2 : nd, h[1] ? nd / 2 : nd, this.cfg.ndim === 2 ? 1 : h[2] ? nd / 2 : nd];
  }
  /** The mirrored axes halved on a rectangular grid (src/domain.ts gridDims). */
  halvedAxes(): boolean[] {
    const gd = gridDims(this.cfg);
    return [0, 1, 2].map((a) => gd[a] * 2 === this.cfg.n);
  }
  static readonly fieldShader = 'amr/field.wgsl';
  readonly fieldShader = AmrSim.fieldShader;
  get clockBuffer(): GPUBuffer {
    return this.clock;
  }
  displayAtlas(): DisplayAtlas | null {
    if (this.cfg.ndim !== 3) return null;
    return { side: this.slotsPerSide * PAD, slotsPerSide: this.slotsPerSide, roots: this.n0 / BS, halved: this.halvedAxes().reduce((m, h, a) => m | (h ? 1 << a : 0), 0), levels: this.levels, idx: this.idx, list: this.listLeaf, leaves: this.nLeaf };
  }
  fieldResources(parity: number, speciesParity: number): Record<number, GPUBuffer | GPUTextureView> {
    // With a magnetic field the primitives come from its own assembly (the same layout), and the
    // display reads the faces; without, one placeholder texel.
    const r: Record<number, GPUBuffer | GPUTextureView> = {
      2: this.state[parity][0].createView(), 3: this.state[parity][1].createView(), 5: this.idx.createView(),
      6: (this.mhd?.prim ?? this.prim[0]).createView(), 7: this.gfield.createView(), 10: this.pred[1].createView(),
      13: (this.mhd?.faces[parity] ?? this.noFaces()).createView(), 29: this.bins, 30: this.bt, 20: this.bh,
      14: this.mhd?.magResult ?? this.noMag(), 15: this.heat.createView(),
    };
    if (this.speciesGroups) r[9] = this.species[speciesParity].createView();
    return { ...r, ...this.compIn };
  }
  get speciesStride(): number {
    return this.speciesGroups ? this.speciesDepth : 0;
  }
  /** The display pass's face field without a magnetic field: one texel of nothing. */
  private placeholderFaces?: GPUTexture;
  private placeholderMag?: GPUBuffer;
  /** Without a field, the display's magnetic reduction binding: four words nothing reads. */
  private noMag(): GPUBuffer {
    this.placeholderMag ??= this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return this.placeholderMag;
  }
  encodeMagnetic(pass: GPUComputePassEncoder, parity: number) {
    this.mhd?.encodeMagnetic(pass, parity);
  }
  private noFaces(): GPUTexture {
    this.placeholderFaces ??= tex3d(this.device, 1, 'rgba32float', 'no face field', 1);
    return this.placeholderFaces;
  }

  status(): string {
    const s = this.tree.stats();
    const full = this.tree.levelLimit < this.levels
      ? `, block budget limits refinement to ${this.tree.levelLimit} of ${this.levels} levels`
      : this.tree.freeSlots < this.tree.nchild ? `, block budget full (${this.cap} with their parents)` : '';
    // Per-block timesteps: what the last cycles measured saved, and whether they are in use.
    const share = this.binShare;
    const bins = this.builtBins === 0 || share === null ? ''
      : this.binPause > 0 ? `, local timesteps paused (would save ${Math.max(0, 100 * (1 - share)).toFixed(0)}% of block updates)`
      : `, local timesteps save ${Math.max(0, 100 * (1 - share)).toFixed(0)}% of block updates`;
    return `${s.leaves} blocks, ${((100 * s.cells) / this.n ** this.cfg.ndim).toFixed(0)}% of uniform${full}${bins}`;
  }

  /**
   * Boxes of `w` texels per side (w^3; w^2 in 2D) at the first interior cell of each block's
   * slot in an atlas texture (offset by `offset` texels, for the species' groups), packed
   * block by block as [z][y][x][channel]. Each box is copied on its own, into staging buffers
   * of at most READ_CHUNK_BYTES: a whole atlas texture of a large mesh is several GB, more
   * than the browser maps at once.
   */
  private async readBoxes(tex: GPUTexture, blocks: Block[], channels: 1 | 4, w = BS, offset: readonly number[] = [0, 0, 0]): Promise<Float32Array> {
    const d = this.device;
    const depth = this.cfg.ndim === 2 ? 1 : w;
    const rowBytes = w * channels * 4;
    const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
    const perBlock = bytesPerRow * w * depth;
    const per = w * w * depth * channels;
    const out = new Float32Array(blocks.length * per);
    const chunk = Math.max(1, Math.floor(READ_CHUNK_BYTES / perBlock));
    const rowF = rowBytes / 4;
    const padF = bytesPerRow / 4;
    for (let start = 0; start < blocks.length; start += chunk) {
      const part = blocks.slice(start, start + chunk);
      const buf = d.createBuffer({ label: 'atlas readback', size: perBlock * part.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = d.createCommandEncoder();
      part.forEach((b, i) => {
        const o = this.origin(b.slot);
        enc.copyTextureToBuffer({ texture: tex, origin: [o[0] + offset[0], o[1] + offset[1], o[2] + offset[2]] }, { buffer: buf, offset: i * perBlock, bytesPerRow, rowsPerImage: w }, [w, w, depth]);
      });
      d.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Float32Array(buf.getMappedRange());
      for (let i = 0; i < part.length; i++) {
        const from = (i * perBlock) / 4;
        const to = (start + i) * per;
        for (let r = 0; r < w * depth; r++) out.set(src.subarray(from + r * padF, from + r * padF + rowF), to + r * rowF);
      }
      buf.unmap();
      buf.destroy();
    }
    return out;
  }

  /** Every leaf cell with its position and size, the leaves in tree order. */
  async readLeaves(): Promise<LeafCells> {
    this.flush();
    const [a, b] = this.state[this.cur];
    const leaves = this.tree.leaves();
    const u1 = await this.readBoxes(a, leaves, 4);
    const u2 = await this.readBoxes(b, leaves, 4);
    const three = this.cfg.ndim === 3;
    const depth = three ? BS : 1;
    const count = leaves.length * BS * BS * depth;
    const out: LeafCells = {
      x: new Float32Array(count), y: new Float32Array(count), z: new Float32Array(count), dx: new Float32Array(count),
      u1, u2, count,
    };
    const lo = domainLo(this.cfg);
    let q = 0;
    for (const blk of leaves) {
      const dx = this.dx0 / (1 << blk.level);
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, q++) {
            out.x[q] = (blk.c[0] * BS + i + 0.5) * dx + lo[0];
            out.y[q] = (blk.c[1] * BS + j + 0.5) * dx + lo[1];
            out.z[q] = three ? (blk.c[2] * BS + k + 0.5) * dx + lo[2] : 0;
            out.dx[q] = dx;
          }
    }
    return out;
  }

  /**
   * Write every leaf's conserved state back, in the order of readLeaves(), on the mesh it was read
   * from, and start the clock again from 0 with the potential of that state, as Sim.setState does
   * (validation: a test that stops the gas to settle it).
   */
  setLeaves(u1: Float32Array, u2: Float32Array) {
    this.flush();
    const depth = this.cfg.ndim === 3 ? BS : 1;
    const per = BS * BS * depth * 4;
    const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
    this.tree.leaves().forEach((b, i) => {
      const origin = this.origin(b.slot);
      this.device.queue.writeTexture({ texture: this.state[this.cur][0], origin }, u1.subarray(i * per, (i + 1) * per) as Float32Array<ArrayBuffer>, layout, [BS, BS, depth]);
      this.device.queue.writeTexture({ texture: this.state[this.cur][1], origin }, u2.subarray(i * per, (i + 1) * per) as Float32Array<ArrayBuffer>, layout, [BS, BS, depth]);
    });
    this.device.queue.writeBuffer(this.clock, 0, clockState(this.cfg));
    this.refreshPotential();
  }

  /**
   * Mass fractions of every leaf cell (validation): X[i][cell], the cells in the order of
   * readLeaves(). Empty without species.
   */
  async readLeafSpecies(): Promise<Float32Array[]> {
    const G = this.speciesGroups;
    if (!G) return [];
    this.flush();
    const leaves = this.tree.leaves();
    const tex = this.species[this.speciesCur];
    // Each group of four species is a stack of the atlas's depth along z.
    const raw: Float32Array[] = [];
    for (let g = 0; g < G; g++) raw.push(await this.readBoxes(tex, leaves, 4, BS, speciesGroupOffset(g, this.speciesDepth)));
    const depth = this.cfg.ndim === 3 ? BS : 1;
    const count = leaves.length * BS * BS * depth;
    const ns = this.cfg.problem.species!.length;
    const X = Array.from({ length: ns }, () => new Float32Array(count));
    for (let q = 0; q < count; q++) {
      let sum = 0;
      for (let g = 0; g < G; g++) for (let c = 0; c < 4; c++) sum += raw[g][q * 4 + c];
      for (let s = 0; s < ns; s++) X[s][q] = raw[Math.floor(s / SPECIES_PER_GROUP)][q * 4 + (s % SPECIES_PER_GROUP)] / sum;
    }
    return X;
  }

  /** The potential on every leaf cell, in the same order as readLeaves() (validation only). */
  async readLeafPotential(): Promise<Float32Array> {
    this.flush();
    return this.readBoxes(this.grav!.phi, this.tree.leaves(), 1);
  }

  /**
   * Flux consistency check on the current mesh. Evaluates one step's face mass fluxes and
   * compares the two sides of every leaf face: bit for bit between same-level leaves, and the
   * coarse flux against the mean of the fine ones at refinement boundaries.
   */
  async fluxMismatch(): Promise<{ faces: number; mismatched: number; coarseFine: number; worstCoarseFine: number }> {
    const d = this.device;
    this.flush();
    const pipe = await computePipeline(d, assemble('amr/update.wgsl', this.slots), 'amr update (flux debug)', { DEBUG_FLUX: 1, ...this.tileConstants() });
    const p = this.cur;
    const enc = d.createCommandEncoder();
    const pass = beginPass(enc);
    pass.setPipeline(this.pAssemble);
    pass.setBindGroup(0, this.assembleGroup(p));
    this.flat(pass, this.nAll, NG);
    pass.setPipeline(this.pPredict);
    pass.setBindGroup(0, this.gPredict);
    this.flat(pass, this.nLeaf, 1);
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this.updateGroup(pipe, p));
    this.interior(pass, this.nLeaf);
    pass.end();
    d.queue.submit([enc.finish()]);
    const side = this.slotsPerSide * PAD;
    const depthTex = this.cfg.ndim === 2 ? this.layers : side;
    const f1 = await readTexture(d, this.state[1 - p][0], side, 4, depthTex);
    const f2 = await readTexture(d, this.state[1 - p][1], side, 4, depthTex);
    const three = this.cfg.ndim === 3;
    const flux = (b: Block, l: number[], f: number) => {
      const [ox, oy, oz] = this.origin(b.slot);
      const cell = (((oz + (three ? l[2] : 0)) * side + oy + l[1]) * side + ox + l[0]) * 4;
      return f < 4 ? f1[cell + f] : f2[cell + f - 4];
    };
    let faces = 0;
    let mismatched = 0;
    let coarseFine = 0;
    let worst = 0;
    const depth = three ? BS : 1;
    for (const b of this.tree.leaves()) {
      for (let axis = 0; axis < this.cfg.ndim; axis++) {
        for (let side = 0; side < 2; side++) {
          const nb = this.tree.neighbor(b, 2 * axis + side);
          // Physical boundaries have no partner; a coarser partner is checked from its own side.
          if (nb.kind !== 'same') continue;
          const refined = !!nb.block.children;
          if (!refined && side === 0) continue; // same-level pairs are counted once, from the lower block
          for (let k = 0; k < depth; k++)
            for (let j = 0; j < BS; j++)
              for (let i = 0; i < BS; i++) {
                const l = [i, j, k];
                if (l[axis] !== (side ? BS - 1 : 0)) continue;
                const mine = flux(b, l, 2 * axis + side);
                if (!refined) {
                  const m = l.slice();
                  m[axis] = 0;
                  faces++;
                  if (mine !== flux(nb.block, m, 2 * axis)) mismatched++;
                  continue;
                }
                // Mean of the fine fluxes through the same face, from the children beyond it.
                let sum = 0;
                let cnt = 0;
                const t = [0, 1, 2].filter((q) => q !== axis && (three || q !== 2));
                const sub = t.length === 2 ? [[0, 0], [1, 0], [0, 1], [1, 1]] : [[0], [1]];
                for (const sb of sub) {
                  const bits = [0, 0, 0];
                  const m = [0, 0, 0];
                  bits[axis] = side ? 0 : 1;
                  m[axis] = side ? 0 : BS - 1;
                  t.forEach((q, n) => {
                    bits[q] = l[q] >= BS / 2 ? 1 : 0;
                    m[q] = 2 * (l[q] - (BS / 2) * bits[q]) + sb[n];
                  });
                  const ch = nb.block.children![bits[0] | (bits[1] << 1) | (bits[2] << 2)];
                  sum += flux(ch, m, 2 * axis + (side ? 0 : 1));
                  cnt++;
                }
                coarseFine++;
                worst = Math.max(worst, Math.abs(mine - sum / cnt));
              }
        }
      }
    }
    return { faces, mismatched, coarseFine, worstCoarseFine: worst };
  }

  /** Ideal MHD on the blocks (mhd/amr_mhd.ts); none without a magnetic field. */
  mhd?: AmrMhd;

  /**
   * Validation: every face two leaves both hold, compared. Between leaves at the same level the
   * copies should agree bit for bit; across a refinement boundary the coarse face should be the
   * mean of the fine faces on it (to rounding). Returns how many same-level copies differ, and
   * the largest coarse-fine difference over the largest face value.
   */
  async faceConsistency(): Promise<{ shared: number; mismatched: number; coarseFine: number; cfFaces: number; worst: string[] }> {
    const bins = await this.readBins();
    const worst: string[] = [];
    const side = this.slotsPerSide * PAD;
    const depth = this.layers * (this.cfg.ndim === 2 ? 1 : PAD);
    const f = await this.mhd!.readAtlas(this.cur, side, depth);
    const flat = this.cfg.ndim === 2;
    const at = (slot: number, l: number[], a: number) => {
      const o = this.origin(slot);
      const x = o[0] + l[0];
      const y = o[1] + l[1];
      const z = flat ? o[2] : o[2] + l[2];
      return f[4 * ((z * side + y) * side + x) + a];
    };
    let shared = 0;
    let mismatched = 0;
    let coarseFine = 0;
    let cfFaces = 0;
    let big = 0;
    const axes = flat ? 2 : 3;
    for (const b of this.tree.leaves()) {
      for (let k = 0; k < 2 * axes; k++) {
        const a = k >> 1;
        const up = (k & 1) === 1;
        const nb = this.tree.neighbor(b, k);
        if (nb.kind === 'bc') continue;
        // Same-level pairs are counted once, from the lower block; refined neighbors on either side.
        if (!up && !(nb.kind === 'same' && nb.block.children)) continue;
        const t1 = (a + 1) % 3;
        const t2 = (a + 2) % 3;
        const n1 = flat && t1 === 2 ? 1 : BS;
        const n2 = flat && t2 === 2 ? 1 : BS;
        for (let j = 0; j < n2; j++)
          for (let i = 0; i < n1; i++) {
            const l = [0, 0, 0];
            l[a] = up ? BS : 0;
            l[t1] = i;
            l[t2] = j;
            const mine = at(b.slot, l, a);
            big = Math.max(big, Math.abs(mine));
            if (nb.kind === 'same' && !nb.block.children) {
              const m = [...l];
              m[a] = 0;
              shared++;
              if (at(nb.block.slot, m, a) !== mine) {
                mismatched++;
                // The first few, for a failed test's report: where, and the two blocks' bins.
                if (worst.length < 8) {
                  const d = Math.abs(at(nb.block.slot, m, a) - mine);
                  worst.push(`L${b.level} [${b.c}] bin ${bins[b.slot]} / [${nb.block.c}] bin ${bins[nb.block.slot]}: axis ${a} at ${i},${j} off by ${d.toExponential(2)}`);
                }
              }
            } else if (nb.kind === 'same') {
              // Refined beyond: the mean of the children's faces on this one.
              let sum = 0;
              let count = 0;
              for (const ch of nb.block.children!) {
                if ((ch.c[a] & 1) !== (up ? 0 : 1)) continue; // the children on the near side
                for (let q = 0; q < (flat && t2 === 2 ? 1 : 2); q++)
                  for (let p = 0; p < (flat && t1 === 2 ? 1 : 2); p++) {
                    const gl = [0, 0, 0];
                    gl[a] = up ? 0 : BS;
                    gl[t1] = 2 * (b.c[t1] * BS + i) + p - ch.c[t1] * BS;
                    gl[t2] = flat && t2 === 2 ? 0 : 2 * (b.c[t2] * BS + j) + q - ch.c[t2] * BS;
                    if (gl[t1] < 0 || gl[t1] >= BS || gl[t2] < 0 || gl[t2] >= (flat && t2 === 2 ? 1 : BS)) continue;
                    sum += at(ch.slot, gl, a);
                    count++;
                  }
              }
              if (count) {
                const d = Math.abs(sum / count - mine);
                if (d > coarseFine && d > 1e-7 * big && worst.length < 8) {
                  worst.push(`coarse [${b.c}] bin ${bins[b.slot]} / fine bins ${nb.block.children!.map((c) => bins[c.slot])}: axis ${a} at ${i},${j} off by ${d.toExponential(2)}`);
                }
                coarseFine = Math.max(coarseFine, d);
                cfFaces++;
              }
            }
          }
      }
    }
    return { shared, mismatched, coarseFine: big > 0 ? coarseFine / big : 0, cfFaces, worst };
  }

  /**
   * The faces of each of `leaves` (amr/amr_mhd.ts blockFaces' layout: (BS + 1)^d texels of four
   * floats, the fourth unused), as the run holds them now.
   */
  async leafFaces(leaves: Block[]): Promise<Float32Array[]> {
    const W = BS + 1;
    const per = W * W * (this.cfg.ndim === 2 ? 1 : W) * 4;
    const f = await this.readBoxes(this.mhd!.faces[this.cur], leaves, 4, W);
    return leaves.map((_, i) => f.subarray(i * per, (i + 1) * per));
  }

  /** Each slot's time bin (0 when they are off). */
  private async readBins(): Promise<Uint32Array> {
    const d = this.device;
    const buf = d.createBuffer({ size: this.bins.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.bins, 0, buf, 0, this.bins.size);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const r = new Uint32Array(buf.getMappedRange().slice(0)).subarray(1);
    buf.destroy();
    return r;
  }

  /** Magnetic energy and divergence of the current state; null without a magnetic field. */
  async magnetic() {
    return this.mhd ? this.mhd.diagnostics(this.cur) : null;
  }

  destroy() {
    this.mhd?.destroy();
    this.prs?.destroy();
    this.placeholderFaces?.destroy();
    this.destroyed = true;
    this.lumpReader?.destroy();
    this.settleEpoch++; // abandon a settling in progress: its textures are going
    for (const s of this.state) for (const t of s) t.destroy();
    for (const t of [...this.prim, ...this.pred, this.idx, this.gfield, this.dummyPhi, ...this.species]) t.destroy();
    this.heat?.destroy();
    this.comp?.destroy();
    this.speciesX?.destroy();
    this.speciesHalf?.destroy();
    this.burn?.destroy();
    this.massFlux?.destroy();
    this.debrisBuf?.destroy();
    for (const b of [this.bins, this.binsB, this.listAct, this.listUpd, this.listEnd, this.listDep, this.binCnt, this.binArgs, this.binStaging]) b.destroy();
    this.grav?.destroy();
  }
}
