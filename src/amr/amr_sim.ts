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
// With self-gravity (3D), the potential is solved on every tree level after the update
// (src/amr/gravity.ts), and the second half of the gravity kick runs as its own pass.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d, type Bindable, type Gpu } from '../gpu/device';
import { MAX, Reducer, SUM } from '../gpu/reduce';
import { beginPass } from '../gpu/timing';
import { memoryBudget, noteAllocation } from '../gpu/memory';
import { AmrGravity, FIELD_SLOTS, FIELD_SLOTS_BINS } from './gravity';
import { makeEos } from '../eos/eos';
import { BurnPass, burnIndicator, burnIndices, flagBurnSlot } from '../burn/burn';
import { cellMark, isSelfGravity, PERIODIC } from '../problems/types';
import { Settler, settlingOf, SETTLE_ROUNDS, SETTLE_TOL, type SettleCells } from '../problems/settle';
import { type CellProbe, type DisplayAtlas, readTexels, completeMass, MassReader, MAX_MASSES, BRUSH_BYTES, DIAG_FLOATS, DIAG_OPS, diagnosticsFrom, hydroSlots, initialMasses, MASS_FLOATS, MASS_RED, MASS_STRIDE, MAX_SOURCES, outlineContains, packBrush, packMasses, packParams, packSources, PARAM_BYTES, readRestart, SOURCE_BYTES, type RestartState, type BrushOp, type Diagnostics, type Outline, type SimConfig, type Simulation, type SourceSpec, writeMassRadii, burnSpec } from '../sim';
import { BlockTree, BS, BT_STRIDE, NG, PAD, RG_STRIDE, type Block, gridBox, type GridBox, type RegridOp } from './tree';
import { flagSpeciesSlot, packSpecies, speciesGroups, SPECIES_JUMP_SCALE, SPECIES_PER_GROUP } from '../species/species';
import { DebrisBuffers, debrisSpec, type DebrisLedger, type DebrisReport } from '../debris/debris';
import { compSrcSlot, hydrogenParticles, mixMoments, problemCoefficients, type Moments } from '../species/mixture';

/**
 * Steps between refinement checks, and how far the flow may run past an unanswered check.
 * Signals cross at most about a third of a cell per step, and a block is 16 cells with a
 * buffer block kept around flagged ones, so 24 steps cannot carry a feature out of its refinement.
 */
const REGRID_EVERY = 8;
const MAX_STEPS_AHEAD = 24;
/** Samples of the initial conditions per pass when building the starting mesh, shared by its leaves. */
const INDICATOR_SAMPLES_2D = 1 << 25;
const INDICATOR_SAMPLES_3D = 1 << 24;
/** Most bins of per-block timesteps: a cycle of 2^3 substeps is one refinement interval. */
const MAX_TIME_BINS = 3;
/** The dispatch-size table of per-block timesteps (amr/bins_args.wgsl): 64 words, triple i at 12 i bytes. */
const BIN_ARGS_BYTES = 256;
const ARG = (i: number) => 12 * i;
/** Slabs of a long 2D block list (amr/list.wgsl): must match SLAB_QUADS and FLAT_ROWS there. */
const SLAB_QUADS = 16384;
const FLAT_ROWS = 32768;
/** Workgroups per block of a flat dispatch (amr/list.wgsl, inv_flat) out to `reach` ghost layers. */
const flatGroups = (ndim: number, reach: number) => Math.ceil((BS + 2 * reach) ** ndim / 64);
const BIN_TOTALS = 60;
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
const BIN_PAUSE_MAX = 1536;

interface RegridPlan {
  ops: RegridOp[];
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
  private pSpecies: GPUComputePipeline[] = [];
  /** per pipeline (assemble, predict, move, regrid): [state parity][species parity] */
  private gSpecies: GPUBindGroup[][][] = [];
  /** Burning of fuel to ash (burn/burn.ts), over the whole atlas after the species pass; none without. */
  private burn?: BurnPass;
  private idx: GPUTexture;
  private params: GPUBuffer;
  private clock: GPUBuffer;
  /** A copy of the clock, for the timestep of a step not yet taken (previewDt). */
  private clockPreview: GPUBuffer;
  private gClockPreview!: GPUBindGroup;
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
  private dummyPhi: GPUTexture;
  private redMp?: Reducer;
  private redBh?: Reducer;
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
  private binCnt: GPUBuffer;
  private binArgs: GPUBuffer;
  private pTick?: GPUComputePipeline;
  private pAdvance?: GPUComputePipeline;
  private gAdvance?: GPUBindGroup;
  private gCorrectB: GPUBindGroup[] = [];
  private gCommitL: GPUBindGroup[] = [];
  /** the point masses' reduction, in which a sink takes gas from the blocks finishing a step */
  private redBhB?: Reducer;
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
  private binPause = 0;
  private binPauseNext = BIN_PAUSE_STEPS;
  /** share of block updates made in the last cycles measured (1 = no saving), or null */
  binShare: number | null = null;

  private pAssemble!: GPUComputePipeline;
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
    const fit = Math.floor(memoryBudget() / AmrSim.blockBytes(cfg.ndim));
    if (cfg.ndim === 3) {
      const auto = Math.max(6, Math.ceil(Math.cbrt(Math.max(full, roots * 2)) - 1e-9));
      // 3D passes run three workgroups per block along x (and gravity's one per block along y),
      // so the blocks must stay within a third of what a dispatch may have along an axis.
      const most = Math.floor(Math.cbrt(Math.floor(gpu.limits.maxComputeWorkgroupsPerDimension / 3)) + 1e-9);
      this.slotsPerSide = Math.min(limit, most, explicit ? Math.ceil(Math.cbrt(explicit) - 1e-9) : auto, Math.max(1, Math.floor(Math.cbrt(fit) + 1e-9)));
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
    noteAllocation(this.cap * AmrSim.blockBytes(cfg.ndim));

    const side = this.slotsPerSide * PAD;
    const t = (label: string) => tex3d(d, side, 'rgba32float', label, this.layers * (cfg.ndim === 2 ? 1 : PAD));
    for (let p = 0; p < 2; p++) this.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    this.prim = [t('W1'), t('W2')];
    this.pred = [t('PA'), t('PB')];
    const G = this.speciesGroups;
    if (G > 0) {
      this.speciesDepth = this.layers * (cfg.ndim === 2 ? 1 : PAD);
      const depth = this.speciesDepth * G;
      if (depth > d.limits.maxTextureDimension3D) {
        throw new Error(`${cfg.problem.species!.length} species need an atlas ${depth} texels deep, more than this GPU allows (${d.limits.maxTextureDimension3D}): pick a lower resolution`);
      }
      const ts = (label: string) => tex3d(d, side, 'rgba32float', label, depth);
      this.species = [ts('species 0'), ts('species 1')];
      this.speciesX = ts('species fractions');
      this.speciesHalf = ts('species half step');
      const faces = (BS + 1) ** 2 * (cfg.ndim === 3 ? BS + 1 : 1) * 3 * 4;
      this.massFlux = d.createBuffer({ label: 'face mass fluxes', size: this.cap * faces, usage: GPUBufferUsage.STORAGE });
      if (this.compCoeffs) this.comp = t('composition');
    }
    const m = nr << this.levels;
    this.idx = d.createTexture({
      label: 'leaf index',
      size: [m, m, cfg.ndim === 3 ? m : 1],
      dimension: '3d',
      format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clock = d.createBuffer({ size: 16, usage: storage });
    this.clockPreview = d.createBuffer({ size: 16, usage: storage });
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
    this.binCnt = d.createBuffer({ size: 16, usage: storage });
    this.binArgs = d.createBuffer({ size: BIN_ARGS_BYTES, usage: storage | GPUBufferUsage.INDIRECT });
    this.binStaging = d.createBuffer({ size: BIN_ARGS_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.staging = d.createBuffer({ size: DIAG_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  /**
   * GPU bytes per atlas block: nine rgba32float textures (two state pairs, primitives,
   * predictor, gravity field) and two r32float ones (gravity's previous potential and its rate).
   */
  static blockBytes(ndim: 2 | 3): number {
    return PAD ** ndim * (9 * 16 + 2 * 4);
  }

  /** Build the simulation; if the GPU runs out of memory, try again with a smaller atlas. */
  static async create(gpu: Gpu, cfg: SimConfig): Promise<AmrSim> {
    for (let slots = 0; ; ) {
      gpu.device.pushErrorScope('out-of-memory');
      let sim: AmrSim | null = null;
      let failure: unknown = null;
      try {
        sim = await AmrSim.build(gpu, cfg, slots);
        await gpu.device.queue.onSubmittedWorkDone();
      } catch (e) {
        failure = e;
      }
      const oom = await gpu.device.popErrorScope();
      if (!oom && sim) return sim;
      const tried = sim?.cap ?? slots;
      sim?.destroy();
      const next = Math.floor(tried * 0.85 ** cfg.ndim);
      const roots = ((cfg.n >> cfg.amrLevels) / BS) ** cfg.ndim;
      if (!oom || !tried || next < roots * 2) throw failure ?? new Error('not enough GPU memory for the adaptive mesh at this resolution');
      console.warn(`adaptive mesh: out of GPU memory with ${tried} blocks, retrying with ${next}`);
      slots = next;
    }
  }

  private static async build(gpu: Gpu, cfg: SimConfig, slots: number): Promise<AmrSim> {
    const sim = new AmrSim(gpu, cfg, slots);
    const d = sim.device;
    const v = (t: GPUTexture) => t.createView();
    const depth = cfg.ndim === 2 ? 1 : BS;
    sim.redDt = await Reducer.create(
      d,
      [sim.cap * BS, BS, depth],
      [MAX, MAX],
      '',
      {},
      [],
      'amr timestep',
      [flatGroups(cfg.ndim, NG), sim.cap, 1], // one record per workgroup of the assembly pass
    );
    sim.pClock = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock', { PER_DX: 1 });
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock, 3: sim.bh });
    sim.gClockPreview = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clockPreview, 3: sim.bh });
    // With species the indicator also reads their fractions, ghosts included (amr/species_assemble.wgsl).
    // With burning, also how fast it burns (burn/burn.ts).
    const burn = burnSpec(cfg);
    const flagSlots: Record<string, string> = { flag_species: flagSpeciesSlot(sim.speciesGroups, sim.speciesDepth) };
    if (burn) Object.assign(flagSlots, { flag_burn: flagBurnSlot(burn, cfg.problem.species), eos: hydroSlots(cfg).eos });
    sim.pFlag = await computePipeline(d, assemble('amr/flag.wgsl', flagSlots), 'amr flag');
    sim.gFlag = bindGroup(d, sim.pFlag, {
      0: sim.params, 1: v(sim.prim[0]), 2: v(sim.prim[1]), 3: sim.flags, 4: sim.bh, 30: sim.bt, 31: sim.listLeaf,
      ...((sim.speciesX ? { 5: v(sim.speciesX) } : {}) as Record<number, GPUTextureView>),
    });
    // Point masses, in 2D and 3D: their reduction over the leaf cells and their orbit passes.
    sim.redBh = await Reducer.create(
      d, [sim.cap * BS, BS, depth], new Array(MASS_RED).fill(SUM), 'amr/bh_moment.wgsl', { bins: 'amr/bins_off.wgsl' },
      [0, 1].map((p) => ({ 0: sim.params, 1: v(sim.state[p][0]), 2: sim.bh, 3: v(sim.state[p][1]), 4: sim.clock, 30: sim.bt })), 'amr point masses',
    );
    sim.pBhDrift = await computePipeline(d, assemble('gravity/bh_drift.wgsl'), 'point mass drift');
    sim.gBhDrift = bindGroup(d, sim.pBhDrift, { 0: sim.params, 1: sim.clock, 2: sim.bh });
    sim.pBhKick = await computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass kick');
    sim.pBhEval = await computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'point mass accelerations', { LIVE: 0 });
    const kick = { 0: sim.params, 1: sim.clock, 2: sim.redBh.result, 3: sim.bh };
    sim.gBhKick = bindGroup(d, sim.pBhKick, kick);
    sim.gBhEval = bindGroup(d, sim.pBhEval, kick);
    if (cfg.ndim === 2) {
      // No self-gravity in 2D, so no solver: the field atlas holds the point masses alone.
      // The potential bindings are never read (there is none), but the pass declares them.
      sim.pMassField = await computePipeline(d, assemble('amr/gravity/field.wgsl', FIELD_SLOTS), 'amr point mass field');
      sim.gMassField = bindGroup(d, sim.pMassField, {
        0: sim.params, 1: v(sim.dummyPhi), 2: v(sim.dummyPhi), 3: v(sim.gfield), 4: sim.bh, 5: sim.clock, 30: sim.bt, 31: sim.listAll,
      });
    }
    if (cfg.ndim === 3) {
      // Self-gravity machinery is built for every 3D run, so gravity can be switched on later.
      const u1 = [0, 1].map((p) => v(sim.state[p][0]));
      sim.redMp = await Reducer.create(d, [sim.cap * BS, BS, BS], new Array(10).fill(SUM), 'amr/multipole.wgsl', {}, u1.map((t) => ({ 0: sim.params, 1: t, 30: sim.bt })), 'amr multipole');
      sim.grav = await AmrGravity.create({
        device: d, slotsPerSide: sim.slotsPerSide, levels: sim.levels, params: sim.params, bt: sim.bt,
        listAll: sim.listAll, listLeaf: sim.listLeaf, listLevel: sim.listLevel, rg: sim.rg, clock: sim.clock, bh: sim.bh,
        multipole: sim.redMp.result, stateU1: [sim.state[0][0], sim.state[1][0]], gfield: sim.gfield,
        bins: sim.bins, listAct: sim.listAct, listUpd: sim.listUpd, binArgs: sim.binArgs,
      });
    }
    const spec = debrisSpec(cfg);
    if (spec) {
      sim.debrisBuf = new DebrisBuffers(d, spec);
      sim.pCensus = await computePipeline(d, assemble('debris/census_amr.wgsl'), 'amr debris census');
    }
    await sim.buildHydro();
    sim.reset();
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

  /** (Re)build every pipeline that depends on the EOS, Riemann solver or limiter. */
  async buildHydro() {
    const d = this.device;
    const slots = this.slots;
    const v = (t: GPUTexture) => t.createView();
    [this.pAssemble, this.pPredict, this.pUpdate, this.pRegrid, this.pCorrect, this.pBrush] = await Promise.all([
      computePipeline(d, assemble('amr/assemble.wgsl', slots), 'amr assemble'),
      computePipeline(d, assemble('amr/predict.wgsl', slots), 'amr predict'),
      computePipeline(d, assemble('amr/update.wgsl', slots), 'amr update'),
      computePipeline(d, assemble('amr/regrid.wgsl', slots), 'amr regrid'),
      computePipeline(d, assemble('amr/correct.wgsl', slots), 'amr gravity correct'),
      computePipeline(d, assemble('amr/brush.wgsl', slots), 'amr brush'),
    ]);
    this.gBrush = [0, 1].map((p) =>
      bindGroup(d, this.pBrush, {
        0: this.params, 1: this.brush, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]),
        30: this.bt, 31: this.listLeaf, ...this.compIn,
      }),
    );
    const K = (this.builtBins = this.timeBins);
    if (K > 0) await this.buildBins(K);
    // With per-block timesteps the passes also read the bins, and the assembly the half-step
    // predictions of blocks part way through a step.
    const bin: Record<number, Bindable> = K > 0 ? { 29: this.bins! } : {};
    const mid: Record<number, Bindable> = K > 0 ? { 7: v(this.pred[0]), 8: v(this.pred[1]) } : {};
    const predict = (list: GPUBuffer) =>
      bindGroup(d, this.pPredict, {
        0: this.params, 1: this.clock, 2: v(this.prim[0]), 3: v(this.prim[1]), 4: v(this.gfield), 5: v(this.pred[0]), 6: v(this.pred[1]), 30: this.bt, 31: list, ...bin,
        ...this.compIn,
      });
    const assembly = (p: number, list: GPUBuffer, q = -1) =>
      bindGroup(d, this.pAssemble, {
        0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]),
        5: v(this.gfield), 6: this.redDt.first, 30: this.bt, 31: list, ...bin, ...mid,
        ...((q >= 0 ? { 25: v(this.species[q]), 26: v(this.comp!) } : {}) as Record<number, GPUTextureView>),
      });
    this.gPredict = predict(this.listLeaf);
    this.gAssemble = [];
    this.gUpdate = [];
    this.gRegrid = [];
    this.gCorrect = [];
    this.gAssembleB = [];
    this.gUpdateB = [];
    this.gCommit = [];
    this.gCommitB = [];
    this.gCorrectB = [];
    this.gCommitL = [];
    if (K > 0) this.gPredictB = predict(this.listEnd);
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // Second half kick: from the updated state (the other parity) back into this one.
      const correct = (list: GPUBuffer) =>
        bindGroup(d, this.pCorrect, {
          0: this.params, 1: this.clock, 2: v(this.state[o][0]), 3: v(this.state[o][1]), 4: v(this.gfield),
          5: v(this.state[p][0]), 6: v(this.state[p][1]), 7: this.bh, 30: this.bt, 31: list, ...bin, ...this.compIn,
        });
      if (K > 0) {
        this.gAssembleB.push(assembly(p, this.listAct));
        this.gUpdateB.push(this.updateGroup(this.pUpdate, p, this.listUpd));
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
      this.gUpdate.push(this.updateGroup(this.pUpdate, p));
      this.gRegrid.push(
        bindGroup(d, this.pRegrid, {
          0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]),
          5: v(this.state[o][0]), 6: v(this.state[o][1]), 7: this.rg, 31: this.listLeaf,
        }),
      );
    }
    if (this.speciesGroups > 0) await this.buildSpecies();
    this.redDiag = await Reducer.create(
      d,
      [this.cap * BS, BS, this.cfg.ndim === 2 ? 1 : BS],
      DIAG_OPS,
      'amr/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.grav?.phi ?? this.dummyPhi), 30: this.bt, ...this.compIn })),
      'amr diagnostics',
    );
  }

  /** Pipelines and bind groups of the species passes (amr/species_*.wgsl). */
  private async buildSpecies() {
    const d = this.device;
    const v = (t: GPUTexture) => t.createView();
    const ns = { species_n: `const NG: u32 = ${this.speciesGroups}u;\nconst SP_DEPTH: i32 = ${this.speciesDepth};\n`, comp: hydroSlots(this.cfg).comp };
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
      per(() => bindGroup(d, pP, { 0: this.params, 1: this.clock, 2: v(this.prim[0]), 4: sx, 5: v(this.speciesHalf!), 30: this.bt, 31: this.listLeaf })),
      per((p, q) =>
        bindGroup(d, pM, {
          0: this.params, 1: this.clock, 3: v(this.state[p][1]), 4: v(this.state[1 - p][0]), 5: v(this.species[q]), 6: v(this.species[1 - q]),
          7: this.massFlux!, 8: sx, 9: sh, 30: this.bt, 31: this.listLeaf, ...kout, ...kflag,
        }),
      ),
      // Re-meshing goes from the state of parity p to the other.
      per((p, q) => bindGroup(d, pR, { 0: this.params, 2: v(this.species[q]), 4: v(this.state[1 - p][0]), 6: v(this.species[1 - q]), 7: this.rg, 31: this.listLeaf, ...kout })),
    ];
    this.burn?.destroy();
    this.burn = undefined;
    const burn = burnSpec(this.cfg);
    if (burn) {
      if (this.comp) throw new Error('burning with species that set the equation of state is not supported');
      const t = this.state[0][0];
      // Every texel of the atlas: the pass is pointwise, and copies what it does not burn.
      this.burn = await BurnPass.create(d, burn, this.cfg.problem.species, {
        dims: [t.width, t.height, t.depthOrArrayLayers], stride: this.speciesDepth, groups: this.speciesGroups, slots: hydroSlots(this.cfg),
        params: this.params, clock: this.clock, state: this.state, species: this.species, pb: this.pred[1],
      });
    }
  }

  /** Burn the state as it is now into the other parities (burn/burn.ts), in a pass of its own. */
  private encodeBurn(enc: GPUCommandEncoder) {
    if (!this.burn) return;
    const pass = beginPass(enc);
    this.burn.encode(pass, this.cur, this.speciesCur);
    pass.end();
    this.cur = 1 - this.cur;
    this.speciesCur = 1 - this.speciesCur;
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
    pass.setPipeline(pM);
    pass.setBindGroup(0, this.gSpecies[2][p][q]);
    this.interior(pass, this.nLeaf);
    this.speciesCur = 1 - q;
  }

  private updateGroup(pipe: GPUComputePipeline, p: number, list = this.listLeaf): GPUBindGroup {
    const v = (t: GPUTexture) => t.createView();
    const o = 1 - p;
    const bin: Record<number, Bindable> = this.builtBins > 0 ? { 29: this.bins! } : {};
    return bindGroup(this.device, pipe, {
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 4: v(this.prim[0]), 5: v(this.prim[1]), 6: v(this.gfield),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 11: this.sources, 30: this.bt, 31: list,
      ...bin,
      ...this.compIn,
      ...((this.massFlux ? { 12: this.massFlux } : {}) as Record<number, GPUBuffer>),
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
      computePipeline(d, assemble('amr/bins_relax.wgsl'), 'bins dilate', { KMAX: K, DILATE: 1 }),
      computePipeline(d, assemble('amr/bins_relax.wgsl'), 'bins relax', { KMAX: K }),
      computePipeline(d, assemble('amr/bins_lists.wgsl'), 'bins lists'),
      computePipeline(d, assemble('amr/bins_args.wgsl'), 'bins dispatch sizes'),
      computePipeline(d, assemble('amr/commit.wgsl'), 'amr commit'),
    ]);
    const A = this.bins;
    const B = this.binsB;
    this.gTick = bindGroup(d, this.pTick, { 1: this.clock, 2: A, 3: this.binCnt });
    this.gAdvance = bindGroup(d, this.pAdvance, { 1: this.clock, 2: A, 3: this.binCnt });
    await this.grav?.setBins();
    const v = (t: GPUTexture) => t.createView();
    this.redBhB ??= await Reducer.create(
      d, [this.cap * BS, BS, this.cfg.ndim === 2 ? 1 : BS], new Array(MASS_RED).fill(SUM), 'amr/bh_moment.wgsl', { bins: 'amr/bins_on.wgsl' },
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: this.bh, 3: v(this.state[p][1]), 4: this.clock, 29: A, 30: this.bt })), 'amr point masses (bins)',
    );
    this.gBhKickB = bindGroup(d, this.pBhKick!, { 0: this.params, 1: this.clock, 2: this.redBhB.result, 3: this.bh });
    if (!this.grav && !this.pMassFieldB) {
      // 2D: no potential (the bindings are declared but never read).
      this.pMassFieldB = await computePipeline(d, assemble('amr/gravity/field.wgsl', FIELD_SLOTS_BINS), 'amr point mass field (bins)');
      this.gMassFieldB = [this.listAct, this.listAll].map((list) =>
        bindGroup(d, this.pMassFieldB!, { 0: this.params, 1: v(this.dummyPhi), 3: v(this.gfield), 4: this.bh, 5: this.clock, 6: v(this.dummyPhi), 29: A, 30: this.bt, 31: list }),
      );
    }
    this.gSetDt = bindGroup(d, this.pSetDt, { 0: this.params, 1: this.redDt.result, 2: this.clock, 3: this.bh });
    this.gBinsInit = bindGroup(d, this.pBinsInit, { 0: this.params, 1: this.clock, 2: this.redDt.first, 3: A, 30: this.bt, 31: this.listAll });
    this.gBinsDilate = bindGroup(d, this.pBinsDilate, { 0: this.params, 1: A, 2: B, 30: this.bt, 31: this.listAll });
    this.gBinsRelax = [
      bindGroup(d, this.pBinsRelax, { 0: this.params, 1: B, 2: A, 30: this.bt, 31: this.listAll }),
      bindGroup(d, this.pBinsRelax, { 0: this.params, 1: A, 2: B, 30: this.bt, 31: this.listAll }),
    ];
    this.gBinsLists = bindGroup(d, this.pBinsLists, { 0: this.params, 1: this.listAct, 2: this.listUpd, 3: this.binCnt, 4: this.listEnd, 29: A, 30: this.bt, 31: this.listAll });
    this.gBinsArgs = bindGroup(d, this.pBinsArgs, { 0: this.params, 1: this.listAct, 2: this.listUpd, 3: this.binCnt, 4: this.binArgs, 5: this.listEnd });
    this.binsPrimed = null;
    // What was measured with other bins says nothing about these.
    this.binPause = 0;
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
      run(pass, this.pMassFieldB!, this.gMassFieldB[count < 0 ? 0 : 1]);
      if (count < 0) pass.dispatchWorkgroupsIndirect(this.binArgs, ARG(0));
      else this.padded(pass, count);
    };
    // The point masses move with the smallest step, every substep.
    const stepMasses = (pass: GPUComputePassEncoder) => {
      run(pass, this.pBhDrift!, this.gBhDrift!);
      pass.dispatchWorkgroups(1);
      this.redBhB!.encode(pass, p);
      run(pass, this.pBhKick!, this.gBhKickB!);
      pass.dispatchWorkgroups(1);
    };
    for (let s = 0; s < n; s++) {
      const pass = beginPass(enc);
      run(pass, this.pTick!, this.gTick!);
      pass.dispatchWorkgroups(1);
      if (s === 0) {
        run(pass, this.pAssemble, this.assembleGroup(p));
        this.flat(pass, this.nAll, NG);
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

  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [this.n0, this.n0, c.ndim === 2 ? 1 : this.n0],
      dx: this.dx0,
      rhoMean: this.rhoMean,
      selfGravity: this.selfGravity,
      masses: this.hasMasses ? this.nMasses : 0,
      amr: [this.slotsPerSide, this.n0 / BS, this.levels, this.layers],
      refine: [c.refineHi, c.jeansRefine],
      solids: this.solids,
      sources: this.nSources > 0,
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
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
  private get hasMasses(): boolean {
    return this.nMasses > 0;
  }

  /** Acceleration and half-step potential for every block, into the field atlas. */
  private encodeField(pass: GPUComputePassEncoder) {
    if (this.grav) {
      this.grav.encodeField(pass, this.nAll);
    } else {
      pass.setPipeline(this.pMassField!);
      pass.setBindGroup(0, this.gMassField!);
      this.padded(pass, this.nAll);
    }
  }
  /** The gravity field atlas is in use: by self-gravity, point masses or both. */
  private get fieldOn(): boolean {
    return this.selfGravity || this.hasMasses;
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
    if (this.hasMasses && perturber !== 'none') {
      const live = perturber === 'step';
      if (live) {
        pass.setPipeline(this.pBhDrift!);
        pass.setBindGroup(0, this.gBhDrift!);
        pass.dispatchWorkgroups(1);
      }
      this.redBh!.encode(pass, parity);
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
    pass.end();
    // No history on this mesh: the half-step extrapolation starts from the potential itself.
    if (this.selfGravity) this.grav!.copyPhiToOld(enc);
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

  /**
   * Refinement indicator of a block from the analytic setup, for building the initial mesh:
   * the same normalized jump across two cell widths that the GPU pass measures, but sampled
   * more finely than the block's own cells. A feature smaller than a coarse cell (a point
   * explosion, say) would otherwise fall between cell centers and never trigger refinement.
   */
  /**
   * Samples per cell side when building the starting mesh: down to the finest level, within a
   * budget. A few blocks share a total budget per pass, so a deep mesh still sees a feature a few
   * finest cells across (a blast) that coarse sampling of a root block would step over; many
   * blocks get the floor per block.
   */
  private samplesPerCell(level: number, leaves: number): number {
    const three = this.cfg.ndim === 3;
    const floor = three ? 2 : 4;
    const budget = three ? INDICATOR_SAMPLES_3D : INDICATOR_SAMPLES_2D;
    const share = Math.floor(Math.pow(budget / Math.max(leaves, 1), 1 / this.cfg.ndim) / (BS + 2));
    return Math.min(1 << (this.levels - level), Math.max(floor, share > 0 ? 1 << Math.floor(Math.log2(share)) : 0));
  }

  private indicator(b: Block, fn: (x: number, y: number, z: number) => { rho: number; p: number; X?: readonly number[] }, sub: number): number {
    const three = this.cfg.ndim === 3;
    const dx = this.dx0 / (1 << b.level);
    const half = this.cfg.problem.box / 2;
    const h = dx / sub;
    const m = (BS + 2) * sub; // one cell of margin on each side
    const mz = three ? m : 1;
    const rho = new Float64Array(m * m * mz);
    const pr = new Float64Array(m * m * mz);
    // The species' mass fractions, normalized as packSpecies stores them.
    const ns = this.cfg.problem.species?.length ?? 0;
    const xs = Array.from({ length: ns }, () => new Float64Array(m * m * mz));
    for (let k = 0; k < mz; k++)
      for (let j = 0; j < m; j++)
        for (let i = 0; i < m; i++) {
          const w = fn(
            b.c[0] * BS * dx + (i - sub + 0.5) * h - half,
            b.c[1] * BS * dx + (j - sub + 0.5) * h - half,
            three ? b.c[2] * BS * dx + (k - sub + 0.5) * h - half : 0,
          );
          rho[(k * m + j) * m + i] = w.rho;
          pr[(k * m + j) * m + i] = w.p;
          if (ns) {
            let sum = 0;
            for (let s = 0; s < ns; s++) sum += Math.max(w.X?.[s] ?? 0, 0);
            for (let s = 0; s < ns; s++) xs[s][(k * m + j) * m + i] = sum > 0 ? Math.max(w.X?.[s] ?? 0, 0) / sum : s === 0 ? 1 : 0;
          }
        }
    // Gas dense enough that the preset wants it at the finest level (stars): as amr/flag.wgsl.
    const dense = this.cfg.refineDense;
    if (dense > 0 && rho.some((r) => r > dense)) return Infinity;
    // The normalized second derivative of amr/flag.wgsl, at the finest spacing sampled.
    let ind = 0;
    const strides = three ? [sub, sub * m, sub * m * m] : [sub, sub * m];
    const lohner = (q: Float64Array, a: number) => {
      let num = 0;
      let den = 0;
      for (const s of strides) {
        const [qp, q0, qm] = [q[a + s], q[a], q[a - s]];
        num += (qp - 2 * q0 + qm) ** 2;
        den += (Math.abs(qp - q0) + Math.abs(q0 - qm) + 0.01 * (Math.abs(qp) + 2 * Math.abs(q0) + Math.abs(qm))) ** 2;
      }
      return den > 0 ? Math.sqrt(num / den) : 0;
    };
    // The mass fractions' jumps between neighbors, as species_indicator (species/species.ts).
    const jump = (q: Float64Array, a: number) => {
      let j = 0;
      for (const s of strides) j = Math.max(j, Math.abs(q[a + s] - q[a]), Math.abs(q[a] - q[a - s]));
      return SPECIES_JUMP_SCALE * j;
    };
    // How fast it burns, against the block's cells (burnIndicator, as amr/flag.wgsl).
    const burn = burnSpec(this.cfg);
    const fuel = burn ? xs[burnIndices(burn, this.cfg.problem.species)[0]] : null;
    const eos = burn ? makeEos(this.cfg.eos, this.cfg.eosParams) : null;
    for (let k = three ? sub : 0; k < (three ? m - sub : 1); k++)
      for (let j = sub; j < m - sub; j++)
        for (let i = sub; i < m - sub; i++) {
          const a = (k * m + j) * m + i;
          ind = Math.max(ind, lohner(rho, a), lohner(pr, a));
          for (const x of xs) ind = Math.max(ind, jump(x, a));
          if (fuel) ind = Math.max(ind, burnIndicator(burn!, eos!, rho[a], pr[a], fuel[a], dx));
        }
    return ind;
  }

  private finerInSnapshot(b: Block, levelAt: (x: number, y: number, z: number) => number): boolean {
    const three = this.cfg.ndim === 3;
    const size = (BS * this.dx0) / (1 << b.level);
    const half = this.cfg.problem.box / 2;
    for (let o = 0; o < (three ? 8 : 4); o++) {
      const at = (a: number) => (b.c[a] + (((o >> a) & 1) + 0.5) / 2) * size - half;
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

  /** Load the problem's initial conditions on a mesh refined to fit them. */
  reset() {
    const c = this.cfg;
    const d = this.device;
    const eos = makeEos(c.eos, c.eosParams);
    const fn = c.problem.init(this.n, eos, c.ndim, c.params);
    const prim = (x: number, y: number, z: number) => {
      const w = fn(x, y, z);
      return eos.hasEnergy ? w : { ...w, p: eos.pressure(w.rho, 0) };
    };
    // For the mesh, an obstacle counts as a jump in density, so its surface starts refined.
    const surface = (x: number, y: number, z: number) => {
      const w = prim(x, y, z);
      return w.solid ? { rho: 4 * w.rho, p: w.p } : w;
    };
    // With self-gravity a level that only partly fits the block budget is left out altogether:
    // a refinement boundary that cuts a steep hydrostatic gradient at an arbitrary place
    // disturbs the balance far more than resolving all of it one level coarser.
    const strict = this.selfGravity;
    const levelAt = c.restart?.levelAt;
    const sampled = new Map<Block, { sub: number; ind: number }>();
    for (let limit = this.levels; ; limit--) {
      this.tree = new BlockTree(c.ndim, this.n0 / BS, this.levels, this.cap, c.bcLo, c.bcHi);
      this.tree.levelLimit = limit;
      for (let pass = 0; pass < limit; pass++) {
        // From a snapshot: the mesh it was saved on. A block is refined wherever the snapshot
        // is finer than it at the center of one of its children.
        const leaves = levelAt ? 0 : this.tree.leaves().length;
        const changed = levelAt
          ? this.tree.adapt((b) => (this.finerInSnapshot(b, levelAt) ? 1 : 0), 0.5, -1)
          : this.tree.adapt((b) => {
              // A block kept from the last pass was sampled at least as finely then (the
              // budget is shared by more leaves now): only new blocks are sampled.
              const sub = this.samplesPerCell(b.level, leaves);
              const seen = sampled.get(b);
              if (seen && seen.sub >= sub) return seen.ind;
              const ind = this.indicator(b, surface, sub);
              sampled.set(b, { sub, ind });
              return ind;
            }, c.refineHi, c.refineLo);
        if (!changed) break;
      }
      if (levelAt || !strict || limit === 0 || this.tree.freeSlots >= this.tree.nchild) break;
    }

    const three = c.ndim === 3;
    const depth = three ? BS : 1;
    const half = c.problem.box / 2;
    const u1 = new Float32Array(BS * BS * depth * 4);
    const u2 = new Float32Array(BS * BS * depth * 4);
    const G = this.speciesGroups;
    const blockCells = BS * BS * depth;
    const sp = new Float32Array(blockCells * 4 * G);
    const kc = this.compCoeffs;
    const km = kc ? new Float32Array(blockCells * 4) : undefined;
    let mass = 0;
    this.solids = false;
    this.nSources = 0;
    d.queue.writeBuffer(this.sources, 0, packSources([]));
    for (const b of this.tree.leaves()) {
      const dx = this.dx0 / (1 << b.level);
      let o = 0;
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, o += 4) {
            const w = prim((b.c[0] * BS + i + 0.5) * dx - half, (b.c[1] * BS + j + 0.5) * dx - half, three ? (b.c[2] * BS + k + 0.5) * dx - half : 0);
            // Where the species set the equation of state, the cell's own.
            let ce = eos;
            if (km) {
              const m = mixMoments(kc!, w.X);
              km.set(m, o);
              ce = makeEos(c.eos, c.eosParams, m, hydrogenParticles(c.problem.composition));
            }
            const eint = ce.eintFromP(w.rho, w.p);
            const mark = cellMark(w);
            if (mark) this.solids = true;
            const go = mark === 1 ? 0 : 1; // the gas kept inside an obstacle is at rest
            mass += w.rho * dx ** c.ndim;
            u1[o] = w.rho;
            u1[o + 1] = w.rho * w.vx * go;
            u1[o + 2] = w.rho * w.vy * go;
            u1[o + 3] = w.rho * w.vz * go;
            u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz) * go;
            u2[o + 1] = eint;
            u2[o + 2] = w.rho * w.s;
            u2[o + 3] = mark;
            if (G) packSpecies(sp, o, blockCells * 4, G, w.rho, w.X);
          }
      const origin = this.origin(b.slot);
      const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
      d.queue.writeTexture({ texture: this.state[0][0], origin }, u1, layout, [BS, BS, depth]);
      d.queue.writeTexture({ texture: this.state[0][1], origin }, u2, layout, [BS, BS, depth]);
      if (km) d.queue.writeTexture({ texture: this.comp!, origin }, km, layout, [BS, BS, depth]);
      for (let g = 0; g < G; g++) {
        const og = [origin[0], origin[1], origin[2] + g * this.speciesDepth];
        d.queue.writeTexture({ texture: this.species[0], origin: og }, sp.subarray(g * blockCells * 4, (g + 1) * blockCells * 4), layout, [BS, BS, depth]);
      }
    }
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
    this.binPause = 0;
    this.binPauseNext = BIN_PAUSE_STEPS;
    this.binShare = null;
    d.queue.writeBuffer(this.binArgs, 0, new Uint32Array(BIN_ARGS_BYTES / 4));
    this.rhoMean = mass / c.problem.box ** c.ndim;
    // The clock is (dt, t, Kahan compensation, steps).
    d.queue.writeBuffer(this.clock, 0, new Float32Array(c.restart ? [0, c.restart.time, 0, c.restart.steps] : [0, 0, 0, 0]));
    const masses = initialMasses(c, this.massDx);
    this.nMasses = this.nOwnMasses = masses.length;
    d.queue.writeBuffer(this.bh, 0, packMasses(masses));
    this.debrisBuf?.reset(c.restart?.debris);
    this.writeParams();
    this.uploadTree();
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
    return this.settling;
  }
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
    const d = this.device;
    this.settling = false;
    this.settled = Promise.resolve();
    if (c.restart || !c.wellBalanced || !this.selfGravity) return;
    const eos = makeEos(c.eos, c.eosParams);
    const set = c.problem.settle ? c.problem.settle(c.params, eos) : c.problem.star ? settlingOf(c.problem.star, c.G) : null;
    if (!set) return;
    this.settling = true;
    const tree = this.tree;
    const t0 = performance.now();
    this.settled = (async () => {
      try {
        const leaf = await this.readLeaves();
        if (epoch !== this.settleEpoch) return;
        const cells: SettleCells = {
          count: leaf.count,
          pos: (q) => [leaf.x[q], leaf.y[q], leaf.z[q]],
          vol: (q) => leaf.dx[q] ** 3,
          u1: leaf.u1,
          u2: leaf.u2,
        };
        const { u1, u2 } = cells;
        let peak = 0;
        for (let q = 0; q < cells.count; q++) peak = Math.max(peak, u1[q * 4]);
        const settler = new Settler(set, cells);
        const per = BS ** 3;
        // Only the blocks a star may reach are ever rewritten.
        const leaves = tree.leaves().filter((_, i) => settler.owner.subarray(i * per, (i + 1) * per).some((o) => o >= 0));
        const index = new Map(tree.leaves().map((b, i) => [b, i]));
        const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
        const upload = (tex: GPUTexture, u: Float32Array) =>
          leaves.forEach((b) => {
            const i = index.get(b)!;
            d.queue.writeTexture({ texture: tex, origin: this.origin(b.slot) }, u.subarray(i * per * 4, (i + 1) * per * 4) as Float32Array<ArrayBuffer>, layout, [BS, BS, BS]);
          });
        this.settleLog = [];
        for (let round = 0; round < SETTLE_ROUNDS; round++) {
          const phi = await this.readLeafPotential();
          if (epoch !== this.settleEpoch || tree !== this.tree) return;
          const change = settler.round(phi);
          this.settleLog.push(change / peak);
          upload(this.state[this.cur][0], u1);
          this.refreshPotential();
          if (change < (set.tol ?? SETTLE_TOL) * peak) break;
        }
        settler.finish(eos.eintFromP);
        upload(this.state[this.cur][0], u1);
        upload(this.state[this.cur][1], u2);
        this.refreshPotential();
      } finally {
        if (epoch === this.settleEpoch) {
          this.settling = false;
          this.settleMs = performance.now() - t0;
        }
      }
    })();
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
    } else pass.dispatchWorkgroups(count * 3, 3, 5);
  }
  /** Dispatch for inv_flat() (amr/list.wgsl): cells and `reach` ghost layers of `count` blocks. */
  private flat(pass: GPUComputePassEncoder, count: number, reach: number) {
    if (count > 0) pass.dispatchWorkgroups(flatGroups(this.cfg.ndim, reach), Math.min(count, FLAT_ROWS), Math.ceil(count / FLAT_ROWS));
  }
  private interior(pass: GPUComputePassEncoder, count: number) {
    if (this.cfg.ndim === 2) {
      const q = Math.ceil(count / 4);
      pass.dispatchWorkgroups(Math.min(q, SLAB_QUADS) * 2, 2, Math.ceil(q / SLAB_QUADS));
    } else pass.dispatchWorkgroups(count * 2, 2, 4);
  }

  private applyRegrid(enc: GPUCommandEncoder) {
    const plan = this.pending!;
    this.pending = null;
    this.uploadTree();
    this.device.queue.writeBuffer(this.rg, 0, BlockTree.regridTable(plan.ops, this.cap) as Int32Array<ArrayBuffer>);
    const pass = beginPass(enc);
    pass.setPipeline(this.pRegrid);
    pass.setBindGroup(0, this.gRegrid[this.cur]);
    this.interior(pass, this.nLeaf);
    if (this.speciesGroups) {
      pass.setPipeline(this.pSpecies[3]);
      pass.setBindGroup(0, this.gSpecies[3][this.cur][this.speciesCur]);
      this.interior(pass, this.nLeaf);
      this.speciesCur = 1 - this.speciesCur;
    }
    pass.end();
    this.cur = 1 - this.cur;
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
    if (this.settling) return 0; // a star is still being settled
    if (this.pending) this.applyRegrid(enc);
    if (this.dtStale) {
      // The block list changed: drop timestep records of workgroups it no longer covers.
      enc.clearBuffer(this.redDt.first);
      this.dtStale = false;
    }
    this.primeBins(enc);
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
      pass.end();
      if (this.fieldOn) {
        // Keep the old potential, solve for that of the updated state (warm-started), move
        // the point masses, rebuild the field, and apply the second half kick (and the
        // masses' sinks) back into this parity.
        if (this.selfGravity) this.grav!.copyPhiToOld(enc);
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
    if (this.levels === 0 || this.flagInFlight || this.wantMap || this.stepsSinceFlag < REGRID_EVERY) return false;
    enc.clearBuffer(this.flags);
    const fp = beginPass(enc);
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
    this.device.queue.writeBuffer(this.brush, 0, packBrush(op));
    const pass = beginPass(enc);
    pass.setPipeline(this.pBrush);
    pass.setBindGroup(0, this.gBrush[this.cur]);
    this.interior(pass, this.nLeaf);
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

  addMass(m: Parameters<Simulation['addMass']>[0]): boolean {
    if (this.nMasses >= MAX_MASSES) return false;
    this.device.queue.writeBuffer(this.bh, 16 + 4 * MASS_STRIDE * this.nMasses, packMasses([completeMass(m, this.dx0)]), 4, MASS_STRIDE);
    this.nMasses++;
    this.writeParams();
    this.refreshPotential();
    return true;
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
      const across = this.n0 << b.level; // cells across the box at this level
      const index = f.slice(0, axes).map((x) => Math.min(across - 1, Math.max(0, Math.floor(x * across))));
      if (!index.every((i, a) => Math.floor(i / BS) === b.c[a])) continue;
      const o = this.origin(b.slot);
      const at = [o[0] + index[0] - b.c[0] * BS, o[1] + index[1] - b.c[1] * BS, three ? o[2] + index[2] - b.c[2] * BS : o[2]] as [number, number, number];
      const tree = this.tree;
      const [u1, u2] = await readTexels(this.device, this.state[this.cur], at);
      const dx = this.dx0 / (1 << b.level);
      const half = this.cfg.problem.box / 2;
      const pos = [0, 1, 2].map((a) => (a < axes ? (index[a] + 0.5) * dx - half : 0)) as [number, number, number];
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
        for (const b of tree.leaves()) this.indicatorOf.set(b, ind[b.slot]);
        // Exactly 0: no gas in the block, only the inside of an obstacle (see amr/flag.wgsl).
        const ops = tree.adapt((b) => (ind[b.slot] === 0 ? -1 : ind[b.slot]), this.cfg.refineHi, this.cfg.refineLo);
        if (ops) this.pending = { ops };
      })
      .finally(() => {
        this.flagInFlight = false;
      });
  }

  stepsResume(): Promise<void> | null {
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
    enc.copyBufferToBuffer(this.clock, 0, this.staging, 0, 16);
    enc.copyBufferToBuffer(this.redDiag.result, 0, this.staging, 16, DIAG_OPS.length * 4);
    enc.copyBufferToBuffer(this.bh, 0, this.staging, 16 + DIAG_OPS.length * 4, MASS_FLOATS * 4);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      const diag = diagnosticsFrom(a, this.hasMasses ? this.nMasses : 0);
      if (diag.steps === 0 && diag.dt === 0) diag.dt = await this.previewDt();
      return diag;
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
   * encode work on its destroyed textures (previewDt, after readDiagnostics' map).
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
    enc.copyBufferToBuffer(this.clock, 0, this.clockPreview, 0, 16);
    const pass = beginPass(enc);
    pass.setPipeline(this.pAssemble);
    pass.setBindGroup(0, this.assembleGroup(this.cur));
    this.flat(pass, this.nAll, NG);
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
    await this.settled;
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      for (;;) {
        const enc = this.device.createCommandEncoder();
        this.encodeSteps(enc, this.levels > 0 ? REGRID_EVERY : 16);
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
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
    return [nd, nd, this.cfg.ndim === 2 ? 1 : nd];
  }
  readonly fieldShader = 'amr/field.wgsl';
  displayAtlas(): DisplayAtlas | null {
    if (this.cfg.ndim !== 3) return null;
    return { side: this.slotsPerSide * PAD, slotsPerSide: this.slotsPerSide, roots: this.n0 / BS, levels: this.levels, idx: this.idx, list: this.listLeaf, leaves: this.nLeaf };
  }
  fieldResources(parity: number, speciesParity: number): Record<number, GPUBuffer | GPUTextureView> {
    const r: Record<number, GPUBuffer | GPUTextureView> = { 2: this.state[parity][0].createView(), 3: this.state[parity][1].createView(), 5: this.idx.createView(), 6: this.prim[0].createView(), 7: this.gfield.createView(), 10: this.pred[1].createView(), 30: this.bt };
    if (this.speciesGroups) r[9] = this.species[speciesParity].createView();
    return { ...r, ...this.compIn };
  }
  get speciesStride(): number {
    return this.speciesGroups ? this.speciesDepth : 0;
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

  /** Every leaf cell with its position and size (validation only; reads the whole atlas). */
  async readLeaves(): Promise<LeafCells> {
    this.flush();
    const side = this.slotsPerSide * PAD;
    const depthTex = this.cfg.ndim === 2 ? this.layers : side;
    const [a, b] = this.state[this.cur];
    const t1 = await readTexture(this.device, a, side, 4, depthTex);
    const t2 = await readTexture(this.device, b, side, 4, depthTex);
    const leaves = this.tree.leaves();
    const three = this.cfg.ndim === 3;
    const depth = three ? BS : 1;
    const count = leaves.length * BS * BS * depth;
    const out: LeafCells = {
      x: new Float32Array(count), y: new Float32Array(count), z: new Float32Array(count), dx: new Float32Array(count),
      u1: new Float32Array(count * 4), u2: new Float32Array(count * 4), count,
    };
    const half = this.cfg.problem.box / 2;
    let q = 0;
    for (const blk of leaves) {
      const dx = this.dx0 / (1 << blk.level);
      const [ox, oy, oz] = this.origin(blk.slot);
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, q++) {
            const src = (((oz + k) * side + oy + j) * side + ox + i) * 4;
            out.x[q] = (blk.c[0] * BS + i + 0.5) * dx - half;
            out.y[q] = (blk.c[1] * BS + j + 0.5) * dx - half;
            out.z[q] = three ? (blk.c[2] * BS + k + 0.5) * dx - half : 0;
            out.dx[q] = dx;
            out.u1.set(t1.subarray(src, src + 4), q * 4);
            out.u2.set(t2.subarray(src, src + 4), q * 4);
          }
    }
    return out;
  }

  /**
   * Mass fractions of every leaf cell (validation): X[i][cell], the cells in the order of
   * readLeaves(). Empty without species.
   */
  async readLeafSpecies(): Promise<Float32Array[]> {
    const G = this.speciesGroups;
    if (!G) return [];
    this.flush();
    const side = this.slotsPerSide * PAD;
    const raw = await readTexture(this.device, this.species[this.speciesCur], side, 4, this.speciesDepth * G);
    const leaves = this.tree.leaves();
    const depth = this.cfg.ndim === 3 ? BS : 1;
    const count = leaves.length * BS * BS * depth;
    const ns = this.cfg.problem.species!.length;
    const X = Array.from({ length: ns }, () => new Float32Array(count));
    let q = 0;
    for (const blk of leaves) {
      const [ox, oy, oz] = this.origin(blk.slot);
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, q++) {
            let sum = 0;
            const at = (g: number) => (((oz + k + g * this.speciesDepth) * side + oy + j) * side + ox + i) * 4;
            for (let g = 0; g < G; g++) for (let c = 0; c < 4; c++) sum += raw[at(g) + c];
            for (let s = 0; s < ns; s++) X[s][q] = raw[at(Math.floor(s / SPECIES_PER_GROUP)) + (s % SPECIES_PER_GROUP)] / sum;
          }
    }
    return X;
  }

  /** The potential on every leaf cell, in the same order as readLeaves() (validation only). */
  async readLeafPotential(): Promise<Float32Array> {
    this.flush();
    const side = this.slotsPerSide * PAD;
    const t = await readTexture(this.device, this.grav!.phi, side, 1);
    const leaves = this.tree.leaves();
    const out = new Float32Array(leaves.length * BS ** 3);
    let q = 0;
    for (const blk of leaves) {
      const [ox, oy, oz] = this.origin(blk.slot);
      for (let k = 0; k < BS; k++) for (let j = 0; j < BS; j++) for (let i = 0; i < BS; i++, q++) out[q] = t[((oz + k) * side + oy + j) * side + ox + i];
    }
    return out;
  }

  /**
   * Flux consistency check on the current mesh. Evaluates one step's face mass fluxes and
   * compares the two sides of every leaf face: bit for bit between same-level leaves, and the
   * coarse flux against the mean of the fine ones at refinement boundaries.
   */
  async fluxMismatch(): Promise<{ faces: number; mismatched: number; coarseFine: number; worstCoarseFine: number }> {
    const d = this.device;
    this.flush();
    const pipe = await computePipeline(d, assemble('amr/update.wgsl', this.slots), 'amr update (flux debug)', { DEBUG_FLUX: 1 });
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

  destroy() {
    this.destroyed = true;
    this.settleEpoch++; // abandon a settling in progress: its textures are going
    for (const s of this.state) for (const t of s) t.destroy();
    for (const t of [...this.prim, ...this.pred, this.idx, this.gfield, this.dummyPhi, ...this.species]) t.destroy();
    this.comp?.destroy();
    this.speciesX?.destroy();
    this.speciesHalf?.destroy();
    this.burn?.destroy();
    this.massFlux?.destroy();
    this.debrisBuf?.destroy();
    for (const b of [this.bins, this.binsB, this.listAct, this.listUpd, this.listEnd, this.binCnt, this.binArgs, this.binStaging]) b.destroy();
    this.grav?.destroy();
  }
}
