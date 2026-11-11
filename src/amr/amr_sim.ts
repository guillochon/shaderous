// Block-structured adaptive mesh refinement on the GPU.
//
// The grid is a tree of 16^3-cell blocks (src/amr/tree.ts) packed into one 3D texture atlas
// per field, with two ghost layers stored around each block. All levels take the same
// timestep. A step is:
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
import { bindGroup, computePipeline, readTexture, tex3d, type Gpu } from '../gpu/device';
import { MAX, Reducer, SUM } from '../gpu/reduce';
import { AmrGravity } from './gravity';
import { makeEos } from '../eos/eos';
import { cellMark } from '../problems/types';
import { completeMass, MassReader, MAX_MASSES, BRUSH_BYTES, DIAG_FLOATS, DIAG_OPS, diagnosticsFrom, hydroSlots, initialMasses, MASS_FLOATS, MASS_RED, MAX_SOURCES, outlineContains, packBrush, packMasses, packParams, packSources, PARAM_BYTES, readRestart, SOURCE_BYTES, type RestartState, type BrushOp, type Diagnostics, type Outline, type SimConfig, type Simulation, type SourceSpec } from '../sim';
import { BlockTree, BS, BT_STRIDE, NG, PAD, RG_STRIDE, type Block, type RegridOp } from './tree';

/**
 * Steps between refinement checks, and how far the flow may run past an unanswered check.
 * Signals cross at most about a third of a cell per step, and a block is 16 cells with a
 * buffer block kept around flagged ones, so 24 steps cannot carry a feature out of its refinement.
 */
const REGRID_EVERY = 8;
const MAX_STEPS_AHEAD = 24;

interface RegridPlan {
  ops: RegridOp[];
}

export interface LeafCells {
  /** per leaf cell: position of the centre, cell size, and the conserved state */
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
  /** atlas slots per side, and in total */
  readonly slotsPerSide: number;
  readonly cap: number;
  tree!: BlockTree;
  cur = 0;

  state: GPUTexture[][] = [];
  private prim: GPUTexture[] = [];
  private pred: GPUTexture[] = [];
  private idx: GPUTexture;
  private params: GPUBuffer;
  private clock: GPUBuffer;
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

  private pAssemble!: GPUComputePipeline;
  private pPredict!: GPUComputePipeline;
  private pUpdate!: GPUComputePipeline;
  private pClock!: GPUComputePipeline;
  private pFlag!: GPUComputePipeline;
  private pRegrid!: GPUComputePipeline;
  private gAssemble: GPUBindGroup[] = [];
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

  get paramBuffer(): GPUBuffer {
    return this.params;
  }

  private constructor(gpu: Gpu, cfg: SimConfig, slots = 0) {
    const d = (this.device = gpu.device);
    this.cfg = cfg;
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
    // Default budget: what refining everywhere would need (leaves plus their parents) in 3D,
    // three quarters of that in 2D, capped so the 3D atlases stay within about 3 GB (2197
    // blocks; create() falls back to a smaller atlas if the GPU cannot hold that).
    const full = roots * (1 << (cfg.ndim * this.levels)) * (cfg.ndim === 3 ? 8 / 7 : 4 / 3);
    const auto = Math.ceil(Math.pow(Math.max(full * (cfg.ndim === 3 ? 1 : 0.75), roots * 2), 1 / cfg.ndim));
    const [lo, hi] = cfg.ndim === 3 ? [6, 13] : [12, 100];
    this.slotsPerSide = Math.min(limit, slots || cfg.amrBudget || Math.min(hi, Math.max(lo, auto)));
    this.cap = this.slotsPerSide ** cfg.ndim;
    if (roots > this.cap) throw new Error(`${roots} root blocks do not fit in an atlas of ${this.cap}`);

    const side = this.slotsPerSide * PAD;
    const t = (label: string) => tex3d(d, side, 'rgba32float', label, cfg.ndim === 2 ? 1 : side);
    for (let p = 0; p < 2; p++) this.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    this.prim = [t('W1'), t('W2')];
    this.pred = [t('PA'), t('PB')];
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
    this.staging = d.createBuffer({ size: DIAG_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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
      const tried = sim?.slotsPerSide ?? slots;
      sim?.destroy();
      const next = Math.floor(tried * 0.85);
      const roots = ((cfg.n >> cfg.amrLevels) / BS) ** cfg.ndim;
      if (!oom || !tried || next ** cfg.ndim < roots * 2) throw failure ?? new Error('not enough GPU memory for the adaptive mesh at this resolution');
      console.warn(`adaptive mesh: out of GPU memory with ${tried ** cfg.ndim} blocks, retrying with ${next ** cfg.ndim}`);
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
      [sim.cap * 3, 3, cfg.ndim === 2 ? 1 : 5], // one record per workgroup of the assembly pass
    );
    sim.pClock = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock', { PER_DX: 1 });
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock });
    sim.pFlag = await computePipeline(d, assemble('amr/flag.wgsl'), 'amr flag');
    sim.gFlag = bindGroup(d, sim.pFlag, { 0: sim.params, 1: v(sim.prim[0]), 2: v(sim.prim[1]), 3: sim.flags, 30: sim.bt, 31: sim.listLeaf });
    // Point masses, in 2D and 3D: their reduction over the leaf cells and their orbit passes.
    sim.redBh = await Reducer.create(
      d, [sim.cap * BS, BS, depth], new Array(MASS_RED).fill(SUM), 'amr/bh_moment.wgsl', {},
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
      sim.pMassField = await computePipeline(d, assemble('amr/gravity/field.wgsl'), 'amr point mass field');
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
      });
    }
    await sim.buildHydro();
    sim.reset();
    return sim;
  }

  private get slots() {
    return hydroSlots(this.cfg);
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
        30: this.bt, 31: this.listLeaf,
      }),
    );
    this.gPredict = bindGroup(d, this.pPredict, {
      0: this.params, 1: this.clock, 2: v(this.prim[0]), 3: v(this.prim[1]), 4: v(this.gfield), 5: v(this.pred[0]), 6: v(this.pred[1]), 30: this.bt, 31: this.listAll,
    });
    this.gAssemble = [];
    this.gUpdate = [];
    this.gRegrid = [];
    this.gCorrect = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // Second half kick: from the updated state (the other parity) back into this one.
      this.gCorrect.push(
        bindGroup(d, this.pCorrect, {
          0: this.params, 1: this.clock, 2: v(this.state[o][0]), 3: v(this.state[o][1]), 4: v(this.gfield),
          5: v(this.state[p][0]), 6: v(this.state[p][1]), 7: this.bh, 30: this.bt, 31: this.listLeaf,
        }),
      );
      this.gAssemble.push(
        bindGroup(d, this.pAssemble, {
          0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]),
          5: v(this.gfield), 6: this.redDt.first, 30: this.bt, 31: this.listAll,
        }),
      );
      this.gUpdate.push(this.updateGroup(this.pUpdate, p));
      this.gRegrid.push(
        bindGroup(d, this.pRegrid, {
          0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]),
          5: v(this.state[o][0]), 6: v(this.state[o][1]), 7: this.rg, 31: this.listLeaf,
        }),
      );
    }
    this.redDiag = await Reducer.create(
      d,
      [this.cap * BS, BS, this.cfg.ndim === 2 ? 1 : BS],
      DIAG_OPS,
      'amr/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.grav?.phi ?? this.dummyPhi), 30: this.bt })),
      'amr diagnostics',
    );
  }

  private updateGroup(pipe: GPUComputePipeline, p: number): GPUBindGroup {
    const v = (t: GPUTexture) => t.createView();
    const o = 1 - p;
    return bindGroup(this.device, pipe, {
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 4: v(this.prim[0]), 5: v(this.prim[1]), 6: v(this.gfield),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 11: this.sources, 30: this.bt, 31: this.listLeaf,
    });
  }

  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [this.n0, this.n0, c.ndim === 2 ? 1 : this.n0],
      dx: this.dx0,
      rhoMean: this.rhoMean,
      selfGravity: this.selfGravity,
      masses: this.hasMasses ? this.nMasses : 0,
      amr: [this.slotsPerSide, this.n0 / BS, this.levels, 0],
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
    return this.cfg.gravity !== 'off' && !!this.grav;
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
    let pass = enc.beginComputePass();
    if (newBlocks && this.selfGravity) this.grav!.encodeNewBlocks(pass, this.nLeaf);
    // Without new blocks to seed this is a cold start: each level begins from the coarser one.
    this.encodeSolve(pass, this.cur, cycles, perturber, !newBlocks);
    pass.end();
    // No history on this mesh: the half-step extrapolation starts from the potential itself.
    if (this.selfGravity) this.grav!.copyPhiToOld(enc);
    pass = enc.beginComputePass();
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
   * the same normalised jump across two cell widths that the GPU pass measures, but sampled
   * more finely than the block's own cells. A feature smaller than a coarse cell (a point
   * explosion, say) would otherwise fall between cell centres and never trigger refinement.
   */
  private indicator(b: Block, fn: (x: number, y: number, z: number) => { rho: number; p: number }): number {
    const three = this.cfg.ndim === 3;
    const dx = this.dx0 / (1 << b.level);
    const half = this.cfg.problem.box / 2;
    // Samples per cell: down to the finest level, within a budget per block.
    const sub = Math.min(1 << (this.levels - b.level), three ? 2 : 4);
    const h = dx / sub;
    const m = (BS + 2) * sub; // one cell of margin on each side
    const mz = three ? m : 1;
    const rho = new Float64Array(m * m * mz);
    const pr = new Float64Array(m * m * mz);
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
        }
    let ind = 0;
    const jump = (q: Float64Array, a: number, s: number) => Math.abs(q[a + s] - q[a - s]) / (q[a + s] + q[a - s]);
    for (let k = three ? sub : 0; k < (three ? m - sub : 1); k++)
      for (let j = sub; j < m - sub; j++)
        for (let i = sub; i < m - sub; i++) {
          const a = (k * m + j) * m + i;
          ind = Math.max(ind, jump(rho, a, sub), jump(pr, a, sub), jump(rho, a, sub * m), jump(pr, a, sub * m));
          if (three) ind = Math.max(ind, jump(rho, a, sub * m * m), jump(pr, a, sub * m * m));
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
    for (let limit = this.levels; ; limit--) {
      this.tree = new BlockTree(c.ndim, this.n0 / BS, this.levels, this.cap, c.bcLo, c.bcHi);
      this.tree.levelLimit = limit;
      for (let pass = 0; pass < limit; pass++) {
        // From a snapshot: the mesh it was saved on. A block is refined wherever the snapshot
        // is finer than it at the centre of one of its children.
        const changed = levelAt
          ? this.tree.adapt((b) => (this.finerInSnapshot(b, levelAt) ? 1 : 0), 0.5, -1)
          : this.tree.adapt((b) => this.indicator(b, surface), c.refineHi, c.refineLo);
        if (!changed) break;
      }
      if (levelAt || !strict || limit === 0 || this.tree.freeSlots >= this.tree.nchild) break;
    }

    const three = c.ndim === 3;
    const depth = three ? BS : 1;
    const half = c.problem.box / 2;
    const u1 = new Float32Array(BS * BS * depth * 4);
    const u2 = new Float32Array(BS * BS * depth * 4);
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
            const eint = eos.eintFromP(w.rho, w.p);
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
          }
      const origin = this.origin(b.slot);
      const layout = { bytesPerRow: BS * 16, rowsPerImage: BS };
      d.queue.writeTexture({ texture: this.state[0][0], origin }, u1, layout, [BS, BS, depth]);
      d.queue.writeTexture({ texture: this.state[0][1], origin }, u2, layout, [BS, BS, depth]);
    }
    this.cur = 0;
    this.pending = null;
    this.stepsSinceFlag = 0;
    this.regrids = 0;
    this.rhoMean = mass / c.problem.box ** c.ndim;
    // The clock is (dt, t, Kahan compensation, steps).
    d.queue.writeBuffer(this.clock, 0, new Float32Array(c.restart ? [0, c.restart.time, 0, c.restart.steps] : [0, 0, 0, 0]));
    const masses = initialMasses(c, this.dx0);
    this.nMasses = this.nOwnMasses = masses.length;
    d.queue.writeBuffer(this.bh, 0, packMasses(masses));
    this.writeParams();
    this.uploadTree();
    if (this.fieldOn) {
      // Cold start: enough cycles per level to converge from zero. The clock's dt is zero, so
      // the point masses' accelerations are only evaluated.
      const enc = d.createCommandEncoder();
      this.encodeFreshPotential(enc, 12, 'kick', false);
      d.queue.submit([enc.finish()]);
    }
  }

  /** Atlas texel of a slot's first interior cell. */
  private origin(slot: number): [number, number, number] {
    const a = this.slotsPerSide;
    if (this.cfg.ndim === 2) return [(slot % a) * PAD + NG, Math.floor(slot / a) * PAD + NG, 0];
    return [(slot % a) * PAD + NG, (Math.floor(slot / a) % a) * PAD + NG, Math.floor(slot / (a * a)) * PAD + NG];
  }

  private uploadTree() {
    const d = this.device;
    const { all, leaves, byLevel } = this.tree.lists();
    this.nAll = all.length;
    this.dtStale = true;
    this.nLeaf = leaves.length;
    this.nLevel = byLevel.map((l) => l.length);
    const pack = (list: Uint32Array) => {
      const out = new Uint32Array(this.cap + 1);
      out[0] = list.length;
      out.set(list, 1);
      return out;
    };
    d.queue.writeBuffer(this.bt, 0, this.tree.table() as Int32Array<ArrayBuffer>);
    d.queue.writeBuffer(this.listAll, 0, pack(all));
    d.queue.writeBuffer(this.listLeaf, 0, pack(leaves));
    byLevel.forEach((l, i) => d.queue.writeBuffer(this.listLevel[i], 0, pack(l)));
    const m = (this.n0 / BS) << this.levels;
    d.queue.writeTexture({ texture: this.idx }, this.tree.indexGrid() as Uint32Array<ArrayBuffer>, { bytesPerRow: m * 4, rowsPerImage: m }, [
      m,
      m,
      this.cfg.ndim === 3 ? m : 1,
    ]);
  }

  // ---------------------------------------------------------------- stepping

  private padded(pass: GPUComputePassEncoder, count: number) {
    pass.dispatchWorkgroups(count * 3, 3, this.cfg.ndim === 2 ? 1 : 5);
  }
  private interior(pass: GPUComputePassEncoder, count: number) {
    pass.dispatchWorkgroups(count * 2, 2, this.cfg.ndim === 2 ? 1 : 4);
  }

  private applyRegrid(enc: GPUCommandEncoder) {
    const plan = this.pending!;
    this.pending = null;
    this.uploadTree();
    this.device.queue.writeBuffer(this.rg, 0, BlockTree.regridTable(plan.ops, this.cap) as Int32Array<ArrayBuffer>);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pRegrid);
    pass.setBindGroup(0, this.gRegrid[this.cur]);
    this.interior(pass, this.nLeaf);
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
    if (this.pending) this.applyRegrid(enc);
    if (this.dtStale) {
      // The block list changed: drop timestep records of workgroups it no longer covers.
      enc.clearBuffer(this.redDt.first);
      this.dtStale = false;
    }
    let done = 0;
    for (; done < count; done++) {
      if (this.levels > 0 && (this.flagInFlight || this.wantMap) && this.stepsSinceFlag >= MAX_STEPS_AHEAD) break;
      const p = this.cur;
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pAssemble);
      pass.setBindGroup(0, this.gAssemble[p]);
      this.padded(pass, this.nAll);
      this.redDt.encode(pass);
      pass.setPipeline(this.pClock);
      pass.setBindGroup(0, this.gClock);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pPredict);
      pass.setBindGroup(0, this.gPredict);
      this.padded(pass, this.nAll);
      pass.setPipeline(this.pUpdate);
      pass.setBindGroup(0, this.gUpdate[p]);
      this.interior(pass, this.nLeaf);
      pass.end();
      if (this.fieldOn) {
        // Keep the old potential, solve for that of the updated state (warm-started), move
        // the point masses, rebuild the field, and apply the second half kick (and the
        // masses' sinks) back into this parity.
        if (this.selfGravity) this.grav!.copyPhiToOld(enc);
        const gp = enc.beginComputePass();
        this.encodeSolve(gp, 1 - p, this.cfg.vcycles, 'step', false, true);
        this.encodeField(gp);
        gp.setPipeline(this.pCorrect);
        gp.setBindGroup(0, this.gCorrect[p]);
        this.interior(gp, this.nLeaf);
        gp.end();
      } else {
        this.cur = 1 - p;
      }
      this.stepsSinceFlag++;

      if (this.levels > 0 && !this.flagInFlight && !this.wantMap && this.stepsSinceFlag >= REGRID_EVERY) {
        // Refinement indicator from this step's primitives.
        enc.clearBuffer(this.flags);
        const fp = enc.beginComputePass();
        fp.setPipeline(this.pFlag);
        fp.setBindGroup(0, this.gFlag);
        this.interior(fp, this.nLeaf);
        fp.end();
        enc.copyBufferToBuffer(this.flags, 0, this.flagStaging, 0, this.cap * 4);
        this.wantMap = true;
        this.stepsSinceFlag = 0;
      }
    }
    return done;
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
      // Refine where the edge passes; merge nothing.
      const ops = this.tree.adapt((b) => (b.level < this.tree.levelLimit && this.crossesOutline(b, outline) ? 1 : 0), 0.5, -1);
      if (!ops) break;
      this.pending = { ops };
      this.flush();
    }
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
    const pass = enc.beginComputePass();
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
    this.device.queue.writeBuffer(this.bh, 16 + 64 * this.nMasses, packMasses([completeMass(m, this.dx0)]), 4, 16);
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

  massChanged(mass: number) {
    this.rhoMean = mass / this.cfg.problem.box ** this.cfg.ndim;
    this.writeParams();
    this.refreshPotential();
  }

  /** Call after submitting an encoder that encodeSteps() wrote to. */
  submitted() {
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
        const ops = tree.adapt((b) => ind[b.slot], this.cfg.refineHi, this.cfg.refineLo);
        if (ops) this.pending = { ops };
      })
      .finally(() => {
        this.flagInFlight = false;
      });
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
    const pass = enc.beginComputePass();
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
      return diagnosticsFrom(a, this.hasMasses ? this.nMasses : 0);
    } finally {
      this.stagingBusy = false;
    }
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
  fieldResources(parity: number): Record<number, GPUBuffer | GPUTextureView> {
    return { 2: this.state[parity][0].createView(), 3: this.state[parity][1].createView(), 5: this.idx.createView(), 30: this.bt };
  }

  status(): string {
    const s = this.tree.stats();
    const full = this.tree.levelLimit < this.levels
      ? `, block budget limits refinement to ${this.tree.levelLimit} of ${this.levels} levels`
      : this.tree.freeSlots < this.tree.nchild ? ', budget full' : '';
    return `${s.leaves} blocks, ${((100 * s.cells) / this.n ** this.cfg.ndim).toFixed(0)}% of uniform${full}`;
  }

  /** Every leaf cell with its position and size (validation only; reads the whole atlas). */
  async readLeaves(): Promise<LeafCells> {
    this.flush();
    const side = this.slotsPerSide * PAD;
    const depthTex = this.cfg.ndim === 2 ? 1 : side;
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
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pAssemble);
    pass.setBindGroup(0, this.gAssemble[p]);
    this.padded(pass, this.nAll);
    pass.setPipeline(this.pPredict);
    pass.setBindGroup(0, this.gPredict);
    this.padded(pass, this.nAll);
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this.updateGroup(pipe, p));
    this.interior(pass, this.nLeaf);
    pass.end();
    d.queue.submit([enc.finish()]);
    const side = this.slotsPerSide * PAD;
    const depthTex = this.cfg.ndim === 2 ? 1 : side;
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
          const nb = this.tree.neighbour(b, 2 * axis + side);
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
    for (const s of this.state) for (const t of s) t.destroy();
    for (const t of [...this.prim, ...this.pred, this.idx, this.gfield, this.dummyPhi]) t.destroy();
    this.grav?.destroy();
  }
}
