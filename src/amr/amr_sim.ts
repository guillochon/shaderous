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
// Self-gravity is not implemented on the block grid yet; a uniform external field is.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d, type Gpu } from '../gpu/device';
import { MAX, Reducer } from '../gpu/reduce';
import { EOS_MODULES, makeEos } from '../eos/eos';
import { DIAG_FLOATS, DIAG_OPS, diagnosticsFrom, packParams, PARAM_BYTES, type Diagnostics, type SimConfig, type Simulation } from '../sim';
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

  private constructor(gpu: Gpu, cfg: SimConfig) {
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
    // Default budget: three quarters of what refining everywhere would need (leaves plus their
    // parents), capped so the atlases stay within about 1 GB in 3D.
    const full = roots * (1 << (cfg.ndim * this.levels)) * (cfg.ndim === 3 ? 8 / 7 : 4 / 3);
    const auto = Math.ceil(Math.pow(Math.max(full * 0.75, roots * 2), 1 / cfg.ndim));
    const [lo, hi] = cfg.ndim === 3 ? [6, 10] : [12, 100];
    this.slotsPerSide = Math.min(limit, cfg.amrBudget || Math.min(hi, Math.max(lo, auto)));
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
    this.bt = d.createBuffer({ size: this.cap * BT_STRIDE * 4, usage: storage });
    this.listAll = d.createBuffer({ size: (this.cap + 1) * 4, usage: storage });
    this.listLeaf = d.createBuffer({ size: (this.cap + 1) * 4, usage: storage });
    this.flags = d.createBuffer({ size: this.cap * 4, usage: storage });
    this.rg = d.createBuffer({ size: this.cap * RG_STRIDE * 4, usage: storage });
    this.flagStaging = d.createBuffer({ size: this.cap * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.staging = d.createBuffer({ size: DIAG_FLOATS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  static async create(gpu: Gpu, cfg: SimConfig): Promise<AmrSim> {
    if (cfg.gravity !== 'off') throw new Error('Self-gravity is not available on the adaptive grid yet.');
    const sim = new AmrSim(gpu, cfg);
    const d = sim.device;
    const v = (t: GPUTexture) => t.createView();
    const depth = cfg.ndim === 2 ? 1 : BS;
    sim.redDt = await Reducer.create(
      d,
      [sim.cap * BS, BS, depth],
      [MAX, MAX],
      'amr/timestep.wgsl',
      {},
      [{ 0: sim.params, 1: v(sim.prim[0]), 2: v(sim.prim[1]), 30: sim.bt }],
      'amr timestep',
    );
    sim.pClock = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock', { PER_DX: 1 });
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock });
    sim.pFlag = await computePipeline(d, assemble('amr/flag.wgsl'), 'amr flag');
    sim.gFlag = bindGroup(d, sim.pFlag, { 0: sim.params, 1: v(sim.prim[0]), 2: v(sim.prim[1]), 3: sim.flags, 31: sim.listLeaf });
    await sim.buildHydro();
    sim.reset();
    return sim;
  }

  private get slots() {
    const c = this.cfg;
    return {
      eos: EOS_MODULES[c.eos],
      riemann: `hydro/riemann/${c.riemann}.wgsl`,
      limiter: `hydro/limiters/${c.limiter}.wgsl`,
    };
  }

  /** (Re)build every pipeline that depends on the EOS, Riemann solver or limiter. */
  async buildHydro() {
    const d = this.device;
    const slots = this.slots;
    const v = (t: GPUTexture) => t.createView();
    [this.pAssemble, this.pPredict, this.pUpdate, this.pRegrid] = await Promise.all([
      computePipeline(d, assemble('amr/assemble.wgsl', slots), 'amr assemble'),
      computePipeline(d, assemble('amr/predict.wgsl', slots), 'amr predict'),
      computePipeline(d, assemble('amr/update.wgsl', slots), 'amr update'),
      computePipeline(d, assemble('amr/regrid.wgsl', slots), 'amr regrid'),
    ]);
    this.gPredict = bindGroup(d, this.pPredict, {
      0: this.params, 1: this.clock, 2: v(this.prim[0]), 3: v(this.prim[1]), 5: v(this.pred[0]), 6: v(this.pred[1]), 30: this.bt, 31: this.listAll,
    });
    this.gAssemble = [];
    this.gUpdate = [];
    this.gRegrid = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      this.gAssemble.push(
        bindGroup(d, this.pAssemble, {
          0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(this.prim[0]), 4: v(this.prim[1]), 30: this.bt, 31: this.listAll,
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
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 30: this.bt })),
      'amr diagnostics',
    );
  }

  private updateGroup(pipe: GPUComputePipeline, p: number): GPUBindGroup {
    const v = (t: GPUTexture) => t.createView();
    const o = 1 - p;
    return bindGroup(this.device, pipe, {
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 4: v(this.prim[0]), 5: v(this.prim[1]),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 30: this.bt, 31: this.listLeaf,
    });
  }

  writeParams() {
    const c = this.cfg;
    const buf = packParams(c, {
      dims: [this.n0, this.n0, c.ndim === 2 ? 1 : this.n0],
      dx: this.dx0,
      rhoMean: 0,
      selfGravity: false,
      perturberGM: 0,
      amr: [this.slotsPerSide, this.n0 / BS, this.levels, 0],
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  refreshPotential() {}

  // ---------------------------------------------------------------- initial conditions

  /** Primitive (rho, p) on a block's cells plus one layer around it, from the analytic setup. */
  private indicator(b: Block, fn: (x: number, y: number, z: number) => { rho: number; p: number }): number {
    const three = this.cfg.ndim === 3;
    const dx = this.dx0 / (1 << b.level);
    const half = this.cfg.problem.box / 2;
    const m = BS + 2;
    const mz = three ? m : 1;
    const rho = new Float64Array(m * m * mz);
    const pr = new Float64Array(m * m * mz);
    for (let k = 0; k < mz; k++)
      for (let j = 0; j < m; j++)
        for (let i = 0; i < m; i++) {
          const w = fn(
            (b.c[0] * BS + i - 0.5) * dx - half,
            (b.c[1] * BS + j - 0.5) * dx - half,
            three ? (b.c[2] * BS + k - 0.5) * dx - half : 0,
          );
          rho[(k * m + j) * m + i] = w.rho;
          pr[(k * m + j) * m + i] = w.p;
        }
    let ind = 0;
    const jump = (q: Float64Array, a: number, s: number) => Math.abs(q[a + s] - q[a - s]) / (q[a + s] + q[a - s]);
    for (let k = three ? 1 : 0; k < (three ? m - 1 : 1); k++)
      for (let j = 1; j < m - 1; j++)
        for (let i = 1; i < m - 1; i++) {
          const a = (k * m + j) * m + i;
          ind = Math.max(ind, jump(rho, a, 1), jump(pr, a, 1), jump(rho, a, m), jump(pr, a, m));
          if (three) ind = Math.max(ind, jump(rho, a, m * m), jump(pr, a, m * m));
        }
    return ind;
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
    this.tree = new BlockTree(c.ndim, this.n0 / BS, this.levels, this.cap, c.bcLo, c.bcHi);
    for (let pass = 0; pass < this.levels; pass++) {
      if (!this.tree.adapt((b) => this.indicator(b, prim), c.refineHi, c.refineLo)) break;
    }

    const three = c.ndim === 3;
    const depth = three ? BS : 1;
    const half = c.problem.box / 2;
    const u1 = new Float32Array(BS * BS * depth * 4);
    const u2 = new Float32Array(BS * BS * depth * 4);
    for (const b of this.tree.leaves()) {
      const dx = this.dx0 / (1 << b.level);
      let o = 0;
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, o += 4) {
            const w = prim((b.c[0] * BS + i + 0.5) * dx - half, (b.c[1] * BS + j + 0.5) * dx - half, three ? (b.c[2] * BS + k + 0.5) * dx - half : 0);
            const eint = eos.eintFromP(w.rho, w.p);
            u1[o] = w.rho;
            u1[o + 1] = w.rho * w.vx;
            u1[o + 2] = w.rho * w.vy;
            u1[o + 3] = w.rho * w.vz;
            u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz);
            u2[o + 1] = eint;
            u2[o + 2] = w.rho * w.s;
            u2[o + 3] = 0;
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
    d.queue.writeBuffer(this.clock, 0, new Float32Array(4));
    this.writeParams();
    this.uploadTree();
  }

  /** Atlas texel of a slot's first interior cell. */
  private origin(slot: number): [number, number, number] {
    const a = this.slotsPerSide;
    if (this.cfg.ndim === 2) return [(slot % a) * PAD + NG, Math.floor(slot / a) * PAD + NG, 0];
    return [(slot % a) * PAD + NG, (Math.floor(slot / a) % a) * PAD + NG, Math.floor(slot / (a * a)) * PAD + NG];
  }

  private uploadTree() {
    const d = this.device;
    const { all, leaves } = this.tree.lists();
    this.nAll = all.length;
    this.nLeaf = leaves.length;
    const pack = (list: Uint32Array) => {
      const out = new Uint32Array(this.cap + 1);
      out[0] = list.length;
      out.set(list, 1);
      return out;
    };
    d.queue.writeBuffer(this.bt, 0, this.tree.table() as Int32Array<ArrayBuffer>);
    d.queue.writeBuffer(this.listAll, 0, pack(all));
    d.queue.writeBuffer(this.listLeaf, 0, pack(leaves));
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
      this.cur = 1 - p;
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

  /** Call after submitting an encoder that encodeSteps() wrote to. */
  submitted() {
    if (!this.wantMap) return;
    this.wantMap = false;
    this.flagInFlight = true;
    const tree = this.tree;
    this.flagWait = this.flagStaging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const ind = new Float32Array(this.flagStaging.getMappedRange().slice(0));
        this.flagStaging.unmap();
        if (tree !== this.tree) return; // the run was reset meanwhile
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
    this.device.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      return diagnosticsFrom(a, false);
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
    const full = this.tree.freeSlots < this.tree.nchild ? ', budget full' : '';
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
    for (const t of [...this.prim, ...this.pred, this.idx]) t.destroy();
  }
}
