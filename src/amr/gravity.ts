// Self-gravity on the block grid: a Poisson solve per tree level, coarsest first, each
// accelerated by multigrid within the blocks. See amr/gravity/common.wgsl for the scheme.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d } from '../gpu/device';
import { BS, PAD } from './tree';

const PRE = 2; // smoothing sweeps before restriction
const POST = 2; // and after prolongation
const COARSE = 8; // sweeps on the coarsest sublevel (one cell per block)
const NSUB = 5; // sublevels: 16, 8, 4, 2, 1 cells per block side
/** The field pass with one timestep for all blocks. */
export const FIELD_SLOTS = { bins: 'amr/bins_off.wgsl', half: 'amr/gravity/half_off.wgsl' };
/** And with per-block timesteps: the half step is each block's own. */
export const FIELD_SLOTS_BINS = { bins: 'amr/bins_on.wgsl', half: '#include "amr/gravity/half_on.wgsl"\n#include "amr/gravity/half_field.wgsl"' };
const SEQUENTIAL = !!(globalThis as { AMR_GRAVITY_SEQUENTIAL?: boolean }).AMR_GRAVITY_SEQUENTIAL; // debugging: always level by level
/**
 * Cycles after the first of a warm-started solve run only while the residual stays above
 * this, relative to the largest source term (amr/gravity/gate.wgsl).
 */
export const VCYCLE_TOL = Number(new URLSearchParams(globalThis.location?.search ?? '').get('vtol')) || 1e-4;
/** A dispatch count that stands for "as many as the gate says" (cells(), color(), faces()). */
const GATED = -2;

export interface GravityContext {
  device: GPUDevice;
  /** atlas slots per side */
  slotsPerSide: number;
  /** deepest tree level */
  levels: number;
  params: GPUBuffer;
  bt: GPUBuffer;
  listAll: GPUBuffer;
  listLeaf: GPUBuffer;
  /** one list per tree level: the blocks at that level, parents and leaves */
  listLevel: GPUBuffer[];
  rg: GPUBuffer;
  clock: GPUBuffer;
  bh: GPUBuffer;
  /** multipole moments for isolated boundaries */
  multipole: GPUBuffer;
  /** density-carrying state atlas, per parity */
  stateU1: GPUTexture[];
  /** gravity field atlas (acceleration, half-step potential) */
  gfield: GPUTexture;
  /** per-block timesteps: the bins, the lists of the coming substep, and their dispatch sizes */
  bins: GPUBuffer;
  listAct: GPUBuffer;
  listUpd: GPUBuffer;
  binArgs: GPUBuffer;
}

/** Byte offset in binArgs of the dispatch sizes of kind i (amr/bins_args.wgsl). */
const ARG = (i: number) => 12 * i;

interface Sub {
  bsize: number;
  phi: GPUTexture;
  rhs: GPUTexture;
  smooth: GPUBindGroup[][]; // [tree level][color]
  restrict?: GPUBindGroup[]; // residual of this sublevel onto the coarser one, [tree level]
  prolong?: GPUBindGroup[]; // into this sublevel from the coarser one, [tree level]
}

export class AmrGravity {
  private sub: Sub[] = [];
  phiOld!: GPUTexture;
  private pSmooth!: GPUComputePipeline;
  private pRestrict!: GPUComputePipeline;
  private pProlong!: GPUComputePipeline;
  private pRhs!: GPUComputePipeline;
  private pRhsRestrict!: GPUComputePipeline;
  private pGhostCoarse!: GPUComputePipeline;
  private pGhostRest!: GPUComputePipeline;
  private pField!: GPUComputePipeline;
  private pNew!: GPUComputePipeline;
  private pInit!: GPUComputePipeline;
  private gInit: GPUBindGroup[] = [];
  private gRhs: GPUBindGroup[] = [];
  private gRhsRestrict: GPUBindGroup[] = [];
  private gGhostCoarse: GPUBindGroup[] = [];
  private gGhostRest!: GPUBindGroup;
  private gGhostCoarseAll!: GPUBindGroup;
  private gField!: GPUBindGroup;
  private gNew!: GPUBindGroup;
  // Adaptive cycles: the residual after a cycle, and the gate that sizes the next one.
  private pResidual!: GPUComputePipeline;
  private pGate!: GPUComputePipeline;
  private pGateRecord!: GPUComputePipeline;
  private gGateRecord!: GPUBindGroup;
  private gResidual!: GPUBindGroup;
  private gGate!: GPUBindGroup;
  private res!: GPUBuffer;
  private gateArgs!: GPUBuffer;
  /** [0] cycles run beyond the first, [1] the last relative residual (float bits), [2] residuals measured */
  stats!: GPUBuffer;
  // Per-block timesteps: the passes of a solve restricted to the blocks starting a step.
  private dphi?: GPUTexture;
  private pFieldB?: GPUComputePipeline;
  private pExtrap?: GPUComputePipeline;
  private pSync?: GPUComputePipeline;
  private pForget?: GPUComputePipeline;
  private gFieldB?: GPUBindGroup;
  private gFieldAllB?: GPUBindGroup;
  private gExtrap?: GPUBindGroup;
  private gSync?: GPUBindGroup;
  private gSyncAll?: GPUBindGroup;
  private gForget?: GPUBindGroup;
  private gRhsB: GPUBindGroup[] = [];
  private gRhsRestrictB!: GPUBindGroup;
  private gGhostCoarseB!: GPUBindGroup;
  private gGhostRestB!: GPUBindGroup;

  private constructor(private ctx: GravityContext) {}

  get phi(): GPUTexture {
    return this.sub[0].phi;
  }

  static async create(ctx: GravityContext): Promise<AmrGravity> {
    const g = new AmrGravity(ctx);
    const d = ctx.device;
    const A = ctx.slotsPerSide;
    const v = (t: GPUTexture) => t.createView();
    [g.pSmooth, g.pRestrict, g.pProlong, g.pRhs, g.pRhsRestrict, g.pGhostCoarse, g.pGhostRest, g.pField, g.pNew, g.pInit] = await Promise.all([
      computePipeline(d, assemble('amr/gravity/smooth.wgsl'), 'amr mg smooth'),
      computePipeline(d, assemble('amr/gravity/restrict.wgsl'), 'amr mg residual + restrict'),
      computePipeline(d, assemble('amr/gravity/prolong.wgsl'), 'amr mg prolong'),
      computePipeline(d, assemble('amr/gravity/rhs.wgsl'), 'amr mg source'),
      computePipeline(d, assemble('amr/gravity/rhs_restrict.wgsl'), 'amr mg source restrict'),
      computePipeline(d, assemble('amr/gravity/ghost.wgsl'), 'amr mg ghost (coarse)', { MODE: 0 }),
      computePipeline(d, assemble('amr/gravity/ghost.wgsl'), 'amr mg ghost (rest)', { MODE: 1 }),
      computePipeline(d, assemble('amr/gravity/field.wgsl', FIELD_SLOTS), 'amr gravity field'),
      computePipeline(d, assemble('amr/gravity/new_blocks.wgsl'), 'amr mg new blocks'),
      computePipeline(d, assemble('amr/gravity/new_blocks.wgsl'), 'amr mg level init', { ALL: 1 }),
    ]);

    // Sublevel geometry. Sublevel 0 shares the hydro atlas layout; the rest keep one ghost layer.
    const geom = Array.from({ length: NSUB }, (_, k) => {
      const bsize = BS >> k;
      return k === 0 ? { bsize, pad: (PAD - BS) / 2, stride: PAD } : { bsize, pad: 1, stride: bsize + 2 };
    });
    const uniform = (k: number, color: number, other: number) => {
      const buf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const a = geom[k];
      const o = geom[Math.min(Math.max(other, 0), NSUB - 1)];
      d.queue.writeBuffer(buf, 0, new Int32Array([a.bsize, a.pad, a.stride, k, color, o.bsize, o.pad, o.stride]));
      return buf;
    };

    g.phiOld = tex3d(d, A * PAD, 'r32float', 'amr phi old');
    for (let k = 0; k < NSUB; k++) {
      const side = A * geom[k].stride;
      g.sub.push({
        bsize: geom[k].bsize,
        phi: tex3d(d, side, 'r32float', `amr phi k${k}`),
        rhs: tex3d(d, side, 'r32float', `amr rhs k${k}`),
        smooth: [],
      });
    }
    // Bind groups are indexed by tree level; one more entry covers every block at once, and
    // another the blocks starting a step (per-block timesteps).
    const lists = [...ctx.listLevel, ctx.listAll, ctx.listAct];
    for (let k = 0; k < NSUB; k++) {
      const s = g.sub[k];
      const uSmooth = [uniform(k, 0, k + 1), uniform(k, 1, k + 1)]; // "other" = the coarser sublevel
      for (let l = 0; l <= ctx.levels + 2; l++) {
        const list = lists[l];
        s.smooth.push(
          uSmooth.map((u) => bindGroup(d, g.pSmooth, { 0: ctx.params, 1: u, 2: v(s.phi), 3: v(s.rhs), 10: ctx.multipole, 30: ctx.bt, 31: list })),
        );
        if (k < NSUB - 1) {
          const c = g.sub[k + 1];
          (s.restrict ??= []).push(
            bindGroup(d, g.pRestrict, { 0: ctx.params, 1: uSmooth[0], 2: v(s.phi), 3: v(s.rhs), 4: v(c.rhs), 5: v(c.phi), 10: ctx.multipole, 30: ctx.bt, 31: list }),
          );
        }
        if (k < NSUB - 1) {
          (s.prolong ??= []).push(bindGroup(d, g.pProlong, { 0: ctx.params, 1: uSmooth[0], 2: v(g.sub[k + 1].phi), 3: v(s.phi), 30: ctx.bt, 31: list }));
        }
      }
    }

    const phi0 = v(g.sub[0].phi);
    const rhs0 = v(g.sub[0].rhs);
    g.gRhs = ctx.stateU1.map((t) => bindGroup(d, g.pRhs, { 0: ctx.params, 1: v(t), 2: rhs0, 31: ctx.listLeaf }));
    g.gGhostCoarseAll = bindGroup(d, g.pGhostCoarse, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listAll });
    for (let l = 0; l <= ctx.levels; l++) {
      g.gRhsRestrict.push(bindGroup(d, g.pRhsRestrict, { 0: ctx.params, 2: rhs0, 30: ctx.bt, 31: ctx.listLevel[l] }));
      g.gGhostCoarse.push(bindGroup(d, g.pGhostCoarse, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listLevel[l] }));
      g.gInit.push(bindGroup(d, g.pInit, { 0: ctx.params, 2: phi0, 7: ctx.rg, 30: ctx.bt, 31: ctx.listLevel[l] }));
    }
    g.gGhostRest = bindGroup(d, g.pGhostRest, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listAll });
    g.gRhsB = ctx.stateU1.map((t) => bindGroup(d, g.pRhs, { 0: ctx.params, 1: v(t), 2: rhs0, 31: ctx.listUpd }));
    g.gRhsRestrictB = bindGroup(d, g.pRhsRestrict, { 0: ctx.params, 2: rhs0, 30: ctx.bt, 31: ctx.listAct });
    g.gGhostCoarseB = bindGroup(d, g.pGhostCoarse, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listAct });
    g.gGhostRestB = bindGroup(d, g.pGhostRest, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listAct });
    g.gField = bindGroup(d, g.pField, { 0: ctx.params, 1: phi0, 2: v(g.phiOld), 3: v(ctx.gfield), 4: ctx.bh, 5: ctx.clock, 30: ctx.bt, 31: ctx.listAll });
    g.gNew = bindGroup(d, g.pNew, { 0: ctx.params, 2: phi0, 7: ctx.rg, 30: ctx.bt, 31: ctx.listLeaf });

    [g.pResidual, g.pGate, g.pGateRecord] = await Promise.all([
      computePipeline(d, assemble('amr/gravity/residual.wgsl'), 'amr mg residual check'),
      computePipeline(d, assemble('amr/gravity/gate.wgsl'), 'amr mg cycle gate', { TOL: VCYCLE_TOL }),
      computePipeline(d, assemble('amr/gravity/gate.wgsl'), 'amr mg cycle gate', { RECORD: 1 }),
    ]);
    g.res = d.createBuffer({ label: 'amr mg residual', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    g.gateArgs = d.createBuffer({ label: 'amr mg gated sizes', size: 33 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    g.stats = d.createBuffer({ label: 'amr mg cycle stats', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const uRes = uniform(0, 0, 1);
    g.gResidual = bindGroup(d, g.pResidual, { 0: ctx.params, 1: uRes, 2: phi0, 3: rhs0, 4: g.res, 10: ctx.multipole, 30: ctx.bt, 31: ctx.listAll });
    g.gGate = bindGroup(d, g.pGate, { 1: g.res, 2: g.gateArgs, 3: g.stats, 31: ctx.listAll });
    g.gGateRecord = bindGroup(d, g.pGateRecord, { 1: g.res, 2: g.gateArgs, 3: g.stats, 31: ctx.listAll });
    return g;
  }

  /**
   * Dispatch over the cells of `count` blocks at a sublevel. A negative count stands for the
   * blocks starting a step (per-block timesteps): their number is only known on the GPU, and
   * the sizes come from the table amr/bins_args.wgsl fills.
   */
  private cells(pass: GPUComputePassEncoder, count: number, bsize: number) {
    if (count === GATED) return pass.dispatchWorkgroupsIndirect(this.gateArgs, ARG(Math.log2(BS / bsize)));
    if (count < 0) return pass.dispatchWorkgroupsIndirect(this.ctx.binArgs, ARG(3 + Math.log2(BS / bsize)));
    const span = Math.max(bsize, 8);
    pass.dispatchWorkgroups(count * (span / 8), Math.ceil(bsize / 8), Math.ceil(bsize / 4));
  }
  /** Dispatch over the cells of one color (see smooth.wgsl). */
  private color(pass: GPUComputePassEncoder, count: number, bsize: number) {
    if (count === GATED) return pass.dispatchWorkgroupsIndirect(this.gateArgs, ARG(5 + Math.log2(BS / bsize)));
    if (count < 0) return pass.dispatchWorkgroupsIndirect(this.ctx.binArgs, ARG(8 + Math.log2(BS / bsize)));
    pass.dispatchWorkgroups(Math.ceil((count * Math.max(bsize / 2, 1)) / 8), Math.ceil(bsize / 8), Math.ceil(bsize / 4));
  }
  /** Dispatch over the face ghosts of `count` blocks (ghost.wgsl): 6 x 2 x 16 x 16 cells, in rows of 64. */
  private faces(pass: GPUComputePassEncoder, count: number) {
    if (count === GATED) return pass.dispatchWorkgroupsIndirect(this.gateArgs, ARG(10));
    if (count < 0) return pass.dispatchWorkgroupsIndirect(this.ctx.binArgs, ARG(15));
    if (count > 0) pass.dispatchWorkgroups((6 * 2 * BS * BS) / 64, count, 1);
  }
  private padded(pass: GPUComputePassEncoder, count: number) {
    if (count < 0) return pass.dispatchWorkgroupsIndirect(this.ctx.binArgs, ARG(0));
    pass.dispatchWorkgroups(count * 3, 3, 5);
  }

  private smooth(pass: GPUComputePassEncoder, k: number, level: number, count: number, sweeps: number) {
    pass.setPipeline(this.pSmooth);
    for (let i = 0; i < sweeps; i++) {
      for (const g of this.sub[k].smooth[level]) {
        pass.setBindGroup(0, g);
        this.color(pass, count, this.sub[k].bsize);
      }
    }
  }

  private vcycle(pass: GPUComputePassEncoder, level: number, count: number) {
    const last = NSUB - 1;
    for (let k = 0; k < last; k++) {
      this.smooth(pass, k, level, count, PRE);
      pass.setPipeline(this.pRestrict);
      pass.setBindGroup(0, this.sub[k].restrict![level]);
      this.cells(pass, count, this.sub[k + 1].bsize);
    }
    this.smooth(pass, last, level, count, COARSE);
    for (let k = last - 1; k >= 0; k--) {
      pass.setPipeline(this.pProlong);
      pass.setBindGroup(0, this.sub[k].prolong![level]);
      this.cells(pass, count, this.sub[k].bsize);
      this.smooth(pass, k, level, count, POST);
    }
  }

  /**
   * Source from the state of the given parity, restricted onto every parent.
   * @param nLevel blocks per tree level  @param nLeaf leaf count
   */
  encodeSource(pass: GPUComputePassEncoder, parity: number, nLevel: number[], nLeaf: number) {
    pass.setPipeline(this.pRhs);
    pass.setBindGroup(0, this.gRhs[parity]);
    this.cells(pass, nLeaf, BS);
    pass.setPipeline(this.pRhsRestrict);
    for (let l = this.ctx.levels - 1; l >= 0; l--) {
      pass.setBindGroup(0, this.gRhsRestrict[l]);
      this.cells(pass, nLevel[l], BS);
    }
  }

  /**
   * Solve every level in turn, then fill the remaining ghost cells. Normally each level starts
   * from its previous potential; a cold start begins each from the coarser level's solution.
   * `together` relaxes all levels in the same dispatches instead (for small changes only).
   */
  encodeSolve(pass: GPUComputePassEncoder, nLevel: number[], nAll: number, cycles: number, cold = false, together = false) {
    together &&= !cold && !SEQUENTIAL;
    for (let l = 0; l <= this.ctx.levels; l++) {
      if (together) break;
      if (nLevel[l] === 0) continue;
      if (l > 0) {
        if (cold) {
          pass.setPipeline(this.pInit);
          pass.setBindGroup(0, this.gInit[l]);
          this.cells(pass, nLevel[l], BS);
        }
        // Boundary condition from the coarser level's finished solution.
        pass.setPipeline(this.pGhostCoarse);
        pass.setBindGroup(0, this.gGhostCoarse[l]);
        this.faces(pass, nLevel[l]);
      }
      for (let c = 0; c < cycles; c++) this.vcycle(pass, l, nLevel[l]);
    }
    if (together) {
      // Warm start (every step): all levels relax together, each against the boundary values
      // its coarser level had at the start of the cycle. The lag costs nothing measurable when
      // the potential changes little per step, and the number of dispatches no longer grows
      // with the depth of the tree.
      //
      // `cycles` is the most it runs. After the first, each further cycle runs only if the
      // residual the last one left is above VCYCLE_TOL of the largest source term and that
      // cycle halved it at least: a residual pass measures it and a gate pass sizes the next
      // cycle's dispatches, all on the GPU. (The first is measured too, before the first cycle.)
      const all = this.ctx.levels + 1;
      if (cycles > 1) {
        pass.setPipeline(this.pResidual);
        pass.setBindGroup(0, this.gResidual);
        this.cells(pass, nAll, BS);
        pass.setPipeline(this.pGateRecord);
        pass.setBindGroup(0, this.gGateRecord);
        pass.dispatchWorkgroups(1);
      }
      for (let c = 0; c < cycles; c++) {
        const count = c === 0 ? nAll : GATED;
        if (c > 0) {
          pass.setPipeline(this.pResidual);
          pass.setBindGroup(0, this.gResidual);
          this.cells(pass, c === 1 ? nAll : GATED, BS);
          pass.setPipeline(this.pGate);
          pass.setBindGroup(0, this.gGate);
          pass.dispatchWorkgroups(1);
        }
        pass.setPipeline(this.pGhostCoarse);
        pass.setBindGroup(0, this.gGhostCoarseAll);
        this.faces(pass, count);
        this.vcycle(pass, all, count);
      }
    }
    pass.setPipeline(this.pGhostRest);
    pass.setBindGroup(0, this.gGhostRest);
    this.faces(pass, nAll);
  }

  /** Acceleration and half-step potential for every block. */
  encodeField(pass: GPUComputePassEncoder, nAll: number) {
    pass.setPipeline(this.pField);
    pass.setBindGroup(0, this.gField);
    this.padded(pass, nAll);
  }

  // ---------------------------------------------------------------- per-block timesteps
  //
  // The potential is solved on every block when a cycle ends, and in between only on the
  // blocks that start a step, which see across their faces the potential of the others
  // carried forward in time at the rate it last changed there.

  /** Build the passes of per-block timesteps (once; they do not depend on the number of bins). */
  async setBins() {
    if (this.pFieldB) return;
    const ctx = this.ctx;
    const d = ctx.device;
    const v = (t: GPUTexture) => t.createView();
    this.dphi = tex3d(d, ctx.slotsPerSide * PAD, 'r32float', 'amr phi rate');
    const half = FIELD_SLOTS_BINS.half;
    [this.pFieldB, this.pExtrap, this.pSync, this.pForget] = await Promise.all([
      computePipeline(d, assemble('amr/gravity/field.wgsl', { ...FIELD_SLOTS_BINS, half }), 'amr gravity field (bins)'),
      computePipeline(d, assemble('amr/gravity/extrap.wgsl'), 'amr potential extrapolation'),
      computePipeline(d, assemble('amr/gravity/sync.wgsl'), 'amr potential rate'),
      computePipeline(d, assemble('amr/gravity/sync.wgsl'), 'amr potential rate reset', { FORGET: 1 }),
    ]);
    const phi0 = v(this.phi);
    const field = (list: GPUBuffer) =>
      bindGroup(d, this.pFieldB!, { 0: ctx.params, 1: phi0, 3: v(ctx.gfield), 4: ctx.bh, 5: ctx.clock, 6: v(this.dphi!), 29: ctx.bins, 30: ctx.bt, 31: list });
    this.gFieldB = field(ctx.listAct);
    this.gFieldAllB = field(ctx.listAll);
    this.gExtrap = bindGroup(d, this.pExtrap, { 0: ctx.params, 1: ctx.clock, 2: phi0, 6: v(this.dphi), 29: ctx.bins, 31: ctx.listAll });
    const sync = (pipe: GPUComputePipeline, list: GPUBuffer) =>
      bindGroup(d, pipe, { 0: ctx.params, 1: ctx.clock, 2: phi0, 3: v(this.phiOld), 4: v(this.dphi!), 29: ctx.bins, 31: list });
    this.gSync = sync(this.pSync, ctx.listAct);
    this.gSyncAll = sync(this.pSync, ctx.listAll);
    this.gForget = sync(this.pForget, ctx.listAll);
  }

  /** Source from the blocks the last substep changed, restricted onto their ancestors. */
  encodeSourceB(pass: GPUComputePassEncoder, parity: number) {
    pass.setPipeline(this.pRhs);
    pass.setBindGroup(0, this.gRhsB[parity]);
    pass.dispatchWorkgroupsIndirect(this.ctx.binArgs, ARG(1));
    // One pass handles a parent of leaves; each further pass reaches one level higher.
    pass.setPipeline(this.pRhsRestrict);
    pass.setBindGroup(0, this.gRhsRestrictB);
    for (let l = 0; l < this.ctx.levels; l++) this.cells(pass, -1, BS);
  }

  /** Carry the potential one substep forward on the blocks that are not starting a step. */
  encodeExtrap(pass: GPUComputePassEncoder, nAll: number) {
    pass.setPipeline(this.pExtrap!);
    pass.setBindGroup(0, this.gExtrap!);
    this.cells(pass, nAll, BS);
  }

  /** Relax the potential on the blocks starting a step, all levels together (a warm start). */
  encodeSolveB(pass: GPUComputePassEncoder, cycles: number) {
    const act = this.ctx.levels + 2;
    for (let c = 0; c < cycles; c++) {
      pass.setPipeline(this.pGhostCoarse);
      pass.setBindGroup(0, this.gGhostCoarseB);
      this.faces(pass, -1);
      this.vcycle(pass, act, -1);
    }
    pass.setPipeline(this.pGhostRest);
    pass.setBindGroup(0, this.gGhostRestB);
    this.faces(pass, -1);
  }

  /** The field with per-block timesteps: for every block, or for the blocks starting a step. */
  encodeFieldB(pass: GPUComputePassEncoder, nAll: number) {
    pass.setPipeline(this.pFieldB!);
    pass.setBindGroup(0, nAll < 0 ? this.gFieldB! : this.gFieldAllB!);
    this.padded(pass, nAll);
  }

  /** After a solve: the rate at which the potential changed over the step just finished. */
  encodeSync(pass: GPUComputePassEncoder, nAll: number) {
    pass.setPipeline(this.pSync!);
    pass.setBindGroup(0, nAll < 0 ? this.gSync! : this.gSyncAll!);
    this.padded(pass, nAll);
  }

  /** Start the potential's history afresh (a new mesh or a new run). */
  encodeForget(pass: GPUComputePassEncoder, nAll: number) {
    if (!this.pForget) return;
    pass.setPipeline(this.pForget);
    pass.setBindGroup(0, this.gForget!);
    this.padded(pass, nAll);
  }

  /** First guess for the potential in blocks the last re-mesh created. */
  encodeNewBlocks(pass: GPUComputePassEncoder, nLeaf: number) {
    pass.setPipeline(this.pNew);
    pass.setBindGroup(0, this.gNew);
    this.cells(pass, nLeaf, BS);
  }

  /** Keep the potential as the "previous" one, for extrapolation to the half step. */
  copyPhiToOld(enc: GPUCommandEncoder) {
    const side = this.ctx.slotsPerSide * PAD;
    enc.copyTextureToTexture({ texture: this.phi }, { texture: this.phiOld }, [side, side, side]);
  }

  destroy() {
    for (const s of this.sub) {
      s.phi.destroy();
      s.rhs.destroy();
    }
    this.phiOld.destroy();
    this.dphi?.destroy();
  }
}
