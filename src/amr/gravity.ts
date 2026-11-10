// Self-gravity on the block grid: a Poisson solve per tree level, coarsest first, each
// accelerated by multigrid within the blocks. See amr/gravity/common.wgsl for the scheme.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d } from '../gpu/device';
import { BS, PAD } from './tree';

const PRE = 2; // smoothing sweeps before restriction
const POST = 2; // and after prolongation
const COARSE = 8; // sweeps on the coarsest sublevel (one cell per block)
const NSUB = 5; // sublevels: 16, 8, 4, 2, 1 cells per block side

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
}

interface Sub {
  bsize: number;
  phi: GPUTexture;
  rhs: GPUTexture;
  res?: GPUTexture;
  smooth: GPUBindGroup[][]; // [tree level][colour]
  residual: GPUBindGroup[]; // [tree level]
  restrict?: GPUBindGroup[]; // into this sublevel from the finer one, [tree level]
  prolong?: GPUBindGroup[]; // into this sublevel from the coarser one, [tree level]
}

export class AmrGravity {
  private sub: Sub[] = [];
  phiOld!: GPUTexture;
  private pSmooth!: GPUComputePipeline;
  private pResidual!: GPUComputePipeline;
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
  private gField!: GPUBindGroup;
  private gNew!: GPUBindGroup;

  private constructor(private ctx: GravityContext) {}

  get phi(): GPUTexture {
    return this.sub[0].phi;
  }

  static async create(ctx: GravityContext): Promise<AmrGravity> {
    const g = new AmrGravity(ctx);
    const d = ctx.device;
    const A = ctx.slotsPerSide;
    const v = (t: GPUTexture) => t.createView();
    [g.pSmooth, g.pResidual, g.pRestrict, g.pProlong, g.pRhs, g.pRhsRestrict, g.pGhostCoarse, g.pGhostRest, g.pField, g.pNew, g.pInit] = await Promise.all([
      computePipeline(d, assemble('amr/gravity/smooth.wgsl'), 'amr mg smooth'),
      computePipeline(d, assemble('amr/gravity/residual.wgsl'), 'amr mg residual'),
      computePipeline(d, assemble('amr/gravity/restrict.wgsl'), 'amr mg restrict'),
      computePipeline(d, assemble('amr/gravity/prolong.wgsl'), 'amr mg prolong'),
      computePipeline(d, assemble('amr/gravity/rhs.wgsl'), 'amr mg source'),
      computePipeline(d, assemble('amr/gravity/rhs_restrict.wgsl'), 'amr mg source restrict'),
      computePipeline(d, assemble('amr/gravity/ghost.wgsl'), 'amr mg ghost (coarse)', { MODE: 0 }),
      computePipeline(d, assemble('amr/gravity/ghost.wgsl'), 'amr mg ghost (rest)', { MODE: 1 }),
      computePipeline(d, assemble('amr/gravity/field.wgsl'), 'amr gravity field'),
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
        res: k < NSUB - 1 ? tex3d(d, side, 'r32float', `amr res k${k}`) : undefined,
        smooth: [],
        residual: [],
      });
    }
    for (let k = 0; k < NSUB; k++) {
      const s = g.sub[k];
      const uSmooth = [uniform(k, 0, k + 1), uniform(k, 1, k + 1)]; // "other" = the coarser sublevel
      const uFromFine = uniform(k, 0, k - 1); // "other" = the finer sublevel
      for (let l = 0; l <= ctx.levels; l++) {
        const list = ctx.listLevel[l];
        s.smooth.push(
          uSmooth.map((u) => bindGroup(d, g.pSmooth, { 0: ctx.params, 1: u, 2: v(s.phi), 3: v(s.rhs), 10: ctx.multipole, 30: ctx.bt, 31: list })),
        );
        if (s.res) {
          s.residual.push(
            bindGroup(d, g.pResidual, { 0: ctx.params, 1: uSmooth[0], 2: v(s.phi), 3: v(s.rhs), 4: v(s.res), 10: ctx.multipole, 30: ctx.bt, 31: list }),
          );
        }
        if (k > 0) {
          (s.restrict ??= []).push(bindGroup(d, g.pRestrict, { 0: ctx.params, 1: uFromFine, 2: v(g.sub[k - 1].res!), 3: v(s.rhs), 4: v(s.phi), 31: list }));
        }
        if (k < NSUB - 1) {
          (s.prolong ??= []).push(bindGroup(d, g.pProlong, { 0: ctx.params, 1: uSmooth[0], 2: v(g.sub[k + 1].phi), 3: v(s.phi), 30: ctx.bt, 31: list }));
        }
      }
    }

    const phi0 = v(g.sub[0].phi);
    const rhs0 = v(g.sub[0].rhs);
    g.gRhs = ctx.stateU1.map((t) => bindGroup(d, g.pRhs, { 0: ctx.params, 1: v(t), 2: rhs0, 31: ctx.listLeaf }));
    for (let l = 0; l <= ctx.levels; l++) {
      g.gRhsRestrict.push(bindGroup(d, g.pRhsRestrict, { 0: ctx.params, 2: rhs0, 30: ctx.bt, 31: ctx.listLevel[l] }));
      g.gGhostCoarse.push(bindGroup(d, g.pGhostCoarse, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listLevel[l] }));
      g.gInit.push(bindGroup(d, g.pInit, { 0: ctx.params, 2: phi0, 7: ctx.rg, 30: ctx.bt, 31: ctx.listLevel[l] }));
    }
    g.gGhostRest = bindGroup(d, g.pGhostRest, { 0: ctx.params, 2: phi0, 30: ctx.bt, 31: ctx.listAll });
    g.gField = bindGroup(d, g.pField, { 0: ctx.params, 1: phi0, 2: v(g.phiOld), 3: v(ctx.gfield), 4: ctx.bh, 5: ctx.clock, 30: ctx.bt, 31: ctx.listAll });
    g.gNew = bindGroup(d, g.pNew, { 0: ctx.params, 2: phi0, 7: ctx.rg, 30: ctx.bt, 31: ctx.listLeaf });
    return g;
  }

  /** Dispatch over the cells of `count` blocks at a sublevel. */
  private cells(pass: GPUComputePassEncoder, count: number, bsize: number) {
    const span = Math.max(bsize, 8);
    pass.dispatchWorkgroups(count * (span / 8), Math.ceil(bsize / 8), Math.ceil(bsize / 4));
  }
  private padded(pass: GPUComputePassEncoder, count: number) {
    pass.dispatchWorkgroups(count * 3, 3, 5);
  }

  private smooth(pass: GPUComputePassEncoder, k: number, level: number, count: number, sweeps: number) {
    pass.setPipeline(this.pSmooth);
    for (let i = 0; i < sweeps; i++) {
      for (const g of this.sub[k].smooth[level]) {
        pass.setBindGroup(0, g);
        this.cells(pass, count, this.sub[k].bsize);
      }
    }
  }

  private vcycle(pass: GPUComputePassEncoder, level: number, count: number) {
    const last = NSUB - 1;
    for (let k = 0; k < last; k++) {
      this.smooth(pass, k, level, count, PRE);
      pass.setPipeline(this.pResidual);
      pass.setBindGroup(0, this.sub[k].residual[level]);
      this.cells(pass, count, this.sub[k].bsize);
      pass.setPipeline(this.pRestrict);
      pass.setBindGroup(0, this.sub[k + 1].restrict![level]);
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
   */
  encodeSolve(pass: GPUComputePassEncoder, nLevel: number[], nAll: number, cycles: number, cold = false) {
    for (let l = 0; l <= this.ctx.levels; l++) {
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
        this.padded(pass, nLevel[l]);
      }
      for (let c = 0; c < cycles; c++) this.vcycle(pass, l, nLevel[l]);
    }
    pass.setPipeline(this.pGhostRest);
    pass.setBindGroup(0, this.gGhostRest);
    this.padded(pass, nAll);
  }

  /** Acceleration and half-step potential for every block. */
  encodeField(pass: GPUComputePassEncoder, nAll: number) {
    pass.setPipeline(this.pField);
    pass.setBindGroup(0, this.gField);
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
      s.res?.destroy();
    }
    this.phiOld.destroy();
  }
}
