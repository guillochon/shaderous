// Geometric multigrid Poisson solver: V-cycles with red-black Gauss-Seidel smoothing,
// 8-cell-average restriction and trilinear prolongation. One r32float texture per level for
// each of phi, the right-hand side and the residual.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d } from '../gpu/device';
import { Reducer, SUM } from '../gpu/reduce';

const PRE = 2; // smoothing sweeps before restriction
const POST = 2; // and after prolongation
const COARSE = 8; // sweeps on the coarsest (2^3) level

interface Level {
  n: number;
  phi: GPUTexture;
  rhs: GPUTexture;
  res: GPUTexture;
  smooth: GPUBindGroup[]; // [red, black]
  residual: GPUBindGroup;
  restrict?: GPUBindGroup; // from the next finer level into this one
  prolong?: GPUBindGroup; // from the next coarser level into this one
}

export class Multigrid {
  levels: Level[] = [];
  phiOld!: GPUTexture; // potential one step back, for extrapolating to the half step
  multipole!: Reducer;
  private pSmooth!: GPUComputePipeline;
  private pResidual!: GPUComputePipeline;
  private pRestrict!: GPUComputePipeline;
  private pProlong!: GPUComputePipeline;
  private pRhs!: GPUComputePipeline;
  private rhsGroups: GPUBindGroup[] = [];

  get phi(): GPUTexture {
    return this.levels[0].phi;
  }

  /** @param stateU1 the density-carrying state texture for each parity */
  static async create(
    device: GPUDevice,
    n: number,
    box: number,
    params: GPUBuffer,
    stateU1: GPUTexture[],
  ): Promise<Multigrid> {
    if (n & (n - 1)) throw new Error('multigrid needs a power-of-two grid');
    const mg = new Multigrid();
    mg.phiOld = tex3d(device, n, 'r32float', 'phi old');

    mg.multipole = await Reducer.create(
      device,
      [n, n, n],
      new Array(10).fill(SUM),
      'gravity/multipole.wgsl',
      {},
      stateU1.map((t) => ({ 0: params, 1: t.createView() })),
      'multipole',
    );

    [mg.pSmooth, mg.pResidual, mg.pRestrict, mg.pProlong, mg.pRhs] = await Promise.all([
      computePipeline(device, assemble('gravity/smooth.wgsl'), 'mg smooth'),
      computePipeline(device, assemble('gravity/residual.wgsl'), 'mg residual'),
      computePipeline(device, assemble('gravity/restrict.wgsl'), 'mg restrict'),
      computePipeline(device, assemble('gravity/prolong.wgsl'), 'mg prolong'),
      computePipeline(device, assemble('gravity/rhs.wgsl'), 'mg rhs'),
    ]);

    const mp = mg.multipole.result;
    for (let l = 0, m = n; m >= 2; l++, m >>= 1) {
      const h = box / m;
      const uni = [0, 1].map((color) => {
        const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const data = new ArrayBuffer(32);
        new Uint32Array(data, 0, 4).set([m, color, l, 0]);
        new Float32Array(data, 16, 4).set([h, 0, 0, 0]);
        device.queue.writeBuffer(buf, 0, data);
        return buf;
      });
      const phi = tex3d(device, m, 'r32float', `phi L${l}`);
      const rhs = tex3d(device, m, 'r32float', `rhs L${l}`);
      const res = tex3d(device, m, 'r32float', `res L${l}`);
      const lev: Level = {
        n: m,
        phi,
        rhs,
        res,
        smooth: uni.map((u) =>
          bindGroup(device, mg.pSmooth, { 0: params, 1: u, 2: phi.createView(), 3: rhs.createView(), 10: mp }),
        ),
        residual: bindGroup(device, mg.pResidual, {
          0: params,
          1: uni[0],
          2: phi.createView(),
          3: rhs.createView(),
          4: res.createView(),
          10: mp,
        }),
      };
      if (l > 0) {
        const fine = mg.levels[l - 1];
        lev.restrict = bindGroup(device, mg.pRestrict, {
          1: uni[0],
          2: fine.res.createView(),
          3: rhs.createView(),
          4: phi.createView(),
        });
        // Prolongation into the finer level uses that level's parameters; find them again.
        const fineUni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const d = new ArrayBuffer(32);
        new Uint32Array(d, 0, 4).set([fine.n, 0, l - 1, 0]);
        new Float32Array(d, 16, 4).set([box / fine.n, 0, 0, 0]);
        device.queue.writeBuffer(fineUni, 0, d);
        fine.prolong = bindGroup(device, mg.pProlong, {
          0: params,
          1: fineUni,
          2: phi.createView(),
          3: fine.phi.createView(),
        });
      }
      mg.levels.push(lev);
    }

    mg.rhsGroups = stateU1.map((t) =>
      bindGroup(device, mg.pRhs, { 0: params, 1: t.createView(), 2: mg.levels[0].rhs.createView() }),
    );
    return mg;
  }

  private dispatch(pass: GPUComputePassEncoder, n: number) {
    pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), Math.ceil(n / 4));
  }

  private smooth(pass: GPUComputePassEncoder, lev: Level, sweeps: number) {
    pass.setPipeline(this.pSmooth);
    for (let i = 0; i < sweeps; i++) {
      for (const g of lev.smooth) {
        pass.setBindGroup(0, g);
        // Each colour is half the cells; the shader spreads the dispatch over every other x.
        pass.dispatchWorkgroups(Math.ceil(lev.n / 16), Math.ceil(lev.n / 8), Math.ceil(lev.n / 4));
      }
    }
  }

  /** Build the source from the state of the given parity (and the multipoles if isolated). */
  encodeSource(pass: GPUComputePassEncoder, parity: number, isolated: boolean) {
    pass.setPipeline(this.pRhs);
    pass.setBindGroup(0, this.rhsGroups[parity]);
    this.dispatch(pass, this.levels[0].n);
    if (isolated) this.multipole.encode(pass, parity);
  }

  encodeVCycle(pass: GPUComputePassEncoder) {
    const L = this.levels;
    const last = L.length - 1;
    for (let l = 0; l < last; l++) {
      this.smooth(pass, L[l], PRE);
      pass.setPipeline(this.pResidual);
      pass.setBindGroup(0, L[l].residual);
      this.dispatch(pass, L[l].n);
      pass.setPipeline(this.pRestrict);
      pass.setBindGroup(0, L[l + 1].restrict!);
      this.dispatch(pass, L[l + 1].n);
    }
    this.smooth(pass, L[last], COARSE);
    for (let l = last - 1; l >= 0; l--) {
      pass.setPipeline(this.pProlong);
      pass.setBindGroup(0, L[l].prolong!);
      this.dispatch(pass, L[l].n);
      this.smooth(pass, L[l], POST);
    }
  }

  /** Residual of the current potential on the finest level, left in levels[0].res. */
  encodeResidual(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pResidual);
    pass.setBindGroup(0, this.levels[0].residual);
    this.dispatch(pass, this.levels[0].n);
  }

  destroy() {
    for (const l of this.levels) {
      l.phi.destroy();
      l.rhs.destroy();
      l.res.destroy();
    }
    this.phiOld.destroy();
  }
}
