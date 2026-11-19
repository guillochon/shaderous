// Geometric multigrid Poisson solver: V-cycles with red-black Gauss-Seidel smoothing,
// 8-cell-average restriction and trilinear prolongation. One r32float texture per level for
// each of phi, the right-hand side and the residual.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d } from '../gpu/device';
import { Multipoles } from './multipole';

const PRE = 2; // smoothing sweeps before restriction
const POST = 2; // and after prolongation
const COARSE = 8; // sweeps on the coarsest (2^3) level

interface Level {
  /** cells along the longest axis */
  n: number;
  /** cells along each axis (B6: half along a mirrored axis of a rectangular grid) */
  dims: [number, number, number];
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
  multipole!: Multipoles;
  private pSmooth!: GPUComputePipeline;
  private pResidual!: GPUComputePipeline;
  private pRestrict!: GPUComputePipeline;
  private pProlong!: GPUComputePipeline;
  private pRhs!: GPUComputePipeline;
  private rhsGroups: GPUBindGroup[] = [];
  // Isolated boundaries: the finest level's face values, once per solve (gravity/boundary.wgsl).
  private pBoundary!: GPUComputePipeline;
  private gBoundary!: GPUBindGroup;
  private boundary!: GPUBuffer;

  get phi(): GPUTexture {
    return this.levels[0].phi;
  }

  /** @param stateU1 the density-carrying state texture for each parity */
  static async create(
    device: GPUDevice,
    size: number | readonly [number, number, number],
    box: number,
    params: GPUBuffer,
    stateU1: GPUTexture[],
    symmetry: readonly boolean[] | undefined,
    clock: GPUBuffer,
    flat = false,
  ): Promise<Multigrid> {
    // A cube of n, or (docs/numerics.md section 22) a box of three counts, each a power
    // of two; the longest is n, and h = box / n.
    const dims0: [number, number, number] = typeof size === 'number' ? [size, size, size] : [size[0], size[1], size[2]];
    // An axisymmetric (R, z) run's levels are one cell deep (gravity/stencil.wgsl).
    if (flat) dims0[2] = 1;
    const n = Math.max(...dims0);
    if (dims0.some((m) => m & (m - 1))) throw new Error('multigrid needs a power-of-two grid');
    const mg = new Multigrid();
    mg.flat = flat;
    mg.phiOld = tex3d(device, dims0[0], 'r32float', 'phi old', dims0[2], dims0[1]);

    mg.multipole = await Multipoles.create(
      device,
      dims0,
      'gravity/multipole.wgsl',
      stateU1.map((t) => ({ 0: params, 1: t.createView() })),
      'multipole',
      symmetry,
    );

    [mg.pSmooth, mg.pResidual, mg.pRestrict, mg.pProlong, mg.pRhs, mg.pBoundary] = await Promise.all([
      computePipeline(device, assemble('gravity/smooth.wgsl'), 'mg smooth'),
      computePipeline(device, assemble('gravity/residual.wgsl'), 'mg residual'),
      computePipeline(device, assemble('gravity/restrict.wgsl'), 'mg restrict'),
      computePipeline(device, assemble('gravity/prolong.wgsl'), 'mg prolong'),
      computePipeline(device, assemble('gravity/rhs.wgsl'), 'mg rhs'),
      computePipeline(device, assemble('gravity/boundary.wgsl'), 'mg isolated boundary'),
    ]);

    const mp = mg.multipole.result;
    // Six faces of n x n (gravity/boundary.wgsl), bound to every level's passes; only the
    // finest level's read it.
    const bf = (mg.boundary = device.createBuffer({ label: 'mg isolated boundary', size: 6 * n * n * 4, usage: GPUBufferUsage.STORAGE }));
    // Levels halve every axis until the shortest (in the plane, for an axisymmetric run) has two
    // cells: the coarsest of a rectangular grid is 2 x 2 x 4 or the like.
    const shortest = (d: number[]) => Math.min(d[0], d[1], flat ? Infinity : d[2]);
    const uniform = (d: readonly number[], m: number, color: number, l: number, h: number) => {
      const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const data = new ArrayBuffer(32);
      new Uint32Array(data, 0, 4).set([m, color, l, flat ? 1 : 0]);
      new Float32Array(data, 16, 1).set([h]);
      new Uint32Array(data, 20, 3).set([d[0], d[1], d[2]]);
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    for (let l = 0, m = n, dl: [number, number, number] = [...dims0]; shortest(dl) >= 2; l++, m >>= 1, dl = [dl[0] >> 1, dl[1] >> 1, flat ? 1 : dl[2] >> 1]) {
      const h = box / m;
      const uni = [0, 1].map((color) => uniform(dl, m, color, l, h));
      const phi = tex3d(device, dl[0], 'r32float', `phi L${l}`, dl[2], dl[1]);
      const rhs = tex3d(device, dl[0], 'r32float', `rhs L${l}`, dl[2], dl[1]);
      const res = tex3d(device, dl[0], 'r32float', `res L${l}`, dl[2], dl[1]);
      const lev: Level = {
        n: m,
        dims: [...dl],
        phi,
        rhs,
        res,
        smooth: uni.map((u) =>
          bindGroup(device, mg.pSmooth, { 0: params, 1: u, 2: phi.createView(), 3: rhs.createView(), 11: bf }),
        ),
        residual: bindGroup(device, mg.pResidual, {
          0: params,
          1: uni[0],
          2: phi.createView(),
          3: rhs.createView(),
          4: res.createView(),
          11: bf,
        }),
      };
      if (l === 0) mg.gBoundary = bindGroup(device, mg.pBoundary, { 0: params, 1: uni[0], 10: mp, 11: bf });
      if (l > 0) {
        const fine = mg.levels[l - 1];
        lev.restrict = bindGroup(device, mg.pRestrict, {
          0: params,
          1: uni[0],
          2: fine.res.createView(),
          3: rhs.createView(),
          4: phi.createView(),
        });
        // Prolongation into the finer level uses that level's parameters; find them again.
        const fineUni = uniform(fine.dims, fine.n, 0, l - 1, box / fine.n);
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
      bindGroup(device, mg.pRhs, { 0: params, 1: t.createView(), 2: mg.levels[0].rhs.createView(), 3: clock }),
    );
    return mg;
  }

  /** The levels are one cell deep (an axisymmetric run). */
  flat = false;

  private dispatch(pass: GPUComputePassEncoder, d: readonly number[]) {
    pass.dispatchWorkgroups(Math.ceil(d[0] / 8), Math.ceil(d[1] / 8), Math.ceil(d[2] / 4));
  }

  private smooth(pass: GPUComputePassEncoder, lev: Level, sweeps: number) {
    pass.setPipeline(this.pSmooth);
    for (let i = 0; i < sweeps; i++) {
      for (const g of lev.smooth) {
        pass.setBindGroup(0, g);
        // Each color is half the cells; the shader spreads the dispatch over every other x.
        pass.dispatchWorkgroups(Math.ceil(lev.dims[0] / 16), Math.ceil(lev.dims[1] / 8), Math.ceil(lev.dims[2] / 4));
      }
    }
  }

  /** Build the source from the state of the given parity (and the multipoles if isolated). */
  encodeSource(pass: GPUComputePassEncoder, parity: number, isolated: boolean) {
    pass.setPipeline(this.pRhs);
    pass.setBindGroup(0, this.rhsGroups[parity]);
    this.dispatch(pass, this.levels[0].dims);
    if (isolated) {
      this.multipole.encode(pass, parity);
      // The boundary of the finest level, from the moments just found, for the whole solve.
      const n = this.levels[0].n;
      pass.setPipeline(this.pBoundary);
      pass.setBindGroup(0, this.gBoundary);
      pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), 6);
    }
  }

  encodeVCycle(pass: GPUComputePassEncoder) {
    const L = this.levels;
    const last = L.length - 1;
    for (let l = 0; l < last; l++) {
      this.smooth(pass, L[l], PRE);
      pass.setPipeline(this.pResidual);
      pass.setBindGroup(0, L[l].residual);
      this.dispatch(pass, L[l].dims);
      pass.setPipeline(this.pRestrict);
      pass.setBindGroup(0, L[l + 1].restrict!);
      this.dispatch(pass, L[l + 1].dims);
    }
    this.smooth(pass, L[last], COARSE);
    for (let l = last - 1; l >= 0; l--) {
      pass.setPipeline(this.pProlong);
      pass.setBindGroup(0, L[l].prolong!);
      this.dispatch(pass, L[l].dims);
      this.smooth(pass, L[l], POST);
    }
  }

  /** Residual of the current potential on the finest level, left in levels[0].res. */
  encodeResidual(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pResidual);
    pass.setBindGroup(0, this.levels[0].residual);
    this.dispatch(pass, this.levels[0].dims);
  }

  destroy() {
    for (const l of this.levels) {
      l.phi.destroy();
      l.rhs.destroy();
      l.res.destroy();
    }
    this.phiOld.destroy();
    this.boundary.destroy();
  }
}
