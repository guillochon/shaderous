// Geometric multigrid for the pressure equation of the implicit step (plans/incompressible.md
// 4.6):  alpha p - div(beta grad p) = b,  with the coefficients a texture per level
// (pressure/common.wgsl). V-cycles with red-black Gauss-Seidel smoothing, restriction by the
// average of the children and trilinear prolongation, in 3D or, with one-layer levels, 2D.
// Beside gravity/multigrid.ts, from which it differs by its coefficients (walls and obstacles
// are faces with beta = 0, carried to every level by coarsening), its 2D form, and its coarsest
// level: a few cells, solved in one invocation, with the mean fixed where no face of the box is
// open.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d } from '../gpu/device';

const PRE = 2; // smoothing sweeps before restriction
const POST = 2; // and after prolongation
/** The coarsest level has at most this many cells: one invocation solves it (pressure/coarse_solve.wgsl). */
const COARSEST_CELLS = 16;

/** How the solve sees a face of the box (pressure/common.wgsl). */
export const PF_PERIODIC = 0;
export const PF_CLOSED = 1;
export const PF_OPEN = 2;

interface Level {
  dims: [number, number, number];
  h: number;
  p: GPUTexture;
  rhs: GPUTexture;
  res: GPUTexture;
  coef: GPUTexture;
  smooth: GPUBindGroup[]; // [red, black]
  residual: GPUBindGroup;
  restrict?: GPUBindGroup; // from the next finer level into this one
  coarsen?: GPUBindGroup; // this level's coefficients from the next finer level's
  prolong?: GPUBindGroup; // from the next coarser level into this one
}

export class PressureSolver {
  levels: Level[] = [];
  /** the density at the start of the step, kept for the correction (pressure/source.wgsl) */
  rho0!: GPUTexture;
  private device!: GPUDevice;
  private wz = 4;
  private pSmooth!: GPUComputePipeline;
  private pResidual!: GPUComputePipeline;
  private pRestrict!: GPUComputePipeline;
  private pCoarsen!: GPUComputePipeline;
  private pProlong!: GPUComputePipeline;
  private pCoarsest!: GPUComputePipeline;
  private gCoarsest!: GPUBindGroup;
  private uniforms: GPUBuffer[] = [];
  // Where each solve ended (pressure/measure.wgsl, pressure/tally.wgsl).
  private pMeasure!: GPUComputePipeline;
  private gMeasure!: GPUBindGroup;
  private pTally!: GPUComputePipeline;
  private gTally!: GPUBindGroup;
  private res!: GPUBuffer;
  private tallies!: GPUBuffer;

  /** The solved pressure on the grid. */
  get p(): GPUTexture {
    return this.levels[0].p;
  }

  /**
   * @param dims cells along each axis, each a power of two (1 along z in 2D)
   * @param h cell size
   * @param lo, hi the kind of each face of the box (PF_*)
   */
  static async create(device: GPUDevice, dims: readonly [number, number, number], h: number, lo: readonly number[], hi: readonly number[]): Promise<PressureSolver> {
    if (dims.some((m) => m & (m - 1))) throw new Error('the pressure multigrid needs a power-of-two grid');
    const s = new PressureSolver();
    s.device = device;
    const flat = dims[2] === 1;
    s.wz = flat ? 1 : 4;
    const k = { PWZ: s.wz };
    [s.pSmooth, s.pResidual, s.pRestrict, s.pCoarsen, s.pProlong, s.pCoarsest, s.pMeasure, s.pTally] = await Promise.all([
      computePipeline(device, assemble('pressure/smooth.wgsl'), 'pressure smooth', k),
      computePipeline(device, assemble('pressure/residual.wgsl'), 'pressure residual', k),
      computePipeline(device, assemble('pressure/restrict.wgsl'), 'pressure restrict', k),
      computePipeline(device, assemble('pressure/coarsen.wgsl'), 'pressure coarsen', k),
      computePipeline(device, assemble('pressure/prolong.wgsl'), 'pressure prolong', k),
      computePipeline(device, assemble('pressure/coarse_solve.wgsl'), 'pressure coarsest level'),
      computePipeline(device, assemble('pressure/measure.wgsl'), 'pressure residual measure', k),
      computePipeline(device, assemble('pressure/tally.wgsl'), 'pressure solve tally'),
    ]);
    const fixMean = [...lo, ...hi].every((f, i) => f !== PF_OPEN || (flat && i % 3 === 2)) ? 1 : 0;
    const uniform = (d: readonly number[], color: number, level: number, hl: number) => {
      const buf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const data = new ArrayBuffer(64);
      const u = new Uint32Array(data);
      u.set([d[0], d[1], d[2], color, lo[0], lo[1], lo[2], level, hi[0], hi[1], hi[2], fixMean]);
      new Float32Array(data)[12] = hl;
      device.queue.writeBuffer(buf, 0, data);
      s.uniforms.push(buf);
      return buf;
    };
    s.rho0 = tex3d(device, dims[0], 'r32float', 'pressure rho0', dims[2], dims[1]);
    const shortest = (d: readonly number[]) => Math.min(d[0], d[1], flat ? Infinity : d[2]);
    const cells = (d: readonly number[]) => d[0] * d[1] * d[2];
    const v = (t: GPUTexture) => t.createView();
    // Levels halve until one has at most COARSEST_CELLS cells (or two along its shortest axis).
    for (let l = 0, d: [number, number, number] = [dims[0], dims[1], dims[2]], hl = h; l === 0 || (shortest(d) >= 2 && cells(s.levels[l - 1].dims) > COARSEST_CELLS); l++, d = [d[0] >> 1, d[1] >> 1, flat ? 1 : d[2] >> 1], hl *= 2) {
      const uni = [0, 1].map((color) => uniform(d, color, l, hl));
      const mk = (name: string, format: GPUTextureFormat = 'r32float') => tex3d(device, d[0], format, `pressure ${name} L${l}`, d[2], d[1]);
      const lev: Level = { dims: [...d], h: hl, p: mk('p'), rhs: mk('rhs'), res: mk('res'), coef: mk('coef', 'rgba32float'), smooth: [], residual: null as unknown as GPUBindGroup };
      lev.smooth = uni.map((u) => bindGroup(device, s.pSmooth, { 0: u, 1: v(lev.p), 2: v(lev.rhs), 3: v(lev.coef) }));
      lev.residual = bindGroup(device, s.pResidual, { 0: uni[0], 1: v(lev.p), 2: v(lev.rhs), 3: v(lev.coef), 4: v(lev.res) });
      if (l > 0) {
        const fine = s.levels[l - 1];
        lev.restrict = bindGroup(device, s.pRestrict, { 0: uni[0], 1: v(fine.res), 2: v(lev.rhs), 3: v(lev.p) });
        lev.coarsen = bindGroup(device, s.pCoarsen, { 0: uni[0], 1: v(fine.coef), 2: v(lev.coef) });
        // Prolongation into the finer level takes that level's parameters.
        fine.prolong = bindGroup(device, s.pProlong, { 0: uniform(fine.dims, 0, l - 1, fine.h), 1: v(lev.p), 2: v(fine.p), 3: v(fine.coef) });
      }
      s.levels.push(lev);
      s.gCoarsest = bindGroup(device, s.pCoarsest, { 0: uni[0], 1: v(lev.p), 2: v(lev.rhs), 3: v(lev.coef) });
    }
    const buffer = (n: number, usage: number, label: string) => device.createBuffer({ label, size: 4 * n, usage });
    s.res = buffer(2, GPUBufferUsage.STORAGE, 'pressure residual');
    s.tallies = buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'pressure solver tallies');
    s.setTolerance(1e-4);
    const l0 = s.levels[0];
    s.gMeasure = bindGroup(device, s.pMeasure, { 0: s.uniforms[0], 1: v(l0.p), 2: v(l0.rhs), 3: v(l0.coef), 4: s.res });
    s.gTally = bindGroup(device, s.pTally, { 0: s.res, 1: s.tallies });
    return s;
  }

  private dispatch(pass: GPUComputePassEncoder, d: readonly number[], halfX = false) {
    pass.dispatchWorkgroups(Math.ceil(d[0] / (halfX ? 16 : 8)), Math.ceil(d[1] / 8), Math.ceil(d[2] / this.wz));
  }

  private smooth(pass: GPUComputePassEncoder, lev: Level, sweeps: number) {
    pass.setPipeline(this.pSmooth);
    for (let i = 0; i < sweeps; i++)
      for (const g of lev.smooth) {
        pass.setBindGroup(0, g);
        this.dispatch(pass, lev.dims, true);
      }
  }

  /** The relative residual above which a solve counts as unconverged (stats). */
  setTolerance(tol: number) {
    this.device.queue.writeBuffer(this.tallies, 0, new Float32Array([tol]));
  }

  /** The coarse levels' coefficients from the finest level's, which the source pass has just written. */
  encodeCoarsen(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pCoarsen);
    for (const lev of this.levels.slice(1)) {
      pass.setBindGroup(0, lev.coarsen!);
      this.dispatch(pass, lev.dims);
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
    pass.setPipeline(this.pCoarsest);
    pass.setBindGroup(0, this.gCoarsest);
    pass.dispatchWorkgroups(1);
    for (let l = last - 1; l >= 0; l--) {
      pass.setPipeline(this.pProlong);
      pass.setBindGroup(0, L[l].prolong!);
      this.dispatch(pass, L[l].dims);
      this.smooth(pass, L[l], POST);
    }
  }

  /** A solve: `cycles` V-cycles, then a measurement of where it ended (pressure/tally.wgsl). */
  encodeSolve(pass: GPUComputePassEncoder, cycles: number) {
    for (let c = 0; c < cycles; c++) this.encodeVCycle(pass);
    pass.setPipeline(this.pMeasure);
    pass.setBindGroup(0, this.gMeasure);
    this.dispatch(pass, this.levels[0].dims);
    pass.setPipeline(this.pTally);
    pass.setBindGroup(0, this.gTally);
    pass.dispatchWorkgroups(1);
  }

  /** Since the solver was made: solves, the last one's relative residual, and how many ended above the tolerance. */
  async stats(): Promise<{ solves: number; residual: number; unconverged: number }> {
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.tallies, 0, buf, 0, 16);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(buf.getMappedRange().slice(0));
    buf.destroy();
    return { solves: u[2], residual: new Float32Array(u.buffer)[1], unconverged: u[3] };
  }

  /** Residual of the current pressure on the finest level, left in levels[0].res. */
  encodeResidual(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pResidual);
    pass.setBindGroup(0, this.levels[0].residual);
    this.dispatch(pass, this.levels[0].dims);
  }

  /** One of the finest level's textures, read back (tests and inspection: slow). */
  async read(which: 'p' | 'rhs' | 'res'): Promise<Float32Array> {
    const l = this.levels[0];
    return readTexture(this.device, l[which], l.dims[0], 1, l.dims[2], l.dims[1]);
  }

  /** Write the finest level's right-hand side, coefficients (four per cell) and starting pressure (tests). */
  write(rhs: Float32Array, coef: Float32Array, p?: Float32Array) {
    const l = this.levels[0];
    const [nx, ny, nz] = l.dims;
    const put = (t: GPUTexture, data: Float32Array, ch: number) =>
      this.device.queue.writeTexture({ texture: t }, data.buffer, { offset: data.byteOffset, bytesPerRow: nx * 4 * ch, rowsPerImage: ny }, [nx, ny, nz]);
    put(l.rhs, rhs, 1);
    put(l.coef, coef, 4);
    put(l.p, p ?? new Float32Array(nx * ny * nz), 1);
  }

  destroy() {
    for (const l of this.levels) for (const t of [l.p, l.rhs, l.res, l.coef]) t.destroy();
    this.rho0.destroy();
    for (const b of [...this.uniforms, this.res, this.tallies]) b.destroy();
  }
}
