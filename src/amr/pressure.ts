// Implicit pressure on the block grid (plans/incompressible.md, stage 3): the implicit step of
// src/pressure, the liquid's and the ideal gas's, on the leaves of the tree. See amr/pressure/common.wgsl for the scheme:
// multigrid over the tree's levels in full-approximation form, with the fine fluxes matched at
// refinement faces, and the root level solved by the uniform grid's multigrid.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d, type Bindable } from '../gpu/device';
import { PressureSolver } from '../pressure/multigrid';

const PRE = 2; // red-black sweeps of a level before restricting it
const POST = 2; // and after prolonging into it
const ROOT_CYCLES = 2; // V-cycles of the uniform multigrid on the root level

export interface AmrPressureContext {
  device: GPUDevice;
  params: GPUBuffer;
  clock: GPUBuffer;
  bt: GPUBuffer;
  listLeaf: GPUBuffer;
  /** one list per tree level: the blocks at that level, parents and leaves */
  listLevel: GPUBuffer[];
  /** deepest tree level */
  levels: number;
  /** the state atlas, per parity: [rho, momentum] and [energies, scalar, mark] */
  state: GPUTexture[][];
  /** the primitives atlas (amr/assemble.wgsl): [rho, velocity], and [p, c_s, scalar, e_int] */
  prim: GPUTexture;
  prim2: GPUTexture;
  /**
   * A gas with an energy (amr/pressure/*_gas.wgsl): the atlases of the face velocities the
   * update keeps (lower and upper faces), and of the carried solved pressure, which the block
   * predictor also reads. Absent for the liquid.
   */
  gas?: { faceLo: GPUTexture; faceHi: GPUTexture; lag: GPUTexture };
  /** dispatch over the cells and `reach` ghost layers of `count` blocks of the bound list (amr/list.wgsl inv_flat) */
  flat(pass: GPUComputePassEncoder, count: number, reach: number): void;
  /** the modules the hydro shaders are assembled with (the equation of state) */
  slots: Record<string, string>;
  /** the root level as a uniform grid: its cells along each axis (1 along z in 2D), its cell size, and the kinds of the box's faces (pressure/multigrid.ts PF_*) */
  rootDims: [number, number, number];
  dx0: number;
  lo: number[];
  hi: number[];
  /** dispatch over the interior cells of `count` blocks of the bound list (amr/list.wgsl inv_interior) */
  interior(pass: GPUComputePassEncoder, count: number): void;
}

export class AmrPressure {
  private ctx!: AmrPressureContext;
  /** the root level's solver */
  root!: PressureSolver;
  private tex: GPUTexture[] = [];
  private pSmooth: GPUComputePipeline[] = [];
  private pRestrict: GPUComputePipeline[] = [];
  private pProlong!: GPUComputePipeline;
  private pRoot: GPUComputePipeline[] = [];
  private pMeasure!: GPUComputePipeline;
  private pTally!: GPUComputePipeline;
  private pSource!: GPUComputePipeline;
  private pCorrect!: GPUComputePipeline;
  private pKeep?: GPUComputePipeline;
  private gKeep?: GPUBindGroup;
  private pLagInit?: GPUComputePipeline;
  private gLagInit: GPUBindGroup[] = [];
  private pLagFill?: GPUComputePipeline;
  private gLagFill: GPUBindGroup[] = []; // [p]
  private pCoef?: GPUComputePipeline;
  private gCoef: GPUBindGroup[] = []; // [level]
  // Per parity p of the step (the passes read the cells' marks from the other one's state).
  private gSmooth: GPUBindGroup[][][] = []; // [p][level][color]
  private gRestrict: GPUBindGroup[][][] = []; // [p][level][stage]
  private gProlong: GPUBindGroup[][] = []; // [p][level]
  private gRoot: GPUBindGroup[][] = []; // [p][to, from]
  private gMeasure: GPUBindGroup[] = [];
  private gTally!: GPUBindGroup;
  private gSource: GPUBindGroup[] = [];
  private gCorrect: GPUBindGroup[] = [];
  private res!: GPUBuffer;
  private tallies!: GPUBuffer;

  static async create(ctx: AmrPressureContext): Promise<AmrPressure> {
    const s = new AmrPressure();
    s.ctx = ctx;
    const d = ctx.device;
    const v0 = (x: GPUTexture) => x.createView();
    const atlas = ctx.state[0][0];
    const mk = (name: string) => tex3d(d, atlas.width, 'r32float', `amr pressure ${name}`, atlas.depthOrArrayLayers, atlas.height);
    const [prs, rhs, tau, auxr, auxp, rho0] = (s.tex = ['p', 'rhs', 'tau', 'restricted residual', 'restricted p', 'rho0'].map(mk));
    const gas = ctx.gas;
    const slots = { ...ctx.slots, ap_coef: gas ? 'amr/pressure/coef_gas.wgsl' : 'amr/pressure/coef_liquid.wgsl' };
    const pipe = (file: string, label: string, k?: Record<string, number>) => computePipeline(d, assemble(`amr/pressure/${file}.wgsl`, slots), `amr pressure ${label}`, k);
    [s.root, s.pSmooth[0], s.pSmooth[1], s.pRestrict[0], s.pRestrict[1], s.pProlong, s.pRoot[0], s.pRoot[1], s.pMeasure, s.pTally, s.pSource, s.pCorrect] = await Promise.all([
      PressureSolver.create(d, ctx.rootDims, ctx.dx0, ctx.lo, ctx.hi),
      pipe('smooth', 'smooth red', { COLOR: 0 }),
      pipe('smooth', 'smooth black', { COLOR: 1 }),
      pipe('restrict', 'restrict', { STAGE: 0 }),
      pipe('restrict', 'coarse right-hand side', { STAGE: 1 }),
      pipe('prolong', 'prolong'),
      pipe('root', 'to the root solver', { TO_ROOT: 1 }),
      pipe('root', 'from the root solver', { TO_ROOT: 0 }),
      pipe('measure', 'residual measure'),
      computePipeline(d, assemble('pressure/tally.wgsl'), 'amr pressure solve tally'),
      pipe(gas ? 'source_gas' : 'source', 'source'),
      pipe(gas ? 'correct_gas' : 'correct', 'correct'),
    ]);
    // A gas: its pressure's advected change, and its new pressure on the way to the carried one.
    // And its coefficients on the parents (amr/pressure/coef_restrict.wgsl).
    const [pd, pn, kr, ka] = gas ? ['advected p', 'new p', 'parent rho', 'parent alpha'].map(mk) : [];
    if (gas) {
      s.tex.push(pd, pn, kr, ka);
      s.pCoef = await pipe('coef_restrict', 'parent coefficients');
      s.gCoef = ctx.listLevel.map((list) => bindGroup(d, s.pCoef!, { 0: ctx.params, 30: ctx.bt, 31: list, 15: v0(ctx.prim), 22: v0(ctx.prim2), 7: v0(kr), 8: v0(ka) }));
      [s.pKeep, s.pLagInit, s.pLagFill] = await Promise.all([pipe('keep', 'advect the old pressure'), pipe('lag_init', 'start the carried pressure'), pipe('lag_fill', 'carry the pressure')]);
      s.gKeep = bindGroup(d, s.pKeep, { 0: ctx.params, 1: ctx.clock, 30: ctx.bt, 31: ctx.listLeaf, 15: v0(ctx.prim), 22: v0(ctx.prim2), 24: v0(pd) });
      s.gLagInit = [0, 1].map((p) => bindGroup(d, s.pLagInit!, { 0: ctx.params, 30: ctx.bt, 31: ctx.listLeaf, 14: v0(ctx.state[p][0]), 16: v0(ctx.state[p][1]), 29: v0(pn), 2: v0(s.tex[0]) }));
      s.gLagFill = [0, 1].map((p) => bindGroup(d, s.pLagFill!, { 0: ctx.params, 30: ctx.bt, 31: ctx.listLeaf, 25: v0(gas.lag), 29: v0(pn), 14: v0(ctx.state[p][0]) }));
    }
    const v = (t: GPUTexture) => t.createView();
    const r0 = s.root.levels[0];
    s.res = d.createBuffer({ label: 'amr pressure residual', size: 8, usage: GPUBufferUsage.STORAGE });
    s.tallies = d.createBuffer({ label: 'amr pressure solver tallies', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    s.setTolerance(1e-4);
    s.gTally = bindGroup(d, s.pTally, { 0: s.res, 1: s.tallies });
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // The marks: of the advected state's parity, which no pass of the step writes.
      // (A gas's coefficients are the advected state's primitives: amr/pressure/coef_gas.wgsl.)
      const prims: Record<number, Bindable> = gas ? { 15: v(ctx.prim), 22: v(ctx.prim2) } : {};
      const coef: Record<number, Bindable> = gas ? { ...prims, 7: v(kr), 8: v(ka) } : {};
      const bare = (list: GPUBuffer): Record<number, Bindable> => ({ 0: ctx.params, 1: ctx.clock, 21: v(ctx.state[o][1]), 30: ctx.bt, 31: list });
      const base = (list: GPUBuffer): Record<number, Bindable> => ({ ...bare(list), ...coef });
      s.gSmooth[p] = ctx.listLevel.map((list) => s.pSmooth.map((pl) => bindGroup(d, pl, { ...base(list), 2: v(prs), 3: v(rhs), 4: v(tau) })));
      s.gRestrict[p] = ctx.listLevel.map((list) => s.pRestrict.map((pl) => bindGroup(d, pl, { ...base(list), 2: v(prs), 3: v(rhs), 4: v(tau), 5: v(auxr), 6: v(auxp) })));
      // (Prolongation asks about no neighbor, and so binds no marks.)
      s.gProlong[p] = ctx.listLevel.map((list) => bindGroup(d, s.pProlong, { 0: ctx.params, 1: ctx.clock, 30: ctx.bt, 31: list, 2: v(prs), 6: v(auxp) }));
      s.gRoot[p] = s.pRoot.map((pl) => bindGroup(d, pl, { ...base(ctx.listLevel[0]), 2: v(prs), 3: v(rhs), 4: v(tau), 10: v(r0.p), 11: v(r0.rhs), 12: v(r0.coef) }));
      s.gMeasure[p] = bindGroup(d, s.pMeasure, { ...base(ctx.listLeaf), 2: v(prs), 3: v(rhs), 9: s.res });
      if (gas) {
        const faces = { 25: v(gas.lag), 27: v(gas.faceLo), 28: v(gas.faceHi) };
        s.gSource.push(bindGroup(d, s.pSource, { ...base(ctx.listLeaf), 3: v(rhs), 4: v(tau), 26: v(pd), 14: v(ctx.state[p][0]), 16: v(ctx.state[p][1]), ...faces }));
        s.gCorrect.push(bindGroup(d, s.pCorrect, { ...bare(ctx.listLeaf), ...prims, 2: v(prs), 16: v(ctx.state[o][0]), 17: v(ctx.state[o][1]), 18: v(ctx.state[p][0]), 19: v(ctx.state[p][1]), 29: v(pn), ...faces }));
        continue;
      }
      s.gSource.push(bindGroup(d, s.pSource, { ...base(ctx.listLeaf), 2: v(prs), 3: v(rhs), 4: v(tau), 13: v(rho0), 14: v(ctx.state[p][0]), 15: v(ctx.prim), 16: v(ctx.state[p][1]) }));
      s.gCorrect.push(
        bindGroup(d, s.pCorrect, {
          ...base(ctx.listLeaf), 2: v(prs), 13: v(rho0), 15: v(ctx.prim), 16: v(ctx.state[o][0]), 17: v(ctx.state[o][1]), 18: v(ctx.state[p][0]), 19: v(ctx.state[p][1]),
        }),
      );
    }
    return s;
  }

  /** The relative residual above which a solve counts as unconverged (stats). */
  setTolerance(tol: number) {
    this.ctx.device.queue.writeBuffer(this.tallies, 0, new Float32Array([tol]));
  }

  private run(pass: GPUComputePassEncoder, pipe: GPUComputePipeline, group: GPUBindGroup, count: number) {
    if (count <= 0) return;
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    this.ctx.interior(pass, count);
  }

  private smooth(pass: GPUComputePassEncoder, p: number, l: number, count: number, sweeps: number) {
    for (let i = 0; i < sweeps; i++) for (let color = 0; color < 2; color++) this.run(pass, this.pSmooth[color], this.gSmooth[p][l][color], count);
  }

  /**
   * One cycle over the tree's levels from level `l` down to the root and back. `nLevel` is the
   * number of blocks at each level.
   */
  private encodeCycle(pass: GPUComputePassEncoder, p: number, nLevel: readonly number[], l = this.ctx.levels) {
    if (l === 0) {
      this.run(pass, this.pRoot[0], this.gRoot[p][0], nLevel[0]);
      this.root.encodeCoarsen(pass);
      for (let c = 0; c < ROOT_CYCLES; c++) this.root.encodeVCycle(pass);
      this.run(pass, this.pRoot[1], this.gRoot[p][1], nLevel[0]);
      return;
    }
    this.smooth(pass, p, l, nLevel[l], PRE);
    for (let stage = 0; stage < 2; stage++) this.run(pass, this.pRestrict[stage], this.gRestrict[p][l - 1][stage], nLevel[l - 1]);
    this.encodeCycle(pass, p, nLevel, l - 1);
    this.run(pass, this.pProlong, this.gProlong[p][l], nLevel[l]);
    this.smooth(pass, p, l, nLevel[l], POST);
  }

  /**
   * A gas with an energy: before the caller assembles the advected state's primitives over
   * those of the state the step started from, keep that state's pressure (amr/pressure/keep.wgsl).
   */
  encodeKeep(pass: GPUComputePassEncoder, nLeaf: number) {
    if (!this.pKeep || nLeaf <= 0) return;
    pass.setPipeline(this.pKeep);
    pass.setBindGroup(0, this.gKeep!);
    this.ctx.flat(pass, nLeaf, 0);
  }

  /** A gas with an energy: the carried pressure and its ghosts, from the leaves' new pressure (of the state of parity `p`). */
  private encodeFill(pass: GPUComputePassEncoder, nLeaf: number, p: number) {
    if (!this.pLagFill || nLeaf <= 0) return;
    pass.setPipeline(this.pLagFill);
    pass.setBindGroup(0, this.gLagFill[p]);
    this.ctx.flat(pass, nLeaf, 2);
  }

  /**
   * A gas with an energy: start the carried pressure from the state's (parity `p`), at the
   * start of a run and after every regrid (amr/pressure/lag_init.wgsl).
   */
  encodeInit(pass: GPUComputePassEncoder, p: number, nLeaf: number) {
    if (!this.pLagInit) return;
    this.run(pass, this.pLagInit, this.gLagInit[p], nLeaf);
    this.encodeFill(pass, nLeaf, p);
  }

  /**
   * The implicit half of a step: the equation from the old state (parity `p`) and the advected
   * one (parity 1 - p, whose primitives the caller has just assembled, ghosts and all),
   * `cycles` cycles, a measurement of where the solve ended, and the correction back into
   * parity `p`.
   */
  encode(pass: GPUComputePassEncoder, p: number, nLevel: readonly number[], nLeaf: number, cycles: number) {
    if (this.pCoef) for (let l = this.ctx.levels - 1; l >= 0; l--) this.run(pass, this.pCoef, this.gCoef[l], nLevel[l]);
    this.run(pass, this.pSource, this.gSource[p], nLeaf);
    for (let c = 0; c < cycles; c++) this.encodeCycle(pass, p, nLevel);
    this.run(pass, this.pMeasure, this.gMeasure[p], nLeaf);
    pass.setPipeline(this.pTally);
    pass.setBindGroup(0, this.gTally);
    pass.dispatchWorkgroups(1);
    this.run(pass, this.pCorrect, this.gCorrect[p], nLeaf);
    this.encodeFill(pass, nLeaf, p);
  }

  /** Since the solver was made: solves, the last one's relative residual, and how many ended above the tolerance. */
  async stats(): Promise<{ solves: number; residual: number; unconverged: number }> {
    const d = this.ctx.device;
    const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.tallies, 0, buf, 0, 16);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(buf.getMappedRange().slice(0));
    buf.destroy();
    return { solves: u[2], residual: new Float32Array(u.buffer)[1], unconverged: u[3] };
  }

  destroy() {
    for (const t of this.tex) t.destroy();
    this.root.destroy();
    this.res.destroy();
    this.tallies.destroy();
  }
}
