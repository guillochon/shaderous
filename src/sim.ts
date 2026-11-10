// The simulation: owns the state textures and encodes hydro + gravity steps.
import { assemble } from './gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture, type Gpu } from './gpu/device';
import { MAX, MIN, Reducer, SUM } from './gpu/reduce';
import { Multigrid } from './gravity/multigrid';
import { defaultEosParams, EOS_MODULES, makeEos, type EosKind, type EosParams } from './eos/eos';
import type { Bc3, GravityMode, Problem } from './problems/types';

export type RiemannKind = 'hllc' | 'hll';
export type LimiterKind = 'mc' | 'vanleer' | 'minmod';

export interface SimConfig {
  /** Cells per side. The grid is n x n x n, or n x n x 1 (periodic in z) when ndim is 2. */
  n: number;
  ndim: 2 | 3;
  /** Uniform external gravitational acceleration. */
  gext: [number, number, number];
  problem: Problem;
  eos: EosKind;
  eosParams: EosParams;
  riemann: RiemannKind;
  limiter: LimiterKind;
  gravity: GravityMode;
  G: number;
  bcLo: Bc3;
  bcHi: Bc3;
  cfl: number;
  tEnd: number; // Infinity runs forever
  rhoFloor: number;
  pFloor: number;
  etaDual: number;
  shockThresh: number;
  jeansFloor: boolean;
  vcycles: number; // V-cycles per step (warm-started)
  /** Flux-form gravitational work (conserves total energy) instead of kick-consistent energy. */
  gravEnergyFlux: boolean;
}

export function configFor(problem: Problem, n: number, over: Partial<SimConfig> = {}): SimConfig {
  const ndim = over.ndim ?? 3;
  // 2D: one periodic layer in z, and no self-gravity (the multigrid solver is 3D only).
  const bc2 = problem.bc2d ?? {
    lo: [problem.bcLo[0], problem.bcLo[1], 0] as Bc3,
    hi: [problem.bcHi[0], problem.bcHi[1], 0] as Bc3,
  };
  return {
    n,
    ndim,
    gext: problem.gext?.(ndim) ?? [0, 0, 0],
    problem,
    eos: problem.eos,
    eosParams: { ...defaultEosParams(), ...problem.eosParams },
    riemann: 'hllc',
    limiter: 'mc',
    gravity: ndim === 2 ? 'off' : problem.gravity,
    G: problem.G,
    bcLo: ndim === 2 ? bc2.lo : problem.bcLo,
    bcHi: ndim === 2 ? bc2.hi : problem.bcHi,
    cfl: 0.8,
    tEnd: problem.tEnd,
    rhoFloor: problem.rhoFloor ?? 1e-10,
    pFloor: problem.pFloor ?? 1e-14,
    etaDual: 1e-3,
    shockThresh: 0.5,
    jeansFloor: problem.jeansFloor ?? false,
    vcycles: 1, // warm-started from the previous step; a second cycle changes the tests by < 0.2%
    gravEnergyFlux: true,
    ...over,
  };
}

// Channel layout of the diagnostics reduction (hydro/diagnostics.wgsl).
const DIAG_OPS = [SUM, SUM, SUM, SUM, SUM, SUM, SUM, MAX, SUM, SUM, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX];

export interface Diagnostics {
  t: number;
  dt: number;
  steps: number;
  mass: number;
  momentum: [number, number, number];
  eKin: number;
  eInt: number;
  eGrav: number;
  eTot: number;
  maxMach: number;
  floored: number;
  jeansUnresolved: number;
  /** [min, max] per renderable field, in FIELDS order */
  ranges: [number, number][];
}


export class Sim {
  readonly device: GPUDevice;
  cfg: SimConfig;
  readonly n: number;
  /** cells along z: n, or 1 in 2D */
  readonly nz: number;
  readonly dx: number;

  /** state[parity] = [U1, U2]; `cur` is the parity holding the current state */
  state: GPUTexture[][] = [];
  prim: GPUTexture[] = [];
  /** half-step predicted primitives and per-cell flags (hydro/predict.wgsl) */
  pred: GPUTexture[] = [];
  /** gravitational acceleration and half-step potential (gravity/field.wgsl) */
  gfield!: GPUTexture;
  cur = 0;
  mg!: Multigrid;
  rhoMean = 0;

  private params: GPUBuffer;
  get paramBuffer(): GPUBuffer {
    return this.params;
  }
  private clock: GPUBuffer;
  private pPrim!: GPUComputePipeline;
  private pUpdate!: GPUComputePipeline;
  private pClock!: GPUComputePipeline;
  private pCorrect!: GPUComputePipeline;
  private pPredict!: GPUComputePipeline;
  private pField!: GPUComputePipeline;
  private gPredict!: GPUBindGroup;
  private gField!: GPUBindGroup;
  private gPrim: GPUBindGroup[] = [];
  private gUpdate: GPUBindGroup[] = [];
  private gCorrect: GPUBindGroup[] = [];
  private gClock!: GPUBindGroup;
  private redDt!: Reducer;
  private redDiag!: Reducer;
  private staging: GPUBuffer;
  private stagingBusy = false;

  private constructor(gpu: Gpu, cfg: SimConfig) {
    this.device = gpu.device;
    this.cfg = cfg;
    this.n = cfg.n;
    this.nz = cfg.ndim === 2 ? 1 : cfg.n;
    this.dx = cfg.problem.box / cfg.n;
    const d = this.device;
    this.params = d.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clock = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.staging = d.createBuffer({
      size: 16 + DIAG_OPS.length * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  static async create(gpu: Gpu, cfg: SimConfig): Promise<Sim> {
    const sim = new Sim(gpu, cfg);
    const d = sim.device;
    const n = cfg.n;
    const nz = sim.nz;
    const t = (label: string) => tex3d(d, n, 'rgba32float', label, nz);
    for (let p = 0; p < 2; p++) sim.state.push([t(`U1 ${p}`), t(`U2 ${p}`)]);
    sim.prim = [t('W1'), t('W2')];
    sim.pred = [t('PA'), t('PB')];
    sim.gfield = t('gravity field');
    sim.writeParams();
    // Self-gravity is 3D only. A 2D run keeps a 2^3 placeholder solver so the shared bindings
    // (potential, multipoles) still exist; it is never dispatched.
    sim.mg = await Multigrid.create(d, cfg.ndim === 2 ? 2 : n, cfg.problem.box, sim.params, [sim.state[0][0], sim.state[1][0]]);

    const v = (t: GPUTexture) => t.createView();
    const phi = sim.mg.phi;
    sim.redDt = await Reducer.create(
      d,
      n,
      nz,
      [MAX, MAX],
      'hydro/timestep.wgsl',
      {},
      [{ 0: sim.params, 1: v(sim.prim[0]), 2: v(sim.prim[1]), 3: v(sim.gfield) }],
      'timestep',
    );
    sim.pClock = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock');
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock });
    sim.pField = await computePipeline(d, assemble('gravity/field.wgsl'), 'gravity field');
    sim.gField = bindGroup(d, sim.pField, { 0: sim.params, 1: v(phi), 2: v(sim.mg.phiOld), 3: v(sim.gfield) });
    await sim.buildHydro();
    sim.reset();
    return sim;
  }

  /** (Re)build every pipeline that depends on the EOS, Riemann solver or limiter. */
  async buildHydro() {
    const d = this.device;
    const c = this.cfg;
    const slots = {
      eos: EOS_MODULES[c.eos],
      riemann: `hydro/riemann/${c.riemann}.wgsl`,
      limiter: `hydro/limiters/${c.limiter}.wgsl`,
    };
    const v = (t: GPUTexture) => t.createView();
    const phi = this.mg.phi;
    [this.pPrim, this.pPredict, this.pUpdate, this.pCorrect] = await Promise.all([
      computePipeline(d, assemble('hydro/primitives.wgsl', slots), 'primitives'),
      computePipeline(d, assemble('hydro/predict.wgsl', slots), 'predict'),
      computePipeline(d, assemble('hydro/update.wgsl', slots), 'update'),
      computePipeline(d, assemble('gravity/correct.wgsl', slots), 'gravity correct'),
    ]);
    this.gPrim = [];
    this.gUpdate = [];
    this.gCorrect = [];
    this.gPredict = bindGroup(d, this.pPredict, {
      0: this.params,
      1: this.clock,
      2: v(this.prim[0]),
      3: v(this.prim[1]),
      4: v(this.gfield),
      5: v(this.pred[0]),
      6: v(this.pred[1]),
    });
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      this.gPrim.push(
        bindGroup(d, this.pPrim, {
          0: this.params,
          1: v(this.state[p][0]),
          2: v(this.state[p][1]),
          3: v(this.prim[0]),
          4: v(this.prim[1]),
        }),
      );
      this.gUpdate.push(
        bindGroup(d, this.pUpdate, {
          0: this.params,
          1: this.clock,
          2: v(this.state[p][0]),
          3: v(this.state[p][1]),
          4: v(this.prim[0]),
          5: v(this.prim[1]),
          6: v(this.gfield),
          7: v(this.state[o][0]),
          8: v(this.state[o][1]),
          9: v(this.pred[0]),
          10: v(this.pred[1]),
        }),
      );
      this.gCorrect.push(
        bindGroup(d, this.pCorrect, {
          0: this.params,
          1: this.clock,
          2: v(this.state[o][0]),
          3: v(this.state[o][1]),
          4: v(this.gfield),
          5: v(this.state[p][0]),
          6: v(this.state[p][1]),
        }),
      );
    }
    this.redDiag = await Reducer.create(
      d,
      this.n,
      this.nz,
      DIAG_OPS,
      'hydro/diagnostics.wgsl',
      slots,
      [0, 1].map((p) => ({ 0: this.params, 1: v(this.state[p][0]), 2: v(this.state[p][1]), 3: v(phi) })),
      'diagnostics',
    );
  }

  /** Push the current configuration to the GPU parameter block. */
  writeParams() {
    const c = this.cfg;
    const buf = new ArrayBuffer(160);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    const selfGravity = c.gravity !== 'off' && c.ndim === 3;
    u.set([...c.bcLo, 0], 0);
    u.set([...c.bcHi, 0], 4);
    f.set([c.eosParams.gamma, c.eosParams.K, c.eosParams.R, c.eosParams.arad], 8);
    u.set([c.n, c.n, this.nz, 0], 12);
    // 2D quantities are per unit length along z.
    f.set([c.ndim === 2 ? this.dx ** 2 : this.dx ** 3, c.ndim, 0, 0], 16);
    f.set([...c.gext, 0], 20);
    u.set([c.n, selfGravity ? 1 : 0, c.gravity === 'isolated' ? 1 : 0, c.gravEnergyFlux ? 1 : 0], 24);
    f.set(
      [
        this.dx,
        c.cfl,
        c.G,
        this.rhoMean,
        c.rhoFloor,
        c.pFloor,
        c.etaDual,
        Number.isFinite(c.tEnd) ? c.tEnd : 1e30,
        1e30,
        c.shockThresh,
        c.jeansFloor ? 1 : 0,
        c.problem.box,
      ],
      28,
    );
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Load the problem's initial conditions and solve for the initial potential. */
  reset() {
    const c = this.cfg;
    const n = this.n;
    const nz = this.nz;
    const eos = makeEos(c.eos, c.eosParams);
    const fn = c.problem.init(n, eos, c.ndim);
    const u1 = new Float32Array(n * n * nz * 4);
    const u2 = new Float32Array(n * n * nz * 4);
    const half = c.problem.box / 2;
    let mass = 0;
    let o = 0;
    for (let k = 0; k < nz; k++) {
      const z = nz === 1 ? 0 : (k + 0.5) * this.dx - half;
      for (let j = 0; j < n; j++) {
        const y = (j + 0.5) * this.dx - half;
        for (let i = 0; i < n; i++, o += 4) {
          const x = (i + 0.5) * this.dx - half;
          const w = fn(x, y, z);
          const p = eos.hasEnergy ? w.p : eos.pressure(w.rho, 0);
          const eint = eos.eintFromP(w.rho, p);
          u1[o] = w.rho;
          u1[o + 1] = w.rho * w.vx;
          u1[o + 2] = w.rho * w.vy;
          u1[o + 3] = w.rho * w.vz;
          u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz);
          u2[o + 1] = eint;
          u2[o + 2] = w.rho * w.s;
          mass += w.rho;
        }
      }
    }
    this.setState(u1, u2, mass / (n * n * nz));
  }

  /** Replace the state with raw conserved arrays (rho, m), (E, e_int, rho s, 0). */
  setState(u1: Float32Array, u2: Float32Array, rhoMean: number) {
    const d = this.device;
    const n = this.n;
    this.cur = 0;
    this.rhoMean = rhoMean;
    writeTexture(d, this.state[0][0], n, 4, u1, this.nz);
    writeTexture(d, this.state[0][1], n, 4, u2, this.nz);
    if (this.cfg.ndim === 3) writeTexture(d, this.mg.phi, n, 1, new Float32Array(n * n * n));
    d.queue.writeBuffer(this.clock, 0, new Float32Array(4));
    this.writeParams();
    this.refreshPotential();
  }

  /** Solve for the potential of the current state from the current guess. */
  solvePotential(cycles: number) {
    if (!this.selfGravity) return;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.mg.encodeSource(pass, this.cur, this.cfg.gravity === 'isolated');
    for (let i = 0; i < cycles; i++) this.mg.encodeVCycle(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Call after changing the gravity mode or G at run time. */
  refreshPotential() {
    if (!this.selfGravity) return;
    this.solvePotential(16);
    const enc = this.device.createCommandEncoder();
    this.copyPhiToOld(enc);
    const pass = enc.beginComputePass();
    this.encodeField(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private get selfGravity(): boolean {
    return this.cfg.gravity !== 'off' && this.cfg.ndim === 3;
  }

  private dispatch(pass: GPUComputePassEncoder) {
    pass.dispatchWorkgroups(Math.ceil(this.n / 8), Math.ceil(this.n / 8), Math.ceil(this.nz / 4));
  }

  /** Acceleration and half-step potential from the current and previous potentials. */
  private encodeField(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.pField);
    pass.setBindGroup(0, this.gField);
    this.dispatch(pass);
  }

  private copyPhiToOld(enc: GPUCommandEncoder) {
    const n = this.n;
    enc.copyTextureToTexture({ texture: this.mg.phi }, { texture: this.mg.phiOld }, [n, n, n]);
  }

  /** Encode `count` full steps: primitives, timestep, predictor, update, then gravity if enabled. */
  encodeSteps(enc: GPUCommandEncoder, count: number) {
    const selfGravity = this.selfGravity;
    const anyGravity = selfGravity || this.cfg.gext.some((g) => g !== 0);
    for (let i = 0; i < count; i++) {
      const p = this.cur;
      let pass = enc.beginComputePass();
      pass.setPipeline(this.pPrim);
      pass.setBindGroup(0, this.gPrim[p]);
      this.dispatch(pass);
      this.redDt.encode(pass);
      pass.setPipeline(this.pClock);
      pass.setBindGroup(0, this.gClock);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.pPredict);
      pass.setBindGroup(0, this.gPredict);
      this.dispatch(pass);
      pass.setPipeline(this.pUpdate);
      pass.setBindGroup(0, this.gUpdate[p]);
      this.dispatch(pass);
      if (selfGravity) {
        // Keep the old potential and solve for that of the updated density (warm-started).
        pass.end();
        this.copyPhiToOld(enc);
        pass = enc.beginComputePass();
        this.mg.encodeSource(pass, 1 - p, this.cfg.gravity === 'isolated');
        for (let v = 0; v < this.cfg.vcycles; v++) this.mg.encodeVCycle(pass);
        this.encodeField(pass);
      }
      if (anyGravity) {
        // Second half kick, with the acceleration of the updated state.
        pass.setPipeline(this.pCorrect);
        pass.setBindGroup(0, this.gCorrect[p]);
        this.dispatch(pass);
      } else {
        this.cur = 1 - p;
      }
      pass.end();
    }
  }

  /**
   * Time each pass in isolation: milliseconds per execution, measured with a readback so the
   * number is real GPU time in every browser. The state is scratch afterwards; call reset().
   */
  async profile(reps = 20): Promise<Record<string, number>> {
    const d = this.device;
    const n = this.n;
    const p = this.cur;
    const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const time = async (count: number, fn: (pass: GPUComputePassEncoder) => void) => {
      const run = async (k: number) => {
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let i = 0; i < k; i++) fn(pass);
        pass.end();
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
        d.queue.submit([enc.finish()]);
        const t0 = performance.now();
        await buf.mapAsync(GPUMapMode.READ);
        buf.unmap();
        return performance.now() - t0;
      };
      // Difference of two batch sizes cancels the fixed readback latency.
      const a = await run(count);
      const b = await run(3 * count);
      return (b - a) / (2 * count);
    };
    const out: Record<string, number> = {};
    out.primitives = await time(reps, (pass) => {
      pass.setPipeline(this.pPrim);
      pass.setBindGroup(0, this.gPrim[p]);
      this.dispatch(pass);
    });
    out.timestep = await time(reps, (pass) => this.redDt.encode(pass));
    out.predict = await time(reps, (pass) => {
      pass.setPipeline(this.pPredict);
      pass.setBindGroup(0, this.gPredict);
      this.dispatch(pass);
    });
    out.update = await time(reps, (pass) => {
      pass.setPipeline(this.pUpdate);
      pass.setBindGroup(0, this.gUpdate[p]);
      this.dispatch(pass);
    });
    out.diagnostics = await time(reps, (pass) => this.redDiag.encode(pass, p));
    if (this.selfGravity) {
      out.gravity_source = await time(reps, (pass) => this.mg.encodeSource(pass, p, this.cfg.gravity === 'isolated'));
      out.vcycle = await time(reps, (pass) => this.mg.encodeVCycle(pass));
      out.gravity_field = await time(reps, (pass) => this.encodeField(pass));
      out.gravity_kick = await time(reps, (pass) => {
        pass.setPipeline(this.pCorrect);
        pass.setBindGroup(0, this.gCorrect[1 - p]);
        this.dispatch(pass);
      });
    }
    buf.destroy();
    for (const k of Object.keys(out)) out[k] = +out[k].toFixed(3);
    return out;
  }

  step(count = 1) {
    const enc = this.device.createCommandEncoder();
    this.encodeSteps(enc, count);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Reduce and read back the diagnostics. Returns null if a readback is already in flight,
   * so the frame loop can call this freely without ever waiting on the GPU.
   */
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
      const r = a.subarray(4);
      const ranges: [number, number][] = [];
      for (let i = 0; i < 5; i++) ranges.push([r[10 + 2 * i], r[11 + 2 * i]]);
      return {
        t: a[1] + a[0],
        dt: a[0],
        steps: a[3] + (a[0] > 0 ? 1 : 0),
        mass: r[0],
        momentum: [r[1], r[2], r[3]],
        eKin: r[4],
        eInt: r[5],
        eGrav: r[6],
        eTot: r[4] + r[5] + r[6],
        maxMach: r[7],
        floored: r[8],
        jeansUnresolved: r[9],
        ranges,
      };
    } finally {
      this.stagingBusy = false;
    }
  }

  /** Diagnostics, waiting for any in-flight readback first (tests). */
  async diagnostics(): Promise<Diagnostics> {
    for (;;) {
      const d = await this.readDiagnostics();
      if (d) return d;
      await new Promise((r) => setTimeout(r, 1));
    }
  }

  /**
   * Step until the clock reaches cfg.tEnd. Returns the number of steps taken.
   *
   * Steps are submitted in batches with one clock readback per batch. The batch grows while a
   * round trip is short (Firefox reports completion only every ~100 ms, so small batches would
   * be latency-bound) but never beyond the steps estimated to remain, since steps past tEnd
   * are no-ops that still cost a full pass over the grid.
   */
  async runToEnd(batch = 16, maxSteps = 1e6): Promise<number> {
    if (!Number.isFinite(this.cfg.tEnd)) throw new Error('runToEnd needs a finite tEnd');
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let steps = 0;
    let size = batch;
    try {
      for (;;) {
        const enc = this.device.createCommandEncoder();
        this.encodeSteps(enc, size);
        enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
        this.device.queue.submit([enc.finish()]);
        const t0 = performance.now();
        await buf.mapAsync(GPUMapMode.READ);
        const ms = performance.now() - t0;
        const c = new Float32Array(buf.getMappedRange().slice(0));
        buf.unmap();
        const [dt, t] = c;
        steps = c[3];
        if (!(dt === dt) || !(t === t)) throw new Error(`clock is NaN after ${steps} steps`);
        if (dt === 0 && t > 0) return steps;
        if (steps > maxSteps) throw new Error(`no convergence to tEnd after ${steps} steps (t = ${t})`);
        if (ms < 150) size = Math.min(size * 2, 512);
        else if (ms > 400) size = Math.max(batch, size >> 1);
        if (dt > 0) size = Math.max(1, Math.min(size, Math.ceil(((this.cfg.tEnd - t - dt) / dt) * 1.1) + 1));
      }
    } finally {
      buf.destroy();
    }
  }

  /**
   * Flux determinism check: evaluates one step's face mass fluxes and counts the
   * faces where the two cells sharing the face disagree in any bit. The state is not advanced.
   */
  async fluxMismatch(): Promise<{ faces: number; mismatched: number; maxDiff: number }> {
    const d = this.device;
    const n = this.n;
    const c = this.cfg;
    const slots = {
      eos: EOS_MODULES[c.eos],
      riemann: `hydro/riemann/${c.riemann}.wgsl`,
      limiter: `hydro/limiters/${c.limiter}.wgsl`,
    };
    const pipe = await computePipeline(d, assemble('hydro/update.wgsl', slots), 'update (flux debug)', { DEBUG_FLUX: 1 });
    const p = this.cur;
    const o = 1 - p;
    const v = (t: GPUTexture) => t.createView();
    const group = bindGroup(d, pipe, {
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]),
      4: v(this.prim[0]), 5: v(this.prim[1]), 6: v(this.gfield),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]),
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pPrim);
    pass.setBindGroup(0, this.gPrim[p]);
    this.dispatch(pass);
    pass.setPipeline(this.pPredict);
    pass.setBindGroup(0, this.gPredict);
    this.dispatch(pass);
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    this.dispatch(pass);
    pass.end();
    d.queue.submit([enc.finish()]);
    const f1 = await readTexture(d, this.state[o][0], n, 4, this.nz);
    const f2 = await readTexture(d, this.state[o][1], n, 4, this.nz);
    const flux = (i: number, j: number, k: number, f: number) => {
      const cell = ((k * n + j) * n + i) * 4;
      return f < 4 ? f1[cell + f] : f2[cell + f - 4];
    };
    let faces = 0;
    let mismatched = 0;
    let maxDiff = 0;
    const dims = [n, n, this.nz];
    for (let k = 0; k < this.nz; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++)
          for (let a = 0; a < 3; a++) {
            const q = [i, j, k];
            q[a]++;
            if (q[a] >= dims[a]) {
              if (c.bcLo[a] !== 0) continue; // only periodic faces have a partner cell
              q[a] = 0;
            }
            const mine = flux(i, j, k, 2 * a + 1); // my upper face
            const theirs = flux(q[0], q[1], q[2], 2 * a); // neighbour's lower face
            faces++;
            if (mine !== theirs) {
              mismatched++;
              maxDiff = Math.max(maxDiff, Math.abs(mine - theirs));
            }
          }
    return { faces, mismatched, maxDiff };
  }

  /** Current conserved state as float arrays (validation only). */
  async readState(): Promise<{ u1: Float32Array; u2: Float32Array }> {
    const [a, b] = this.state[this.cur];
    return {
      u1: await readTexture(this.device, a, this.n, 4, this.nz),
      u2: await readTexture(this.device, b, this.n, 4, this.nz),
    };
  }

  async readPotential(): Promise<Float32Array> {
    return readTexture(this.device, this.mg.phi, this.n, 1);
  }

  /** Max |residual| of the Poisson solve on the finest level. */
  async poissonResidual(): Promise<number> {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.mg.encodeResidual(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const r = await readTexture(this.device, this.mg.levels[0].res, this.n, 1);
    let m = 0;
    for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i]));
    return m;
  }

  destroy() {
    for (const s of this.state) for (const t of s) t.destroy();
    for (const t of [...this.prim, ...this.pred, this.gfield]) t.destroy();
    this.mg.destroy();
  }
}
