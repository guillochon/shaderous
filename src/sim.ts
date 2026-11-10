// The simulation: owns the state textures and encodes hydro + gravity steps.
import { assemble } from './gpu/shaders';
import { RegionPass, type RegionGroups } from './hydro/region_pass';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture, type Gpu } from './gpu/device';
import { MAX, MIN, Reducer, SUM } from './gpu/reduce';
import { Multigrid } from './gravity/multigrid';
import { defaultEosParams, EOS_MODULES, makeEos, type EosKind, type EosParams } from './eos/eos';
import { cellMark, type Bc3, type GravityMode, type Problem } from './problems/types';
import type { Restart } from './io/snapshot';

export type RiemannKind = 'hllc' | 'hll';
export type LimiterKind = 'koren' | 'mc' | 'vanleer' | 'minmod';

export interface SimConfig {
  /** Cells per side. The grid is n x n x n, or n x n x 1 (periodic in z) when ndim is 2. */
  n: number;
  ndim: 2 | 3;
  /** Uniform external gravitational acceleration. */
  gext: [number, number, number];
  /** Values of the preset's own parameters (Problem.params). */
  params: Record<string, number>;
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
  /** Cells per Jeans length that the pressure floor enforces (Truelove's minimum is 4). */
  jeansCells: number;
  vcycles: number; // V-cycles per step (warm-started)
  /** Flux-form gravitational work (conserves total energy) instead of kick-consistent energy. */
  gravEnergyFlux: boolean;
  /**
   * Adaptive mesh: number of refinement levels below the coarsest (0 = uniform grid). `n` is
   * then the finest resolution and the coarsest level has n >> amrLevels cells per side.
   */
  amrLevels: number;
  /** Atlas slots per side for the adaptive mesh (0 = choose from the root grid). */
  amrBudget: number;
  /** Start from a snapshot: its time, step count and perturber rather than the preset's. */
  restart?: Restart;
  /** A block is split where its refinement indicator exceeds refineHi, merged below refineLo. */
  refineHi: number;
  refineLo: number;
  /** With self-gravity, refine where the Jeans length is under this many cells (0 = off). */
  jeansRefine: number;
}

/** What the front end, renderer and tests need from a simulation, uniform or adaptive. */
export interface Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  /** finest cells per side, and along z (1 in 2D) */
  readonly n: number;
  readonly nz: number;
  /** parity of the state textures holding the current state */
  cur: number;
  readonly paramBuffer: GPUBuffer;
  /** Encode up to `count` steps; returns how many were encoded. */
  encodeSteps(enc: GPUCommandEncoder, count: number): number;
  /** Call after submitting an encoder that encodeSteps() wrote to. */
  submitted(): void;
  readDiagnostics(): Promise<Diagnostics | null>;
  diagnostics(): Promise<Diagnostics>;
  runToEnd(batch?: number, maxSteps?: number): Promise<number>;
  reset(): void;
  writeParams(): void;
  buildHydro(): Promise<void>;
  refreshPotential(): void;
  /** Encode one application of an editing brush to the current state (running or paused). */
  encodeBrush(enc: GPUCommandEncoder, op: BrushOp): void;
  /** Call once gas has been added or removed, with the new total mass: self-gravity is re-solved. */
  massChanged(mass: number): void;
  /** Replace the sources and sinks (at most MAX_SOURCES; a reset removes them all). */
  setSources(list: SourceSpec[]): void;
  destroy(): void;
  /** Size of the uniform volume the renderer displays, and the pass that fills it. */
  readonly displayDims: [number, number, number];
  readonly fieldShader: string;
  fieldResources(parity: number): Record<number, GPUBuffer | GPUTextureView>;
  /** Extra text for the status line. */
  status(): string;
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
    params: Object.fromEntries((problem.params ?? []).map((q) => [q.id, q.value])),
    problem,
    eos: problem.eos,
    eosParams: { ...defaultEosParams(), ...problem.eosParams },
    riemann: 'hllc',
    limiter: 'koren',
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
    jeansCells: problem.jeansCells ?? 4,
    // Warm-started from the previous step. One cycle is enough for the validation tests, but
    // leaves total momentum drifting about ten times faster in long collapse runs.
    vcycles: 2,
    gravEnergyFlux: true,
    amrLevels: 0,
    amrBudget: 0,
    refineHi: 0.06,
    refineLo: 0.025,
    jeansRefine: 16,
    ...over,
  };
}

/** One application of an editing brush (edit/brush_core.wgsl). */
export interface BrushOp {
  kind: 'mass' | 'velocity' | 'temperature' | 'obstacle';
  /** centre and radius, in fractions of the box */
  pos: [number, number, number];
  radius: number;
  /** 0: hard edge; 1: falls off all the way from the centre */
  softness: number;
  /**
   * mass: density added at the centre (negative removes); temperature: log of the factor;
   * obstacle: positive makes the cells solid, negative turns them back into gas
   */
  amount: number;
  /** velocity tool: the velocity to set */
  vel: [number, number, number];
  /** mass tool: new gas brings no momentum (otherwise it arrives with the local velocity) */
  atRest?: boolean;
  /** obstacle tool: the outline, drawn on the screen and cut through the box along the line of sight */
  outline?: Outline;
}

export interface Outline {
  /**
   * Takes a point of the box (in fractions of it, with a 1 appended) to homogeneous screen
   * coordinates, the units the outline is given in. Column-major, 16 numbers.
   */
  matrix: number[];
  /** an ellipse: centre and half-axes ... */
  ellipse?: [number, number, number, number];
  /** ... or a polygon: x, y of each vertex, at most MAX_OUTLINE_VERTICES of them */
  polygon?: number[];
}

/** Whether a point of the box (in fractions of it) lies behind an outline: as in edit/brush_core.wgsl. */
export function outlineContains(o: Outline, x: number, y: number, z: number): boolean {
  const m = o.matrix;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w <= 0) return false;
  const sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
  const sy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
  if (o.ellipse) {
    const [cx, cy, hx, hy] = o.ellipse;
    return ((sx - cx) / hx) ** 2 + ((sy - cy) / hy) ** 2 <= 1;
  }
  const p = o.polygon ?? [];
  let inside = false;
  for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) {
    if (p[i + 1] > sy !== p[j + 1] > sy && sx < ((p[j] - p[i]) * (sy - p[i + 1])) / (p[j + 1] - p[i + 1]) + p[i]) inside = !inside;
  }
  return inside;
}

/** A source or sink of gas (hydro/sources.wgsl). */
export interface SourceSpec {
  /** centre and radius, in fractions of the box */
  pos: [number, number, number];
  radius: number;
  softness: number;
  /** velocity of the gas a source adds; zero for one that spreads evenly */
  vel: [number, number, number];
  /** a source: density added per unit time at its centre */
  rate: number;
  /** a sink: fraction of the gas removed per unit time at its centre (0 for a source) */
  decay: number;
}

export const MAX_SOURCES = 8;
export const SOURCE_BYTES = 16 + 48 * MAX_SOURCES;

/** Pack the Sources block. The layout must match the struct in hydro/sources.wgsl. */
export function packSources(list: SourceSpec[]): ArrayBuffer {
  const buf = new ArrayBuffer(SOURCE_BYTES);
  const f = new Float32Array(buf);
  const use = list.slice(0, MAX_SOURCES);
  new Uint32Array(buf)[0] = use.length;
  use.forEach((q, i) => f.set([...q.pos, q.radius, ...q.vel, q.softness, q.rate, q.decay, 0, 0], 4 + 12 * i));
  return buf;
}

export const MAX_OUTLINE_VERTICES = 32;
export const BRUSH_BYTES = 384;

/** Pack the Brush block. The layout must match the struct in edit/brush_core.wgsl. */
export function packBrush(b: BrushOp): ArrayBuffer {
  const buf = new ArrayBuffer(BRUSH_BYTES);
  const f = new Float32Array(buf);
  f.set([...b.pos, b.radius, ...b.vel, b.softness], 0);
  const u = new Uint32Array(buf);
  u[8] = ['mass', 'velocity', 'temperature', 'obstacle'].indexOf(b.kind);
  f[9] = b.amount;
  u[10] = b.atRest ? 1 : 0;
  if (b.outline) {
    const poly = (b.outline.polygon ?? []).slice(0, 2 * MAX_OUTLINE_VERTICES);
    u[11] = poly.length / 2;
    f.set(b.outline.matrix, 12);
    f.set(b.outline.ellipse ?? [0, 0, 1, 1], 28);
    f.set(poly, 32);
  }
  return buf;
}

/** Number of floats the diagnostics readback holds: clock (4), reduction, perturber position (4). */
export const DIAG_FLOATS = 4 + 23 + 4;

/** Decode a diagnostics readback: [clock (4), reduction channels, perturber position (4)]. */
export function diagnosticsFrom(a: Float32Array, hasPerturber: boolean): Diagnostics {
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
    com: [r[20] / r[0], r[21] / r[0], r[22] / r[0]],
    perturber: hasPerturber ? [r[23], r[24], r[25]] : null,
    ranges,
  };
}

/** Size of the Params uniform block (hydro/common.wgsl). */
export const PARAM_BYTES = 208;

/** Pack the Params block. The layout must match the struct in hydro/common.wgsl. */
export function packParams(
  c: SimConfig,
  x: { dims: number[]; dx: number; rhoMean: number; selfGravity: boolean; perturberGM: number; amr?: number[]; refine?: number[]; solids?: boolean; sources?: boolean },
): ArrayBuffer {
  const buf = new ArrayBuffer(PARAM_BYTES);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u.set([...c.bcLo, 0], 0);
  u.set([...c.bcHi, 0], 4);
  f.set([c.eosParams.gamma, c.eosParams.K, c.eosParams.R, c.eosParams.arad], 8);
  u.set([...x.dims, 0], 12);
  // 2D quantities are per unit length along z.
  f.set([c.ndim === 2 ? x.dx ** 2 : x.dx ** 3, c.ndim, ...(x.refine ?? [0, 0])], 16);
  f.set([...c.gext, 0], 20);
  // Perturber: G M, softening of two cells, and whether it is active.
  f.set([x.perturberGM, 2 * x.dx, x.perturberGM > 0 ? 1 : 0, 0], 24);
  const fl = c.problem.fluff;
  f.set(fl ? [fl.rho, fl.u, fl.v, 0] : [0, 0, 0, 0], 28);
  u.set(x.amr ?? [0, 0, 0, 0], 32);
  u.set([c.n, x.selfGravity ? 1 : 0, c.gravity === 'isolated' ? 1 : 0, (c.gravEnergyFlux ? 1 : 0) | (x.solids ? 2 : 0) | (x.sources ? 4 : 0)], 36);
  f.set(
    [
      x.dx,
      c.cfl,
      c.G,
      x.rhoMean,
      c.rhoFloor,
      c.pFloor,
      c.etaDual,
      Number.isFinite(c.tEnd) ? c.tEnd : 1e30,
      1e30,
      c.shockThresh,
      c.jeansFloor ? (c.jeansCells / 4) ** 2 : 0, // scales the 4-cell floor pressure
      c.problem.box,
    ],
    40,
  );
  return buf;
}

// Channel layout of the diagnostics reduction (hydro/diagnostics.wgsl).
export const DIAG_OPS = [SUM, SUM, SUM, SUM, SUM, SUM, SUM, MAX, SUM, SUM, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, MIN, MAX, SUM, SUM, SUM];

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
  /** centre of mass of the gas, relative to the box centre */
  com: [number, number, number];
  /** position of the point-mass perturber relative to the box centre, if there is one */
  perturber: [number, number, number] | null;
  /** [min, max] per renderable field, in FIELDS order */
  ranges: [number, number][];
}


export class Sim implements Simulation {
  readonly device: GPUDevice;
  cfg: SimConfig;
  readonly n: number;
  /** cells along z: n, or 1 in 2D */
  readonly nz: number;
  readonly dx: number;

  /** state[parity] = [U1, U2]; `cur` is the parity holding the current state */
  state: GPUTexture[][] = [];
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
  private pUpdate!: RegionPass;
  private pClock!: GPUComputePipeline;
  private pCorrect!: GPUComputePipeline;
  private pPredict!: RegionPass;
  private pField!: GPUComputePipeline;
  private pBhDrift!: GPUComputePipeline;
  private pBhKick!: GPUComputePipeline;
  private gBhDrift!: GPUBindGroup;
  private gBhKick!: GPUBindGroup;
  private redBh!: Reducer;
  /** perturber state: position, velocity, acceleration, frame acceleration (4 x vec4) */
  private bh: GPUBuffer;
  private gPredict: RegionGroups[] = [];
  private gField!: GPUBindGroup;
  private gPrim: GPUBindGroup[] = [];
  private gUpdate: RegionGroups[] = [];
  private gCorrect: GPUBindGroup[] = [];
  private gClock!: GPUBindGroup;
  private redDt!: Reducer;
  private redDiag!: Reducer;
  private staging: GPUBuffer;
  private stagingBusy = false;
  private brush: GPUBuffer;
  private pBrush!: GPUComputePipeline;
  private gBrush: GPUBindGroup[] = [];
  /** the grid holds solid or fixed cells, so the passes must look for them */
  private solids = false;
  private sources: GPUBuffer;
  private nSources = 0;

  private constructor(gpu: Gpu, cfg: SimConfig) {
    this.device = gpu.device;
    this.cfg = cfg;
    this.n = cfg.n;
    this.nz = cfg.ndim === 2 ? 1 : cfg.n;
    this.dx = cfg.problem.box / cfg.n;
    const d = this.device;
    this.params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clock = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.bh = d.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.brush = d.createBuffer({ size: BRUSH_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sources = d.createBuffer({ size: SOURCE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.staging = d.createBuffer({
      size: 16 + DIAG_OPS.length * 4 + 16,
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
      [n, n, nz],
      [MAX, MAX],
      '',
      {},
      [],
      'timestep',
      [n, n, nz].map((m) => Math.ceil(m / 4)), // the primitives pass does the first stage
    );
    sim.pClock = await computePipeline(d, assemble('hydro/clock.wgsl'), 'clock');
    sim.gClock = bindGroup(d, sim.pClock, { 0: sim.params, 1: sim.redDt.result, 2: sim.clock });
    sim.pField = await computePipeline(d, assemble('gravity/field.wgsl'), 'gravity field');
    sim.gField = bindGroup(d, sim.pField, { 0: sim.params, 1: v(phi), 2: v(sim.mg.phiOld), 3: v(sim.gfield), 4: sim.bh, 5: sim.clock });
    // Point-mass perturber: its pull on the gas is reduced each step and drives both the
    // frame's acceleration and the perturber's orbit.
    sim.redBh = await Reducer.create(
      d,
      [n, n, nz],
      [SUM, SUM, SUM, SUM],
      'gravity/bh_moment.wgsl',
      {},
      [0, 1].map((p) => ({ 0: sim.params, 1: v(sim.state[p][0]), 2: sim.bh })),
      'perturber',
    );
    sim.pBhDrift = await computePipeline(d, assemble('gravity/bh_drift.wgsl'), 'perturber drift');
    sim.gBhDrift = bindGroup(d, sim.pBhDrift, { 1: sim.clock, 2: sim.bh });
    sim.pBhKick = await computePipeline(d, assemble('gravity/bh_kick.wgsl'), 'perturber kick');
    sim.gBhKick = bindGroup(d, sim.pBhKick, { 0: sim.params, 1: sim.clock, 2: sim.redBh.result, 3: sim.bh });
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
    const dims = [this.n, this.n, this.nz];
    this.pPredict?.destroy();
    this.pUpdate?.destroy();
    [this.pPrim, this.pPredict, this.pUpdate, this.pCorrect, this.pBrush] = await Promise.all([
      computePipeline(d, assemble('hydro/rates.wgsl', slots), 'timestep rates'),
      RegionPass.create(d, assemble('hydro/predict.wgsl', slots), 'predict', dims),
      RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update', dims),
      computePipeline(d, assemble('gravity/correct.wgsl', slots), 'gravity correct'),
      computePipeline(d, assemble('edit/brush.wgsl', slots), 'brush'),
    ]);
    this.gBrush = [0, 1].map((p) =>
      bindGroup(d, this.pBrush, {
        0: this.params, 1: this.brush, 2: v(this.state[p][0]), 3: v(this.state[p][1]), 5: v(this.state[1 - p][0]), 6: v(this.state[1 - p][1]),
      }),
    );
    this.gPrim = [];
    this.gUpdate = [];
    this.gCorrect = [];
    this.gPredict = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      this.gPrim.push(
        bindGroup(d, this.pPrim, {
          0: this.params,
          1: v(this.state[p][0]),
          2: v(this.state[p][1]),
          5: v(this.gfield),
          6: this.redDt.first,
        }),
      );
      this.gPredict.push(
        this.pPredict.groups({
          0: this.params,
          1: this.clock,
          2: v(this.state[p][0]),
          3: v(this.state[p][1]),
          4: v(this.gfield),
          5: v(this.pred[0]),
          6: v(this.pred[1]),
        }),
      );
      this.gUpdate.push(
        this.pUpdate.groups({
          0: this.params,
          1: this.clock,
          2: v(this.state[p][0]),
          3: v(this.state[p][1]),
          6: v(this.gfield),
          7: v(this.state[o][0]),
          8: v(this.state[o][1]),
          9: v(this.pred[0]),
          10: v(this.pred[1]),
          11: this.sources,
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
      [this.n, this.n, this.nz],
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
    const bh = this.hasPerturber ? c.problem.perturber!(c.params, c.G) : null;
    const buf = packParams(c, {
      dims: [c.n, c.n, this.nz],
      dx: this.dx,
      rhoMean: this.rhoMean,
      selfGravity: this.selfGravity,
      perturberGM: bh ? bh.gm : 0,
      solids: this.solids,
      sources: this.nSources > 0,
    });
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Load the problem's initial conditions and solve for the initial potential. */
  reset() {
    const c = this.cfg;
    const n = this.n;
    const nz = this.nz;
    const eos = makeEos(c.eos, c.eosParams);
    const fn = c.problem.init(n, eos, c.ndim, c.params);
    if (c.problem.perturber) {
      const bh = c.restart?.perturber ?? c.problem.perturber(c.params, c.G);
      this.device.queue.writeBuffer(this.bh, 0, new Float32Array([...bh.pos, 0, ...bh.vel, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    }
    const u1 = new Float32Array(n * n * nz * 4);
    const u2 = new Float32Array(n * n * nz * 4);
    const half = c.problem.box / 2;
    let mass = 0;
    let o = 0;
    this.solids = false;
    this.nSources = 0;
    this.device.queue.writeBuffer(this.sources, 0, packSources([]));
    for (let k = 0; k < nz; k++) {
      const z = nz === 1 ? 0 : (k + 0.5) * this.dx - half;
      for (let j = 0; j < n; j++) {
        const y = (j + 0.5) * this.dx - half;
        for (let i = 0; i < n; i++, o += 4) {
          const x = (i + 0.5) * this.dx - half;
          const w = fn(x, y, z);
          const p = eos.hasEnergy ? w.p : eos.pressure(w.rho, 0);
          const eint = eos.eintFromP(w.rho, p);
          const mark = cellMark(w);
          if (mark) this.solids = true;
          const go = mark === 1 ? 0 : 1; // the gas kept inside an obstacle is at rest
          u1[o] = w.rho;
          u1[o + 1] = w.rho * w.vx * go;
          u1[o + 2] = w.rho * w.vy * go;
          u1[o + 3] = w.rho * w.vz * go;
          u2[o] = eint + 0.5 * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz) * go;
          u2[o + 1] = eint;
          u2[o + 2] = w.rho * w.s;
          u2[o + 3] = mark;
          mass += w.rho;
        }
      }
    }
    this.setState(u1, u2, mass / (n * n * nz));
    // The clock is (dt, t, Kahan compensation, steps).
    if (c.restart) this.device.queue.writeBuffer(this.clock, 0, new Float32Array([0, c.restart.time, 0, c.restart.steps]));
  }

  /** Time, step count and perturber state, for a snapshot (see io/snapshot.ts). */
  async readRestart(): Promise<{ time: number; steps: number; perturber?: { pos: number[]; vel: number[] } }> {
    const d = this.device;
    const buf = d.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.clock, 0, buf, 0, 16);
    enc.copyBufferToBuffer(this.bh, 0, buf, 16, 32);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    const out: { time: number; steps: number; perturber?: { pos: number[]; vel: number[] } } = { time: f[1], steps: f[3] };
    if (this.cfg.problem.perturber) out.perturber = { pos: [f[4], f[5], f[6]], vel: [f[8], f[9], f[10]] };
    return out;
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
    // The clock's dt is zero here, so this only evaluates the accelerations.
    if (this.hasPerturber) this.encodePerturberKick(pass, this.cur);
    this.encodeField(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private get selfGravity(): boolean {
    return this.cfg.gravity !== 'off' && this.cfg.ndim === 3;
  }

  /** A perturber needs the self-gravity path (its field shares that texture). */
  private get hasPerturber(): boolean {
    return this.selfGravity && !!this.cfg.problem.perturber;
  }

  /** Reduce the perturber's pull on the gas of the given parity, then finish its step. */
  private encodePerturberKick(pass: GPUComputePassEncoder, parity: number) {
    this.redBh.encode(pass, parity);
    pass.setPipeline(this.pBhKick);
    pass.setBindGroup(0, this.gBhKick);
    pass.dispatchWorkgroups(1);
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
  encodeSteps(enc: GPUCommandEncoder, count: number): number {
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
      this.pPredict.encode(pass, this.gPredict[p]);
      this.pUpdate.encode(pass, this.gUpdate[p]);
      if (selfGravity) {
        // Keep the old potential and solve for that of the updated density (warm-started).
        pass.end();
        this.copyPhiToOld(enc);
        pass = enc.beginComputePass();
        this.mg.encodeSource(pass, 1 - p, this.cfg.gravity === 'isolated');
        for (let v = 0; v < this.cfg.vcycles; v++) this.mg.encodeVCycle(pass);
        if (this.hasPerturber) {
          // Move the perturber to the new time, then its pull on the updated gas.
          pass.setPipeline(this.pBhDrift);
          pass.setBindGroup(0, this.gBhDrift);
          pass.dispatchWorkgroups(1);
          this.encodePerturberKick(pass, 1 - p);
        }
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
    return count;
  }

  submitted() {}

  encodeBrush(enc: GPUCommandEncoder, op: BrushOp) {
    if (op.kind === 'obstacle' && !this.solids) {
      this.solids = true;
      this.writeParams();
    }
    this.device.queue.writeBuffer(this.brush, 0, packBrush(op));
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pBrush);
    pass.setBindGroup(0, this.gBrush[this.cur]);
    this.dispatch(pass);
    pass.end();
    this.cur = 1 - this.cur;
  }

  setSources(list: SourceSpec[]) {
    this.nSources = Math.min(list.length, MAX_SOURCES);
    this.device.queue.writeBuffer(this.sources, 0, packSources(list));
    this.writeParams();
  }

  massChanged(mass: number) {
    // The periodic solver subtracts the mean density, so it has to follow the edit.
    this.rhoMean = mass / this.cfg.problem.box ** this.cfg.ndim;
    this.writeParams();
    this.refreshPotential();
  }

  get displayDims(): [number, number, number] {
    return [this.n, this.n, this.nz];
  }
  readonly fieldShader = 'render/field.wgsl';
  fieldResources(parity: number): Record<number, GPUBuffer | GPUTextureView> {
    return { 2: this.state[parity][0].createView(), 3: this.state[parity][1].createView() };
  }
  status(): string {
    return '';
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
    out.rates = await time(reps, (pass) => {
      pass.setPipeline(this.pPrim);
      pass.setBindGroup(0, this.gPrim[p]);
      this.dispatch(pass);
    });
    out.timestep = await time(reps, (pass) => this.redDt.encode(pass));
    out.predict = await time(reps, (pass) => {
      this.pPredict.encode(pass, this.gPredict[p]);
    });
    out.update = await time(reps, (pass) => {
      this.pUpdate.encode(pass, this.gUpdate[p]);
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
    enc.copyBufferToBuffer(this.bh, 0, this.staging, 16 + DIAG_OPS.length * 4, 16);
    this.device.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const a = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      return diagnosticsFrom(a, this.hasPerturber);
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
    // Split by region like the real pass, so faces between the two variants are checked too.
    const pipe = await RegionPass.create(d, assemble('hydro/update.wgsl', slots), 'update (flux debug)', [n, n, this.nz], { DEBUG_FLUX: 1 });
    const p = this.cur;
    const o = 1 - p;
    const v = (t: GPUTexture) => t.createView();
    const group = pipe.groups({
      0: this.params, 1: this.clock, 2: v(this.state[p][0]), 3: v(this.state[p][1]),
      6: v(this.gfield),
      7: v(this.state[o][0]), 8: v(this.state[o][1]), 9: v(this.pred[0]), 10: v(this.pred[1]), 11: this.sources,
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pPrim);
    pass.setBindGroup(0, this.gPrim[p]);
    this.dispatch(pass);
    this.pPredict.encode(pass, this.gPredict[p]);
    pipe.encode(pass, group);
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
    for (const t of [...this.pred, this.gfield]) t.destroy();
    this.mg.destroy();
  }
}
