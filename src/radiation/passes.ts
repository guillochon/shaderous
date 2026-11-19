// The radiation passes of a uniform grid (radiation/radiation.ts): their textures, pipelines and
// bind groups, and how a step encodes them. Sim (src/sim.ts) owns one when radiation is on.
//
// R = (E, F) and G = (rho e, momentum gained) each have two parities, which advance together:
// the advection reads R[cur] and writes R[1 - cur]; each sweep reads R and G at cur and writes
// both at 1 - cur; the commit reads G[cur] and writes the gas into the state's other parity.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture } from '../gpu/device';
import { Reducer, MAX, SUM } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { radSweeps } from './radiation';

/** Channels of the radiation diagnostics (radiation/diagnostics.wgsl). */
export const RAD_DIAG_OPS = [SUM, SUM, SUM, SUM, SUM, SUM, MAX, MAX];

/** What a run with radiation reports beside the gas's diagnostics. */
export interface RadDiagnostics {
  /** total radiation energy */
  eRad: number;
  /** sum of E / r: with the gas's total energy, what the scheme conserves */
  eRadReduced: number;
  /** the share of the radiation energy in cells whose speed of light is reduced */
  reducedShare: number;
  /** luminosity through the box's open faces, as the readout finds it */
  luminosity: number;
  /**
   * The radiation energy that has left through the open faces since the run started, summed
   * step by step on the GPU (radiation/escape.wgsl): its change between two readouts over the
   * time between them is the mean luminosity, with nothing missed.
   */
  escaped: number;
  /**
   * The time `escaped` is the total at. A readout made while the run goes on reads the clock
   * first and this some steps later: a luminosity from the change of `escaped` divides by the
   * change of this, not of the readout's t.
   */
  escapedAt: number;
  /** cells at the streaming limit |F| = c E */
  streaming: number;
  /** cells where |v| + c_s > chat / 3: the reduced speed of light is too slow for the gas there */
  tooFast: number;
  maxReducedFlux: number;
  maxTrad: number;
}

/** Floats of the escaped-energy record (radiation/escape.wgsl's ESC), read after the reduction's. */
export const ESCAPE_FLOATS = 4;

/** From the reduction's channels followed by the escaped-energy record. */
export function radDiagnosticsFrom(r: Float32Array): RadDiagnostics {
  const k = RAD_DIAG_OPS.length;
  return {
    eRad: r[0], eRadReduced: r[1], reducedShare: r[0] > 0 ? r[2] / r[0] : 0, luminosity: r[3],
    escaped: r.length > k ? r[k] : 0,
    escapedAt: r.length > k ? r[k + 3] : 0,
    streaming: r[4], tooFast: r[5], maxReducedFlux: r[6], maxTrad: r[7],
  };
}

export interface RadPassesInit {
  device: GPUDevice;
  cfg: SimConfig;
  dims: [number, number, number];
  params: GPUBuffer;
  clock: GPUBuffer;
  /** state[parity] = [U1, U2] */
  state: GPUTexture[][];
  massFlux: GPUBuffer;
  slots: Record<string, string>;
  /** the composition moments (binding 27), where they vary */
  comp: Record<number, GPUTextureView>;
}

export class RadPasses {
  /** GPU bytes per cell: R and G, two parities each. */
  static readonly cellBytes = 4 * 16;
  readonly R: GPUTexture[];
  readonly G: GPUTexture[];
  /** the parity of R (and G) holding the current radiation */
  cur = 0;
  private info: GPUBuffer[];
  private pAdvect!: GPUComputePipeline;
  private pSweep!: GPUComputePipeline;
  private pCommit!: GPUComputePipeline;
  private pEscape!: GPUComputePipeline;
  /** gEscape[state parity][R parity] */
  private gEscape: GPUBindGroup[][] = [];
  /** the energy that has left the box (radiation/escape.wgsl) */
  private escaped: GPUBuffer;
  /** gAdvect[state parity][R parity] */
  private gAdvect: GPUBindGroup[][] = [];
  /** gSweep[state parity][R parity][first] */
  private gSweep: GPUBindGroup[][][] = [];
  /** gCommit[state parity][G parity] */
  private gCommit: GPUBindGroup[][] = [];
  private red!: Reducer;
  private staging: GPUBuffer;
  private stagingBusy = false;

  private constructor(private o: RadPassesInit) {
    const [nx, ny, nz] = o.dims;
    const d = o.device;
    const t = (label: string) => tex3d(d, nx, 'rgba32float', label, nz, ny);
    this.R = [t('radiation 0'), t('radiation 1')];
    this.G = [t('radiation gas 0'), t('radiation gas 1')];
    this.info = [0, 1].map((first) => {
      const b = d.createBuffer({ label: 'radiation sweep info', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, new Uint32Array([first, 0, 0, 0]));
      return b;
    });
    this.staging = d.createBuffer({ size: (RAD_DIAG_OPS.length + ESCAPE_FLOATS) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.escaped = d.createBuffer({ label: 'radiation escaped', size: ESCAPE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  static async create(o: RadPassesInit): Promise<RadPasses> {
    const r = new RadPasses(o);
    await r.rebuild(o.slots, o.comp);
    return r;
  }

  /** (Re)build the pipelines, for new equation-of-state, opacity or composition slots. */
  async rebuild(slots: Record<string, string>, comp: Record<number, GPUTextureView>) {
    const { device: d, params, clock, state, massFlux, dims } = this.o;
    const v = (t: GPUTexture) => t.createView();
    const red = Reducer.create(
      d, dims, RAD_DIAG_OPS, 'radiation/diagnostics.wgsl', slots,
      [0, 1].flatMap((p) => [0, 1].map((q) => ({ 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 3: v(this.R[q]), ...comp }))),
      'radiation diagnostics',
    );
    [this.pAdvect, this.pSweep, this.pCommit, this.pEscape] = await Promise.all([
      computePipeline(d, assemble('radiation/advect.wgsl', slots), 'radiation advection'),
      computePipeline(d, assemble('radiation/sweep.wgsl', slots), 'radiation sweep'),
      computePipeline(d, assemble('radiation/commit.wgsl', slots), 'radiation commit'),
      computePipeline(d, assemble('radiation/escape.wgsl', slots), 'radiation escaped'),
    ]);
    this.gEscape = [0, 1].map((p) => [0, 1].map((q) => bindGroup(d, this.pEscape, { 0: params, 1: clock, 2: v(state[p][1]), 3: v(this.R[q]), 4: this.escaped })));
    this.red = await red;
    this.gAdvect = [0, 1].map((p) =>
      [0, 1].map((q) =>
        bindGroup(d, this.pAdvect, {
          0: params, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 5: v(this.R[q]), 6: v(this.R[1 - q]), 7: massFlux, ...comp,
        }),
      ),
    );
    this.gSweep = [0, 1].map((p) =>
      [0, 1].map((q) =>
        [0, 1].map((first) =>
          bindGroup(d, this.pSweep, {
            0: params, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 4: v(this.R[q]), 5: v(this.G[q]), 6: v(this.R[1 - q]), 7: v(this.G[1 - q]), 8: this.info[first], ...comp,
          }),
        ),
      ),
    );
    this.gCommit = [0, 1].map((p) =>
      [0, 1].map((q) =>
        bindGroup(d, this.pCommit, { 0: params, 2: v(state[p][0]), 3: v(state[p][1]), 5: v(this.G[q]), 7: v(state[1 - p][0]), 8: v(state[1 - p][1]), ...comp }),
      ),
    );
  }

  private dispatch(pass: GPUComputePassEncoder) {
    const [nx, ny, nz] = this.o.dims;
    pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 4));
  }

  /** Carry the radiation with the gas: the update took the state from parity p to 1 - p. */
  encodeAdvect(pass: GPUComputePassEncoder, p: number) {
    pass.setPipeline(this.pAdvect);
    pass.setBindGroup(0, this.gAdvect[p][this.cur]);
    this.dispatch(pass);
    this.cur = 1 - this.cur;
  }

  /** The sweeps of a step, on the gas at parity s, then the commit into 1 - s. */
  encodeSweeps(pass: GPUComputePassEncoder, s: number) {
    const n = radSweeps(this.o.cfg);
    pass.setPipeline(this.pSweep);
    for (let k = 0; k < n; k++) {
      pass.setBindGroup(0, this.gSweep[s][this.cur][k === 0 ? 1 : 0]);
      this.dispatch(pass);
      this.cur = 1 - this.cur;
    }
    pass.setPipeline(this.pCommit);
    pass.setBindGroup(0, this.gCommit[s][this.cur]);
    this.dispatch(pass);
    // What left the box over this step (radiation/escape.wgsl).
    pass.setPipeline(this.pEscape);
    pass.setBindGroup(0, this.gEscape[s][this.cur]);
    pass.dispatchWorkgroups(1);
  }

  /** The current radiation for a pass that reads it (the timestep's rates: binding 30). */
  view(): GPUTextureView {
    return this.R[this.cur].createView();
  }

  /** Set the radiation from (E, Fx, Fy, Fz) per cell. */
  set(data: Float32Array) {
    const [nx, ny, nz] = this.o.dims;
    this.cur = 0;
    writeTexture(this.o.device, this.R[0], nx, 4, data, nz, ny);
  }

  /** Start the count of the energy that has left the box again (a new run). */
  resetEscaped() {
    this.o.device.queue.writeBuffer(this.escaped, 0, new Float32Array(ESCAPE_FLOATS));
  }

  /** (E, Fx, Fy, Fz) per cell, as it is now. */
  async read(): Promise<Float32Array> {
    const [nx, ny, nz] = this.o.dims;
    return readTexture(this.o.device, this.R[this.cur], nx, 4, nz, ny);
  }

  /** Reduce and read the radiation diagnostics of the state at `parity`; null while one is in flight. */
  async diagnostics(parity: number): Promise<RadDiagnostics | null> {
    if (this.stagingBusy) return null;
    this.stagingBusy = true;
    const d = this.o.device;
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.red.encode(pass, 2 * parity + this.cur);
    pass.end();
    enc.copyBufferToBuffer(this.red.result, 0, this.staging, 0, RAD_DIAG_OPS.length * 4);
    enc.copyBufferToBuffer(this.escaped, 0, this.staging, RAD_DIAG_OPS.length * 4, ESCAPE_FLOATS * 4);
    d.queue.submit([enc.finish()]);
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const r = new Float32Array(this.staging.getMappedRange().slice(0));
      this.staging.unmap();
      return radDiagnosticsFrom(r);
    } finally {
      this.stagingBusy = false;
    }
  }

  destroy() {
    for (const t of [...this.R, ...this.G]) t.destroy();
    for (const b of this.info) b.destroy();
    this.staging.destroy();
    this.escaped.destroy();
  }
}
