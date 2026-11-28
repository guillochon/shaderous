// The radiation passes of a block grid (radiation/radiation.ts; docs/numerics.md section 24):
// their atlas textures, pipelines and bind groups, and how a step encodes them. AmrSim owns one
// when radiation is on. One timestep for all blocks.
//
// R = (E, F) and G = (rho e, momentum gained) hold the leaves' cells, two parities each, which
// advance together as on the uniform grid (radiation/passes.ts). RA and RK are the radiation
// and kappa_R rho of every leaf's cells with two layers of ghosts, rebuilt from R before the
// advection and before every sweep (amr/rad_assemble.wgsl): the sweeps, the advection, the
// timestep (amr/rad_rate_on.wgsl) and re-meshing read them.
import { assemble } from '../gpu/shaders';
import { bindGroupUsed, computePipeline, type Bindable } from '../gpu/device';
import { Reducer } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { radSweeps } from '../radiation/radiation';
import { ESCAPE_FLOATS, RAD_DIAG_OPS, radDiagnosticsFrom, type RadDiagnostics } from '../radiation/passes';
import { BS, PAD } from './tree';

export interface AmrRadInit {
  device: GPUDevice;
  cfg: SimConfig;
  /** blocks the atlas holds */
  cap: number;
  /** an atlas texture laid out as the state's */
  texture: (label: string, format?: GPUTextureFormat) => GPUTexture;
  params: GPUBuffer;
  clock: GPUBuffer;
  bt: GPUBuffer;
  listLeaf: GPUBuffer;
  rg: GPUBuffer;
  /** state[parity] = [U1, U2] */
  state: GPUTexture[][];
  /** the primitives with ghosts, as the hydro's assembly leaves them: (density, velocity) */
  prim: GPUTexture;
  /** the face mass fluxes the update records (AmrSim.massFluxWindows) */
  massFlux: { lo: Bindable; hi: Bindable | null };
  /** dispatches over `count` leaves: cells and `reach` ghost layers, and interior cells */
  flat: (pass: GPUComputePassEncoder, count: number, reach: number) => void;
  interior: (pass: GPUComputePassEncoder, count: number) => void;
}

export class AmrRad {
  /** GPU bytes per atlas texel: R and G at two parities, RA, and RK's one channel. */
  static readonly texelBytes = 5 * 16 + 4;
  readonly R: GPUTexture[];
  readonly G: GPUTexture[];
  readonly RA: GPUTexture;
  readonly RK: GPUTexture;
  /** the parity of R (and G) holding the current radiation */
  cur = 0;
  private info: GPUBuffer[];
  private pAssemble!: GPUComputePipeline;
  private pAdvect!: GPUComputePipeline;
  private pSweep!: GPUComputePipeline;
  private pCommit!: GPUComputePipeline;
  private pRegrid!: GPUComputePipeline;
  private pEscape!: GPUComputePipeline;
  /** gEscape[state parity][R parity] */
  private gEscape: GPUBindGroup[][] = [];
  /** the energy that has left the box (amr/rad_escape.wgsl) */
  private escaped: GPUBuffer;
  /** gAssemble[state parity][R parity][first] */
  private gAssemble: GPUBindGroup[][][] = [];
  /** gAdvect[state parity][R parity] */
  private gAdvect: GPUBindGroup[][] = [];
  /** gSweep[state parity][R parity][first] */
  private gSweep: GPUBindGroup[][][] = [];
  /** gCommit[state parity][G parity] */
  private gCommit: GPUBindGroup[][] = [];
  /** gRegrid[R parity] */
  private gRegrid: GPUBindGroup[] = [];
  private red!: Reducer;
  private staging: GPUBuffer;
  private stagingBusy = false;

  private constructor(private o: AmrRadInit) {
    const d = o.device;
    this.R = [o.texture('radiation 0'), o.texture('radiation 1')];
    this.G = [o.texture('radiation gas 0'), o.texture('radiation gas 1')];
    this.RA = o.texture('radiation with ghosts');
    this.RK = o.texture('opacity with ghosts', 'r32float');
    this.info = [0, 1].map((first) => {
      const b = d.createBuffer({ label: 'radiation sweep info', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, new Uint32Array([first, 0, 0, 0]));
      return b;
    });
    this.staging = d.createBuffer({ size: (RAD_DIAG_OPS.length + ESCAPE_FLOATS) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.escaped = d.createBuffer({ label: 'radiation escaped', size: ESCAPE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  static async create(o: AmrRadInit, slots: Record<string, string>, comp: Record<number, GPUTextureView>): Promise<AmrRad> {
    const r = new AmrRad(o);
    await r.rebuild(slots, comp);
    return r;
  }

  /** (Re)build the pipelines, for new equation-of-state, opacity or composition slots. */
  async rebuild(slots: Record<string, string>, comp: Record<number, GPUTextureView>) {
    const { device: d, params, clock, state, bt, listLeaf, cfg } = this.o;
    const v = (t: GPUTexture) => t.createView();
    const red = Reducer.create(
      d, [this.o.cap * BS, cfg.ndim === 1 ? 1 : BS, cfg.ndim < 3 ? 1 : BS], RAD_DIAG_OPS, 'amr/rad_diagnostics.wgsl', slots,
      [0, 1].flatMap((p) => [0, 1].map((q) => ({ 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 3: v(this.R[q]), 30: bt, ...comp }))),
      'amr radiation diagnostics',
    );
    red.catch(() => {});
    const pipe = (file: string, label: string) => computePipeline(d, assemble(file, slots), label);
    [this.pAssemble, this.pAdvect, this.pSweep, this.pCommit, this.pRegrid, this.pEscape] = await Promise.all([
      pipe('amr/rad_assemble.wgsl', 'amr radiation assembly'),
      pipe('amr/rad_advect.wgsl', 'amr radiation advection'),
      pipe('amr/rad_sweep.wgsl', 'amr radiation sweep'),
      pipe('amr/rad_commit.wgsl', 'amr radiation commit'),
      pipe('amr/rad_regrid.wgsl', 'amr radiation regrid'),
      pipe('amr/rad_escape.wgsl', 'amr radiation escaped'),
    ]);
    this.red = await red;
    const tree = { 30: bt, 31: listLeaf };
    const both = <T>(f: (a: number, b: number) => T) => [0, 1].map((a) => [0, 1].map((b) => f(a, b)));
    const mf = { 7: this.o.massFlux.lo, ...((this.o.massFlux.hi ? { 32: this.o.massFlux.hi } : {}) as Record<number, Bindable>) };
    const ra = v(this.RA);
    const rk = v(this.RK);
    this.gAssemble = both((p, q) =>
      [0, 1].map((first) =>
        bindGroupUsed(d, this.pAssemble, { 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 3: v(this.R[q]), 4: v(this.G[q]), 5: ra, 6: rk, 8: this.info[first], ...tree, ...comp }),
      ),
    );
    this.gAdvect = both((p, q) =>
      bindGroupUsed(d, this.pAdvect, { 0: params, 1: clock, 2: v(this.o.prim), 3: v(state[p][1]), 5: ra, 6: v(this.R[1 - q]), ...mf, ...tree, ...comp }),
    );
    this.gSweep = both((p, q) =>
      [0, 1].map((first) =>
        bindGroupUsed(d, this.pSweep, {
          0: params, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 4: ra, 5: v(this.G[q]), 6: v(this.R[1 - q]), 7: v(this.G[1 - q]), 8: this.info[first], 9: rk, ...tree, ...comp,
        }),
      ),
    );
    this.gCommit = both((p, q) =>
      bindGroupUsed(d, this.pCommit, { 0: params, 2: v(state[p][0]), 3: v(state[p][1]), 5: v(this.G[q]), 7: v(state[1 - p][0]), 8: v(state[1 - p][1]), ...tree, ...comp }),
    );
    this.gEscape = both((p, q) => bindGroupUsed(d, this.pEscape, { 0: params, 1: clock, 2: v(state[p][1]), 3: v(this.R[q]), 4: this.escaped, ...tree }));
    this.gRegrid = [0, 1].map((q) => bindGroupUsed(d, this.pRegrid, { 0: params, 2: v(this.R[q]), 3: ra, 6: v(this.R[1 - q]), 7: this.o.rg, ...tree }));
  }

  /**
   * The radiation and opacity of `leaves` leaves with their ghosts (RA, RK), from the radiation as
   * it is and the gas at parity p: its temperature the state's (`first`), or the sweeps' own.
   */
  encodeAssemble(pass: GPUComputePassEncoder, p: number, leaves: number, first = true) {
    pass.setPipeline(this.pAssemble);
    pass.setBindGroup(0, this.gAssemble[p][this.cur][first ? 1 : 0]);
    this.o.flat(pass, leaves, 2);
  }

  /**
   * Carry the radiation with the gas: the update took the state from parity p, whose primitives
   * the hydro's assembly left, and encodeAssemble(p) ran as the step started.
   */
  encodeAdvect(pass: GPUComputePassEncoder, p: number, leaves: number) {
    pass.setPipeline(this.pAdvect);
    pass.setBindGroup(0, this.gAdvect[p][this.cur]);
    this.o.interior(pass, leaves);
    this.cur = 1 - this.cur;
  }

  /** The sweeps of a step, on the gas at parity s, then the commit into 1 - s. */
  encodeSweeps(pass: GPUComputePassEncoder, s: number, leaves: number) {
    const n = radSweeps(this.o.cfg);
    for (let k = 0; k < n; k++) {
      this.encodeAssemble(pass, s, leaves, k === 0);
      pass.setPipeline(this.pSweep);
      pass.setBindGroup(0, this.gSweep[s][this.cur][k === 0 ? 1 : 0]);
      this.o.interior(pass, leaves);
      this.cur = 1 - this.cur;
    }
    pass.setPipeline(this.pCommit);
    pass.setBindGroup(0, this.gCommit[s][this.cur]);
    this.o.interior(pass, leaves);
    // What left the box over this step (amr/rad_escape.wgsl).
    pass.setPipeline(this.pEscape);
    pass.setBindGroup(0, this.gEscape[s][this.cur]);
    pass.dispatchWorkgroups(1);
  }

  /** Start the count of the energy that has left the box again (a new run). */
  resetEscaped() {
    this.o.device.queue.writeBuffer(this.escaped, 0, new Float32Array(ESCAPE_FLOATS));
  }

  /**
   * Re-meshing: the new tree's leaves from the old radiation. encodeAssemble must have run on the
   * old mesh, in an encoder submitted before the new tree was uploaded.
   */
  encodeRegrid(pass: GPUComputePassEncoder, leaves: number) {
    pass.setPipeline(this.pRegrid);
    pass.setBindGroup(0, this.gRegrid[this.cur]);
    this.o.interior(pass, leaves);
    this.cur = 1 - this.cur;
  }

  /** The texture holding the current radiation of the leaves' cells. */
  get current(): GPUTexture {
    return this.R[this.cur];
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
    } catch (e) {
      // Destroyed while the read was in flight (a new setup loaded): nothing to report. Uncaught,
      // the rejected map reached the page as an error (random test seed 10437, after 10436).
      if (this.destroyed) return null;
      throw e;
    } finally {
      this.stagingBusy = false;
    }
  }

  /** GPU bytes per block. */
  static blockBytes(ndim: 1 | 2 | 3): number {
    return PAD ** ndim * AmrRad.texelBytes;
  }

  private destroyed = false;

  destroy() {
    this.destroyed = true;
    for (const t of [...this.R, ...this.G, this.RA, this.RK]) t.destroy();
    for (const b of this.info) b.destroy();
    this.staging.destroy();
    this.escaped.destroy();
  }
}
