// Ideal MHD on the block grid (docs/mhd-plan.md, section 4b): the van Leer + constrained
// transport step of mhd/mhd.ts, block by block. Per stage:
//
//   assemble  every leaf's cells, centered field and faces, ghosts and edge ghosts included
//   fluxes    per chunk of leaves: interior fluxes to the scratch, boundary ones to the register
//   edges     per chunk: edge impulses into the edge atlas
//   update    per chunk (fluxes recomputed first when the leaves span several chunks): cells
//             and faces, with fine fluxes and fine edges taken at refinement boundaries
//
// Re-meshing moves the faces with the cells (encodeRegrid()). With time bins, a cycle's first
// substep steps every leaf (encodeFull()) and the others the leaves of the GPU's lists
// (encodeSubstep(), docs/mhd-plan.md section 5b).
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture } from '../gpu/device';
import { MAX, Reducer, SUM } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { BS, type Block } from '../amr/tree';
import { magneticFrom, mhdRiemann, type MagDiagnostics } from './mhd';

/** Bytes the flux scratch may take, which sets how many leaves a chunk holds. */
let scratchBytes = 256 * 1024 ** 2;
/** Change it for runs built from now on (validation: many chunks on a small mesh); null restores the default. */
export function setMhdScratchBytes(b: number | null) {
  scratchBytes = b ?? 256 * 1024 ** 2;
}
const FBOX = BS + 2;
const FLAT_WG = 64;

/**
 * Per atlas texel: the face field and its compensation at both parities, half-step faces, the
 * two stages' assemblies (faces, primitives and centered field: stage 2's first-order flux
 * correction reads stage 1's), edge impulses.
 */
export const MHD_BLOCK_TEXTURES = 12;

/** Bytes of the flux register per slot: six faces of BS^2 cells (BS in 2D), two vec4 each (mhd/amr_block.wgsl, reg_entry). */
export const registerBytes = (ndim: 2 | 3) => 6 * (ndim === 3 ? BS * BS : BS) * 32;

/** The most bytes one buffer bound to a pass may have on this device. */
export const maxBinding = (d: GPUDevice) => Math.min(d.limits.maxBufferSize, d.limits.maxStorageBufferBindingSize);

interface Deps {
  device: GPUDevice;
  /** gravity: the field atlas (acceleration, potential), and the point masses */
  gfield: GPUTexture;
  bh: GPUBuffer;
  /** with species: the leaves' face mass fluxes the species pass reads (amr/species.wgsl) */
  massFlux?: GPUBuffer;
  cfg: SimConfig;
  cap: number;
  /** a new atlas texture the size of the state's */
  texture: (label: string) => GPUTexture;
  params: GPUBuffer;
  clock: GPUBuffer;
  bt: GPUBuffer;
  listLeaf: GPUBuffer;
  idx: GPUTexture;
  /** the regrid table (amr/tree.ts RG_*) */
  rg: GPUBuffer;
  /** state[parity] = [U1, U2] */
  state: GPUTexture[][];
  /** the half-step conserved state (the hydro predictor's textures) */
  half: GPUTexture[];
  /** the timestep reduction's first-stage records, which the assembly writes */
  rates: GPUBuffer;
  slots: Record<string, string>;
  /** Per-block timesteps (amr/bins_on.wgsl): the bins, the lists of a substep and their dispatch sizes. */
  bins: BinsDeps;
}

/** What the passes of a substep of time bins work from (src/amr/amr_sim.ts, encodeCycle()). */
export interface BinsDeps {
  /** BIN: the substep and each block's bin */
  A: GPUBuffer;
  /** the leaves starting a step (LL), those updated (LU) and those taking a deposit (LD) */
  LL: GPUBuffer;
  LU: GPUBuffer;
  LD: GPUBuffer;
  /** indirect dispatch sizes (amr/bins_args.wgsl) */
  args: GPUBuffer;
}

/** Offsets of the field's dispatches in the indirect arguments (amr/bins_args.wgsl). */
const ARG = (i: number) => 12 * i;
const ARG_ASSEMBLE = 16;
const ARG_COMMIT = 17;
const ARG_DEPOSIT = 18;

export class AmrMhd {
  faces: GPUTexture[] = [];
  comps: GPUTexture[] = [];
  private halfFaces: GPUTexture;
  /** Stage 2's assembly (the half step): faces, and primitives W1, W2 and the centered field. */
  private fa: GPUTexture;
  private w: GPUTexture[];
  /**
   * Stage 1's (the start of the step, or the state as it is for encodeAssembly): kept apart, so
   * stage 2's first-order flux correction can read it (mhd/face_flux.wgsl).
   */
  private fas: GPUTexture;
  private ws: GPUTexture[];
  /**
   * The half step's assembled primitives (rho, v), as the hydro passes' W1 holds them: the
   * species predictor's velocities after the step, and the display's.
   */
  get prim(): GPUTexture {
    return this.w[0];
  }
  /** Stage 1's assembled primitives W1 and W2 (encodeAssembly's), for the refinement indicator and the cells' re-meshing. */
  get primitives(): GPUTexture[] {
    return [this.ws[0], this.ws[1]];
  }
  /** The old tree, copied before a re-mesh replaces it. */
  private btOld: GPUBuffer;
  private idxOld: GPUTexture;
  private pRegrid!: GPUComputePipeline;
  private pEnergy!: GPUComputePipeline;
  private gRegrid: GPUBindGroup[] = [];
  private gEnergy: GPUBindGroup[] = [];
  private emf: GPUTexture;
  private scratch: GPUBuffer;
  private register: GPUBuffer;
  private chunkBufs: GPUBuffer[] = [];
  /** leaves per chunk */
  private perChunk: number;
  private nLeaf = 0;
  private pAssemble: GPUComputePipeline[] = [];
  private pFlux: GPUComputePipeline[] = [];
  private pEmf: GPUComputePipeline[] = [];
  private pUpdate: GPUComputePipeline[] = [];
  /** [parity][stage] for the assembly; [parity][stage][chunk] for the others */
  private gAssemble: GPUBindGroup[][] = [];
  private gFlux: GPUBindGroup[][][] = [];
  private gEmf: GPUBindGroup[][][] = [];
  private gUpdate: GPUBindGroup[][][] = [];
  private redMag!: Reducer;
  private ndim: 2 | 3;
  /** With time bins: the passes built to read them, and their bind groups over the substep's lists. */
  private binsOn = false;
  private pCommit?: GPUComputePipeline;
  /** the dispatch sizes of a substep's fluxes, edges and update, per chunk (mhd/amr_chunk_args.wgsl) */
  private pChunkArgs?: GPUComputePipeline;
  private gChunkArgs?: GPUBindGroup;
  private chunkArgs: GPUBuffer;
  /** a chunk that covers any list (its length bounds it) */
  private allChunk: GPUBuffer;
  /** [parity][stage] over LL */
  private gAssembleL: GPUBindGroup[][] = [];
  /** [parity][stage][chunk] over LL */
  private gFluxL: GPUBindGroup[][][] = [];
  private gEmfL: GPUBindGroup[][][] = [];
  private gUpdateL: GPUBindGroup[][][] = [];
  /** [parity] over LD */
  private gDeposit: GPUBindGroup[] = [];
  /** [parity] faces back from the other parity: over LU, and over every leaf */
  private gCommitU: GPUBindGroup[] = [];
  private gCommitAll: GPUBindGroup[] = [];

  private constructor(private dep: Deps) {
    const d = dep.device;
    this.ndim = dep.cfg.ndim;
    const t = dep.texture;
    this.faces = [t('face field 0'), t('face field 1')];
    this.comps = [t('face compensation 0'), t('face compensation 1')];
    this.halfFaces = t('half-step face field');
    this.fa = t('assembled faces');
    this.w = [t('MHD W1'), t('MHD W2'), t('assembled centered field')];
    this.fas = t('assembled faces at the start');
    this.ws = [t('MHD W1 at the start'), t('MHD W2 at the start'), t('assembled centered field at the start')];
    this.emf = t('edge impulses');
    const perBlock = (this.ndim === 3 ? 3 * FBOX ** 3 : 2 * FBOX ** 2) * 48;
    this.perChunk = Math.max(1, Math.min(dep.cap, Math.floor(Math.min(scratchBytes, maxBinding(d)) / perBlock), 65535));
    this.scratch = d.createBuffer({ label: 'MHD flux scratch', size: this.perChunk * perBlock, usage: GPUBufferUsage.STORAGE });
    this.register = d.createBuffer({ label: 'MHD flux register', size: dep.cap * registerBytes(this.ndim), usage: GPUBufferUsage.STORAGE });
    this.btOld = d.createBuffer({ label: 'old block table', size: dep.bt.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.idxOld = d.createTexture({
      label: 'old leaf index', size: [dep.idx.width, dep.idx.height, dep.idx.depthOrArrayLayers], dimension: '3d', format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.allChunk = d.createBuffer({ label: 'MHD whole list', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.allChunk, 0, new Uint32Array([0, 0xffffffff, 0, 0]));
    for (let i = 0; i * this.perChunk < dep.cap; i++) {
      this.chunkBufs.push(d.createBuffer({ label: `MHD chunk ${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    this.chunkArgs = d.createBuffer({ label: 'MHD chunk dispatch sizes', size: 24 * this.chunkBufs.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
  }

  /** The W textures are the hydro primitives' (W1, W2) when given, so the display can share them. */
  static async create(dep: Deps): Promise<AmrMhd> {
    const m = new AmrMhd(dep);
    await m.build();
    return m;
  }

  private chunks(): number {
    return Math.max(1, Math.ceil(this.nLeaf / this.perChunk));
  }

  /** Call when the leaf list changes. */
  setLeaves(n: number) {
    this.nLeaf = n;
    for (let i = 0; i < this.chunks(); i++) {
      const first = i * this.perChunk;
      this.dep.device.queue.writeBuffer(this.chunkBufs[i], 0, new Uint32Array([first, Math.min(this.perChunk, n - first), 0, 0]));
    }
  }

  /**
   * Build the step's passes with time bins on or off (the run's cfg.timeBins changed), and with
   * the run's configuration and hydro slots as they are now (a panel setting such as viscosity
   * or the Riemann solver may have changed them). On, the passes read each block's bin, and a
   * cycle steps through encodeFull() and encodeSubstep().
   */
  async setBins(on: boolean, cfg: SimConfig, slots: Record<string, string>) {
    const key = (c: SimConfig, s: Record<string, string>) => JSON.stringify([mhdRiemann(c), { ...s, bins: '', mid: '' }]);
    if (on === this.binsOn && key(cfg, slots) === key(this.dep.cfg, this.dep.slots)) return;
    this.binsOn = on;
    this.dep.cfg = cfg;
    this.dep.slots = slots;
    await this.build();
  }

  private async build() {
    const { device: d, cfg, params, clock, bt, listLeaf, idx, state, half, rates } = this.dep;
    const B = this.dep.bins;
    const on = this.binsOn;
    // Re-meshing and the diagnostics never read the bins.
    const still = { ...this.dep.slots, mhd_riemann: mhdRiemann(cfg), bins: 'amr/bins_off.wgsl', mid: 'amr/mid_off.wgsl' };
    const slots = { ...still, bins: on ? 'amr/bins_on.wgsl' : 'amr/bins_off.wgsl' };
    const v = (x: GPUTexture) => x.createView();
    const [rgp, enp, a1, a2, f1, f2, e1, e2, u1, u2, red, cm, ca] = await Promise.all([
      computePipeline(d, assemble('mhd/amr_regrid.wgsl', still), 'MHD re-mesh faces'),
      computePipeline(d, assemble('mhd/amr_regrid_energy.wgsl', still), 'MHD re-mesh energy'),
      computePipeline(d, assemble('mhd/amr_assemble.wgsl', slots), 'MHD block assembly 1', { RATES_ON: 1, ASM_STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_assemble.wgsl', slots), 'MHD block assembly 2', { RATES_ON: 0, ASM_STAGE: 2 }),
      computePipeline(d, assemble('mhd/amr_flux.wgsl', slots), 'MHD block fluxes 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_flux.wgsl', slots), 'MHD block fluxes 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/amr_emf.wgsl', slots), 'MHD block edges 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_emf.wgsl', slots), 'MHD block edges 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/amr_update.wgsl', slots), 'MHD block update 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_update.wgsl', slots), 'MHD block update 2', { STAGE: 2, SPECIES_FLUX: this.dep.massFlux ? 1 : 0 }),
      Reducer.create(
        d,
        [this.dep.cap * BS, BS, this.ndim === 2 ? 1 : BS],
        [SUM, MAX, MAX, MAX],
        'mhd/amr_diagnostics.wgsl',
        still,
        [0, 1].map((p) => ({ 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 4: v(this.faces[p]), 30: bt })),
        'magnetic diagnostics (blocks)',
      ),
      computePipeline(d, assemble('mhd/amr_commit.wgsl'), 'MHD commit faces'),
      computePipeline(d, assemble('mhd/amr_chunk_args.wgsl'), 'MHD chunk dispatch sizes', {
        PER_CHUNK: this.perChunk, CHUNKS: this.chunkBufs.length, X_FLUX: this.groups(FBOX), X_BOX: this.groups(BS + 1),
      }),
    ]);
    this.pCommit = cm;
    this.pChunkArgs = ca;
    this.gChunkArgs = bindGroup(d, ca, { 1: B.LL, 2: this.chunkArgs });
    this.pRegrid = rgp;
    this.pEnergy = enp;
    this.pAssemble = [a1, a2];
    this.pFlux = [f1, f2];
    this.pEmf = [e1, e2];
    this.pUpdate = [u1, u2];
    this.redMag = red;
    const bin: Record<number, GPUBuffer> = on ? { 29: B.A } : {};
    const common = { 0: params, 30: bt, 31: listLeaf, ...bin };
    this.gAssemble = [];
    this.gFlux = [];
    this.gEmf = [];
    this.gUpdate = [];
    this.gRegrid = [];
    this.gEnergy = [];
    this.gAssembleL = [];
    this.gFluxL = [];
    this.gEmfL = [];
    this.gUpdateL = [];
    this.gDeposit = [];
    this.gCommitU = [];
    this.gCommitAll = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // Re-meshing from parity p: faces into o (with the old tree), then energy and all back into p.
      this.gRegrid.push(
        bindGroup(d, rgp, {
          0: params, 30: this.btOld, 31: listLeaf, 7: this.dep.rg, 28: v(this.idxOld), 16: v(this.halfFaces),
          9: v(this.faces[p]), 10: v(this.comps[p]), 11: v(this.faces[o]), 13: v(this.comps[o]),
        }),
      );
      this.gEnergy.push(
        bindGroup(d, enp, {
          0: params, 31: listLeaf, 7: this.dep.rg, 1: v(state[o][0]), 2: v(state[o][1]), 3: v(this.faces[o]), 4: v(this.comps[o]),
          5: v(state[p][0]), 6: v(state[p][1]), 8: v(this.faces[p]), 10: v(this.comps[p]),
        }),
      );
    }
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      // Each stage assembles into its own textures (stage 1 into ws and fas, stage 2 into w and fa).
      const shared = { 12: rates, 28: v(idx), 19: v(this.dep.gfield), 20: this.dep.bh };
      const out = [
        { 3: v(this.ws[0]), 4: v(this.ws[1]), 5: v(this.ws[2]), 6: v(this.fas), ...shared },
        { 3: v(this.w[0]), 4: v(this.w[1]), 5: v(this.w[2]), 6: v(this.fa), ...shared },
      ];
      // Both stages read the state and the half-step stores; ASM_STAGE weighs them.
      const stores = { 1: v(state[p][0]), 2: v(state[p][1]), 9: v(this.faces[p]), 16: v(this.halfFaces), 17: v(half[0]), 18: v(half[1]) };
      const Wst = [this.ws, this.w];
      const W = (s: number) => ({ 2: v(Wst[s][0]), 3: v(Wst[s][1]), 4: v(Wst[s][2]) });
      const FA = [this.fas, this.fa];
      // Stage 1's assembly again, for stage 2's first-order flux correction (both stages bind it).
      const start = { 8: v(this.ws[0]), 9: v(this.ws[1]), 10: v(this.ws[2]), 11: v(this.fas) };
      // The step's bind groups over a list of leaves, per stage (and per chunk, but the assembly's).
      const groups = (list: GPUBuffer, chunks: GPUBuffer[]) => {
        const c0 = { ...common, 31: list };
        const perChunk = (fn: (c: GPUBuffer) => GPUBindGroup) => chunks.map(fn);
        // Stage 1 writes the half-step state; its compensation output (13) is unused and takes the
        // other parity's, which stage 2 then overwrites.
        const upd = (pipe: GPUComputePipeline, outs: Record<number, GPUTextureView>, s: number) =>
          perChunk((c) =>
            bindGroup(d, pipe, {
              ...c0, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 4: v(this.faces[p]), 5: v(this.comps[p]),
              6: this.scratch, 7: this.register, 8: v(this.emf), ...outs, 13: v(this.comps[o]), 14: c,
              15: v(Wst[s][0]), 16: v(Wst[s][1]), 17: v(Wst[s][2]), 28: v(idx), 19: v(this.dep.gfield), 20: this.dep.bh,
              12: this.dep.massFlux ?? this.noFlux(),
            }),
          );
        return {
          assemble: [bindGroup(d, a1, { ...c0, ...out[0], ...stores }), bindGroup(d, a2, { ...c0, ...out[1], ...stores })],
          flux: [f1, f2].map((pipe, s) =>
            perChunk((c) => bindGroup(d, pipe, { ...c0, 1: clock, ...W(s), 5: v(FA[s]), ...start, 6: this.scratch, 7: this.register, 14: c })),
          ),
          emf: [e1, e2].map((pipe, s) => perChunk((c) => bindGroup(d, pipe, { ...c0, 1: clock, ...W(s), 6: this.scratch, 8: v(this.emf), 14: c }))),
          update: [
            upd(u1, { 9: v(half[0]), 10: v(half[1]), 11: v(this.halfFaces) }, 0),
            upd(u2, { 9: v(state[o][0]), 10: v(state[o][1]), 11: v(this.faces[o]) }, 1),
          ],
        };
      };
      const all = groups(listLeaf, this.chunkBufs);
      this.gAssemble.push(all.assemble);
      this.gFlux.push(all.flux);
      this.gEmf.push(all.emf);
      this.gUpdate.push(all.update);
      // Faces back from the other parity into this one, over every leaf: a cycle's first substep,
      // and the step after the second half kick has brought the cells back (amr/correct.wgsl).
      const commit = (list: GPUBuffer) =>
        bindGroup(d, cm, { 0: params, 31: list, 1: v(this.faces[o]), 2: v(this.comps[o]), 3: v(this.faces[p]), 4: v(this.comps[p]) });
      this.gCommitAll.push(commit(listLeaf));
      if (!on) continue;
      // A substep's leaves in the same chunks of rows, with dispatch sizes made on the GPU.
      const sub = groups(B.LL, this.chunkBufs);
      this.gAssembleL.push(sub.assemble);
      this.gFluxL.push(sub.flux);
      this.gEmfL.push(sub.emf);
      this.gUpdateL.push(sub.update);
      // The deposits read no fluxes of their own, only the register and the edge atlas.
      this.gDeposit.push(groups(B.LD, [this.allChunk]).update[1][0]);
      this.gCommitU.push(commit(B.LU));
    }
  }

  /**
   * With time bins, the first substep of a cycle: the step of every leaf over its own bin's
   * step, as encode() takes it; `timestep` sets the clock's dt and the bins between the rates
   * and the fluxes. Then the faces of every leaf back into `parity` (the cells are the caller's).
   */
  encodeFull(pass: GPUComputePassEncoder, parity: number, assembleDispatch: () => void, timestep: () => void) {
    this.encode(pass, parity, assembleDispatch, timestep);
    this.encodeCommitFaces(pass, parity);
  }

  /** Without species, the update's mass-flux binding: one word that nothing reads. */
  private placeholderFlux?: GPUBuffer;
  private noFlux(): GPUBuffer {
    this.placeholderFlux ??= this.dep.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return this.placeholderFlux;
  }

  /** Every leaf's faces (and compensations) from the other parity into `parity`. */
  encodeCommitFaces(pass: GPUComputePassEncoder, parity: number) {
    pass.setPipeline(this.pCommit!);
    pass.setBindGroup(0, this.gCommitAll[parity]);
    pass.dispatchWorkgroups(this.groups(BS + 1), this.nLeaf);
  }

  /**
   * A later substep of a cycle: the step of the leaves that start one (LL), then the deposits
   * of the leaves at their midpoint (LD), from the state at `parity` into the other; then the
   * faces of all of them (LU) back. The cells are the caller's to bring back.
   */
  encodeSubstep(pass: GPUComputePassEncoder, parity: number) {
    const args = this.dep.bins.args;
    const ca = this.chunkArgs;
    // Chunks of the leaf count bound those of the list; the ones past its end dispatch nothing.
    const n = this.chunks();
    pass.setPipeline(this.pChunkArgs!);
    pass.setBindGroup(0, this.gChunkArgs!);
    pass.dispatchWorkgroups(1);
    for (let s = 0; s < 2; s++) {
      pass.setPipeline(this.pAssemble[s]);
      pass.setBindGroup(0, this.gAssembleL[parity][s]);
      pass.dispatchWorkgroupsIndirect(args, ARG(ARG_ASSEMBLE));
      const fluxes = (i: number) => {
        pass.setPipeline(this.pFlux[s]);
        pass.setBindGroup(0, this.gFluxL[parity][s][i]);
        pass.dispatchWorkgroupsIndirect(ca, 24 * i);
      };
      for (let i = 0; i < n; i++) {
        fluxes(i);
        pass.setPipeline(this.pEmf[s]);
        pass.setBindGroup(0, this.gEmfL[parity][s][i]);
        pass.dispatchWorkgroupsIndirect(ca, 24 * i + 12);
      }
      for (let i = 0; i < n; i++) {
        if (n > 1) fluxes(i);
        pass.setPipeline(this.pUpdate[s]);
        pass.setBindGroup(0, this.gUpdateL[parity][s][i]);
        pass.dispatchWorkgroupsIndirect(ca, 24 * i + 12);
      }
    }
    pass.setBindGroup(0, this.gDeposit[parity]);
    pass.dispatchWorkgroupsIndirect(args, ARG(ARG_DEPOSIT));
    pass.setPipeline(this.pCommit!);
    pass.setBindGroup(0, this.gCommitU[parity]);
    pass.dispatchWorkgroupsIndirect(args, ARG(ARG_COMMIT));
  }

  /** Workgroups per block for a box of `w` texels per side. */
  private groups(w: number): number {
    return Math.ceil(w ** this.ndim / FLAT_WG);
  }

  /**
   * Encode a step from the state at `parity` into the other parity. `assembleDispatch` dispatches
   * the assembly over the leaves (amr/list.wgsl's flat layout, two ghost layers), and `timestep`
   * reduces its rates into the clock's dt.
   */
  encode(pass: GPUComputePassEncoder, parity: number, assembleDispatch: () => void, timestep: () => void) {
    const n = this.chunks();
    const sizeOf = (i: number) => Math.min(this.perChunk, this.nLeaf - i * this.perChunk);
    for (let s = 0; s < 2; s++) {
      pass.setPipeline(this.pAssemble[s]);
      pass.setBindGroup(0, this.gAssemble[parity][s]);
      assembleDispatch();
      if (s === 0) timestep();
      const fluxes = (i: number) => {
        pass.setPipeline(this.pFlux[s]);
        pass.setBindGroup(0, this.gFlux[parity][s][i]);
        pass.dispatchWorkgroups(this.groups(FBOX), sizeOf(i));
      };
      for (let i = 0; i < n; i++) {
        fluxes(i);
        pass.setPipeline(this.pEmf[s]);
        pass.setBindGroup(0, this.gEmf[parity][s][i]);
        pass.dispatchWorkgroups(this.groups(BS + 1), sizeOf(i));
      }
      for (let i = 0; i < n; i++) {
        // Several chunks share one scratch: the chunk's own fluxes are gone, compute them again.
        if (n > 1) fluxes(i);
        pass.setPipeline(this.pUpdate[s]);
        pass.setBindGroup(0, this.gUpdate[parity][s][i]);
        pass.dispatchWorkgroups(this.groups(BS + 1), sizeOf(i));
      }
    }
  }

  /** Copy the block table and leaf index, before a re-mesh uploads the new ones. */
  snapshotTree(enc: GPUCommandEncoder) {
    enc.copyBufferToBuffer(this.dep.bt, 0, this.btOld, 0, this.dep.bt.size);
    enc.copyTextureToTexture({ texture: this.dep.idx }, { texture: this.idxOld }, [this.dep.idx.width, this.dep.idx.height, this.dep.idx.depthOrArrayLayers]);
  }

  /**
   * After amr/regrid.wgsl has re-meshed the cells from parity p into the other: re-mesh the faces
   * likewise (mhd/amr_regrid.wgsl), then rebuild the energy and bring state and faces back into
   * parity p (mhd/amr_regrid_energy.wgsl). The run stays in parity p.
   */
  encodeRegrid(pass: GPUComputePassEncoder, p: number) {
    for (const [pipe, g] of [[this.pRegrid, this.gRegrid[p]], [this.pEnergy, this.gEnergy[p]]] as const) {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(this.groups(BS + 1), this.nLeaf);
    }
  }

  /** Encode the start-of-step assembly alone, which measures the timestep rates. */
  encodeAssembly(pass: GPUComputePassEncoder, parity: number, dispatch: () => void) {
    pass.setPipeline(this.pAssemble[0]);
    pass.setBindGroup(0, this.gAssemble[parity][0]);
    dispatch();
  }

  /** Magnetic energy and divergence of the state at `parity`. */
  async diagnostics(parity: number): Promise<MagDiagnostics> {
    const d = this.dep.device;
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.redMag.encode(pass, parity);
    pass.end();
    const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.redMag.result, 0, buf, 0, 16);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return magneticFrom(r);
  }

  /** Write a leaf's faces (blockFaces' layout) at the atlas texel `origin` of its cell (0, 0, 0), compensations zeroed. */
  writeFaces(origin: number[], faces: Float32Array) {
    const d = this.dep.device;
    const depth = this.ndim === 3 ? BS + 1 : 1;
    const layout = { bytesPerRow: (BS + 1) * 16, rowsPerImage: BS + 1 };
    const size = [BS + 1, BS + 1, depth];
    d.queue.writeTexture({ texture: this.faces[0], origin }, faces as Float32Array<ArrayBuffer>, layout, size);
    d.queue.writeTexture({ texture: this.comps[0], origin }, new Float32Array(faces.length), layout, size);
  }

  /** The whole face atlas at `parity` (validation). */
  readAtlas(parity: number, side: number, depth: number): Promise<Float32Array> {
    return readTexture(this.dep.device, this.faces[parity], side, 4, depth);
  }

  destroy() {
    for (const x of [...this.faces, ...this.comps, this.halfFaces, this.fa, ...this.w, this.fas, ...this.ws, this.emf]) x.destroy();
    this.scratch.destroy();
    this.btOld.destroy();
    this.idxOld.destroy();
    this.register.destroy();
    this.chunkArgs.destroy();
    this.allChunk.destroy();
    for (const b of this.chunkBufs) b.destroy();
  }
}

/**
 * The faces of leaf `b` from the problem's field, in a (BS + 1)^3 (2D: (BS + 1)^2) grid of
 * texels, as the atlas holds them: texel l has the lower faces of cell l. From a vector
 * potential, each face is the circulation of A around it over its area, with every edge's line
 * integral summed over the finest level's segments, so that a coarse face is exactly the mean of
 * the fine faces covering it. Direct face values are averaged over the finest level's
 * sub-faces likewise.
 */
export function blockFaces(cfg: SimConfig, b: Block, dx0: number, maxLevel: number): Float32Array {
  const spec = cfg.problem.mhd!;
  if (spec.block) return spec.block(b.level, b.c);
  const flat = cfg.ndim === 2;
  const W = BS + 1;
  const D = flat ? 1 : W;
  const out = new Float32Array(W * W * D * 4);
  const h = dx0 / (1 << b.level);
  const sub = 1 << (maxLevel - b.level);
  const half = cfg.problem.box / 2;
  const at = (i: number, a: number) => (b.c[a] * BS + i) * h - half;
  if (spec.potential) {
    const A = spec.potential(cfg.params, cfg.ndim);
    // Line integral of A_c along the edge from p over one cell along c (2D, c = z: the point value).
    const I = (c: number, p: number[]) => {
      if (flat && c === 2) return A(p[0], p[1], 0)[2];
      let s = 0;
      for (let m = 0; m < sub; m++) {
        const q = [...p];
        q[c] += ((m + 0.5) * h) / sub;
        s += A(q[0], q[1], flat ? 0 : q[2])[c];
      }
      return (s * h) / sub;
    };
    const hz = flat ? 1 : h;
    for (let k = 0; k < D; k++)
      for (let j = 0; j < W; j++)
        for (let i = 0; i < W; i++) {
          const o = 4 * ((k * W + j) * W + i);
          const p = [at(i, 0), at(j, 1), flat ? 0 : at(k, 2)];
          const off = (a: number) => {
            const q = [...p];
            q[a] += h;
            return q;
          };
          // Circulation around each face, over its area (2D: per unit length in z).
          const dzAy = flat ? 0 : I(1, off(2)) - I(1, p);
          const dzAx = flat ? 0 : I(0, off(2)) - I(0, p);
          out[o] = (I(2, off(1)) - I(2, p) - dzAy) / (h * hz);
          out[o + 1] = (dzAx - (I(2, off(0)) - I(2, p))) / (h * hz);
          out[o + 2] = (I(1, off(0)) - I(1, p) - (I(0, off(1)) - I(0, p))) / (h * h);
        }
  } else if (spec.face) {
    const F = spec.face(cfg.params, cfg.ndim);
    const hs = h / sub;
    // Mean of the face's value over sub x sub points of it (sub along a 2D face's one dimension).
    const mean = (a: number, p: number[]) => {
      const t1 = (a + 1) % 3;
      const t2 = (a + 2) % 3;
      const n1 = flat && t1 === 2 ? 1 : sub;
      const n2 = flat && t2 === 2 ? 1 : sub;
      let s = 0;
      for (let v = 0; v < n2; v++)
        for (let u = 0; u < n1; u++) {
          const q = [...p];
          if (n1 > 1 || !(flat && t1 === 2)) q[t1] += (u + 0.5) * hs;
          if (n2 > 1 || !(flat && t2 === 2)) q[t2] += (v + 0.5) * hs;
          s += F(a, q[0], q[1], flat ? 0 : q[2]);
        }
      return s / (n1 * n2);
    };
    for (let k = 0; k < D; k++)
      for (let j = 0; j < W; j++)
        for (let i = 0; i < W; i++) {
          const o = 4 * ((k * W + j) * W + i);
          const p = [at(i, 0), at(j, 1), flat ? 0 : at(k, 2)];
          for (let a = 0; a < 3; a++) out[o + a] = mean(a, p);
        }
  } else {
    throw new Error('MHD on the adaptive mesh needs a vector potential or face values');
  }
  return out;
}
