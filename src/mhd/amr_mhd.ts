// Ideal MHD on the block grid (docs/numerics.md section 21, Block grid): the van Leer + constrained
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
// (encodeSubstep(), docs/numerics.md section 21, Time bins).
import { domainLo } from '../domain';
import { assemble } from '../gpu/shaders';
import { bindGroup, bindGroupUsed, computePipeline, readTexture, type Bindable } from '../gpu/device';
import { Reducer } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { BS, type Block } from '../amr/tree';
import { MAG_OPS, magneticFrom, mhdRiemann, type MagDiagnostics } from './mhd';

/** Bytes the flux scratch may take, which sets how many leaves a chunk holds. */
let scratchBytes = 256 * 1024 ** 2;
/** Change it for runs built from now on (validation: many chunks on a small mesh); null restores the default. */
export function setMhdScratchBytes(b: number | null) {
  scratchBytes = b ?? 256 * 1024 ** 2;
}
const FBOX = BS + 2;
const FLAT_WG = 64;

/**
 * GPU bytes per atlas texel of a field's own textures: the face field and its compensation at
 * both parities, half-step faces, assembled faces and centered field, edge impulses (eight
 * rgba32float), and what stage 2's first-order flux correction keeps of stage 1's assembly (three
 * rgba32float: mhd/amr_keep.wgsl), and the flux correction's round-1 marks (one r32float:
 * mhd/amr_fofc.wgsl). The assembled primitives W1 and W2 are the
 * hydro atlas's, which a run with a field does not otherwise use.
 */
export const MHD_BLOCK_BYTES = 11 * 16 + 4;

/** Bytes of the flux register per slot: six faces of BS^2 cells (BS in 2D), two vec4 each (mhd/amr_block.wgsl, reg_entry). */
export const registerBytes = (ndim: 2 | 3) => 6 * (ndim === 3 ? BS * BS : BS) * 32;

/** The most bytes one buffer bound to a pass may have on this device. */
export const maxBinding = (d: GPUDevice) => Math.min(d.limits.maxBufferSize, d.limits.maxStorageBufferBindingSize);

interface Deps {
  device: GPUDevice;
  /** gravity: the field atlas (acceleration, potential), and the point masses */
  gfield: GPUTexture;
  bh: GPUBuffer;
  /** the hydro atlas's primitives W1 and W2, which the assembly writes (a run with a field has no other use for them) */
  prim: GPUTexture[];
  /** with species: the leaves' face mass fluxes the species pass reads, as two windows (amr/species_index.wgsl) */
  massFlux?: { lo: Bindable; hi: Bindable | null };
  cfg: SimConfig;
  cap: number;
  /** a new atlas texture the size of the state's */
  texture: (label: string, format?: GPUTextureFormat) => GPUTexture;
  params: GPUBuffer;
  clock: GPUBuffer;
  bt: GPUBuffer;
  listLeaf: GPUBuffer;
  idx: GPUTexture;
  /** the editing brush's uniform (edit/brush_core.wgsl), which the field brushes read */
  brush: GPUBuffer;
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
  /** The stage's assembly: faces, and primitives W1, W2 (the hydro atlas's) and the centered field. */
  private fa: GPUTexture;
  private w: GPUTexture[];
  /** What stage 2's first-order flux correction keeps of stage 1's assembly (mhd/amr_keep.wgsl). */
  private kept: GPUTexture[];
  /**
   * The assembled primitives (rho, v), as the hydro passes' W1 holds them: after a step the half
   * step's (the species predictor's velocities), and the display's.
   */
  get prim(): GPUTexture {
    return this.w[0];
  }
  /** The assembled primitives W1 and W2, for the refinement indicator and the cells' re-meshing. */
  get primitives(): GPUTexture[] {
    return [this.w[0], this.w[1]];
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
  /** The magnetic diagnostics' reduction result (MAG_OPS order), for the display's magnetic fields. */
  get magResult(): GPUBuffer {
    return this.redMag.result;
  }
  /** Reduce the magnetic diagnostics of the state at `parity` into magResult (the display's). */
  encodeMagnetic(pass: GPUComputePassEncoder, parity: number) {
    this.redMag.encode(pass, parity);
  }
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
  /** [parity] stage 1's assembly kept for stage 2 (mhd/amr_keep.wgsl): over every leaf, and over a substep's list */
  private pKeep?: GPUComputePipeline;
  private gKeep: GPUBindGroup[] = [];
  /** The field brushes (amr/field_brush.wgsl): faces, then energy; [parity] bind groups. */
  private pBrush: GPUComputePipeline[] = [];
  private gBrush: GPUBindGroup[][] = [];
  private gKeepL: GPUBindGroup[] = [];
  /**
   * The first-order flux correction after the update (mhd/amr_fofc.wgsl): the list of leaves to
   * correct (count, then slots), its chunks' dispatch sizes, the scan that fills it ([parity], over
   * every leaf and over a substep's), the empty, and stage 2's fluxes and update again over it
   * ([parity][chunk]).
   */
  private redoList: GPUBuffer;
  private redoArgs: GPUBuffer;
  /** round 1's dispatch sizes, which the recheck sets as it adds to the list (mhd/amr_fofc_recheck.wgsl) */
  private redoArgs1: GPUBuffer;
  /** round 1's marks, a float a texel (mhd/amr_fofc_recheck.wgsl) */
  private marks1: GPUTexture;
  /** the substep's stamp per slot of the leaves listed (mhd/amr_fofc_scan.wgsl) */
  private redoSeen: GPUBuffer;
  /** [round] */
  private pScan?: GPUComputePipeline;
  private pRecheck?: GPUComputePipeline;
  /** [parity][chunk]: between the rounds, the cells round 0 left all but emptied (mhd/amr_fofc_recheck.wgsl) */
  private gRecheck: GPUBindGroup[][] = [];
  private pScanReset?: GPUComputePipeline;
  private pFluxRedo?: GPUComputePipeline;
  private pUpdateRedo?: GPUComputePipeline;
  /** [parity][round] */
  private gScan: GPUBindGroup[] = [];
  private gScanL: GPUBindGroup[] = [];
  private gScanReset?: GPUBindGroup;
  private gFluxRedo: GPUBindGroup[][] = [];
  private gUpdateRedo: GPUBindGroup[][] = [];

  private constructor(private dep: Deps) {
    const d = dep.device;
    this.ndim = dep.cfg.ndim;
    const t = dep.texture;
    this.faces = [t('face field 0'), t('face field 1')];
    this.comps = [t('face compensation 0'), t('face compensation 1')];
    this.halfFaces = t('half-step face field');
    this.fa = t('assembled faces');
    this.w = [dep.prim[0], dep.prim[1], t('assembled centered field')];
    this.kept = [t('kept start: density, velocity'), t('kept start: pressure, scalar, b_x, b_y'), t('kept start: b_z, centered field')];
    this.emf = t('edge impulses');
    this.marks1 = t('flux correction marks, round 1', 'r32float');
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
    this.redoList = d.createBuffer({ label: 'MHD flux correction list', size: 4 * (dep.cap + 1), usage: GPUBufferUsage.STORAGE });
    this.redoSeen = d.createBuffer({ label: 'MHD flux correction leaves listed', size: 4 * dep.cap, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.redoArgs = d.createBuffer({ label: 'MHD flux correction dispatch sizes', size: 24 * this.chunkBufs.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    // Per chunk, (boxes of the fluxes, rows, 1) and (boxes of the update, rows, 1); the scan sets
    // the rows (mhd/amr_fofc_scan.wgsl).
    const template = new Uint32Array(6 * this.chunkBufs.length);
    for (let i = 0; i < this.chunkBufs.length; i++) template.set([this.groups(FBOX), 0, 1, this.groups(BS + 1), 0, 1], 6 * i);
    d.queue.writeBuffer(this.redoArgs, 0, template);
    this.redoArgs1 = d.createBuffer({ label: 'MHD flux correction dispatch sizes, round 1', size: 24 * this.chunkBufs.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.redoArgs1, 0, template);
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
    const [rgp, enp, a1, a2, f1, f2, e1, e2, u1, u2, red, cm, ca, kp, bf, be, f2r, u2r, scan, scanReset, recheck] = await Promise.all([
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
        MAG_OPS,
        'mhd/amr_diagnostics.wgsl',
        still,
        [0, 1].map((p) => ({ 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 4: v(this.faces[p]), 5: this.dep.bh, 30: bt })),
        'magnetic diagnostics (blocks)',
      ),
      computePipeline(d, assemble('mhd/amr_commit.wgsl'), 'MHD commit faces'),
      computePipeline(d, assemble('mhd/amr_chunk_args.wgsl'), 'MHD chunk dispatch sizes', {
        PER_CHUNK: this.perChunk, CHUNKS: this.chunkBufs.length, X_FLUX: this.groups(FBOX), X_BOX: this.groups(BS + 1),
      }),
      computePipeline(d, assemble('mhd/amr_keep.wgsl', still), 'MHD keep the start'),
      computePipeline(d, assemble('amr/field_brush.wgsl', still), 'field brush faces (blocks)', undefined, 'faces'),
      computePipeline(d, assemble('amr/field_brush.wgsl', still), 'field brush energy (blocks)', undefined, 'energy'),
      computePipeline(d, assemble('mhd/amr_flux.wgsl', slots), 'MHD block fluxes, correction', { STAGE: 2, REDO: 1 }),
      computePipeline(d, assemble('mhd/amr_update.wgsl', slots), 'MHD block update, correction', { STAGE: 2, SPECIES_FLUX: this.dep.massFlux ? 1 : 0, REDO: 1 }),
      computePipeline(d, assemble('mhd/amr_fofc_scan.wgsl', slots), 'MHD flux correction scan', { PER_CHUNK: this.perChunk, CHUNKS: this.chunkBufs.length }),
      computePipeline(d, assemble('mhd/amr_fofc_scan.wgsl', slots), 'MHD flux correction list reset', { PER_CHUNK: this.perChunk, CHUNKS: this.chunkBufs.length }, 'reset_main'),
      computePipeline(d, assemble('mhd/amr_fofc_recheck.wgsl', slots), 'MHD flux correction recheck', { PER_CHUNK: this.perChunk }),
    ]);
    this.pFluxRedo = f2r;
    this.pUpdateRedo = u2r;
    this.pScan = scan;
    this.pRecheck = recheck;
    this.pScanReset = scanReset;
    this.gScanReset = bindGroupUsed(d, scanReset, { 36: this.redoList, 37: this.redoArgs, 40: this.redoArgs1 }, 'MHD flux correction list reset');
    this.pBrush = [bf, be];
    this.gBrush = [0, 1].map((p) => {
      const o = 1 - p;
      return [
        bindGroup(d, bf, { 0: params, 1: this.dep.brush, 30: bt, 31: listLeaf, 11: v(idx), 2: v(this.faces[p]), 3: v(this.comps[p]), 4: v(this.faces[o]), 5: v(this.comps[o]) }),
        bindGroup(d, be, {
          0: params, 31: listLeaf, 2: v(this.faces[p]), 6: v(this.faces[o]), 7: v(state[p][0]), 8: v(state[p][1]), 9: v(state[o][0]), 10: v(state[o][1]),
        }),
      ];
    });
    this.pCommit = cm;
    this.pKeep = kp;
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
    this.gKeep = [];
    this.gKeepL = [];
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
    this.gScan = [];
    this.gScanL = [];
    this.gRecheck = [];
    this.gFluxRedo = [];
    this.gUpdateRedo = [];
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
      const out = { 3: v(this.w[0]), 4: v(this.w[1]), 5: v(this.w[2]), 6: v(this.fa), 12: rates, 28: v(idx), 19: v(this.dep.gfield), 20: this.dep.bh };
      // Both stages read the state and the half-step stores; ASM_STAGE weighs them.
      const stores = { 1: v(state[p][0]), 2: v(state[p][1]), 9: v(this.faces[p]), 16: v(this.halfFaces), 17: v(half[0]), 18: v(half[1]) };
      const W = { 2: v(this.w[0]), 3: v(this.w[1]), 4: v(this.w[2]) };
      // What stage 1's assembly keeps for stage 2's first-order flux correction (both stages bind it).
      const start = { 8: v(this.kept[0]), 9: v(this.kept[1]), 10: v(this.kept[2]) };
      // The step's bind groups over a list of leaves, per stage (and per chunk, but the assembly's).
      const groups = (list: GPUBuffer, chunks: GPUBuffer[]) => {
        const c0 = { ...common, 31: list };
        const perChunk = (fn: (c: GPUBuffer) => GPUBindGroup) => chunks.map(fn);
        // Stage 1 writes the half-step state; its compensation output (13) is unused and takes the
        // other parity's, which stage 2 then overwrites.
        // The flux correction's marks (35, mhd/amr_fofc.wgsl), which only its redo reads, take the
        // new faces where the pass does not write them, and otherwise a texture it leaves alone.
        const upd = (pipe: GPUComputePipeline, outs: Record<number, GPUTextureView>, marks: GPUTexture) =>
          perChunk((c) =>
            bindGroup(d, pipe, {
              ...c0, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 4: v(this.faces[p]), 5: v(this.comps[p]),
              6: this.scratch, 7: this.register, 8: v(this.emf), ...outs, 13: v(this.comps[o]), 14: c,
              15: v(this.w[0]), 16: v(this.w[1]), 17: v(this.w[2]), 28: v(idx), 19: v(this.dep.gfield), 20: this.dep.bh,
              12: this.dep.massFlux?.lo ?? this.noFlux(),
              ...((this.dep.massFlux?.hi ? { 32: this.dep.massFlux.hi } : {}) as Record<number, Bindable>),
              35: v(marks), 39: v(this.marks1),
            }),
          );
        return {
          assemble: [bindGroup(d, a1, { ...c0, ...out, ...stores }), bindGroup(d, a2, { ...c0, ...out, ...stores })],
          keep: bindGroup(d, kp, { 0: params, 31: list, 2: v(this.w[0]), 3: v(this.w[1]), 4: v(this.fa), 5: v(this.kept[0]), 6: v(this.kept[1]), 7: v(this.kept[2]), 8: v(this.w[2]) }),
          flux: [f1, f2].map((pipe) =>
            perChunk((c) => bindGroup(d, pipe, { ...c0, 1: clock, ...W, 5: v(this.fa), ...start, 6: this.scratch, 7: this.register, 14: c, 28: v(idx), 35: v(this.faces[o]), 39: v(this.marks1) })),
          ),
          emf: [e1, e2].map((pipe) => perChunk((c) => bindGroup(d, pipe, { ...c0, 1: clock, ...W, 5: v(this.fa), 6: this.scratch, 8: v(this.emf), 14: c }))),
          update: [
            upd(u1, { 9: v(half[0]), 10: v(half[1]), 11: v(this.halfFaces) }, this.faces[o]),
            upd(u2, { 9: v(state[o][0]), 10: v(state[o][1]), 11: v(this.faces[o]) }, this.halfFaces),
          ],
          // The flux correction's passes over its list (mhd/amr_fofc.wgsl): stage 2's fluxes, and
          // its update writing the cells alone (its faces output (11) a texture it leaves alone),
          // both reading the marks in the new faces.
          fluxRedo: perChunk((c) =>
            bindGroup(d, f2r, { ...c0, 1: clock, ...W, 5: v(this.fa), ...start, 6: this.scratch, 7: this.register, 14: c, 28: v(idx), 35: v(this.faces[o]), 39: v(this.marks1) }),
          ),
          updateRedo: upd(u2r, { 9: v(state[o][0]), 10: v(state[o][1]), 11: v(this.halfFaces) }, this.faces[o]),
          // (Not over its own list, which it fills.)
          scan: list === this.redoList ? undefined : bindGroupUsed(d, scan, { 0: params, 1: clock, 30: bt, 31: list, 28: v(idx), 35: v(this.faces[o]), 36: this.redoList, 37: this.redoArgs, 38: this.redoSeen, 39: v(this.marks1), ...bin }, 'MHD flux correction scan'),
          // Over its own list: what round 0 left, against the start.
          recheck: perChunk((c) =>
            bindGroupUsed(d, recheck, {
              0: params, 1: clock, 30: bt, 14: c, 2: v(state[p][0]), 9: v(state[o][0]), 11: v(this.marks1), 28: v(idx),
              36: this.redoList, 37: this.redoArgs1, 38: this.redoSeen, ...bin,
            }, 'MHD flux correction recheck'),
          ),
        };
      };
      const all = groups(listLeaf, this.chunkBufs);
      const redo = groups(this.redoList, this.chunkBufs);
      this.gFluxRedo.push(redo.fluxRedo);
      this.gRecheck.push(redo.recheck);
      this.gUpdateRedo.push(redo.updateRedo);
      this.gScan.push(all.scan!);
      this.gAssemble.push(all.assemble);
      this.gKeep.push(all.keep);
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
      this.gKeepL.push(sub.keep);
      this.gFluxL.push(sub.flux);
      this.gEmfL.push(sub.emf);
      this.gUpdateL.push(sub.update);
      this.gScanL.push(sub.scan!);
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

  /**
   * Without species, the update's mass-flux binding (MF): a few words that nothing reads.
   */
  private placeholderFlux?: GPUBuffer;
  private noFlux(): GPUBuffer {
    this.placeholderFlux ??= this.dep.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return this.placeholderFlux;
  }

  /**
   * A field brush (amr/field_brush.wgsl, the Brush uniform already written): every leaf's faces
   * from `parity` into the other, changed inside the brush, then the leaves' cells, their energy
   * changed with the field's. `interior` dispatches a pass over the leaves' interior cells.
   */
  encodeFieldBrush(pass: GPUComputePassEncoder, parity: number, interior: (pass: GPUComputePassEncoder) => void) {
    pass.setPipeline(this.pBrush[0]);
    pass.setBindGroup(0, this.gBrush[parity][0]);
    pass.dispatchWorkgroups(this.groups(BS + 1), this.nLeaf);
    pass.setPipeline(this.pBrush[1]);
    pass.setBindGroup(0, this.gBrush[parity][1]);
    interior(pass);
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
      if (s === 0) {
        pass.setPipeline(this.pKeep!);
        pass.setBindGroup(0, this.gKeepL[parity]);
        pass.dispatchWorkgroupsIndirect(args, ARG(ARG_ASSEMBLE));
      }
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
    // Before the deposits, which then take the corrected register.
    this.encodeCorrection(pass, parity, this.gScanL[parity]);
    pass.setPipeline(this.pUpdate[1]);
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
      if (s === 0) {
        // Kept for stage 2's first-order flux correction, before stage 2's assembly overwrites it.
        pass.setPipeline(this.pKeep!);
        pass.setBindGroup(0, this.gKeep[parity]);
        assembleDispatch();
        timestep();
      }
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
    this.encodeCorrection(pass, parity, this.gScan[parity]);
  }

  /**
   * The first-order flux correction after stage 2's update (mhd/amr_fofc.wgsl): list the leaves
   * of the step (`scan`'s list) with a cell the update all but emptied or a neighbor with one,
   * then stage 2's fluxes and update again over that list, chunk by chunk, and empty it. When
   * nothing is listed the fluxes and update dispatch no workgroups.
   */
  private encodeCorrection(pass: GPUComputePassEncoder, parity: number, scan: GPUBindGroup) {
    const n = this.chunks();
    pass.setPipeline(this.pScan!);
    pass.setBindGroup(0, scan);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(this.nLeaf / 64)));
    // Round 0 over the scan's list, sized by redoArgs; round 1 over the list as the recheck left it,
    // sized by redoArgs1.
    for (let r = 0; r < 2; r++) {
      const args = r === 0 ? this.redoArgs : this.redoArgs1;
      const fluxes = (i: number) => {
        pass.setPipeline(this.pFluxRedo!);
        pass.setBindGroup(0, this.gFluxRedo[parity][i]);
        pass.dispatchWorkgroupsIndirect(args, 24 * i);
      };
      // As the step does: with several chunks, every chunk's fluxes first (they correct the
      // register, which a coarser or slower neighbor in another chunk reads), then each chunk's
      // again before its update (the scratch holds one chunk).
      if (n > 1) for (let i = 0; i < n; i++) fluxes(i);
      for (let i = 0; i < n; i++) {
        fluxes(i);
        pass.setPipeline(this.pUpdateRedo!);
        pass.setBindGroup(0, this.gUpdateRedo[parity][i]);
        pass.dispatchWorkgroupsIndirect(args, 24 * i + 12);
      }
      if (r === 0) {
        // Between the rounds: mark what round 0 left all but emptied, and add to the list.
        for (let i = 0; i < n; i++) {
          pass.setPipeline(this.pRecheck!);
          pass.setBindGroup(0, this.gRecheck[parity][i]);
          pass.dispatchWorkgroupsIndirect(this.redoArgs, 24 * i + 12);
        }
      }
    }
    pass.setPipeline(this.pScanReset!);
    pass.setBindGroup(0, this.gScanReset!);
    pass.dispatchWorkgroups(1);
  }



  /**
   * The run starts again (its step count with it): forget which leaves the flux correction listed
   * at which step (mhd/amr_fofc_scan.wgsl), or a leaf of the new run could pass for listed.
   */
  forgetCorrection() {
    this.dep.device.queue.writeBuffer(this.redoSeen, 0, new Uint32Array(this.redoSeen.size / 4));
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
    const size = 4 * MAG_OPS.length;
    const buf = d.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.redMag.result, 0, buf, 0, size);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return magneticFrom(r, !!this.dep.cfg.problem.disk);
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
    // (w[0] and w[1] are the hydro atlas's, which the run destroys.)
    for (const x of [...this.faces, ...this.comps, this.halfFaces, this.fa, this.w[2], ...this.kept, this.emf, this.marks1]) x.destroy();
    this.scratch.destroy();
    this.btOld.destroy();
    this.idxOld.destroy();
    this.register.destroy();
    this.chunkArgs.destroy();
    this.redoList.destroy();
    this.redoSeen.destroy();
    this.redoArgs.destroy();
    this.redoArgs1.destroy();
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
  const dlo = domainLo(cfg);
  const at = (i: number, a: number) => (b.c[a] * BS + i) * h + dlo[a];
  if (spec.potential) {
    const A = spec.potential(cfg.params, cfg.ndim);
    // Line integrals of A_c along every edge of the block, each computed once: E[c] at point
    // (i, j, k) is the integral along c over one cell from that point (2D, c = z: the point
    // value). The texels' faces reach one point past the last texel, so N points per axis, and
    // an edge along c starts at most W points along it. Every face shares each edge's single
    // value, so the circulations add up around a cell exactly as the edges do.
    const N = W + 1;
    const NZ = flat ? 1 : N;
    const idx = (i: number, j: number, k: number) => (k * N + j) * N + i;
    const E = [0, 1, 2].map((c) => {
      const e = new Float64Array(N * N * NZ);
      const point = flat && c === 2;
      const ni = c === 0 ? W : N;
      const nj = c === 1 ? W : N;
      const nk = flat ? 1 : c === 2 ? W : N;
      for (let k = 0; k < nk; k++) {
        const z = flat ? 0 : at(k, 2);
        for (let j = 0; j < nj; j++) {
          const y = at(j, 1);
          for (let i = 0; i < ni; i++) {
            const x = at(i, 0);
            if (point) {
              e[idx(i, j, k)] = A(x, y, 0)[2];
              continue;
            }
            let s = 0;
            for (let m = 0; m < sub; m++) {
              const d = ((m + 0.5) * h) / sub;
              s += A(c === 0 ? x + d : x, c === 1 ? y + d : y, c === 2 ? z + d : z)[c];
            }
            e[idx(i, j, k)] = (s * h) / sub;
          }
        }
      }
      return e;
    });
    const [Ex, Ey, Ez] = E;
    const hz = flat ? 1 : h;
    for (let k = 0; k < D; k++)
      for (let j = 0; j < W; j++)
        for (let i = 0; i < W; i++) {
          const o = 4 * ((k * W + j) * W + i);
          const p = idx(i, j, k);
          const pi = idx(i + 1, j, k);
          const pj = idx(i, j + 1, k);
          const pk = flat ? p : idx(i, j, k + 1);
          // Circulation around each face, over its area (2D: per unit length in z).
          const dzAy = flat ? 0 : Ey[pk] - Ey[p];
          const dzAx = flat ? 0 : Ex[pk] - Ex[p];
          out[o] = (Ez[pj] - Ez[p] - dzAy) / (h * hz);
          out[o + 1] = (dzAx - (Ez[pi] - Ez[p])) / (h * hz);
          out[o + 2] = (Ey[pi] - Ey[p] - (Ex[pj] - Ex[p])) / (h * h);
        }
  }
  if (spec.face) {
    // With a potential too (a uniform field through a star's or a disk's own), the two add.
    const both = !!spec.potential;
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
          for (let a = 0; a < 3; a++) out[o + a] = both ? out[o + a] + mean(a, p) : mean(a, p);
        }
  } else if (!spec.potential) {
    throw new Error('MHD on the adaptive mesh needs a vector potential or face values');
  }
  return out;
}
