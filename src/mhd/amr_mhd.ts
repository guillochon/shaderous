// Ideal MHD on the block grid (docs/mhd-plan.md, section 4b): the van Leer + constrained
// transport step of mhd/mhd.ts, block by block. Per stage:
//
//   assemble  every leaf's cells, centered field and faces, ghosts and edge ghosts included
//   fluxes    per chunk of leaves: interior fluxes to the scratch, boundary ones to the register
//   edges     per chunk: edge impulses into the edge atlas
//   update    per chunk (fluxes recomputed first when the leaves span several chunks): cells
//             and faces, with fine fluxes and fine edges taken at refinement boundaries
//
// The mesh stays as it was built (no regridding with a field until phase 3), and every block
// takes the same timestep (time bins come later in phase 2).
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture } from '../gpu/device';
import { MAX, Reducer, SUM } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { BS, type Block } from '../amr/tree';
import { mhdRiemann, type MagDiagnostics } from './mhd';

/** Bytes the flux scratch may take, which sets how many leaves a chunk holds. */
let scratchBytes = 256 * 1024 ** 2;
/** Change it for runs built from now on (validation: many chunks on a small mesh); null restores the default. */
export function setMhdScratchBytes(b: number | null) {
  scratchBytes = b ?? 256 * 1024 ** 2;
}
const FBOX = BS + 2;
const FLAT_WG = 64;

/** Per atlas texel: the face field and its compensation at both parities, half-step faces, assembled faces and field, edge impulses. */
export const MHD_BLOCK_TEXTURES = 8;

interface Deps {
  device: GPUDevice;
  cfg: SimConfig;
  cap: number;
  /** a new atlas texture the size of the state's */
  texture: (label: string) => GPUTexture;
  params: GPUBuffer;
  clock: GPUBuffer;
  bt: GPUBuffer;
  listLeaf: GPUBuffer;
  idx: GPUTexture;
  /** state[parity] = [U1, U2] */
  state: GPUTexture[][];
  /** the half-step conserved state (the hydro predictor's textures) */
  half: GPUTexture[];
  /** the timestep reduction's first-stage records, which the assembly writes */
  rates: GPUBuffer;
  slots: Record<string, string>;
}

export class AmrMhd {
  faces: GPUTexture[] = [];
  comps: GPUTexture[] = [];
  private halfFaces: GPUTexture;
  private fa: GPUTexture;
  private w: GPUTexture[];
  /** The assembled primitives (rho, v), as the hydro passes' W1 holds them, for the display. */
  get prim(): GPUTexture {
    return this.w[0];
  }
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

  private constructor(private dep: Deps) {
    const d = dep.device;
    this.ndim = dep.cfg.ndim;
    const t = dep.texture;
    this.faces = [t('face field 0'), t('face field 1')];
    this.comps = [t('face compensation 0'), t('face compensation 1')];
    this.halfFaces = t('half-step face field');
    this.fa = t('assembled faces');
    this.w = [t('MHD W1'), t('MHD W2'), t('assembled centered field')];
    this.emf = t('edge impulses');
    const perBlock = (this.ndim === 3 ? 3 * FBOX ** 3 : 2 * FBOX ** 2) * 48;
    this.perChunk = Math.max(1, Math.min(dep.cap, Math.floor(scratchBytes / perBlock), 65535));
    this.scratch = d.createBuffer({ label: 'MHD flux scratch', size: this.perChunk * perBlock, usage: GPUBufferUsage.STORAGE });
    this.register = d.createBuffer({ label: 'MHD flux register', size: dep.cap * 6 * BS * BS * 32, usage: GPUBufferUsage.STORAGE });
    for (let i = 0; i * this.perChunk < dep.cap; i++) {
      this.chunkBufs.push(d.createBuffer({ label: `MHD chunk ${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
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

  private async build() {
    const { device: d, cfg, params, clock, bt, listLeaf, idx, state, half, rates } = this.dep;
    const slots = { ...this.dep.slots, mhd_riemann: mhdRiemann(cfg), bins: 'amr/bins_off.wgsl', mid: 'amr/mid_off.wgsl' };
    const v = (x: GPUTexture) => x.createView();
    const [a1, a2, f1, f2, e1, e2, u1, u2, red] = await Promise.all([
      computePipeline(d, assemble('mhd/amr_assemble.wgsl', slots), 'MHD block assembly 1', { RATES_ON: 1 }),
      computePipeline(d, assemble('mhd/amr_assemble.wgsl', slots), 'MHD block assembly 2', { RATES_ON: 0 }),
      computePipeline(d, assemble('mhd/amr_flux.wgsl', slots), 'MHD block fluxes 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_flux.wgsl', slots), 'MHD block fluxes 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/amr_emf.wgsl', slots), 'MHD block edges 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_emf.wgsl', slots), 'MHD block edges 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/amr_update.wgsl', slots), 'MHD block update 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/amr_update.wgsl', slots), 'MHD block update 2', { STAGE: 2 }),
      Reducer.create(
        d,
        [this.dep.cap * BS, BS, this.ndim === 2 ? 1 : BS],
        [SUM, MAX, MAX],
        'mhd/amr_diagnostics.wgsl',
        slots,
        [0, 1].map((p) => ({ 0: params, 4: v(this.faces[p]), 30: bt })),
        'magnetic diagnostics (blocks)',
      ),
    ]);
    this.pAssemble = [a1, a2];
    this.pFlux = [f1, f2];
    this.pEmf = [e1, e2];
    this.pUpdate = [u1, u2];
    this.redMag = red;
    const W = { 2: v(this.w[0]), 3: v(this.w[1]), 4: v(this.w[2]) };
    const common = { 0: params, 30: bt, 31: listLeaf };
    this.gAssemble = [];
    this.gFlux = [];
    this.gEmf = [];
    this.gUpdate = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      const out = { 3: v(this.w[0]), 4: v(this.w[1]), 5: v(this.w[2]), 6: v(this.fa), 12: rates, 28: v(idx) };
      this.gAssemble.push([
        bindGroup(d, a1, { ...common, ...out, 1: v(state[p][0]), 2: v(state[p][1]), 9: v(this.faces[p]) }),
        bindGroup(d, a2, { ...common, ...out, 1: v(half[0]), 2: v(half[1]), 9: v(this.halfFaces) }),
      ]);
      const perChunk = (fn: (c: GPUBuffer) => GPUBindGroup) => this.chunkBufs.map(fn);
      this.gFlux.push(
        [f1, f2].map((pipe) =>
          perChunk((c) => bindGroup(d, pipe, { ...common, 1: clock, ...W, 5: v(this.fa), 6: this.scratch, 7: this.register, 14: c })),
        ),
      );
      this.gEmf.push(
        [e1, e2].map((pipe) => perChunk((c) => bindGroup(d, pipe, { ...common, 1: clock, ...W, 6: this.scratch, 8: v(this.emf), 14: c }))),
      );
      // Stage 1 writes the half-step state; its compensation output (13) is unused and takes the
      // other parity's, which stage 2 then overwrites.
      const upd = (pipe: GPUComputePipeline, outs: Record<number, GPUTextureView>) =>
        perChunk((c) =>
          bindGroup(d, pipe, {
            ...common, 1: clock, 2: v(state[p][0]), 3: v(state[p][1]), 4: v(this.faces[p]), 5: v(this.comps[p]),
            6: this.scratch, 7: this.register, 8: v(this.emf), ...outs, 13: v(this.comps[o]), 14: c,
            15: v(this.w[0]), 16: v(this.w[1]), 17: v(this.w[2]), 28: v(idx),
          }),
        );
      this.gUpdate.push([
        upd(u1, { 9: v(half[0]), 10: v(half[1]), 11: v(this.halfFaces) }),
        upd(u2, { 9: v(state[o][0]), 10: v(state[o][1]), 11: v(this.faces[o]) }),
      ]);
    }
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
    enc.copyBufferToBuffer(this.redMag.result, 0, buf, 0, 12);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return { eMag: r[0], divB: r[2] > 0 ? r[1] / r[2] : 0, bMax: r[2] };
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
    for (const x of [...this.faces, ...this.comps, this.halfFaces, this.fa, ...this.w, this.emf]) x.destroy();
    this.scratch.destroy();
    this.register.destroy();
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
