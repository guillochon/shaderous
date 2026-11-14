// Ideal MHD on the uniform grid (docs/mhd-plan.md): the van Leer predictor-corrector with
// constrained transport (Stone & Gardiner 2009), the Gardiner & Stone (2005) edge electric
// fields, and HLLE or HLLD fluxes. A step is, after the timestep:
//
//   stage 1: fluxes (first order, from the state at the start), edge fields, half-step update
//   stage 2: fluxes (reconstructed from the half-step state), edge fields, full update
//
// The magnetic field lives on the faces of the cells (mhd/common.wgsl); the conserved state
// keeps its layout, with the magnetic energy B^2 / 2 in the total energy. The half-step state
// uses the hydro predictor's two textures and a face texture of its own.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, readTexture, tex3d, writeTexture } from '../gpu/device';
import { MAX, Reducer, SUM } from '../gpu/reduce';
import type { SimConfig } from '../sim';
import { PERIODIC } from '../problems/types';

/** The run evolves a magnetic field. */
export const mhdOn = (c: SimConfig): boolean => c.problem.mhd !== undefined;

/** The MHD Riemann solver for the run's choice: HLLE for HLL, HLLD (which resolves contacts) otherwise. */
export const mhdRiemann = (c: SimConfig): string => (c.riemann === 'hll' ? 'mhd/riemann/hlle.wgsl' : 'mhd/riemann/hlld.wgsl');

/**
 * Why this configuration cannot run with a magnetic field yet, or null if it can. The field is
 * coupled to the gas dynamics alone so far: gravity, point masses, species, cooling, diffusion,
 * sources and the adaptive mesh come in later phases of docs/mhd-plan.md.
 */
export function mhdUnsupported(c: SimConfig): string | null {
  if (c.gravity !== 'off' || c.gext.some((g) => g !== 0)) return 'gravity is not coupled to the magnetic field yet';
  if (c.problem.masses) return 'point masses are not coupled to the magnetic field yet';
  if (c.problem.species?.length) return 'species are not carried with a magnetic field yet';
  if (c.cooling !== 'off') return 'cooling is not coupled to the magnetic field yet';
  if (c.viscosity > 0 || c.conductivity > 0) return 'viscosity and conduction are not coupled to the magnetic field yet';
  return null;
}

/** Magnetic diagnostics channels: magnetic energy, max |div B| dx, max |B|. */
const MAG_OPS = [SUM, MAX, MAX];

export interface MagDiagnostics {
  /** total magnetic energy, sum of B^2 / 2 over the cells */
  eMag: number;
  /** largest |div B| dx of any cell over the largest |B| on the grid */
  divB: number;
  /** largest |B| at a cell center */
  bMax: number;
}

/**
 * The face field of the problem's initial conditions: texel (i, j, k) of a grid with one more
 * texel than cells along each axis (one along z in 2D) holds (b_x on the lower x face of cell
 * (i, j, k), b_y on its lower y face, b_z on its lower z face, 0). From a vector potential A,
 * b = curl A with A at the edge centers, so the divergence of every cell is zero up to
 * round-off; or the problem's face values directly.
 */
export function initialFaces(c: SimConfig, n: number, nz: number, dx: number): Float32Array {
  const spec = c.problem.mhd!;
  const half = c.problem.box / 2;
  const fx = n + 1;
  const fz = nz === 1 ? 1 : nz + 1;
  const out = new Float32Array(fx * fx * fz * 4);
  const flat = nz === 1;
  if (spec.faces) {
    const f = spec.faces(n, nz);
    if (f.length !== out.length) throw new Error(`MHD: the face field is for another grid (${f.length / 4} texels, not ${out.length / 4})`);
    return f;
  }
  // Face (lower) and center positions along an axis.
  const lo = (i: number) => i * dx - half;
  const mid = (i: number) => (i + 0.5) * dx - half;
  const zc = (k: number) => (flat ? 0 : mid(k));
  const zl = (k: number) => (flat ? 0 : lo(k));
  if (spec.potential) {
    const A = spec.potential(c.params, c.ndim);
    const ax = (i: number, j: number, k: number) => A(mid(i), lo(j), zl(k))[0];
    const ay = (i: number, j: number, k: number) => A(lo(i), mid(j), zl(k))[1];
    const az = (i: number, j: number, k: number) => A(lo(i), lo(j), zc(k))[2];
    for (let k = 0; k < fz; k++)
      for (let j = 0; j < fx; j++)
        for (let i = 0; i < fx; i++) {
          const o = 4 * ((k * fx + j) * fx + i);
          const dzAy = flat ? 0 : ay(i, j, k + 1) - ay(i, j, k);
          const dzAx = flat ? 0 : ax(i, j, k + 1) - ax(i, j, k);
          out[o] = (az(i, j + 1, k) - az(i, j, k) - dzAy) / dx;
          out[o + 1] = (dzAx - (az(i + 1, j, k) - az(i, j, k))) / dx;
          out[o + 2] = (ay(i + 1, j, k) - ay(i, j, k) - (ax(i, j + 1, k) - ax(i, j, k))) / dx;
        }
  } else if (spec.face) {
    const F = spec.face(c.params, c.ndim);
    for (let k = 0; k < fz; k++)
      for (let j = 0; j < fx; j++)
        for (let i = 0; i < fx; i++) {
          const o = 4 * ((k * fx + j) * fx + i);
          out[o] = F(0, lo(i), mid(j), zc(k));
          out[o + 1] = F(1, mid(i), lo(j), zc(k));
          out[o + 2] = F(2, mid(i), mid(j), zl(k));
        }
  }
  return out;
}

/** Cell-centered field of cell (i, j, k) from face texels as initialFaces lays them out. */
export function cellField(c: SimConfig, faces: Float32Array, n: number, nz: number, i: number, j: number, k: number): [number, number, number] {
  const fx = n + 1;
  const up = (a: number, m: number, len: number) => (m + 1 === len && c.bcLo[a] === PERIODIC ? 0 : m + 1);
  const at = (a: number, ii: number, jj: number, kk: number) => faces[4 * ((kk * fx + jj) * fx + ii) + a];
  const kz = nz === 1 ? 0 : up(2, k, nz);
  return [
    0.5 * (at(0, i, j, k) + at(0, up(0, i, n), j, k)),
    0.5 * (at(1, i, j, k) + at(1, i, up(1, j, n), k)),
    0.5 * (at(2, i, j, k) + at(2, i, j, kz)),
  ];
}

interface MhdDeps {
  device: GPUDevice;
  cfg: SimConfig;
  n: number;
  nz: number;
  params: GPUBuffer;
  clock: GPUBuffer;
  /** state[parity] = [U1, U2] */
  state: GPUTexture[][];
  /** the half-step conserved state (the hydro predictor's textures) */
  half: GPUTexture[];
  /** slots of the hydro passes (equation of state, limiter, composition) */
  slots: Record<string, string>;
  /** the composition texture binding, if the composition varies */
  comp: Record<number, GPUTextureView>;
}

/**
 * How the edge fields are formed (mhd/corner.wgsl), from estimates carried to the edge from the
 * four faces around it:
 *   'mean'        Gardiner & Stone (2005), as Athena++ does it: the mean of their estimates
 *   'upwind'      Lee (2013)'s upwind selection of those same estimates
 *   'mec'         Lee & Deane (2009): the mean of Taylor estimates with central derivatives
 *                 along each line of faces
 *   'upwind-mec'  Lee (2013): the upwind selection of Lee & Deane's estimates
 * The last two are on the uniform grid only.
 */
export type EdgeScheme = 'mean' | 'upwind' | 'mec' | 'upwind-mec';
const EDGE_MODES: Record<EdgeScheme, number> = { mean: 0, upwind: 1, mec: 2, 'upwind-mec': 3 };
let edgeScheme: EdgeScheme = 'mean';
/** Change it for runs built from now on (comparisons and tests). */
export function setMhdEdge(e: EdgeScheme) {
  edgeScheme = e;
}

/** Bytes the flux and edge buffers of one slab may take, which sets how many layers it has. */
let slabBytes = 192 * 1024 ** 2;
/** Change it for runs built from now on (validation: many thin slabs on a small grid); null restores the default. */
export function setMhdSlabBytes(b: number | null) {
  slabBytes = b ?? 192 * 1024 ** 2;
}

/** Layers of cells per slab of a 3D grid of side n (the whole grid if it fits). */
export function slabLayers(n: number, nz: number): number {
  if (nz === 1) return 1;
  const layer = 3 * (n + 4) ** 2 * 48 + (n + 1) ** 2 * 16;
  return Math.max(4, Math.min(nz, Math.floor(slabBytes / layer) - 4));
}

export class MhdPasses {
  /** face fields, faces[parity] with the state's parity */
  faces: GPUTexture[] = [];
  /** what each face's float32 rounding has left out (mhd/update.wgsl), by parity */
  comps: GPUTexture[] = [];
  /** half-step face field */
  halfFaces!: GPUTexture;
  private flux: GPUBuffer[] = [];
  private emf!: GPUBuffer;
  /** per slab: its first layer, its layers, and its (first layer, layers, last) uniform */
  private slabs: { z0: number; nz: number; buf: GPUBuffer }[] = [];
  private pFlux: GPUComputePipeline[] = [];
  private pEmf: GPUComputePipeline[] = [];
  private pUpdate: GPUComputePipeline[] = [];
  pRates!: GPUComputePipeline;
  /** [parity][stage][slab] bind groups of each pass */
  private gFlux: GPUBindGroup[][][] = [];
  private gEmf: GPUBindGroup[][][] = [];
  private gUpdate: GPUBindGroup[][][] = [];
  gRates: GPUBindGroup[] = [];
  private redMag!: Reducer;
  private dep: MhdDeps;

  /**
   * Bytes per cell of the MHD textures: five face textures (two parities of the field and of its
   * compensation, and the half step). The flux and edge buffers hold one slab of layers at a
   * time (slabLayers) and are not counted per cell.
   */
  static readonly cellBytes = 5 * 16;

  private constructor(dep: MhdDeps) {
    this.dep = dep;
  }

  static async create(dep: MhdDeps, rates: GPUBuffer): Promise<MhdPasses> {
    const m = new MhdPasses(dep);
    const { device: d, n, nz } = dep;
    const fz = nz === 1 ? 1 : nz + 1;
    for (let p = 0; p < 2; p++) m.faces.push(tex3d(d, n + 1, 'rgba32float', `face field ${p}`, fz));
    for (let p = 0; p < 2; p++) m.comps.push(tex3d(d, n + 1, 'rgba32float', `face compensation ${p}`, fz));
    m.halfFaces = tex3d(d, n + 1, 'rgba32float', 'half-step face field', fz);
    const S = slabLayers(n, nz);
    const flux = (n + 4) * (n + 4) * (nz === 1 ? 1 : S + 4) * 3 * 16;
    for (let a = 0; a < 3; a++) m.flux.push(d.createBuffer({ label: `MHD fluxes ${'xyz'[a]}`, size: flux, usage: GPUBufferUsage.STORAGE }));
    m.emf = d.createBuffer({ label: 'edge impulses', size: (n + 1) * (n + 1) * (nz === 1 ? 1 : S + 1) * 16, usage: GPUBufferUsage.STORAGE });
    for (let z0 = 0; z0 < nz; z0 += S) {
      const k = Math.min(S, nz - z0);
      const buf = d.createBuffer({ label: `MHD slab ${z0}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(buf, 0, new Int32Array([z0, k, z0 + k >= nz ? 1 : 0, 0]));
      m.slabs.push({ z0, nz: k, buf });
    }
    await m.build(rates);
    return m;
  }

  /** Rebuild the pipelines after a change of the equation of state, limiter or Riemann solver. */
  async rebuild(rates: GPUBuffer, slots: Record<string, string>, comp: Record<number, GPUTextureView>) {
    this.dep.slots = slots;
    this.dep.comp = comp;
    await this.build(rates);
  }

  private async build(rates: GPUBuffer) {
    const { device: d, cfg, params, clock, state, half, comp } = this.dep;
    const slots = { ...this.dep.slots, mhd_riemann: mhdRiemann(cfg) };
    const v = (t: GPUTexture) => t.createView();
    const [f1, f2, e1, e2, u1, u2, prates, red] = await Promise.all([
      computePipeline(d, assemble('mhd/flux.wgsl', slots), 'MHD fluxes 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/flux.wgsl', slots), 'MHD fluxes 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/emf.wgsl', slots), 'edge impulses 1', { STAGE: 1, EDGE_MODE: EDGE_MODES[edgeScheme] }),
      computePipeline(d, assemble('mhd/emf.wgsl', slots), 'edge impulses 2', { STAGE: 2, EDGE_MODE: EDGE_MODES[edgeScheme] }),
      computePipeline(d, assemble('mhd/update.wgsl', slots), 'MHD update 1', { STAGE: 1 }),
      computePipeline(d, assemble('mhd/update.wgsl', slots), 'MHD update 2', { STAGE: 2 }),
      computePipeline(d, assemble('mhd/rates.wgsl', slots), 'MHD timestep rates'),
      Reducer.create(
        d,
        [this.dep.n, this.dep.n, this.dep.nz],
        MAG_OPS,
        'mhd/diagnostics.wgsl',
        slots,
        [0, 1].map((p) => ({ 0: params, 4: v(this.faces[p]), ...comp })),
        'magnetic diagnostics',
      ),
    ]);
    this.pFlux = [f1, f2];
    this.pEmf = [e1, e2];
    this.pUpdate = [u1, u2];
    this.pRates = prates;
    this.redMag = red;
    const fl = { 5: this.flux[0], 6: this.flux[1], 7: this.flux[2] };
    this.gFlux = [];
    this.gEmf = [];
    this.gUpdate = [];
    this.gRates = [];
    for (let p = 0; p < 2; p++) {
      const o = 1 - p;
      const start = { 2: v(state[p][0]), 3: v(state[p][1]), 4: v(this.faces[p]) };
      const mid = { 2: v(half[0]), 3: v(half[1]), 4: v(this.halfFaces) };
      const perSlab = (fn: (slab: GPUBuffer) => GPUBindGroup) => this.slabs.map((s) => fn(s.buf));
      this.gFlux.push([
        perSlab((sb) => bindGroup(d, f1, { 0: params, 1: clock, ...start, ...fl, 14: sb, ...comp })),
        perSlab((sb) => bindGroup(d, f2, { 0: params, 1: clock, ...mid, ...fl, 14: sb, ...comp })),
      ]);
      this.gEmf.push([
        perSlab((sb) => bindGroup(d, e1, { 0: params, 1: clock, ...start, ...fl, 8: this.emf, 14: sb, ...comp })),
        perSlab((sb) => bindGroup(d, e2, { 0: params, 1: clock, ...mid, ...fl, 8: this.emf, 14: sb, ...comp })),
      ]);
      // Stage 1 writes the half-step state, so its unused half-step inputs (12, 13) take the
      // start's, and its unused compensation output (16) the other parity's, which stage 2
      // overwrites.
      const upd = (pipe: GPUComputePipeline, out: Record<number, GPUTextureView>, h: GPUTexture[]) =>
        perSlab((sb) =>
          bindGroup(d, pipe, {
            0: params, 1: clock, ...start, ...fl, 8: this.emf, ...out, 12: v(h[0]), 13: v(h[1]), 14: sb, 15: v(this.comps[p]), 16: v(this.comps[o]), ...comp,
          }),
        );
      this.gUpdate.push([
        upd(u1, { 9: v(half[0]), 10: v(half[1]), 11: v(this.halfFaces) }, state[p]),
        upd(u2, { 9: v(state[o][0]), 10: v(state[o][1]), 11: v(this.faces[o]) }, half),
      ]);
      this.gRates.push(bindGroup(d, prates, { 0: params, 1: v(state[p][0]), 2: v(state[p][1]), 4: v(this.faces[p]), 6: rates, ...comp }));
    }
  }

  private dispatch(pass: GPUComputePassEncoder, dims: number[]) {
    pass.dispatchWorkgroups(Math.ceil(dims[0] / 8), Math.ceil(dims[1] / 8), Math.ceil(dims[2] / 4));
  }

  /**
   * Encode the two stages of a step from the state at `parity` into the other parity (the clock
   * already holds dt). Each stage runs slab by slab; stage 2 starts once stage 1 has written the
   * whole half-step state, which its stencils reach across slabs for.
   */
  encode(pass: GPUComputePassEncoder, parity: number) {
    const { n, nz } = this.dep;
    const flat = nz === 1;
    for (let s = 0; s < 2; s++) {
      this.slabs.forEach((sl, i) => {
        const last = sl.z0 + sl.nz >= nz;
        pass.setPipeline(this.pFlux[s]);
        pass.setBindGroup(0, this.gFlux[parity][s][i]);
        this.dispatch(pass, [n + 4, n + 4, flat ? 1 : sl.nz + 4]);
        pass.setPipeline(this.pEmf[s]);
        pass.setBindGroup(0, this.gEmf[parity][s][i]);
        this.dispatch(pass, [n + 1, n + 1, flat ? 1 : sl.nz + 1]);
        pass.setPipeline(this.pUpdate[s]);
        pass.setBindGroup(0, this.gUpdate[parity][s][i]);
        this.dispatch(pass, [n + 1, n + 1, flat ? 1 : sl.nz + (last ? 1 : 0)]);
      });
    }
  }

  /** Encode the first stage of the timestep reduction for the state at `parity`. */
  encodeRates(pass: GPUComputePassEncoder, parity: number) {
    pass.setPipeline(this.pRates);
    pass.setBindGroup(0, this.gRates[parity]);
    const { n, nz } = this.dep;
    this.dispatch(pass, [n, n, nz]);
  }

  /** Upload a face field (initialFaces' layout) as the state at `parity`, with nothing to compensate. */
  setFaces(faces: Float32Array, parity = 0) {
    const { n, nz } = this.dep;
    const fz = nz === 1 ? 1 : nz + 1;
    writeTexture(this.dep.device, this.faces[parity], n + 1, 4, faces, fz);
    writeTexture(this.dep.device, this.comps[parity], n + 1, 4, new Float32Array(faces.length), fz);
  }

  /** The face field at `parity`, in initialFaces' layout (slow: tests and snapshots). */
  readFaces(parity: number): Promise<Float32Array> {
    const { n, nz } = this.dep;
    return readTexture(this.dep.device, this.faces[parity], n + 1, 4, nz === 1 ? 1 : nz + 1);
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

  destroy() {
    for (const t of [...this.faces, ...this.comps]) t.destroy();
    this.halfFaces?.destroy();
    for (const sl of this.slabs) sl.buf.destroy();
    for (const b of this.flux) b.destroy();
    this.emf?.destroy();
  }
}
