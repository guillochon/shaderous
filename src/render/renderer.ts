// Volume, slice and projection rendering on the simulation's own device.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { EOS_MODULES } from '../eos/eos';
import { COOLING_MODULES } from '../cooling/cooling';
import { LINEAR_FIELDS } from '../problems/types';
import { hydroSlots, type Simulation } from '../sim';

/** The display pass's range record (render/field_value.wgsl): four words, then a histogram. */
const HIST_BINS = 128;
const RANGE_WORDS = 4 + HIST_BINS;
/** Edge of histogram bin i, as a value of |field| (HIST_LOG_LO and HIST_PER_DECADE there). */
const binEdge = (i: number) => 10 ** (i / 3.2 - 20);

/** Inverse of range_key() in render/field_value.wgsl: the float whose ordered bits are k. */
function fromKey(k: number): number {
  const bits = k & 0x80000000 ? k ^ 0x80000000 : ~k >>> 0;
  return new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0];
}

export interface ViewState {
  /**
   * Projection: along the slice axis, density summed into a column density and any other field
   * averaged, weighted by mass.
   */
  mode: 'volume' | 'slice' | 'projection';
  field: number;
  /** Mass fraction field: which species (in the order of Problem.species) */
  species: number;
  log: boolean;
  lo: number; // displayed range in field units
  hi: number;
  sliceAxis: number;
  slicePos: number; // [0, 1)
  /**
   * Volume view: the least density drawn, as a fraction of the way from the smallest density in
   * the box to the largest, on a log scale. 0 draws all the gas.
   */
  minRho: number;
  opacity: number;
  yaw: number;
  pitch: number;
  dist: number;
  /** Slice view: magnification (1 fits the box) and the point of the slice at the view center. */
  zoom: number;
  cx: number;
  cy: number;
  /** Outline the blocks of an adaptive mesh, in the slice view and the volume view. */
  showBlocks: boolean;
  /**
   * Slice and projection views: contour lines, nContours of them, evenly spaced in log10 |field|
   * inside the displayed range (contourRange()).
   */
  contours: boolean;
  nContours: number;
}

/** Floats per line vertex: position (box units, centered), color with alpha. */
const LINE_FLOATS = 7;
const BLOCK_COLOR = [0.75, 0.8, 0.9, 0.22];
/** Guides through a tool, one color per axis. */
const GUIDE_COLORS = [
  [1, 0.42, 0.42, 0.85],
  [0.45, 0.9, 0.5, 0.85],
  [0.45, 0.65, 1, 0.85],
];

/**
 * Edges of a set of axis-aligned boxes (lo xyz and size, in fractions of the box), as line
 * vertices; an edge two blocks share is drawn once.
 */
export function boxEdges(boxes: Float32Array): Float32Array<ArrayBuffer> {
  const seen = new Set<string>();
  const out: number[] = [];
  for (let i = 0; i + 3 < boxes.length; i += 4) {
    const [x, y, z, s] = [boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]];
    for (let a = 0; a < 3; a++) {
      const b = (a + 1) % 3;
      const c = (a + 2) % 3;
      for (let k = 0; k < 4; k++) {
        const lo = [x, y, z];
        lo[b] += k & 1 ? s : 0;
        lo[c] += k & 2 ? s : 0;
        const hi = lo.slice();
        hi[a] += s;
        const key = `${a}:${lo.map((v) => Math.round(v * 65536)).join(',')}:${Math.round(s * 65536)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(lo[0] - 0.5, lo[1] - 0.5, lo[2] - 0.5, ...BLOCK_COLOR, hi[0] - 0.5, hi[1] - 0.5, hi[2] - 0.5, ...BLOCK_COLOR);
      }
    }
  }
  return new Float32Array(out);
}

/**
 * Guides through a point (in fractions of the box): a line along each axis from face to face,
 * and a small square where it meets each face.
 */
export function guideLines(p: [number, number, number]): Float32Array<ArrayBuffer> {
  const out: number[] = [];
  const push = (q: number[], col: number[]) => out.push(q[0] - 0.5, q[1] - 0.5, q[2] - 0.5, ...col);
  const h = 0.015;
  for (let a = 0; a < 3; a++) {
    const col = GUIDE_COLORS[a];
    const ends = [0, 1].map((f) => {
      const q = p.slice();
      q[a] = f;
      return q;
    });
    push(ends[0], col);
    push(ends[1], col);
    const b = (a + 1) % 3;
    const c = (a + 2) % 3;
    for (const e of ends) {
      const corner = (i: number) => {
        const q = e.slice();
        q[b] += i === 1 || i === 2 ? h : -h;
        q[c] += i >= 2 ? h : -h;
        return q;
      };
      for (let i = 0; i < 4; i++) {
        push(corner(i), col);
        push(corner((i + 1) % 4), col);
      }
    }
  }
  return new Float32Array(out);
}

export const defaultView = (): ViewState => ({
  mode: 'volume',
  field: 0,
  species: 0,
  log: true,
  lo: 0,
  hi: 1,
  sliceAxis: 2,
  slicePos: 0.5,
  minRho: 0,
  opacity: 40,
  yaw: 0.7,
  pitch: 0.45,
  dist: 2.6,
  zoom: 1,
  cx: 0.5,
  cy: 0.5,
  showBlocks: true,
  contours: false,
  nContours: 8,
});

/**
 * The color scale as the shaders take it: logarithmic or not (a field of either sign is always
 * shown on a linear scale), and its ends, in log10 when logarithmic.
 */
export function colorScale(v: ViewState): { log: boolean; lo: number; hi: number } {
  const log = v.log && !LINEAR_FIELDS.has(v.field);
  const tiny = 1e-30;
  return { log, lo: log ? Math.log10(Math.max(v.lo, tiny)) : v.lo, hi: log ? Math.log10(Math.max(v.hi, tiny)) : v.hi };
}

/** A half-precision float, from its bits. */
function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/**
 * The range of |field|, in log10, that the contour levels divide evenly: the displayed range
 * when it is positive, else (a range reaching zero or below) the four decades under its largest
 * magnitude.
 */
export function contourRange(lo: number, hi: number): [number, number] {
  const top = Math.max(Math.abs(lo), Math.abs(hi), 1e-30);
  const bottom = lo > 0 ? Math.min(lo, hi) : top * 1e-4;
  return [Math.log10(bottom), Math.log10(lo > 0 ? Math.max(lo, hi) : top)];
}

/** The orbit camera: around the box center (the box is [-0.5, 0.5]^3), z up. */
export function cameraFrame(v: ViewState, aspect: number) {
  const cp = Math.cos(v.pitch);
  const eye = [v.dist * cp * Math.cos(v.yaw), v.dist * cp * Math.sin(v.yaw), v.dist * Math.sin(v.pitch)];
  const fwd = eye.map((c) => -c / v.dist);
  const right = [fwd[1], -fwd[0], 0];
  const rl = Math.hypot(right[0], right[1]) || 1;
  right[0] /= rl;
  right[1] /= rl;
  const up = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  // The field of view applies to the shorter side, so the box always fits.
  return { eye, fwd, right, up, tan: Math.tan(0.35) / Math.min(aspect, 1) };
}

export class Renderer {
  private device: GPUDevice;
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private cam: GPUBuffer;
  private viewBuf: GPUBuffer;
  private noSpecies?: GPUTexture;
  /** Volume view: the least density drawn (0 draws all the gas). */
  densityCut = 0;
  /**
   * Slice view at zoom 1, in CSS pixels: the width of the slice, and how far its middle sits
   * left of the canvas's middle. Unset, it is centered with a small margin.
   */
  slice: { span: number; shift: number } | null = null;
  private tf: GPUTexture;
  private sampler: GPUSampler;
  private pVolume!: GPURenderPipeline;
  private pSlice!: GPURenderPipeline;
  private gSlice?: GPUBindGroup;
  /**
   * Slice and projection views: the 2D map they draw, one texel per cell of the finest level
   * (render/project.wgsl), and the passes that fill it.
   */
  private map?: GPUTexture;
  private pPlane?: GPUComputePipeline;
  private gPlane: GPUBindGroup[] = [];
  private pProject?: GPUComputePipeline;
  private gProject: GPUBindGroup[] = [];
  private pLines!: GPURenderPipeline;
  private gLines!: GPUBindGroup;
  private lineBuf: GPUBuffer | null = null;
  /** Block outline last uploaded: the simulation and the version of its mesh. */
  private blocksOf: { sim: Simulation; version: number; count: number } | null = null;
  /** The contour drawn highlighted (its level, counted from 1), or 0 for none. */
  hoverLevel = 0;
  /** Where the guides through a tool cross (in fractions of the box), or null for none. */
  guides: [number, number, number] | null = null;
  private gVolume!: GPUBindGroup;
  private vol?: GPUTexture;
  private pField?: GPUComputePipeline;
  private gField: GPUBindGroup[] = [];
  private dims: number[] = [0, 0, 0];
  private pPick!: GPUComputePipeline;
  private gPick?: GPUBindGroup;
  private pickIn: GPUBuffer;
  private pickOut: GPUBuffer;
  /** Range of the field last drawn (render/field_value.wgsl), and its readback. */
  private rangeBuf: GPUBuffer;
  private rangeStage: GPUBuffer;
  private rangeCopied: string | null = null;
  private rangeBusy = false;
  /**
   * The range of the field drawn in a recent frame: its least and greatest values and its
   * smallest positive value (null if none), for what it was drawn for (rangeKey()).
   */
  range: { key: string; lo: number; hi: number; pos: number | null; hist: Uint32Array } | null = null;

  constructor(
    gpu: Gpu,
    private canvas: HTMLCanvasElement,
  ) {
    this.device = gpu.device;
    this.ctx = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.cam = this.device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewBuf = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tf = this.device.createTexture({
      size: [256, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.pickIn = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickOut = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.rangeBuf = this.device.createBuffer({ size: RANGE_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rangeStage = this.device.createBuffer({ size: RANGE_WORDS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  async init() {
    const make = async (file: string) => {
      const module = this.device.createShaderModule({ code: assemble(file), label: file });
      return this.device.createRenderPipelineAsync({
        label: file,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      });
    };
    const lines = this.device.createShaderModule({ code: assemble('render/lines.wgsl'), label: 'lines' });
    [this.pVolume, this.pSlice, this.pPick, this.pLines] = await Promise.all([
      make('render/raymarch.wgsl'),
      make('render/slice.wgsl'),
      computePipeline(this.device, assemble('render/pick.wgsl'), 'pick'),
      this.device.createRenderPipelineAsync({
        label: 'lines',
        layout: 'auto',
        vertex: {
          module: lines,
          entryPoint: 'line_vs',
          buffers: [{
            arrayStride: LINE_FLOATS * 4,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x4' }],
          }],
        },
        fragment: {
          module: lines,
          entryPoint: 'line_fs',
          targets: [{
            format: this.format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'line-list' },
      }),
    ]);
    this.gLines = bindGroup(this.device, this.pLines, { 0: this.cam });
  }

  setTransfer(rgba: Uint8Array<ArrayBuffer>) {
    this.device.queue.writeTexture({ texture: this.tf }, rgba, { bytesPerRow: 1024 }, [256, 1]);
  }

  /** Attach to a simulation (new grid, or a rebuilt equation of state). */
  async attach(sim: Simulation) {
    const d = this.device;
    const dims = sim.displayDims;
    if (!this.vol || dims.some((v, i) => v !== this.dims[i])) {
      this.vol?.destroy();
      this.vol = d.createTexture({
        label: 'display volume',
        size: dims,
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.dims = dims.slice();
      const res = { 0: this.cam, 1: this.vol.createView(), 2: this.tf.createView(), 3: this.sampler };
      this.gVolume = bindGroup(d, this.pVolume, res);
      this.gPick = bindGroup(d, this.pPick, { ...res, 4: this.pickIn, 5: this.pickOut });
    }
    // The slice's and the projection's map: one texel per cell of the finest level, on either
    // kind of grid, whatever the display volume is capped at.
    const side = Math.min(sim.n, d.limits.maxTextureDimension2D);
    if (!this.map || this.map.width !== side) {
      this.map?.destroy();
      this.map = d.createTexture({
        label: 'slice and projection map',
        size: [side, side],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      this.gSlice = bindGroup(d, this.pSlice, { 0: this.cam, 2: this.tf.createView(), 3: this.sampler, 4: this.map.createView() });
    }
    const code = assemble(sim.fieldShader, { eos: EOS_MODULES[sim.cfg.eos], cooling: COOLING_MODULES[sim.cfg.cooling], diffusion: hydroSlots(sim.cfg).diffusion, comp: hydroSlots(sim.cfg).comp });
    [this.pField, this.pPlane, this.pProject] = await Promise.all([
      computePipeline(d, code, 'field'),
      computePipeline(d, code, 'slice', undefined, 'slice'),
      computePipeline(d, code, 'projection', undefined, 'project'),
    ]);
    // One per parity of the state and of the species (a run without species binds a stand-in).
    this.noSpecies ??= d.createTexture({ label: 'no species', size: [1, 1, 1], dimension: '3d', format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    const common = (k: number) => ({ 0: sim.paramBuffer, 1: this.viewBuf, 9: this.noSpecies!.createView(), ...sim.fieldResources(k >> 1, k & 1) });
    this.gField = [0, 1, 2, 3].map((k) => bindGroup(d, this.pField!, { ...common(k), 4: this.vol!.createView(), 8: this.rangeBuf }));
    // The slice pass records no range: the display pass does, over the whole box.
    this.gPlane = [0, 1, 2, 3].map((k) => bindGroup(d, this.pPlane!, { ...common(k), 11: this.map!.createView() }));
    this.gProject = [0, 1, 2, 3].map((k) => bindGroup(d, this.pProject!, { ...common(k), 8: this.rangeBuf, 11: this.map!.createView() }));
    this.range = null;
  }

  /** Call after submitting an encoder that encode() wrote to: reads the range back. */
  submitted() {
    const key = this.rangeCopied;
    if (key === null || this.rangeBusy) return;
    this.rangeCopied = null;
    this.rangeBusy = true;
    this.rangeStage
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const k = new Uint32Array(this.rangeStage.getMappedRange().slice(0));
        this.rangeStage.unmap();
        if (k[0] === 0xffffffff) return; // nothing drawn but obstacles
        this.range = { key, lo: fromKey(k[0]), hi: fromKey(k[1]), pos: k[2] === 0xffffffff ? null : fromKey(k[2]), hist: k.slice(4) };
      })
      .catch(() => {})
      .finally(() => (this.rangeBusy = false));
  }

  /** Projection view (a 2D run has nothing to project): the axis summed along, or null. */
  projectionAxis(v: ViewState): number | null {
    return v.mode === 'projection' && this.dims[2] > 1 ? v.sliceAxis : null;
  }

  /** What a range is drawn for: the field, and in the projection view the axis it is summed along. */
  rangeKey(v: ViewState): string {
    const axis = this.projectionAxis(v);
    return axis === null ? String(v.field) : `${v.field}@${axis}`;
  }

  /**
   * From the last range read back for `key` (rangeKey()): the value of |field| below which the
   * share `q` of the cells (or columns) drawn lies, to within one histogram bin (a factor of
   * about 2), or null.
   */
  quantile(key: string, q: number): number | null {
    const r = this.range;
    if (!r || r.key !== key) return null;
    let total = 0;
    for (const n of r.hist) total += n;
    if (!total) return null;
    let sum = 0;
    for (let i = 0; i < HIST_BINS; i++) {
      sum += r.hist[i];
      if (sum >= q * total) return binEdge(i + 1);
    }
    return null;
  }

  get aspect(): number {
    return this.canvas.width / this.canvas.height;
  }

  /**
   * Volume view: how far along the ray through a point of the canvas (x, y in [-1, 1], y up)
   * the gas seen there lies, or null if the ray shows next to nothing. Uses the frame last drawn.
   */
  async pick(x: number, y: number): Promise<number | null> {
    if (!this.gPick) return null;
    const d = this.device;
    d.queue.writeBuffer(this.pickIn, 0, new Float32Array([x, y, 0, 0]));
    const out = d.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pPick);
    pass.setBindGroup(0, this.gPick);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.pickOut, 0, out, 0, 16);
    d.queue.submit([enc.finish()]);
    await out.mapAsync(GPUMapMode.READ);
    const t = new Float32Array(out.getMappedRange())[0];
    out.unmap();
    out.destroy();
    return t > 0 ? t : null;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  encode(enc: GPUCommandEncoder, sim: Simulation, v: ViewState) {
    if (!this.pField || !this.vol) return;
    this.resize();
    const d = this.device;

    const [nx, ny, nz] = this.dims;
    const vb = new ArrayBuffer(64);
    // Mass fraction: which species, the texels between groups of four, and how many groups.
    // Slice and projection: the axis (a 2D run shows its plane), the slice's position along it,
    // and the side of their map.
    const projAxis = this.projectionAxis(v);
    const axis = nz > 1 ? v.sliceAxis : 2;
    const side = this.map!.width;
    new Uint32Array(vb, 48, 4).set([v.species, sim.speciesStride, Math.ceil((sim.cfg.problem.species?.length ?? 0) / 4), axis]);
    new Uint32Array(vb, 16, 4).set([nx, ny, nz, side]);
    // Volume view: gas thinner than densityCut is not drawn, whatever the field (0: all of it).
    const marched = v.mode === 'volume' && nz > 1;
    new Float32Array(vb, 32, 3).set([this.densityCut > 0 ? Math.log10(this.densityCut) : -1e30, marched ? 1 : 0, nz > 1 ? v.slicePos : 0]);
    const { log, lo, hi } = colorScale(v);
    new Uint32Array(vb, 0, 2).set([v.field, log ? 1 : 0]);
    new Float32Array(vb, 8, 2).set([lo, hi]);
    d.queue.writeBuffer(this.viewBuf, 0, vb);

    const aspect = this.aspect;
    const { eye, fwd, right, up, tan } = cameraFrame(v, aspect);
    const samples = Math.min(1.5 * nx, 384);
    const cw = Math.max(1, this.canvas.clientWidth);
    const ch = Math.max(1, this.canvas.clientHeight);
    const slice = this.slice ?? { span: 0.94 * Math.min(cw, ch), shift: 0 };
    d.queue.writeBuffer(
      this.cam,
      0,
      new Float32Array([
        // The slice view's layout (see camera.wgsl) in the spare fourth components.
        ...eye, slice.shift / slice.span,
        ...right, cw / (2 * slice.span),
        ...up, ch / (2 * slice.span),
        ...fwd, 0,
        tan, aspect, v.opacity, samples,
        v.sliceAxis, v.slicePos, nx, nz,
        v.showBlocks ? 1 : 0, v.zoom, v.cx, v.cy,
        v.contours ? Math.max(1, Math.round(v.nContours)) : 0, ...contourRange(v.lo, v.hi), v.contours ? this.hoverLevel : 0,
        // The color scale, for the color of each contour's level (as in the View uniform).
        log ? 1 : 0, lo, hi, 0,
      ]),
    );

    // The range starts empty: min and smallest positive at the top, max at the bottom.
    const empty = new Uint32Array(RANGE_WORDS);
    empty.set([0xffffffff, 0, 0xffffffff, 0]);
    d.queue.writeBuffer(this.rangeBuf, 0, empty);
    const cp2 = enc.beginComputePass();
    const parity = 2 * sim.cur + sim.speciesCur;
    if (projAxis !== null) {
      cp2.setPipeline(this.pProject!);
      cp2.setBindGroup(0, this.gProject[parity]);
      cp2.dispatchWorkgroups(Math.ceil(side / 16), Math.ceil(side / 8));
    } else {
      cp2.setPipeline(this.pField);
      cp2.setBindGroup(0, this.gField[parity]);
      cp2.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 4));
      if (v.mode !== 'volume' || nz === 1) {
        cp2.setPipeline(this.pPlane!);
        cp2.setBindGroup(0, this.gPlane[parity]);
        cp2.dispatchWorkgroups(Math.ceil(side / 16), Math.ceil(side / 8));
      }
    }
    cp2.end();
    if (!this.rangeBusy && this.rangeCopied === null) {
      enc.copyBufferToBuffer(this.rangeBuf, 0, this.rangeStage, 0, RANGE_WORDS * 4);
      this.rangeCopied = this.rangeKey(v);
    }

    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
      ],
    });
    // A 2D run has nothing to ray-march: always show the plane.
    const volume = v.mode === 'volume' && nz > 1;
    rp.setPipeline(volume ? this.pVolume : this.pSlice);
    rp.setBindGroup(0, volume ? this.gVolume : this.gSlice!);
    rp.draw(3);
    if (volume) this.drawLines(rp, sim, v.showBlocks);
    rp.end();
  }

  /**
   * Slice and projection views, from the map last drawn: the contour line within `tolPx` screen
   * pixels of the point (a, b) of the map (in fractions of its side), as its level (counted from
   * 1, as the shader counts them) and the value it marks, or null. `pxPerSide`: screen pixels
   * across the whole map. Mirrors contour() in render/slice.wgsl.
   */
  async contourAt(a: number, b: number, v: ViewState, pxPerSide: number, tolPx = 5): Promise<{ level: number; value: number } | null> {
    const map = this.map;
    if (!map || !v.contours || map.width < 2) return null;
    const side = map.width;
    const gx = a * side - 0.5;
    const gy = b * side - 0.5;
    const [ix, iy] = [Math.floor(gx), Math.floor(gy)];
    const [fx, fy] = [gx - ix, gy - iy];
    const clampi = (i: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, i));
    const ox = clampi(ix, 0, side - 2);
    const oy = clampi(iy, 0, side - 2);
    const patch = await this.readMapPatch(map, ox, oy);
    if (!patch || map !== this.map) return null;
    // Texel (x, y) of the map, clamped to it as textureLoad is in the shader: rgba.
    const at = (x: number, y: number) => {
      const i = 4 * (2 * (clampi(y, 0, side - 1) - oy) + (clampi(x, 0, side - 1) - ox));
      return patch.subarray(i, i + 4);
    };
    const [p00, p10, p01, p11] = [at(ix, iy), at(ix + 1, iy), at(ix, iy + 1), at(ix + 1, iy + 1)];
    if (Math.min(p00[0], p10[0], p01[0], p11[0]) < -0.5) return null; // beside a solid
    const lerp = (u: number, w: number, s: number) => u + (w - u) * s;
    const bil = (c: number) => lerp(lerp(p00[c], p10[c], fx), lerp(p01[c], p11[c], fx), fy);
    const n = Math.max(1, Math.round(v.nContours));
    const [l0, l1] = contourRange(v.lo, v.hi);
    const per = (n + 1) / Math.max(l1 - l0, 1e-6); // levels per unit of log10 |value|
    const f = (bil(2) - l0) * per;
    // Gradient of f per texel, then per screen pixel.
    const dfx = lerp(p10[2] - p00[2], p11[2] - p01[2], fy) * per;
    const dfy = lerp(p01[2] - p00[2], p11[2] - p10[2], fx) * per;
    const perPx = (Math.hypot(dfx, dfy) * side) / pxPerSide;
    const k = Math.round(f);
    if (k < 1 || k > n || Math.abs(f - k) > tolPx * perPx) return null;
    // The line is where the value is minus the level when it lies below zero on a linear scale.
    const { log, lo, hi } = colorScale(v);
    const neg = !log && bil(0) < -lo / Math.max(hi - lo, 1e-20);
    const value = 10 ** (l0 + k / per);
    return { level: k, value: neg ? -value : value };
  }

  /** Texels (x, y) to (x + 1, y + 1) of the map, as rgba floats, row by row. */
  private async readMapPatch(map: GPUTexture, x: number, y: number): Promise<Float32Array | null> {
    const d = this.device;
    const buf = d.createBuffer({ size: 256 + 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: map, origin: [x, y] }, { buffer: buf, bytesPerRow: 256 }, [2, 2]);
      d.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const raw = new Uint16Array(buf.getMappedRange().slice(0));
      const out = new Float32Array(16);
      for (let row = 0; row < 2; row++) for (let i = 0; i < 8; i++) out[row * 8 + i] = halfToFloat(raw[row * 128 + i]);
      return out;
    } catch {
      return null;
    } finally {
      buf.destroy();
    }
  }

  /** Volume view: the outline of the mesh's blocks, then the guides through a tool. */
  private drawLines(rp: GPURenderPassEncoder, sim: Simulation, blocks: boolean) {
    const mesh = blocks ? sim.blockBoxes?.() : null;
    if (mesh && (this.blocksOf?.sim !== sim || this.blocksOf.version !== mesh.version)) {
      const edges = boxEdges(mesh.boxes);
      this.reserveLines(edges.length + guideLines([0, 0, 0]).length);
      this.device.queue.writeBuffer(this.lineBuf!, 0, edges);
      this.blocksOf = { sim, version: mesh.version, count: edges.length / LINE_FLOATS };
    }
    // The guides go after the block outline, which stays in the buffer while it is hidden.
    const base = this.blocksOf?.sim === sim ? this.blocksOf.count : 0;
    const guides = this.guides ? guideLines(this.guides) : null;
    if (guides) {
      this.reserveLines(base * LINE_FLOATS + guides.length);
      this.device.queue.writeBuffer(this.lineBuf!, base * LINE_FLOATS * 4, guides);
    }
    if (!(mesh && base) && !guides) return;
    rp.setPipeline(this.pLines);
    rp.setBindGroup(0, this.gLines);
    rp.setVertexBuffer(0, this.lineBuf!);
    if (mesh && base) rp.draw(base);
    if (guides) rp.draw(guides.length / LINE_FLOATS, 1, base);
  }

  /** Room for `floats` floats of line vertices, keeping what the buffer holds. */
  private reserveLines(floats: number) {
    const bytes = floats * 4;
    if (this.lineBuf && this.lineBuf.size >= bytes) return;
    const next = this.device.createBuffer({ size: Math.max(4096, 2 * bytes), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (this.lineBuf) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.lineBuf, 0, next, 0, this.lineBuf.size);
      this.device.queue.submit([enc.finish()]);
      this.lineBuf.destroy();
    }
    this.lineBuf = next;
  }
}
