// Magnetic field lines over the views of a run with a field (docs/mhd-plan.md, section 10).
//
// Each frame the field and the velocity are resampled onto a small volume (render/bsample.wgsl
// or amr/bsample.wgsl). Then:
// - a 2D run: the flux function A_z is integrated from it, and its contours drawn over the
//   plane (render/fluxfunction.wgsl), the same flux between each two lines;
// - a 3D run: the seeds move with the gas and the lines through them are traced
//   (render/fieldlines.wgsl), then drawn (render/fieldtubes.wgsl): as lit tubes in the volume
//   view, which the ray-marcher composites with the gas in front of them, and in the slice view
//   as lines that follow the field's components in the plane.
import { assemble } from '../gpu/shaders';
import { bindGroup } from '../gpu/device';
import { EOS_MODULES } from '../eos/eos';
import type { Simulation } from '../sim';
import type { ViewState } from './renderer';

/** Most lines drawn. */
export const MAX_FIELD_LINES = 1024;
/** Points per line, half each way from its seed, and the step between them (a fraction of the box): 3D, and in a plane. */
const POINTS = { space: 512, plane: 1024 };
const STEP = { space: 1 / 256, plane: 1 / 512 };
/** Voxels per side of the field-line volume, at most: 3D, and in a plane. */
const VOXELS = { space: 128, plane: 1024 };
/** 2D: corners of the flux function's grid per side, at most (render/aflux.wgsl). */
const FLUX_GRID = 2048;
/** Lines in the plane: width of the line drawn, in CSS pixels. */
const LINE_WIDTH = 1.5;
/** Tube radius, in units of the box side: thin, as Guillochon & McCourt (2017) drew theirs. */
const TUBE_RADIUS = 0.0008;

type Entry = GPUBindGroupLayoutEntry;
/** GPUShaderStage.COMPUTE (the global is not there outside a browser, as in the unit tests). */
const C = 4;
const tex = (binding: number, sampleType: GPUTextureSampleType = 'unfilterable-float'): Entry => ({ binding, visibility: C, texture: { sampleType, viewDimension: '3d' } });
const store3d = (binding: number): Entry => ({ binding, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } });
const buf = (binding: number, type: GPUBufferBindingType): Entry => ({ binding, visibility: C, buffer: { type } });

export class FieldLines {
  private device: GPUDevice;
  private params: GPUBuffer;
  private draw: GPUBuffer;
  private seeds: GPUBuffer;
  /** The box around the field where new seeds are drawn (render/fieldlines.wgsl, BB). */
  private bbox: GPUBuffer;
  private pts: GPUBuffer;
  private keep: GPUBuffer;
  private frame: GPUBuffer;
  private sampler: GPUSampler;
  private pBound!: GPUComputePipeline;
  private pAdvect!: GPUComputePipeline;
  private pStamp!: GPUComputePipeline;
  private pTrace!: GPUComputePipeline;
  private pTube!: GPURenderPipeline;
  private pFlat!: GPURenderPipeline;
  private traceLayout!: GPUBindGroupLayout;
  /** 2D: the flux function at the volume's cell corners, its range, and its passes. */
  private aBuf: GPUBuffer | null = null;
  private aRange: GPUBuffer;
  private aGrid: GPUBuffer;
  private drawFlux: GPUBuffer;
  private pFlux!: GPURenderPipeline;
  private gFlux?: GPUBindGroup;
  /** 2D: the flux function's passes for the run (one bind group per parity of its faces), and its grid's side. */
  private aflux: { pRow: GPUComputePipeline; pCols: GPUComputePipeline; groups: GPUBindGroup[]; m: number } | null = null;
  /** What drawFlat() draws this frame: traced lines, the flux function's contours, or nothing. */
  private flatKind: 'lines' | 'flux' | null = null;
  /** The run drawn: its sampling pass (one bind group per parity), volume and trace bind group. */
  private run: {
    sim: Simulation;
    pSample: GPUComputePipeline;
    gSample: GPUBindGroup[];
    bv: GPUTexture;
    vv: GPUTexture;
    gTrace: GPUBindGroup;
  } | null = null;
  private gTube?: GPUBindGroup;
  private gFlat?: GPUBindGroup;
  /** Tubes: color, distance along the ray, and depth, the canvas's size. */
  private targets: { color: GPUTexture; dist: GPUTexture; depth: GPUTexture } | null = null;
  /** Changes whenever the tube targets are made again (the ray-marcher's bind group follows). */
  version = 0;
  private count = 0;
  private plane = false;
  private frameNo = 0;
  /** What the seeds were placed for: when it changes they are placed again. */
  private placedFor = '';

  constructor(device: GPUDevice, private format: GPUTextureFormat) {
    this.device = device;
    const d = device;
    this.params = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.draw = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.seeds = d.createBuffer({ label: 'field-line seeds', size: MAX_FIELD_LINES * 16, usage: GPUBufferUsage.STORAGE });
    const most = Math.max(POINTS.space, POINTS.plane);
    this.pts = d.createBuffer({ label: 'field-line points', size: MAX_FIELD_LINES * most * 16, usage: GPUBufferUsage.STORAGE });
    this.keep = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bbox = d.createBuffer({ label: 'field-line seed box', size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.frame = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.aGrid = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aRange = d.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.drawFlux = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  }

  async init() {
    const d = this.device;
    this.traceLayout = d.createBindGroupLayout({
      label: 'field lines',
      entries: [
        buf(0, 'uniform'), tex(1, 'float'), tex(2, 'float'), { binding: 3, visibility: C, sampler: { type: 'filtering' } },
        buf(4, 'storage'), buf(5, 'storage'), buf(6, 'read-only-storage'), buf(7, 'storage'), buf(8, 'storage'), buf(9, 'storage'),
      ],
    });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.traceLayout] });
    const module = d.createShaderModule({ code: assemble('render/fieldlines.wgsl'), label: 'field lines' });
    const compute = (entryPoint: string) => d.createComputePipelineAsync({ label: `field lines ${entryPoint}`, layout, compute: { module, entryPoint } });
    const draw = d.createShaderModule({ code: assemble('render/fieldtubes.wgsl'), label: 'field-line drawing' });
    const flux = d.createShaderModule({ code: assemble('render/fluxfunction.wgsl'), label: 'flux function' });
    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    [this.pFlux] = await Promise.all([
      d.createRenderPipelineAsync({
        label: 'flux function contours',
        layout: 'auto',
        vertex: { module: flux, entryPoint: 'vs' },
        fragment: { module: flux, entryPoint: 'fs', targets: [{ format: this.format, blend }] },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    [this.pBound, this.pAdvect, this.pStamp, this.pTrace, this.pTube, this.pFlat] = await Promise.all([
      compute('bound'),
      compute('advect'),
      compute('stamp'),
      compute('trace'),
      d.createRenderPipelineAsync({
        label: 'field-line tubes',
        layout: 'auto',
        vertex: { module: draw, entryPoint: 'tube_vs' },
        fragment: { module: draw, entryPoint: 'tube_fs', targets: [{ format: 'rgba16float' }, { format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      }),
      d.createRenderPipelineAsync({
        label: 'field lines in the plane',
        layout: 'auto',
        vertex: { module: draw, entryPoint: 'flat_vs' },
        fragment: {
          module: draw,
          entryPoint: 'flat_fs',
          targets: [{
            format: this.format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
  }

  /** Bind to a run (null, or one without a field, draws nothing). */
  async attach(sim: Simulation | null, cam: GPUBuffer) {
    const d = this.device;
    this.run?.bv.destroy();
    this.run?.vv.destroy();
    this.run = null;
    this.aflux = null;
    this.placedFor = '';
    if (!sim || !sim.cfg.problem.mhd) return;
    const amr = sim.fieldShader === 'amr/field.wgsl';
    const flat = sim.nz === 1;
    const m = Math.min(sim.n, flat ? VOXELS.plane : VOXELS.space);
    const size = [m, m, flat ? 1 : m];
    const vol = (label: string) =>
      d.createTexture({ label, size, dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const bv = vol('field-line field');
    const vv = vol('field-line velocity');
    // The sampling pass reads the state's first texture, the faces and (blocks) the leaf index and
    // block table; its layout is explicit, so what the shader leaves unused does no harm.
    const entries: Entry[] = [buf(0, 'uniform'), tex(2), tex(13), store3d(14), store3d(15), buf(16, 'storage')];
    if (amr) entries.push({ binding: 5, visibility: C, texture: { sampleType: 'uint', viewDimension: '3d' } }, buf(30, 'read-only-storage'));
    else entries.push(tex(3));
    const layout = d.createBindGroupLayout({ label: 'field-line sampling', entries });
    const code = amr ? assemble('amr/bsample.wgsl') : assemble('render/bsample.wgsl', { eos: EOS_MODULES[sim.cfg.eos] });
    const pSample = await d.createComputePipelineAsync({
      label: 'field-line sampling',
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: d.createShaderModule({ code, label: 'field-line sampling' }), entryPoint: 'main' },
    });
    const gSample = [0, 1].map((p) => {
      const r = sim.fieldResources(p, 0);
      const pick: Record<number, GPUBuffer | GPUTextureView> = { 0: sim.paramBuffer, 2: r[2], 13: r[13], 14: bv.createView(), 15: vv.createView(), 16: this.frame };
      if (amr) Object.assign(pick, { 5: r[5], 30: r[30] });
      else pick[3] = r[3];
      return d.createBindGroup({
        layout,
        entries: Object.entries(pick).map(([b, x]) => ({ binding: +b, resource: x instanceof GPUBuffer ? { buffer: x } : x })),
      });
    });
    const gTrace = d.createBindGroup({
      layout: this.traceLayout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: bv.createView() },
        { binding: 2, resource: vv.createView() },
        { binding: 3, resource: this.sampler },
        { binding: 4, resource: { buffer: this.seeds } },
        { binding: 5, resource: { buffer: this.pts } },
        { binding: 6, resource: { buffer: sim.clockBuffer } },
        { binding: 7, resource: { buffer: this.keep } },
        { binding: 8, resource: { buffer: this.frame } },
        { binding: 9, resource: { buffer: this.bbox } },
      ],
    });
    this.run = { sim, pSample, gSample, bv, vv, gTrace };
    const res = { 0: cam, 4: this.pts, 5: this.frame, 6: this.draw };
    this.gTube = bindGroup(d, this.pTube, res);
    this.gFlat = bindGroup(d, this.pFlat, res);
    if (flat) await this.attachFlux(sim, amr, bv, cam);
  }

  /**
   * 2D: the flux function's passes (render/aflux.wgsl), integrating the run's own face field on a
   * grid of up to FLUX_GRID cells per side, and the contour pass that draws it.
   */
  private async attachFlux(sim: Simulation, amr: boolean, bv: GPUTexture, cam: GPUBuffer) {
    const d = this.device;
    const m = Math.min(sim.n, FLUX_GRID);
    d.queue.writeBuffer(this.aGrid, 0, new Uint32Array([m, sim.n / m, 0, 0]));
    const bytes = (m + 1) ** 2 * 4;
    if (!this.aBuf || this.aBuf.size < bytes) {
      this.aBuf?.destroy();
      this.aBuf = d.createBuffer({ label: 'flux function', size: bytes, usage: GPUBufferUsage.STORAGE });
    }
    const entries: Entry[] = [buf(0, 'uniform'), tex(13), buf(17, 'storage'), buf(18, 'storage'), buf(19, 'uniform')];
    if (amr) entries.push({ binding: 5, visibility: C, texture: { sampleType: 'uint', viewDimension: '3d' } }, buf(30, 'read-only-storage'));
    const layout = d.createBindGroupLayout({ label: 'flux function', entries });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = d.createShaderModule({ code: assemble('render/aflux.wgsl', { aface: amr ? 'amr/aface.wgsl' : 'render/aface_uniform.wgsl' }), label: 'flux function' });
    const [pRow, pCols] = await Promise.all(
      ['row', 'cols'].map((entryPoint) => d.createComputePipelineAsync({ label: `flux function ${entryPoint}`, layout: pl, compute: { module, entryPoint } })),
    );
    const groups = [0, 1].map((p) => {
      const r = sim.fieldResources(p, 0);
      const pick: Record<number, GPUBuffer | GPUTextureView> = { 0: sim.paramBuffer, 13: r[13], 17: this.aBuf!, 18: this.aRange, 19: this.aGrid };
      if (amr) Object.assign(pick, { 5: r[5], 30: r[30] });
      return d.createBindGroup({ layout, entries: Object.entries(pick).map(([b, x]) => ({ binding: +b, resource: x instanceof GPUBuffer ? { buffer: x } : x })) });
    });
    this.aflux = { pRow, pCols, groups, m };
    this.gFlux = bindGroup(d, this.pFlux, { 0: cam, 4: bv.createView(), 5: this.aBuf!, 6: this.aRange, 7: this.drawFlux, 8: this.frame });
  }

  /** Lines are drawn for this run and view: it has a field, they are on, and it is not a projection. */
  active(sim: Simulation, v: ViewState): boolean {
    return !!this.run && this.run.sim === sim && v.fieldLines && v.fieldLineCount > 0 && !(v.mode === 'projection' && sim.nz > 1);
  }

  /** Whether the volume view's tubes were drawn this frame (encode() then drew them, before the gas). */
  tubesDrawn = false;

  /**
   * Trace the lines for this frame and, in the volume view, draw the tubes into their targets.
   * `w`, `h`: the canvas in pixels, `dpr` of them per CSS pixel; `volume`: the volume view is drawn.
   */
  encode(enc: GPUCommandEncoder, sim: Simulation, v: ViewState, w: number, h: number, dpr: number, volume: boolean) {
    this.tubesDrawn = false;
    this.flatKind = null;
    const run = this.run;
    if (!run || !this.active(sim, v)) return;
    const d = this.device;
    const flat = sim.nz === 1;
    const count = Math.min(MAX_FIELD_LINES, Math.max(1, Math.round(v.fieldLineCount)));
    if (flat && this.aflux) {
      // 2D: the flux function's contours, one per level.
      const af = this.aflux;
      d.queue.writeBuffer(this.aRange, 0, new Uint32Array([0xffffffff, 0]));
      // The largest in-plane |B|, and the sum and count of its log (render/bsample.wgsl).
      d.queue.writeBuffer(this.frame, 0, new Uint32Array([0, 0, 0, 0]));
      d.queue.writeBuffer(this.drawFlux, 0, new Float32Array([count, LINE_WIDTH * dpr, v.light ? 1 : 0, af.m]));
      const m = run.bv;
      const cp = enc.beginComputePass({ label: 'flux function' });
      cp.setPipeline(run.pSample);
      cp.setBindGroup(0, run.gSample[sim.cur]);
      cp.dispatchWorkgroups(Math.ceil(m.width / 4), Math.ceil(m.height / 4), 1);
      cp.setPipeline(af.pRow);
      cp.setBindGroup(0, af.groups[sim.cur]);
      cp.dispatchWorkgroups(1);
      cp.setPipeline(af.pCols);
      cp.dispatchWorkgroups(Math.ceil((af.m + 1) / 64));
      cp.end();
      this.flatKind = 'flux';
      return;
    }
    const plane = !volume;
    const points = plane ? POINTS.plane : POINTS.space;
    const axis = v.sliceAxis;
    const pos = v.slicePos;
    const key = `${plane}|${axis}|${pos}|${count}`;
    const again = key !== this.placedFor;
    this.placedFor = key;
    this.count = count;
    this.plane = plane;
    const m = run.bv;
    const per = sim.cfg.bcLo.map((b) => (b === 0 ? 1 : 0));
    d.queue.writeBuffer(this.params, 0, new Float32Array([
      m.width, m.height, m.depthOrArrayLayers, sim.cfg.problem.box,
      plane ? 1 : 0, axis, pos, count,
      points, plane ? STEP.plane : STEP.space, this.frameNo++, again ? 1 : 0,
      per[0], per[1], per[2], 0,
    ]));
    // Largest |B| in the volume, and the least and greatest on the lines: from nothing. The box
    // around the field new seeds are drawn in (render/fieldlines.wgsl): empty.
    d.queue.writeBuffer(this.frame, 0, new Uint32Array([0, 0x7f7fffff, 0, 0]));
    d.queue.writeBuffer(this.bbox, 0, new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]));
    const cp = enc.beginComputePass({ label: 'field lines' });
    cp.setPipeline(run.pSample);
    cp.setBindGroup(0, run.gSample[sim.cur]);
    cp.dispatchWorkgroups(Math.ceil(m.width / 4), Math.ceil(m.height / 4), Math.ceil(m.depthOrArrayLayers / 4));
    cp.setBindGroup(0, run.gTrace);
    cp.setPipeline(this.pBound);
    cp.dispatchWorkgroups(Math.ceil(m.width / 4), Math.ceil(m.height / 4), Math.ceil(m.depthOrArrayLayers / 4));
    cp.setPipeline(this.pAdvect);
    cp.dispatchWorkgroups(Math.ceil(count / 64));
    cp.setPipeline(this.pStamp);
    cp.dispatchWorkgroups(1);
    cp.setPipeline(this.pTrace);
    cp.dispatchWorkgroups(Math.ceil(count / 64));
    cp.end();
    d.queue.writeBuffer(this.draw, 0, new Float32Array([points, TUBE_RADIUS, LINE_WIDTH * dpr, axis, w, h, v.light ? 1 : 0, 0]));
    if (plane) {
      this.flatKind = 'lines';
      return;
    }
    // The volume view: tubes into their own targets, which the ray-marcher then reads.
    const t = this.ensureTargets(w, h);
    const rp = enc.beginRenderPass({
      label: 'field-line tubes',
      colorAttachments: [
        { view: t.color.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: t.dist.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: t.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    });
    rp.setPipeline(this.pTube);
    rp.setBindGroup(0, this.gTube!);
    rp.draw(6, count * points);
    rp.end();
    this.tubesDrawn = true;
  }

  /** Slice view and 2D runs: the lines traced by encode(), over the map, in the current render pass. */
  drawFlat(rp: GPURenderPassEncoder) {
    if (!this.run) return;
    if (this.flatKind === 'flux') {
      rp.setPipeline(this.pFlux);
      rp.setBindGroup(0, this.gFlux!);
      rp.draw(3);
    } else if (this.flatKind === 'lines' && this.count > 0) {
      rp.setPipeline(this.pFlat);
      rp.setBindGroup(0, this.gFlat!);
      rp.draw(6, this.count * POINTS.plane);
    }
  }

  /** The tubes' color and distance targets, for the ray-marcher. */
  get tubeViews(): { color: GPUTexture; dist: GPUTexture } | null {
    return this.targets;
  }

  private ensureTargets(w: number, h: number) {
    const t = this.targets;
    if (t && t.color.width === w && t.color.height === h) return t;
    t?.color.destroy();
    t?.dist.destroy();
    t?.depth.destroy();
    const d = this.device;
    const make = (format: GPUTextureFormat, label: string) =>
      d.createTexture({ label, size: [w, h], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.targets = { color: make('rgba16float', 'field-line tubes'), dist: make('rgba16float', 'field-line tube distance'), depth: make('depth32float', 'field-line tube depth') };
    this.version++;
    return this.targets;
  }

  destroy() {
    this.run?.bv.destroy();
    this.run?.vv.destroy();
    for (const b of [this.params, this.draw, this.seeds, this.pts, this.keep, this.bbox, this.frame, this.aRange, this.aGrid, this.drawFlux]) b.destroy();
    this.aBuf?.destroy();
    const t = this.targets;
    if (t) for (const x of [t.color, t.dist, t.depth]) x.destroy();
  }
}
