// Volume and slice rendering on the simulation's own device.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { EOS_MODULES } from '../eos/eos';
import type { Simulation } from '../sim';

export interface ViewState {
  mode: 'volume' | 'slice';
  field: number;
  log: boolean;
  lo: number; // displayed range in field units
  hi: number;
  sliceAxis: number;
  slicePos: number; // [0, 1)
  opacity: number;
  yaw: number;
  pitch: number;
  dist: number;
  /** Slice view: magnification (1 fits the box) and the point of the slice at the view centre. */
  zoom: number;
  cx: number;
  cy: number;
  /** Outline the blocks of an adaptive mesh in the slice view. */
  showBlocks: boolean;
}

export const defaultView = (): ViewState => ({
  mode: 'volume',
  field: 0,
  log: true,
  lo: 0,
  hi: 1,
  sliceAxis: 2,
  slicePos: 0.5,
  opacity: 40,
  yaw: 0.7,
  pitch: 0.45,
  dist: 2.6,
  zoom: 1,
  cx: 0.5,
  cy: 0.5,
  showBlocks: true,
});

/** The orbit camera: around the box centre (the box is [-0.5, 0.5]^3), z up. */
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
  private tf: GPUTexture;
  private sampler: GPUSampler;
  private pVolume!: GPURenderPipeline;
  private pSlice!: GPURenderPipeline;
  private gVolume!: GPUBindGroup;
  private gSlice!: GPUBindGroup;
  private vol?: GPUTexture;
  private pField?: GPUComputePipeline;
  private gField: GPUBindGroup[] = [];
  private dims: number[] = [0, 0, 0];
  private pPick!: GPUComputePipeline;
  private gPick?: GPUBindGroup;
  private pickIn: GPUBuffer;
  private pickOut: GPUBuffer;

  constructor(
    gpu: Gpu,
    private canvas: HTMLCanvasElement,
  ) {
    this.device = gpu.device;
    this.ctx = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.cam = this.device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tf = this.device.createTexture({
      size: [256, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.pickIn = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickOut = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
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
    [this.pVolume, this.pSlice, this.pPick] = await Promise.all([
      make('render/raymarch.wgsl'),
      make('render/slice.wgsl'),
      computePipeline(this.device, assemble('render/pick.wgsl'), 'pick'),
    ]);
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
      this.gSlice = bindGroup(d, this.pSlice, res);
      this.gPick = bindGroup(d, this.pPick, { ...res, 4: this.pickIn, 5: this.pickOut });
    }
    this.pField = await computePipeline(d, assemble(sim.fieldShader, { eos: EOS_MODULES[sim.cfg.eos] }), 'field');
    this.gField = [0, 1].map((p) =>
      bindGroup(d, this.pField!, { 0: sim.paramBuffer, 1: this.viewBuf, 4: this.vol!.createView(), ...sim.fieldResources(p) }),
    );
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
    const vb = new ArrayBuffer(32);
    new Uint32Array(vb, 16, 4).set([nx, ny, nz, 0]);
    new Uint32Array(vb, 0, 2).set([v.field, v.log ? 1 : 0]);
    const tiny = 1e-30;
    const lo = v.log ? Math.log10(Math.max(v.lo, tiny)) : v.lo;
    const hi = v.log ? Math.log10(Math.max(v.hi, tiny)) : v.hi;
    new Float32Array(vb, 8, 2).set([lo, hi]);
    d.queue.writeBuffer(this.viewBuf, 0, vb);

    const aspect = this.aspect;
    const { eye, fwd, right, up, tan } = cameraFrame(v, aspect);
    const samples = Math.min(1.5 * nx, 384);
    d.queue.writeBuffer(
      this.cam,
      0,
      new Float32Array([
        ...eye, 0, ...right, 0, ...up, 0, ...fwd, 0,
        tan, aspect, v.opacity, samples,
        v.sliceAxis, v.slicePos, nx, nz,
        v.showBlocks ? 1 : 0, v.zoom, v.cx, v.cy,
      ]),
    );

    const cp2 = enc.beginComputePass();
    cp2.setPipeline(this.pField);
    cp2.setBindGroup(0, this.gField[sim.cur]);
    cp2.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 4));
    cp2.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
      ],
    });
    // A 2D run has nothing to ray-march: always show the plane.
    const volume = v.mode === 'volume' && nz > 1;
    rp.setPipeline(volume ? this.pVolume : this.pSlice);
    rp.setBindGroup(0, volume ? this.gVolume : this.gSlice);
    rp.draw(3);
    rp.end();
  }
}
