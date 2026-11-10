// Volume and slice rendering on the simulation's own device.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, tex3d, type Gpu } from '../gpu/device';
import { EOS_MODULES } from '../eos/eos';
import type { Sim } from '../sim';

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
});

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
  private n = 0;

  constructor(
    gpu: Gpu,
    private canvas: HTMLCanvasElement,
  ) {
    this.device = gpu.device;
    this.ctx = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.cam = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.viewBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tf = this.device.createTexture({
      size: [256, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
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
    [this.pVolume, this.pSlice] = await Promise.all([make('render/raymarch.wgsl'), make('render/slice.wgsl')]);
  }

  setTransfer(rgba: Uint8Array<ArrayBuffer>) {
    this.device.queue.writeTexture({ texture: this.tf }, rgba, { bytesPerRow: 1024 }, [256, 1]);
  }

  /** Attach to a simulation (new grid, or a rebuilt equation of state). */
  async attach(sim: Sim) {
    const d = this.device;
    if (this.n !== sim.n || !this.vol) {
      this.vol?.destroy();
      this.vol = tex3d(d, sim.n, 'rgba16float', 'display volume');
      this.n = sim.n;
      const res = { 0: this.cam, 1: this.vol.createView(), 2: this.tf.createView(), 3: this.sampler };
      this.gVolume = bindGroup(d, this.pVolume, res);
      this.gSlice = bindGroup(d, this.pSlice, res);
    }
    this.pField = await computePipeline(d, assemble('render/field.wgsl', { eos: EOS_MODULES[sim.cfg.eos] }), 'field');
    this.gField = [0, 1].map((p) =>
      bindGroup(d, this.pField!, {
        0: sim.paramBuffer,
        1: this.viewBuf,
        2: sim.state[p][0].createView(),
        3: sim.state[p][1].createView(),
        4: this.vol!.createView(),
      }),
    );
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

  encode(enc: GPUCommandEncoder, sim: Sim, v: ViewState) {
    if (!this.pField || !this.vol) return;
    this.resize();
    const d = this.device;

    const vb = new ArrayBuffer(16);
    new Uint32Array(vb, 0, 2).set([v.field, v.log ? 1 : 0]);
    const tiny = 1e-30;
    const lo = v.log ? Math.log10(Math.max(v.lo, tiny)) : v.lo;
    const hi = v.log ? Math.log10(Math.max(v.hi, tiny)) : v.hi;
    new Float32Array(vb, 8, 2).set([lo, hi]);
    d.queue.writeBuffer(this.viewBuf, 0, vb);

    // Orbit camera around the box centre, z up.
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
    const aspect = this.canvas.width / this.canvas.height;
    const samples = Math.min(1.5 * sim.n, 384);
    d.queue.writeBuffer(
      this.cam,
      0,
      new Float32Array([
        ...eye, 0, ...right, 0, ...up, 0, ...fwd, 0,
        // The field of view applies to the shorter side, so the box always fits.
        Math.tan(0.35) / Math.min(aspect, 1), aspect, v.opacity, samples,
        v.sliceAxis, v.slicePos, sim.n, 0,
      ]),
    );

    const cp2 = enc.beginComputePass();
    cp2.setPipeline(this.pField);
    cp2.setBindGroup(0, this.gField[sim.cur]);
    cp2.dispatchWorkgroups(Math.ceil(sim.n / 8), Math.ceil(sim.n / 8), Math.ceil(sim.n / 4));
    cp2.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
      ],
    });
    rp.setPipeline(v.mode === 'volume' ? this.pVolume : this.pSlice);
    rp.setBindGroup(0, v.mode === 'volume' ? this.gVolume : this.gSlice);
    rp.draw(3);
    rp.end();
  }
}
