// The damping zone's rate and target state at one point, read from the GPU (hydro/damp_probe.wgsl).
import { computePipeline } from '../gpu/device';
import { assemble } from '../gpu/shaders';

export interface DampTarget {
  /** relaxation rate (per unit time); 0 outside the zone */
  rate: number;
  rho: number;
  vel: [number, number, number];
  p: number;
}

export class DampProbe {
  private constructor(
    private device: GPUDevice,
    private pipeline: GPUComputePipeline,
    private params: GPUBuffer,
    private clock: GPUBuffer,
  ) {}

  /** A probe for a run whose hydro slots (src/sim.ts hydroSlots) hold its damping zone, reading its uniform block `params` and its clock (the target at the run's time). */
  static async create(device: GPUDevice, slots: Record<string, string>, params: GPUBuffer, clock: GPUBuffer): Promise<DampProbe> {
    const pipeline = await computePipeline(device, assemble('hydro/damp_probe.wgsl', slots), 'damping target probe');
    return new DampProbe(device, pipeline, params, clock);
  }

  /** The target at `x` (relative to the box center), in a cell of size dx. */
  async at(x: number[], dx: number): Promise<DampTarget> {
    const d = this.device;
    const at = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(at, 0, new Float32Array([x[0], x[1], x[2] ?? 0, dx]));
    const out = d.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = d.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bg = d.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: at } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: this.clock } },
      ],
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, 32);
    d.queue.submit([enc.finish()]);
    try {
      await read.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      return { rate: f[0], rho: f[1], p: f[2], vel: [f[4], f[5], f[6]] };
    } finally {
      at.destroy();
      out.destroy();
      read.destroy();
    }
  }
}
