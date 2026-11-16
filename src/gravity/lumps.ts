// Reading the gas back as lumps (gravity/lumps.wgsl, amr/lumps.wgsl) for the orbit preview
// (gravity/preview.ts).
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { assemble } from '../gpu/shaders';
import type { Lump } from './preview';

/** Floats per lump in the pass's output. */
const LUMP_FLOATS = 8;

/** Lumps along each axis of a uniform grid of n cells a side (nz along z): 32 in 3D, 64 in 2D. */
export function lumpGrid(n: number, nz: number): [number, number, number] {
  const k = Math.min(n, nz === 1 ? 64 : 32);
  return [k, k, nz === 1 ? 1 : k];
}

/**
 * Runs a lump pass and reads its output without ever waiting on the GPU, as MassReader does
 * (src/sim.ts): a call made while the last is in flight, or while the pass is still
 * compiling, returns null.
 */
export class LumpReader {
  private pipeline?: GPUComputePipeline;
  private out?: GPUBuffer;
  private staging?: GPUBuffer;
  private busy = false;
  /** Set when the run is destroyed: nothing more is encoded on its textures. */
  closed = false;

  constructor(private device: GPUDevice, shader: string) {
    this.busy = true;
    computePipeline(device, assemble(shader), 'gas lumps').then(
      (p) => {
        this.pipeline = p;
        this.busy = false;
      },
      () => {}, // stays busy: no lumps, and the preview goes without the gas
    );
  }

  /**
   * `count` lumps from the pass bound to `resources` (the output buffer is added at binding 4)
   * and dispatched by `dispatch`.
   */
  async read(count: number, resources: Record<number, Bindable>, dispatch: (pass: GPUComputePassEncoder) => void): Promise<Lump[] | null> {
    if (this.busy || this.closed || !this.pipeline || count === 0) return null;
    this.busy = true;
    try {
      const d = this.device;
      const bytes = count * LUMP_FLOATS * 4;
      if (!this.out || this.out.size < bytes) {
        this.out?.destroy();
        this.staging?.destroy();
        // Room to spare, so a growing mesh does not make new buffers at every read.
        const size = Math.ceil((bytes * 1.5) / 4096) * 4096;
        this.out = d.createBuffer({ label: 'gas lumps', size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.staging = d.createBuffer({ label: 'gas lumps readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      }
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass({ label: 'gas lumps' });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup(d, this.pipeline, { ...resources, 4: this.out }));
      dispatch(pass);
      pass.end();
      enc.copyBufferToBuffer(this.out, 0, this.staging!, 0, bytes);
      d.queue.submit([enc.finish()]);
      const staging = this.staging!;
      await staging.mapAsync(GPUMapMode.READ, 0, bytes);
      const f = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
      staging.unmap();
      const lumps: Lump[] = [];
      for (let i = 0; i < count; i++) {
        const o = LUMP_FLOATS * i;
        if (f[o + 3] > 0) lumps.push({ m: f[o + 3], pos: [f[o], f[o + 1], f[o + 2]], h: f[o + 4] });
      }
      return lumps;
    } catch {
      return null; // the run was replaced while the read was in flight
    } finally {
      this.busy = false;
    }
  }

  destroy() {
    this.closed = true;
    this.out?.destroy();
    this.staging?.destroy();
  }
}
