// Multi-stage GPU reduction (see reduce.wgsl). The first stage reads textures through a
// problem-specific red_cell(); later stages fold the previous stage's records.
import { assemble } from './shaders';
import { bindGroup, computePipeline, type Bindable } from './device';
import type { Slots } from './preprocess';

export const SUM = 0;
export const MAX = 1;
export const MIN = 2;

// Block side folded per stage. 4 measured faster than 8 (shorter serial loops, more threads).
export const RED_F = Number(new URLSearchParams(globalThis.location?.search ?? '').get('redf')) || 4;

interface Stage {
  pipeline: GPUComputePipeline;
  groups: GPUBindGroup[]; // one per source variant (state parity)
  nout: number[];
}

export class Reducer {
  readonly k: number;
  readonly result: GPUBuffer; // k floats
  private stages: Stage[] = [];

  private constructor(k: number, result: GPUBuffer) {
    this.k = k;
    this.result = result;
  }

  /**
   * @param source   registry path of the first-stage red_cell() module
   * @param variants input bindings of the first stage, one map per state parity
   */
  static async create(
    device: GPUDevice,
    dims: number[],
    ops: number[],
    source: string,
    slots: Slots,
    variants: Record<number, Bindable>[],
    label: string,
  ): Promise<Reducer> {
    const k = ops.length;
    const def = `const RED_K = ${k}u;\nconst RED_OP = array<u32, ${k}>(${ops.map((o) => o + 'u').join(', ')});`;
    // Each stage shrinks every axis by RED_F until one record is left. The axes may differ in
    // length (a 2D run has a single layer in z; a block grid is a long row of blocks).
    const sizes: number[][] = [dims.slice()];
    while (sizes[sizes.length - 1].some((d) => d > 1)) sizes.push(sizes[sizes.length - 1].map((d) => Math.ceil(d / RED_F)));

    const buffers: GPUBuffer[] = [];
    for (let s = 1; s < sizes.length; s++) {
      buffers.push(
        device.createBuffer({
          label: `${label} stage ${s}`,
          size: Math.max(16, sizes[s][0] * sizes[s][1] * sizes[s][2] * k * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
      );
    }
    const red = new Reducer(k, buffers[buffers.length - 1]);

    const first = assemble('gpu/reduce.wgsl', { ...slots, red_def: def, red_source: source });
    const later = assemble('gpu/reduce.wgsl', { red_def: def, red_source: 'gpu/reduce_buffer.wgsl' });
    for (let s = 1; s < sizes.length; s++) {
      const [ix, iy, iz] = sizes[s - 1];
      const [ox, oy, oz] = sizes[s];
      const constants = { NINX: ix, NINY: iy, NINZ: iz, NOUTX: ox, NOUTY: oy, NOUTZ: oz, RED_F };
      const pipeline = await computePipeline(device, s === 1 ? first : later, `${label} reduce ${s}`, constants);
      const groups =
        s === 1
          ? variants.map((v) => bindGroup(device, pipeline, { ...v, 20: buffers[0] }))
          : [bindGroup(device, pipeline, { 0: buffers[s - 2], 20: buffers[s - 1] })];
      red.stages.push({ pipeline, groups, nout: sizes[s] });
    }
    return red;
  }

  encode(pass: GPUComputePassEncoder, variant = 0) {
    this.stages.forEach((st, i) => {
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.groups[i === 0 ? variant : 0]);
      pass.dispatchWorkgroups(Math.ceil(st.nout[0] / 4), Math.ceil(st.nout[1] / 4), Math.ceil(st.nout[2] / 4));
    });
  }
}
