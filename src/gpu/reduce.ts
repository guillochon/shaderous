// Multi-stage GPU reduction (see reduce.wgsl). The first stage reads textures through a
// problem-specific red_cell(); later stages fold the previous stage's records.
import { assemble } from './shaders';
import { bindGroup, computePipeline, type Bindable } from './device';
import type { Slots } from './preprocess';

export const SUM = 0;
export const MAX = 1;
export const MIN = 2;

interface Stage {
  pipeline: GPUComputePipeline;
  groups: GPUBindGroup[]; // one per source variant (state parity)
  nout: number;
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
    n: number,
    ops: number[],
    source: string,
    slots: Slots,
    variants: Record<number, Bindable>[],
    label: string,
  ): Promise<Reducer> {
    const k = ops.length;
    const def = `const RED_K = ${k}u;\nconst RED_OP = array<u32, ${k}>(${ops.map((o) => o + 'u').join(', ')});`;
    const sizes = [n];
    while (sizes[sizes.length - 1] > 1) sizes.push(Math.ceil(sizes[sizes.length - 1] / 8));

    const buffers: GPUBuffer[] = [];
    for (let s = 1; s < sizes.length; s++) {
      buffers.push(
        device.createBuffer({
          label: `${label} stage ${s}`,
          size: Math.max(16, sizes[s] ** 3 * k * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
      );
    }
    const red = new Reducer(k, buffers[buffers.length - 1]);

    const first = assemble('gpu/reduce.wgsl', { ...slots, red_def: def, red_source: source });
    const later = assemble('gpu/reduce.wgsl', { red_def: def, red_source: 'gpu/reduce_buffer.wgsl' });
    for (let s = 1; s < sizes.length; s++) {
      const constants = { NIN: sizes[s - 1], NOUT: sizes[s] };
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
      const d = Math.ceil(st.nout / 4);
      pass.dispatchWorkgroups(d, d, d);
    });
  }
}
