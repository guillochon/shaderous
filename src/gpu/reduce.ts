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
  /** workgroups dispatched: x in slabs stacked along z where it is longer than the device allows */
  dispatch: [number, number, number];
}

export class Reducer {
  readonly k: number;
  readonly result: GPUBuffer; // k floats
  /** records of the first stage (written externally if the reducer was created that way) */
  first!: GPUBuffer;
  private stages: Stage[] = [];

  private constructor(k: number, result: GPUBuffer) {
    this.k = k;
    this.result = result;
  }

  /**
   * @param source   registry path of the first-stage red_cell() module
   * @param variants input bindings of the first stage, one map per state parity
   * @param external the first stage is done by someone else, who writes its records to `first`
   *                 on a grid of these dimensions; `dims`, `source` and `variants` are then unused
   */
  static async create(
    device: GPUDevice,
    dims: number[],
    ops: number[],
    source: string,
    slots: Slots,
    variants: Record<number, Bindable>[],
    label: string,
    external?: number[],
  ): Promise<Reducer> {
    const k = ops.length;
    // The operation of each channel, as two bit masks, and two more for channels 32 and up. (No
    // array constants or array constructors anywhere in these shaders: Safari's Metal backend
    // fails to compile them.)
    if (k > 64) throw new Error('a reduction has at most 64 channels');
    const mask = (op: number, from: number) => ops.slice(from, from + 32).reduce((m, o, i) => (o === op ? m | (1 << i) : m), 0) >>> 0;
    const def = `const RED_K = ${k}u;\nconst RED_MAX = ${mask(MAX, 0)}u;\nconst RED_MIN = ${mask(MIN, 0)}u;\nconst RED_MAX_HI = ${mask(MAX, 32)}u;\nconst RED_MIN_HI = ${mask(MIN, 32)}u;`;
    // Each stage shrinks every axis by RED_F until one record is left. The axes may differ in
    // length (a 2D run has a single layer in z; a block grid is a long row of blocks).
    const sizes: number[][] = [dims.slice()];
    if (external) sizes.push(external.slice());
    while (sizes[sizes.length - 1].some((d) => d > 1)) sizes.push(sizes[sizes.length - 1].map((d) => Math.ceil(d / RED_F)));
    // A grid of one block still needs a stage to copy its record to the result.
    if (external && sizes.length === 2) sizes.push([1, 1, 1]);

    const buffers: GPUBuffer[] = [];
    for (let s = 1; s < sizes.length; s++) {
      buffers.push(
        device.createBuffer({
          label: `${label} stage ${s}`,
          size: Math.max(16, sizes[s][0] * sizes[s][1] * sizes[s][2] * k * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    const red = new Reducer(k, buffers[buffers.length - 1]);
    red.first = buffers[0];

    const first = external ? '' : assemble('gpu/reduce.wgsl', { ...slots, red_def: def, red_source: source });
    const later = assemble('gpu/reduce.wgsl', { red_def: def, red_source: 'gpu/reduce_buffer.wgsl' });
    // Every stage's pipeline is asked for before any is awaited, so they compile together.
    const stages = [];
    for (let s = external ? 2 : 1; s < sizes.length; s++) {
      const [ix, iy, iz] = sizes[s - 1];
      const [ox, oy, oz] = sizes[s];
      const gx = Math.ceil(ox / 4);
      const gz = Math.ceil(oz / 4);
      const slabs = Math.ceil(gx / device.limits.maxComputeWorkgroupsPerDimension);
      const xg = Math.ceil(gx / slabs);
      const constants = { NINX: ix, NINY: iy, NINZ: iz, NOUTX: ox, NOUTY: oy, NOUTZ: oz, RED_F, RED_XG: xg, RED_ZG: gz };
      const pipeline = computePipeline(device, s === 1 ? first : later, `${label} reduce ${s}`, constants);
      stages.push({ s, pipeline, dispatch: [xg, Math.ceil(oy / 4), gz * slabs] as [number, number, number] });
    }
    for (const { s, pipeline: pending, dispatch } of stages) {
      const pipeline = await pending;
      const groups =
        s === 1
          ? variants.map((v) => bindGroup(device, pipeline, { ...v, 20: buffers[0] }))
          : [bindGroup(device, pipeline, { 0: buffers[s - 2], 20: buffers[s - 1] })];
      red.stages.push({ pipeline, groups, dispatch });
    }
    return red;
  }

  encode(pass: GPUComputePassEncoder, variant = 0) {
    this.stages.forEach((st, i) => {
      pass.setPipeline(st.pipeline);
      pass.setBindGroup(0, st.groups[i === 0 ? Math.min(variant, st.groups.length - 1) : 0]);
      pass.dispatchWorkgroups(...st.dispatch);
    });
  }
}
