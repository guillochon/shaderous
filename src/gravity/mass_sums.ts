// The point masses' sums over the gas: the gas's mass, each mass's pull on the gas, and what
// each sink takes this step (gravity/bh_cell.wgsl), which gravity/bh_kick.wgsl turns into the
// masses' accelerations and growth.
//
// A reduction's record is kept in registers (gpu/reduce.ts), so it cannot hold the ten sums of
// every one of MAX_MASSES masses. The masses are summed in groups of MASS_GROUP instead: one
// reduction over the grid per group in use, each with the same pipelines, told which group it
// is by a uniform and gathered into one buffer with every mass's sums. A run with no more than
// MASS_GROUP masses does one reduction, as it did when that was all a run could hold.
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { Reducer, SUM } from '../gpu/reduce';
import { assemble } from '../gpu/shaders';
import type { Slots } from '../gpu/preprocess';

/** Masses a reduction sums at once. Must match gravity/bh_cell.wgsl. */
export const MASS_GROUP = 4;
/** Channels of one group's reduction: the gas's mass, then ten sums per mass. */
export const MASS_RED = 1 + 10 * MASS_GROUP;
/** The binding of the uniform that tells a reduction which group it is. */
const GROUP_BINDING = 6;

export class MassSums {
  /** Every mass's sums: the gas's mass, then ten floats per mass (gravity/bh_kick.wgsl's RED). */
  readonly result: GPUBuffer;
  private reducer!: Reducer;
  private pGather!: GPUComputePipeline;
  private gGather: GPUBindGroup[] = [];
  private bases: GPUBuffer[] = [];
  private variants = 1;

  private constructor(private device: GPUDevice, private maxMasses: number) {
    this.result = device.createBuffer({ label: 'point mass sums', size: (1 + 10 * maxMasses) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  }

  /**
   * @param dims     the grid the first stage reduces over (as Reducer.create)
   * @param source   registry path of the red_cell() module, which includes gravity/bh_cell.wgsl
   * @param variants input bindings of the first stage, one map per state parity
   */
  static async create(
    device: GPUDevice,
    maxMasses: number,
    dims: number[],
    source: string,
    slots: Slots,
    variants: Record<number, Bindable>[],
    label: string,
  ): Promise<MassSums> {
    const sums = new MassSums(device, maxMasses);
    const groups = Math.ceil(maxMasses / MASS_GROUP);
    sums.variants = variants.length;
    for (let g = 0; g < groups; g++) {
      const b = device.createBuffer({ label: `${label} group ${g}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([g * MASS_GROUP, 0, 0, 0]));
      sums.bases.push(b);
    }
    // One reduction, with a first-stage bind group per group and parity.
    const all = sums.bases.flatMap((b) => variants.map((v) => ({ ...v, [GROUP_BINDING]: b })));
    [sums.reducer, sums.pGather] = await Promise.all([
      Reducer.create(device, dims, new Array(MASS_RED).fill(SUM), source, slots, all, label),
      computePipeline(device, assemble('gravity/bh_gather.wgsl'), 'point mass sums'),
    ]);
    sums.gGather = sums.bases.map((b) => bindGroup(device, sums.pGather, { 0: sums.reducer.result, 1: sums.result, 2: b }));
    return sums;
  }

  /** Sum over the gas of the given parity for the first `masses` point masses. */
  encode(pass: GPUComputePassEncoder, variant: number, masses: number) {
    const groups = Math.max(1, Math.ceil(Math.min(masses, this.maxMasses) / MASS_GROUP));
    for (let g = 0; g < groups; g++) {
      this.reducer.encode(pass, g * this.variants + Math.min(variant, this.variants - 1));
      pass.setPipeline(this.pGather);
      pass.setBindGroup(0, this.gGather[g]);
      pass.dispatchWorkgroups(1);
    }
  }

  destroy() {
    this.result.destroy();
    for (const b of this.bases) b.destroy();
  }
}
