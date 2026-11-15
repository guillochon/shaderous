// Multipole moments for isolated boundaries: a reduction of the gas's moments about the box
// center, then a pass that moves them to its center of mass, where the boundary potential is
// expanded (gravity/multipole_eval.wgsl, to l = 4).
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { Reducer, SUM } from '../gpu/reduce';

/** Moments summed (gravity/mp_index.wgsl, MP_COUNT), and the floats of the result: mass, center of mass, moments. */
export const MP_COUNT = 35;
const MP_FLOATS = 4 + MP_COUNT;

export class Multipoles {
  /** what gravity/multipole_eval.wgsl reads (MP, binding 10) */
  readonly result: GPUBuffer;
  private constructor(
    private reducer: Reducer,
    private pipeline: GPUComputePipeline,
    private group: GPUBindGroup,
    result: GPUBuffer,
  ) {
    this.result = result;
  }

  /** `source`: the reduction's red_cell() module; `variants`: its bindings, one map per state parity. */
  static async create(device: GPUDevice, dims: number[], source: string, variants: Record<number, Bindable>[], label: string): Promise<Multipoles> {
    const [reducer, pipeline] = await Promise.all([
      Reducer.create(device, dims, new Array(MP_COUNT).fill(SUM), source, {}, variants, label),
      computePipeline(device, assemble('gravity/multipole_com.wgsl'), `${label} about the center of mass`),
    ]);
    const result = device.createBuffer({ label: `${label} about the center of mass`, size: MP_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const group = bindGroup(device, pipeline, { 0: reducer.result, 1: result });
    return new Multipoles(reducer, pipeline, group, result);
  }

  encode(pass: GPUComputePassEncoder, parity: number) {
    this.reducer.encode(pass, parity);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(1);
  }
}
