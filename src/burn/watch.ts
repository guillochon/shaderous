// The watch for burning (hydro/burn_watch.wgsl): in a run that carries only he4, c12 and o16
// (burn/network.ts WATCH_SPECIES) with the network's burning set, nothing burns. In its place
// this pass copies the state as a burn pass does and keeps the largest share of a cell's
// internal energy that the network's reactions would release in a step. The simulation reads it
// after each batch of steps (poll); once it passes NetworkBurnSpec.switchAt, triggered() says
// so, and the run stops for the switch to all thirteen species (stopped 'burn'): a checkpoint,
// restarted with them (the runner's --switch-burn).
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { assemble } from '../gpu/shaders';
import { speciesConst } from '../species/species';
import type { BurnTargets, Burner } from './burn';
import { BURN_GRID_ATLAS, BURN_GRID_UNIFORM, NET_SPECIES, NET_T_MAX, netWatchSlot, WATCH_SPECIES, type NetUnits, type NetworkBurnSpec } from './network';

/** The share of a cell's internal energy, released over one step, that stops the run for the switch (NetworkBurnSpec.switchAt). */
export const SWITCH_AT = 0.1;

const WATCH_UNIFORM_BYTES = 48;

export class NetWatchPass implements Burner {
  /** the largest share read back so far, and whether a read is in flight */
  private most = 0;
  private reading = false;

  private constructor(
    private device: GPUDevice,
    private pipe: GPUComputePipeline,
    private groupsBy: GPUBindGroup[][],
    private dims: [number, number, number],
    private max: GPUBuffer,
    private staging: GPUBuffer,
    private uniform: GPUBuffer,
    readonly switchAt: number,
  ) {}

  static async create(d: GPUDevice, spec: NetworkBurnSpec, species: readonly string[] | undefined, units: NetUnits, t: BurnTargets): Promise<NetWatchPass> {
    if (!species || species.join() !== WATCH_SPECIES.join()) throw new Error(`watching for burning needs the species ${WATCH_SPECIES.join(', ')}, in that order`);
    const code = assemble('hydro/burn_watch.wgsl', {
      eos: t.slots.eos, comp: t.slots.comp, species_n: speciesConst(t.groups), burn_watch: netWatchSlot(),
      burn_grid: t.leaves ? BURN_GRID_ATLAS : BURN_GRID_UNIFORM,
    });
    const pipe = await computePipeline(d, code, 'burn watch');
    const uniform = d.createBuffer({ label: 'burn watch', size: WATCH_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const buf = new ArrayBuffer(WATCH_UNIFORM_BYTES);
    new Float32Array(buf, 0, 8).set([units.rho, units.T, units.time, 1 / units.specificEnergy, spec.tMin ?? 1e8, spec.rhoMin ?? 1e4, NET_T_MAX, 0]);
    new Uint32Array(buf, 32, 4).set([t.stride, spec.inShocks ? 1 : 0, 0, 0]);
    d.queue.writeBuffer(uniform, 0, buf);
    const max = d.createBuffer({ label: 'burn watch max', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const staging = d.createBuffer({ label: 'burn watch readback', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const v = (x: GPUTexture) => x.createView();
    const groupsBy = [0, 1].map((s) =>
      [0, 1].map((q) => {
        const r: Record<number, Bindable> = {
          0: t.params, 1: t.clock, 2: v(t.state[s][0]), 3: v(t.state[s][1]), 4: v(t.pb), 5: v(t.species[q]),
          6: v(t.state[1 - s][0]), 7: v(t.state[1 - s][1]), 8: v(t.species[1 - q]), 9: uniform, 12: max,
        };
        if (t.leaves) r[31] = t.leaves;
        return bindGroup(d, pipe, r, 'burn watch');
      }),
    );
    return new NetWatchPass(d, pipe, groupsBy, t.dims, max, staging, uniform, spec.switchAt ?? SWITCH_AT);
  }

  /** As NetBurnPass.encode: the state of parity s with the species of parity q, copied into the other parities, watched on the way. */
  encode(pass: GPUComputePassEncoder, s: number, q: number, dispatch?: (pass: GPUComputePassEncoder) => void) {
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.groupsBy[s][q]);
    if (dispatch) return dispatch(pass);
    const [x, y, z] = this.dims;
    pass.dispatchWorkgroups(Math.ceil(x / 8), Math.ceil(y / 4), Math.ceil(z / 2));
  }

  /** Read the largest share since the last read, and start afresh (after a batch of steps is submitted). At most one read at a time. */
  poll() {
    if (this.reading) return;
    this.reading = true;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.max, 0, this.staging, 0, 16);
    enc.clearBuffer(this.max);
    this.device.queue.submit([enc.finish()]);
    void this.staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const v = new Float32Array(this.staging.getMappedRange().slice(0))[0];
        this.staging.unmap();
        if (v > this.most) this.most = v;
      })
      .catch(() => {})
      .finally(() => {
        this.reading = false;
      });
  }

  /** The largest share read back, if it has passed switchAt: the run should stop for the switch. */
  triggered(): number | null {
    return this.most > this.switchAt ? this.most : null;
  }

  /** The largest share read back so far. */
  get largest(): number {
    return this.most;
  }

  destroy() {
    this.max.destroy();
    this.staging.destroy();
    this.uniform.destroy();
  }
}

/**
 * A setup that watched for burning (he4, c12 and o16 with the network's burning), as it
 * continues after the switch: with all the network's species. Throws for any other setup: a
 * switch asked of a run that did not watch is a mistake, not a no-op.
 */
export function switchedSetup<S extends { physics?: { species?: unknown; burn?: unknown } }>(spec: S): S {
  const sp = spec.physics?.species;
  const burn = spec.physics?.burn as { network?: string } | undefined;
  if (!(Array.isArray(sp) && sp.join() === WATCH_SPECIES.join()) || burn?.network !== 'aprox13') {
    throw new Error(`the switch to all the network's species is for a run that carries ${WATCH_SPECIES.join(', ')} alone with the network's burning`);
  }
  return { ...spec, physics: { ...spec.physics, species: 'aprox13' } };
}

/**
 * A snapshot's species arrays with the network's others added at zero: he4, c12 and o16 map to
 * themselves, and the other ten are not there yet. Returns the names it added.
 */
export function expandSpecies(data: Record<string, ArrayLike<number> | undefined>): string[] {
  const have = data[WATCH_SPECIES[0]];
  if (!have || WATCH_SPECIES.some((k) => !data[k])) throw new Error(`the file holds no ${WATCH_SPECIES.join(', ')} to switch from`);
  const added: string[] = [];
  for (const k of NET_SPECIES) {
    if (data[k]) continue;
    data[k] = new Float32Array(have.length);
    added.push(k);
  }
  return added;
}
