// A stencil pass over a uniform grid, split by region.
//
// Every load of a stencil pass goes through the boundary mapping, but only a thin shell of
// cells can reach a boundary. The pass is therefore compiled from the same source in two
// forms: with the mapping compiled out (INTERIOR), for the box of cells at least `REACH` from
// every face, and in general form for the shell around it.
//
// Two things about the dispatches were measured and matter:
//   - The interior pipeline runs over the whole grid on the usual workgroup grid and skips
//     the shell's cells. Starting it REACH cells in instead was slower: workgroups that
//     straddle the texture's tiles lose more than the skipped invocations cost.
//   - The shell is REACH cells thick, so its six slabs get workgroups that are REACH thick
//     along the slab's normal. With the usual workgroup the slabs launch several times more
//     invocations than they have cells, and those are not free.
//   - The shape of the workgroup hardly matters (Apple M4 Pro, Chrome: every shape of 64 to
//     256 invocations within a few percent of the defaults), so a pass with a face-flux tile in
//     workgroup memory (hydro/tile.wgsl) takes whichever shape lets its tile fit.
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';

const REACH = 2; // cells a stencil may extend from its center
const FACE_BYTES = 48; // one face flux in the tile of hydro/tile.wgsl, viscous part included

/** Bytes of workgroup memory the face-flux tile of hydro/tile.wgsl takes for workgroup `wg`. */
export function tileBytes(wg: number[], flat: boolean): number {
  const [x, y, z] = wg;
  return FACE_BYTES * ((x + 1) * y * z + x * (y + 1) * z + (flat ? 0 : x * y * (z + 1)));
}

interface Dispatch {
  origin: number[]; // first cell
  size: number[]; // cells
  wg: number[]; // workgroup size
}

export interface Split {
  /** interior box [lo, hi) in cells; empty (lo = hi) if the grid is too small to have one */
  lo: number[];
  hi: number[];
  interior: Dispatch | null;
  shell: Dispatch[];
}

/** The interior box and the slabs that hold every cell outside it. */
export function splitRegions(dims: number[], tile = false): Split {
  const flat = dims[2] === 1;
  // 256 invocations per workgroup either way; a single layer gets them all in the plane. A 3D
  // face-flux tile for 8 x 8 x 4 would take 42 KB, more than devices offer: half as many.
  const wg = flat ? [16, 16, 1] : tile ? [8, 4, 4] : [8, 8, 4];
  // A pass with a face-flux tile is not split: one pipeline computes every face, so that the
  // two workgroups sharing a face run the same compiled code on it (hydro/tile.wgsl).
  if (tile) return { lo: [0, 0, 0], hi: [0, 0, 0], interior: null, shell: [{ origin: [0, 0, 0], size: dims, wg }] };
  // An axis with a single layer (2D) has no boundary cells along it.
  const lo = dims.map((n) => (n === 1 ? 0 : REACH));
  const hi = dims.map((n) => (n === 1 ? 1 : n - REACH));
  const all: Dispatch = { origin: [0, 0, 0], size: dims, wg };
  if (hi.some((h, a) => h <= lo[a])) return { lo: [0, 0, 0], hi: [0, 0, 0], interior: null, shell: [all] };
  const slab = (axis: number, from: number[], to: number[]): Dispatch => {
    const w = flat ? [16, 16, 1] : [8, 8, 4];
    w[axis] = REACH;
    return { origin: from, size: to.map((t, a) => t - from[a]), wg: w };
  };
  const shell = [
    slab(2, [0, 0, 0], [dims[0], dims[1], lo[2]]),
    slab(2, [0, 0, hi[2]], dims),
    slab(1, [0, 0, lo[2]], [dims[0], lo[1], hi[2]]),
    slab(1, [0, hi[1], lo[2]], [dims[0], dims[1], hi[2]]),
    slab(0, [0, lo[1], lo[2]], [lo[0], hi[1], hi[2]]),
    slab(0, [hi[0], lo[1], lo[2]], [dims[0], hi[1], hi[2]]),
  ].filter((r) => r.size.every((c) => c > 0));
  return { lo, hi, interior: all, shell };
}

/** Bind groups 0 of one set of resources, one per pipeline of the pass. */
export type RegionGroups = GPUBindGroup[];

interface Part {
  pipeline: GPUComputePipeline;
  regions: { group: GPUBindGroup; count: number[] }[];
}

export class RegionPass {
  private buffers: GPUBuffer[] = [];
  private parts: Part[] = [];

  private constructor(private device: GPUDevice) {}

  /**
   * `tile`: the shader includes hydro/tile.wgsl. The pass then runs as a single pipeline over
   * the whole grid, which computes each face once per workgroup if the tile fits in the
   * device's workgroup memory, and gathers if not.
   */
  static async create(device: GPUDevice, code: string, label: string, dims: number[], constants: Record<string, number> = {}, tile = false): Promise<RegionPass> {
    const split = splitRegions(dims, tile);
    if (tile) {
      const flat = dims[2] === 1;
      const fits = tileBytes(split.shell[0].wg, flat) <= device.limits.maxComputeWorkgroupStorageSize;
      constants = { TILE: fits ? 1 : 0, TILE_FLAT: flat ? 1 : 0, ...constants };
    }
    const pass = new RegionPass(device);
    // One pipeline per distinct (variant, workgroup size).
    const wanted = new Map<string, { interior: boolean; wg: number[]; list: Dispatch[] }>();
    const want = (interior: boolean, r: Dispatch) => {
      const key = `${interior}:${r.wg.join('x')}`;
      if (!wanted.has(key)) wanted.set(key, { interior, wg: r.wg, list: [] });
      wanted.get(key)!.list.push(r);
    };
    if (split.interior) want(true, split.interior);
    for (const r of split.shell) want(false, r);
    pass.parts = await Promise.all(
      [...wanted.values()].map(async (w) => {
        const pipeline = await computePipeline(device, code, `${label} (${w.interior ? 'interior' : 'shell'} ${w.wg.join('x')})`, {
          ...constants,
          INTERIOR: w.interior ? 1 : 0,
          WGX: w.wg[0],
          WGY: w.wg[1],
          WGZ: w.wg[2],
        });
        const regions = w.list.map((r) => {
          const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          const end = r.origin.map((o, a) => o + r.size[a]);
          device.queue.writeBuffer(buffer, 0, new Uint32Array([...r.origin, 0, ...split.lo, 0, ...split.hi, 0, ...end, 0]));
          pass.buffers.push(buffer);
          return {
            group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer } }] }),
            count: r.size.map((c, a) => Math.ceil(c / r.wg[a])),
          };
        });
        return { pipeline, regions };
      }),
    );
    return pass;
  }

  groups(resources: Record<number, Bindable>): RegionGroups {
    return this.parts.map((p) => bindGroup(this.device, p.pipeline, resources));
  }

  encode(pass: GPUComputePassEncoder, g: RegionGroups) {
    this.parts.forEach((p, i) => {
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, g[i]);
      for (const r of p.regions) {
        pass.setBindGroup(1, r.group);
        pass.dispatchWorkgroups(r.count[0], r.count[1], r.count[2]);
      }
    });
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
  }
}
