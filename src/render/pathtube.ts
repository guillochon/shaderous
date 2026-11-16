// Volume view: the path of a point mass being placed, as a tube with an arrowhead
// (render/pathtube.wgsl), drawn into the tube targets the ray-marcher composites
// (render/fieldlines.ts makes them).
import { assemble } from '../gpu/shaders';
import { bindGroup } from '../gpu/device';

/** Points of the path itself, at most (gravity/preview.ts resamples to fewer). */
const MAX_POINTS = 256;
/** Points in all, with the paths of the run's other masses after it. */
const ALL_POINTS = 4096;
/** The tube's radius, in units of the box side; the cone is 9 of them long and 3 wide at its base. */
const RADIUS = 0.0025;
const HEAD_LENGTH = 9;
const HEAD_RADIUS = 3;
/** The sink markers' orange (src/ui/style.css). */
const COLOR = [1, 0.616, 0.36, 1];

export class PathTube {
  private pts: GPUBuffer;
  private draw: GPUBuffer;
  private pSeg?: GPURenderPipeline;
  private pHead?: GPURenderPipeline;
  private gSeg?: GPUBindGroup;
  private gHead?: GPUBindGroup;
  /** The path in the buffers: its version, and how many stretches of tube it has (-1: too short to draw). */
  private loaded = { version: -1, segments: -1 };

  constructor(private device: GPUDevice) {
    this.pts = device.createBuffer({ label: 'orbit path', size: ALL_POINTS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.draw = device.createBuffer({ label: 'orbit path drawing', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(cam: GPUBuffer) {
    const d = this.device;
    const module = d.createShaderModule({ code: assemble('render/pathtube.wgsl'), label: 'orbit path' });
    const pipeline = (name: string) =>
      d.createRenderPipelineAsync({
        label: `orbit path ${name}`,
        layout: 'auto',
        vertex: { module, entryPoint: `${name}_vs` },
        fragment: { module, entryPoint: `${name}_fs`, targets: [{ format: 'rgba16float' }, { format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      });
    [this.pSeg, this.pHead] = await Promise.all([pipeline('seg'), pipeline('head')]);
    this.gSeg = bindGroup(d, this.pSeg, { 0: cam, 4: this.pts, 5: this.draw });
    this.gHead = bindGroup(d, this.pHead, { 0: cam, 5: this.draw });
  }

  /**
   * The tube's points and its cone, from the path: the tube runs to where the cone begins, a
   * cone's length short of the end. Returns the number of stretches of tube, or -1 for a path
   * too short to draw.
   */
  private load(path: { points: Float32Array; others: Float32Array[]; box: number }): number {
    const n = Math.min(MAX_POINTS, path.points.length / 3);
    if (n < 2) return -1;
    const p = (i: number) => [0, 1, 2].map((c) => path.points[3 * i + c] / path.box);
    const arc = [0];
    for (let i = 1; i < n; i++) {
      const [a, b] = [p(i - 1), p(i)];
      arc.push(arc[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    }
    const total = arc[n - 1];
    if (!(total > 1e-5)) return -1;
    // A short path gets a thinner tube and a smaller head, so that it is still mostly path.
    const r = Math.min(RADIUS, total / 12);
    const start = total - HEAD_LENGTH * r;
    const out = new Float32Array(ALL_POINTS * 4);
    let k = 0;
    let base = p(0);
    for (let i = 0; i < n; i++) {
      if (arc[i] < start) {
        out.set([...p(i), arc[i] / total], 4 * k++);
        continue;
      }
      // The point where the cone begins, between this point and the last.
      const f = (start - arc[i - 1]) / Math.max(arc[i] - arc[i - 1], 1e-12);
      const [a, b] = [p(i - 1), p(i)];
      base = a.map((v, c) => v + f * (b[c] - v));
      out.set([...base, start / total], 4 * k++);
      break;
    }
    // The other masses' paths, each after a point that breaks the tube, sharing what room is
    // left: a path with more points than its share is thinned.
    const others = path.others.filter((o) => o.length >= 6);
    const share = others.length ? Math.floor((ALL_POINTS - k) / others.length) - 1 : 0;
    for (const o of others) {
      const m = o.length / 3;
      const take = Math.min(m, share);
      if (take < 2) break;
      out.set([0, 0, 0, -1], 4 * k++);
      for (let i = 0; i < take; i++) {
        const j = Math.round((i * (m - 1)) / (take - 1));
        out.set([o[3 * j] / path.box, o[3 * j + 1] / path.box, o[3 * j + 2] / path.box, 2], 4 * k++);
      }
    }
    const d = this.device;
    d.queue.writeBuffer(this.pts, 0, out, 0, 4 * k);
    d.queue.writeBuffer(this.draw, 0, new Float32Array([k, r, HEAD_RADIUS * r, 0, ...base, 0, ...p(n - 1), 0, ...COLOR]));
    return k - 1;
  }

  /**
   * Draw the path into the tube targets, which are cleared first if nothing has been drawn into
   * them this frame. Returns whether anything was drawn.
   */
  encode(enc: GPUCommandEncoder, path: { points: Float32Array; others: Float32Array[]; box: number; version: number }, t: { color: GPUTexture; dist: GPUTexture; depth: GPUTexture }, clear: boolean): boolean {
    if (!this.pSeg || !this.pHead) return false;
    if (this.loaded.version !== path.version) this.loaded = { version: path.version, segments: this.load(path) };
    const segments = this.loaded.segments;
    if (segments < 0) return false;
    const loadOp: GPULoadOp = clear ? 'clear' : 'load';
    const rp = enc.beginRenderPass({
      label: 'orbit path',
      colorAttachments: [
        { view: t.color.createView(), loadOp, storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: t.dist.createView(), loadOp, storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: t.depth.createView(), depthLoadOp: loadOp, depthStoreOp: 'store', depthClearValue: 1 },
    });
    if (segments > 0) {
      rp.setPipeline(this.pSeg);
      rp.setBindGroup(0, this.gSeg!);
      rp.draw(6, segments);
    }
    rp.setPipeline(this.pHead);
    rp.setBindGroup(0, this.gHead!);
    rp.draw(6);
    rp.end();
    return true;
  }

  destroy() {
    this.pts.destroy();
    this.draw.destroy();
  }
}
