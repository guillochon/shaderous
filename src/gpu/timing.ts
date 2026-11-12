// GPU time per pipeline, from timestamp queries, while the user asks for it.
//
// WebGPU only timestamps whole passes, and the simulation records many dispatches in each
// pass. While timing is on, beginPass() hands out a stand-in for a compute pass that ends the
// real pass and opens a new, timestamped one whenever the pipeline changes, so each run of
// dispatches of one pipeline is timed on its own. Splitting costs a little (the GPU cannot
// overlap work across passes), but only while timing is on: off, beginPass() is the plain
// beginComputePass(). Results come back asynchronously, a frame or two late; a frame whose
// readback buffer is still busy goes untimed rather than waiting.

const QUERIES_PER_SET = 4096; // the WebGPU maximum for one query set
const MAX_SETS = 16; // up to 32768 timed sections a frame
const STAGING = 3; // readbacks in flight at once

export interface TimingRow {
  label: string;
  /** Milliseconds of GPU time per step. */
  msPerStep: number;
  /** Share of the timed GPU time. */
  share: number;
  /** Passes of this pipeline per step. */
  passesPerStep: number;
}

export interface TimingReport {
  rows: TimingRow[];
  /** Timed GPU milliseconds per step, all pipelines together. */
  msPerStep: number;
  steps: number;
  frames: number;
  /** Frames left untimed because every readback buffer was busy. */
  skipped: number;
}

interface Frame {
  labels: string[];
  steps: number;
}

class GpuTimer {
  enabled = false;
  private sets: GPUQuerySet[] = [];
  private resolve: GPUBuffer | null = null;
  private free: GPUBuffer[] = [];
  private frame: Frame | null = null;
  private used = 0; // queries written in the open frame
  private pending: { buf: GPUBuffer; frame: Frame; bytes: number } | null = null;
  // Totals since the last report.
  private ns = new Map<string, number>();
  private passes = new Map<string, number>();
  private steps = 0;
  private frames = 0;
  private skipped = 0;

  constructor(private device: GPUDevice) {}

  /** Start timing the passes of a frame. Returns false (and times nothing) if no readback buffer is free. */
  open(): boolean {
    if (!this.enabled) return false;
    if (!this.resolve) {
      this.resolve = this.device.createBuffer({
        label: 'timing resolve',
        size: QUERIES_PER_SET * MAX_SETS * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      for (let i = 0; i < STAGING; i++) {
        this.free.push(this.device.createBuffer({ label: 'timing readback', size: this.resolve.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      }
    }
    if (!this.free.length) {
      this.skipped++;
      return false;
    }
    this.frame = { labels: [], steps: 0 };
    this.used = 0;
    return true;
  }

  /** Timestamp writes for a new section of the open frame, or undefined when out of queries. */
  section(label: string): GPUComputePassTimestampWrites | undefined {
    const f = this.frame;
    if (!f) return undefined;
    const set = Math.floor(this.used / QUERIES_PER_SET);
    if (set >= MAX_SETS) return undefined;
    while (this.sets.length <= set) {
      this.sets.push(this.device.createQuerySet({ label: `timing ${this.sets.length}`, type: 'timestamp', count: QUERIES_PER_SET }));
    }
    const at = this.used % QUERIES_PER_SET;
    this.used += 2;
    f.labels.push(label);
    return { querySet: this.sets[set], beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 };
  }

  get timing(): boolean {
    return this.frame !== null;
  }

  /** End the frame: resolve its queries into a readback buffer. Call before submitting `enc`. */
  close(enc: GPUCommandEncoder, steps: number) {
    const f = this.frame;
    this.frame = null;
    if (!f) return;
    f.steps = steps;
    if (!this.used) return;
    const buf = this.free.pop()!;
    for (let s = 0; s * QUERIES_PER_SET < this.used; s++) {
      const n = Math.min(QUERIES_PER_SET, this.used - s * QUERIES_PER_SET);
      enc.resolveQuerySet(this.sets[s], 0, n, this.resolve!, s * QUERIES_PER_SET * 8);
    }
    enc.copyBufferToBuffer(this.resolve!, 0, buf, 0, this.used * 8);
    this.pending = { buf, frame: f, bytes: this.used * 8 };
  }

  /** Call after submitting the encoder close() wrote to: reads the times back when they land. */
  submitted() {
    const p = this.pending;
    this.pending = null;
    if (!p) return;
    p.buf
      .mapAsync(GPUMapMode.READ, 0, p.bytes)
      .then(() => {
        const t = new BigInt64Array(p.buf.getMappedRange(0, p.bytes));
        if (this.enabled) this.add(p.frame, t);
        p.buf.unmap();
      })
      .catch(() => {})
      .finally(() => this.free.push(p.buf));
  }

  private add(f: Frame, t: BigInt64Array) {
    f.labels.forEach((label, i) => {
      // Timestamps may be quantized (Chrome rounds to 100 µs unless its developer features are on).
      const d = Math.max(0, Number(t[2 * i + 1] - t[2 * i]));
      this.ns.set(label, (this.ns.get(label) ?? 0) + d);
      this.passes.set(label, (this.passes.get(label) ?? 0) + 1);
    });
    this.steps += f.steps;
    this.frames++;
  }

  /** What was timed since the last call, slowest pipeline first; then start over. */
  report(): TimingReport {
    const total = [...this.ns.values()].reduce((a, b) => a + b, 0);
    const steps = Math.max(this.steps, 1);
    const rows = [...this.ns.entries()]
      .map(([label, ns]) => ({ label, msPerStep: ns / 1e6 / steps, share: total ? ns / total : 0, passesPerStep: this.passes.get(label)! / steps }))
      .sort((a, b) => b.msPerStep - a.msPerStep);
    const out = { rows, msPerStep: total / 1e6 / steps, steps: this.steps, frames: this.frames, skipped: this.skipped };
    this.ns.clear();
    this.passes.clear();
    this.steps = this.frames = this.skipped = 0;
    return out;
  }

  /** Drop what has been gathered (a new run, or timing switched off). */
  clear() {
    this.report();
  }
}

/**
 * Stand-in for a compute pass while timing: the four calls the simulation makes, with a new
 * timestamped pass at each change of pipeline. Bind groups carry over to the new pass, as
 * they would across setPipeline() in a single pass.
 */
class TimedPass {
  private pass: GPUComputePassEncoder | null = null;
  private label: string | null = null;
  private bound = new Map<number, [GPUBindGroup | null, Iterable<number> | undefined]>();

  constructor(private enc: GPUCommandEncoder, private timer: GpuTimer) {}

  private open(label: string) {
    this.pass = this.enc.beginComputePass({ label, timestampWrites: this.timer.section(label) });
    this.label = label;
    for (const [i, [g, offsets]] of this.bound) this.bind(this.pass, i, g, offsets);
  }

  // Chrome rejects an explicit undefined for the offsets: pass them only when there are some.
  private bind(pass: GPUComputePassEncoder, i: number, g: GPUBindGroup | null, offsets?: Iterable<number>) {
    if (offsets) pass.setBindGroup(i, g, offsets);
    else pass.setBindGroup(i, g);
  }

  setPipeline(p: GPUComputePipeline) {
    const label = p.label || '(unlabelled)';
    if (this.pass && this.label !== label) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.pass) this.open(label);
    this.pass!.setPipeline(p);
  }

  setBindGroup(i: number, g: GPUBindGroup | null, offsets?: Iterable<number>) {
    this.bound.set(i, [g, offsets ? [...offsets] : undefined]);
    if (this.pass) this.bind(this.pass, i, g, offsets);
  }

  dispatchWorkgroups(x: number, y?: number, z?: number) {
    this.pass!.dispatchWorkgroups(x, y ?? 1, z ?? 1);
  }

  dispatchWorkgroupsIndirect(buf: GPUBuffer, offset: number) {
    this.pass!.dispatchWorkgroupsIndirect(buf, offset);
  }

  end() {
    this.pass?.end();
    this.pass = null;
  }
}

let timer: GpuTimer | null = null;

/** Set up timing for a device that has the timestamp-query feature (see initGpu). */
export function initTiming(device: GPUDevice) {
  timer = device.features.has('timestamp-query') ? new GpuTimer(device) : null;
}

export function timingAvailable(): boolean {
  return timer !== null;
}

export function setTiming(on: boolean) {
  if (!timer) return;
  timer.enabled = on;
  timer.clear();
}

/** A compute pass: timed per pipeline inside a frame the timer opened, plain otherwise. */
export function beginPass(enc: GPUCommandEncoder): GPUComputePassEncoder {
  if (!timer?.timing) return enc.beginComputePass();
  return new TimedPass(enc, timer) as unknown as GPUComputePassEncoder;
}

/** Frame hooks for the main loop: open before encoding, close before submit, then submitted(). */
export const frameTiming = {
  open: () => timer?.open() ?? false,
  close: (enc: GPUCommandEncoder, steps: number) => timer?.close(enc, steps),
  submitted: () => timer?.submitted(),
  report: (): TimingReport | null => (timer?.enabled ? timer.report() : null),
  clear: () => timer?.clear(),
};
