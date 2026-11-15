// An MP4 movie written frame by frame (plans/mp4-export-plan.md, section 5): each frame is
// painted onto a 2D compositing canvas, encoded as H.264 by WebCodecs as soon as it is added,
// and muxed into a file held in memory. Only the encoded stream is kept, never the images.
// Nothing here knows about the simulation.
import { BufferTarget, Mp4OutputFormat, Output, VideoSample, VideoSampleSource, canEncodeVideo } from 'mediabunny';

export interface MovieOptions {
  width: number;
  height: number;
  /** Playback rate: frame i is shown at i / fps seconds. */
  fps: number;
  /** Bits per second; by default about 0.2 bits per pixel per frame (12 Mbit/s at 1080p30). */
  bitrate?: number;
}

/** Sides rounded down to even numbers (H.264 needs them) and capped at `max`. */
export function movieSize(width: number, height: number, max = 4096): [number, number] {
  const side = (v: number) => Math.max(16, Math.min(max, Math.floor(v / 2) * 2));
  return [side(width), side(height)];
}

/** The default bitrate: smooth color-mapped gradients band at the usual web bitrates. */
export const defaultBitrate = (width: number, height: number, fps: number) => Math.round(0.2 * width * height * fps);

/** Why this browser cannot write the movie asked for, or null if it can. */
export async function movieUnsupported(o: MovieOptions): Promise<string | null> {
  if (typeof VideoEncoder === 'undefined') return 'this browser has no video encoder (WebCodecs)';
  const ok = await canEncodeVideo('avc', { width: o.width, height: o.height, bitrate: o.bitrate ?? defaultBitrate(o.width, o.height, o.fps), frameRate: o.fps }).catch(() => false);
  return ok ? null : `this browser cannot encode H.264 video at ${o.width} × ${o.height}`;
}

export class MovieWriter {
  /** The compositing canvas: paint each frame here, then add() it. */
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  readonly fps: number;
  frames = 0;
  private output: Output<Mp4OutputFormat, BufferTarget>;
  private source: VideoSampleSource;
  private done = false;
  /** The frames handed to the encoder so far, each after the one before it. */
  private queue: Promise<void> = Promise.resolve();

  private constructor(o: MovieOptions) {
    this.fps = o.fps;
    this.canvas = document.createElement('canvas');
    this.canvas.width = o.width;
    this.canvas.height = o.height;
    this.ctx = this.canvas.getContext('2d', { alpha: false })!;
    this.output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    this.source = new VideoSampleSource({
      codec: 'avc',
      bitrate: o.bitrate ?? defaultBitrate(o.width, o.height, o.fps),
      keyFrameInterval: 2,
    });
    this.output.addVideoTrack(this.source, { frameRate: o.fps });
  }

  /** A movie ready for frames, or an error saying why this browser cannot write it. */
  static async open(o: MovieOptions): Promise<MovieWriter> {
    const why = await movieUnsupported(o);
    if (why) throw new Error(`cannot record: ${why}`);
    const m = new MovieWriter(o);
    await m.output.start();
    return m;
  }

  /** Seconds of movie so far. */
  get seconds(): number {
    return this.frames / this.fps;
  }

  /**
   * Encode what the compositing canvas holds now as the next frame: the canvas is copied at
   * once and may be painted again. Resolves once the frame before this one has been taken by the
   * encoder, so this one encodes while the caller goes on and frames do not pile up in memory.
   */
  async add(): Promise<void> {
    if (this.done) throw new Error('the movie is already finished');
    const i = this.frames++;
    const sample = new VideoSample(this.canvas, { timestamp: i / this.fps, duration: 1 / this.fps });
    const before = this.queue;
    this.queue = before.then(() => this.source.add(sample)).finally(() => sample.close());
    // A failure surfaces at the next add() or at finish().
    this.queue.catch(() => {});
    await before;
  }

  /** Flush the encoder and close the file: its bytes. */
  async finish(): Promise<Uint8Array> {
    if (this.done) throw new Error('the movie is already finished');
    this.done = true;
    await this.queue;
    this.source.close();
    await this.output.finalize();
    return new Uint8Array(this.output.target.buffer!);
  }

  /** Abandon the movie. */
  async cancel() {
    if (this.done) return;
    this.done = true;
    await this.output.cancel().catch(() => {});
  }
}
