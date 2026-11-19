// What the tabulated equations of state share (eos/helmholtz.ts, eos/hhe.ts; eos/aneos.ts the
// loading, the texture and the Hermite basis only), in float64 and
// as eos/table.wgsl has it: loading a table and giving it to the GPU, and the bicubic Hermite
// lookup of a free energy h(x, y) and its derivatives from (h, h_x, h_y, h_xy) per node. Each
// module keeps what is its own: its terms outside the table, and its solve for the temperature.
import { setSharedBinding } from '../gpu/device';

/** The binding of the table in eos/table_tex.wgsl (EOS_TAB): whichever the run's equation of state uses. */
export const EOS_TABLE_BINDING = 22;

/** A table's texels (rgba32float, `rows` rows of `width`) and the grid of its free energy. */
export interface EosTable {
  data: Float32Array;
  width: number;
  rows: number;
  /** x (log10 of a density) along the rows, y (log10 T) along the columns: first node, spacing, nodes */
  x0: number;
  dx: number;
  nx: number;
  y0: number;
  dy: number;
  ny: number;
  /**
   * The nodes' first channel holds h's difference to the next node along y, and rows nx to
   * 2 nx - 1 its difference to the next along x, instead of h (TAB_DIFF in eos/table.wgsl).
   */
  diff?: boolean;
}

/** Read a table's file: from the page's assets in a browser, from disk under Node. */
export async function loadTableData(url: URL, floats: number, name: string): Promise<Float32Array> {
  let bytes: ArrayBuffer;
  if (url.protocol === 'file:') {
    // Under Node (the unit tests): read the file. The module's name in a variable keeps the
    // bundler and the type checker from looking for it in a browser build.
    const fsName = 'node:fs/promises';
    const { readFile } = (await import(/* @vite-ignore */ fsName)) as { readFile: (u: URL) => Promise<Uint8Array> };
    const b = await readFile(url);
    bytes = new Uint8Array(b).buffer as ArrayBuffer;
  } else {
    bytes = await (await fetch(url)).arrayBuffer();
  }
  const data = new Float32Array(bytes);
  if (data.length !== floats) throw new Error(`${name}: ${data.length} floats, expected ${floats}`);
  return data;
}

export const tableTexel = (t: EosTable, row: number, col: number, c: number): number => t.data[(row * t.width + col) * 4 + c];

/** The cubic Hermite basis at t in [0, 1]: weights of (value at 0, slope at 0, value at 1, slope at 1) (tab_hb in eos/table_tex.wgsl). */
export const hermite = (t: number): [number, number, number, number] => [2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2];
/** Its derivative in t (tab_hb1). */
export const hermite1 = (t: number): [number, number, number, number] => [6 * t ** 2 - 6 * t, 3 * t ** 2 - 4 * t + 1, -6 * t ** 2 + 6 * t, 3 * t ** 2 - 2 * t];
const hb = hermite;
const hb1 = hermite1;
const hb2 = (t: number) => [12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2];

/**
 * h and its derivatives per ln of the density (n) and ln T at (x, y) = (lx, ly). With
 * differences (`diff`), h is counted from the first node of the cell the point is in.
 */
export function tableLookup(t: EosTable, lx: number, ly: number) {
  const x = Math.min(Math.max((lx - t.x0) / t.dx, 0), t.nx - 1.0001);
  const y = Math.min(Math.max((ly - t.y0) / t.dy, 0), t.ny - 1.0001);
  const i = Math.floor(x);
  const j = Math.floor(y);
  const u = x - i;
  const v = y - j;
  const node = (a: number, b: number) => [0, 1, 2, 3].map((c) => tableTexel(t, i + a, j + b, c));
  const c = [node(0, 0), node(0, 1), node(1, 0), node(1, 1)];
  if (t.diff) {
    const d = tableTexel(t, t.nx + i, j, 0);
    c[3][0] = d + c[2][0];
    c[1][0] = c[0][0];
    c[2][0] = d;
    c[0][0] = 0;
  }
  const comb = (A: number[], B: number[]) => {
    let s = 0;
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 2; b++) {
        const q = c[a * 2 + b];
        s += A[2 * a] * B[2 * b] * q[0] + A[2 * a + 1] * B[2 * b] * q[1] + A[2 * a] * B[2 * b + 1] * q[2] + A[2 * a + 1] * B[2 * b + 1] * q[3];
      }
    return s;
  };
  const sx = 1 / (t.dx * Math.LN10);
  const sy = 1 / (t.dy * Math.LN10);
  return {
    h: comb(hb(u), hb(v)),
    hn: comb(hb1(u), hb(v)) * sx,
    ht: comb(hb(u), hb1(v)) * sy,
    hnn: comb(hb2(u), hb(v)) * sx * sx,
    hnt: comb(hb1(u), hb1(v)) * sx * sy,
    htt: comb(hb(u), hb2(v)) * sy * sy,
  };
}

/** Channel c of the table from row `row`, bilinear at (gx, gy) nodes along the rows and the columns (tab_bilinear). */
export function tableBilinear(t: EosTable, row: number, gx: number, gy: number, c: number): number {
  const i = Math.floor(gx);
  const j = Math.floor(gy);
  const u = gx - i;
  const v = gy - j;
  const at = (a: number, b: number) => tableTexel(t, row + i + a, j + b, c);
  return (1 - u) * ((1 - v) * at(0, 0) + v * at(0, 1)) + u * ((1 - v) * at(1, 0) + v * at(1, 1));
}

/** The table as a texture for EOS_TAB. */
export function tableTexture(device: GPUDevice, t: EosTable, label: string): GPUTexture {
  const tex = device.createTexture({ label, size: [t.width, t.rows], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, t.data as Float32Array<ArrayBuffer>, { bytesPerRow: t.width * 16 }, [t.width, t.rows]);
  return tex;
}

const deviceTextures = new WeakMap<GPUDevice, Map<string, GPUTextureView>>();

/**
 * Make `t` the table of a device's runs from here on: uploaded once (per `key`), and
 * registered as the binding every bind group gets whose shader reaches it (gpu/device.ts).
 * Bind groups made earlier keep the table they were made with.
 */
export function bindTable(device: GPUDevice, key: string, t: EosTable): void {
  let m = deviceTextures.get(device);
  if (!m) deviceTextures.set(device, (m = new Map()));
  let view = m.get(key);
  if (!view) m.set(key, (view = tableTexture(device, t, `equation of state table: ${key}`).createView()));
  setSharedBinding(device, EOS_TABLE_BINDING, view);
}
