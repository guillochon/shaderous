// The tabulated equation of state of white dwarf matter that burns (eos/helmholtz.wgsl): the
// electron-positron gas at any degeneracy and temperature, ideal ions and radiation. Its table
// (helmholtz_table.bin, layout in helmholtz_meta.ts) is made by tools/eos/make_table.py from
// the exact gas (tools/eos/electron_gas.py); this file loads it, gives it to the GPU, and holds
// the float64 mirror the initial conditions and the tests use.
//
// Electrons: the dimensionless thermal free energy h(n, T) = F_th / (n k T), less the pair gas
// at zero net charge, is tabulated in log10(rho Ye) and log10 T with its first derivatives, and
// interpolated by bicubic Hermite polynomials (as Timmes & Swesty's Helmholtz table, with
// one degree less); pressure and energy are its first derivatives, so they are consistent:
//   P_e,th = P_pair + n k T dh/dln n,   E_e,th = P_pair (dln P_pair/dln T - 1) - n k T dh/dln T
// Against the exact gas, in the benchmark's single precision: T to 3e-5 and p to 1e-4 at worst,
// c_s^2 to 5e-4 at the 99th percentile.
import { HELM_META as M } from './helmholtz_meta';
import { setSharedBinding } from '../gpu/device';

/** What the hydro needs at a state, per unit mass, cgs (see helm_thermo in eos/helmholtz.wgsl). */
export interface HelmThermo {
  /** thermal energy per gram above the zero-temperature electrons */
  e: number;
  /** p_th / rho */
  pr: number;
  /** de/dln T at fixed density */
  de: number;
  /** d(p_th/rho)/dln T at fixed density */
  dpr: number;
  /** dp_th/drho at fixed T */
  dprho: number;
}

const { k: K, mu: MU, mec2: MEC2, arad: ARAD } = M.consts;
const NQ = (2 * Math.PI * M.consts.me * K) / M.consts.h ** 2;
const LN10 = Math.LN10;
export const HELM_T_MIN = 1e4;
export const HELM_T_MAX = 1e11;

let table: Float32Array | null = null;
let loading: Promise<Float32Array> | null = null;

/** The table, once loaded (loadHelmholtz). */
export function helmholtzTable(): Float32Array {
  if (!table) throw new Error('the Helmholtz table is not loaded: await loadHelmholtz() first');
  return table;
}

/** Load the table (once): from the page's assets in a browser, from disk under Node. */
export function loadHelmholtz(): Promise<Float32Array> {
  if (table) return Promise.resolve(table);
  loading ??= (async () => {
    const url = new URL('./helmholtz_table.bin', import.meta.url);
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
    table = new Float32Array(bytes);
    if (table.length !== M.width * M.rows * 4) throw new Error(`Helmholtz table: ${table.length} floats, expected ${M.width * M.rows * 4}`);
    return table;
  })();
  return loading;
}

/** The table as a texture for binding 22 (eos/helmholtz.wgsl, HELM_TAB). */
export function helmholtzTexture(device: GPUDevice): GPUTexture {
  const t = helmholtzTable();
  const tex = device.createTexture({ label: 'Helmholtz table', size: [M.width, M.rows], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, t as Float32Array<ArrayBuffer>, { bytesPerRow: M.width * 16 }, [M.width, M.rows]);
  return tex;
}

const texel = (row: number, col: number, c: number): number => helmholtzTable()[(row * M.width + col) * 4 + c];

const hb = (t: number) => [2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2];
const hb1 = (t: number) => [6 * t ** 2 - 6 * t, 3 * t ** 2 - 4 * t + 1, -6 * t ** 2 + 6 * t, 3 * t ** 2 - 2 * t];
const hb2 = (t: number) => [12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2];

/** h and its derivatives per ln n and ln T at log10(rho Ye) = lrye, log10 T = lt. */
export function helmLookup(lrye: number, lt: number) {
  const T = M.table;
  const x = Math.min(Math.max((lrye - T.r0) / T.d, 0), T.nr - 1.0001);
  const y = Math.min(Math.max((lt - T.t0) / T.d, 0), T.nt - 1.0001);
  const i = Math.floor(x);
  const j = Math.floor(y);
  const u = x - i;
  const v = y - j;
  const node = (a: number, b: number) => [0, 1, 2, 3].map((c) => texel(i + a, j + b, c));
  const c = [node(0, 0), node(0, 1), node(1, 0), node(1, 1)];
  const comb = (A: number[], B: number[]) => {
    let s = 0;
    for (let a = 0; a < 2; a++)
      for (let b = 0; b < 2; b++) {
        const q = c[a * 2 + b];
        s += A[2 * a] * B[2 * b] * q[0] + A[2 * a + 1] * B[2 * b] * q[1] + A[2 * a] * B[2 * b + 1] * q[2] + A[2 * a + 1] * B[2 * b + 1] * q[3];
      }
    return s;
  };
  const s = 1 / (T.d * LN10);
  return {
    h: comb(hb(u), hb(v)),
    hn: comb(hb1(u), hb(v)) * s,
    ht: comb(hb(u), hb1(v)) * s,
    hnn: comb(hb2(u), hb(v)) * s * s,
    hnt: comb(hb1(u), hb1(v)) * s * s,
    htt: comb(hb(u), hb2(v)) * s * s,
  };
}

const cheb = (c: readonly number[], x: number): number => {
  let b1 = 0;
  let b2 = 0;
  for (let k = c.length - 1; k >= 1; k--) {
    const b0 = 2 * x * b1 - b2 + c[k];
    b2 = b1;
    b1 = b0;
  }
  return x * b1 - b2 + c[0];
};

/** The pair gas at zero charge per unit mass at density rho: [P_p / rho, dln P_p/dln T, d^2 ln P_p/dln T^2]. */
export function helmPairs(lt: number, rho: number): [number, number, number] {
  const P = M.pair;
  const lnT = lt * LN10;
  const T = Math.exp(lnT);
  const beta = (K * T) / MEC2;
  const x = Math.min(Math.max((lt - P.mid) / P.half, -1), 1);
  const s = 1 / (P.half * LN10);
  const lnP = Math.log((4 * K) / rho) + lnT + 1.5 * Math.log(NQ * T) - 1 / beta + cheb(P.q, x);
  return [Math.exp(lnP), 2.5 + 1 / beta + cheb(P.dq, x) * s, -1 / beta + cheb(P.d2q, x) * s * s];
}

/** Everything above the zero-temperature electrons at density rho (g/cm^3), ln T, Ye and ions per amu ia. */
export function helmThermo(rho: number, lnT: number, ye: number, ia: number): HelmThermo {
  const lt = lnT / LN10;
  const T = Math.exp(lnT);
  const h = helmLookup(Math.log10(rho * ye), lt);
  const [pp, l1, l2] = helmPairs(lt, rho);
  const sm = (ye * K * T) / MU;
  const im = (ia * K * T) / MU;
  const rm = (ARAD * T ** 4) / rho;
  return {
    e: pp * (l1 - 1) - sm * h.ht + 1.5 * im + rm,
    de: pp * (l1 * l1 - l1 + l2) - sm * (h.ht + h.htt) + 1.5 * im + 4 * rm,
    pr: pp + sm * h.hn + im + rm / 3,
    dpr: pp * l1 + sm * (h.hn + h.hnt) + im + (4 * rm) / 3,
    dprho: sm * (h.hn + h.hnn) + im,
  };
}

/** Newton's starting point, log10 T, from the inverse tables (byP: from p_th / rho, else e_th). */
export function helmGuess(rho: number, x: number, ia: number, byP: boolean): number {
  const G = M.guess;
  const gx = Math.min(Math.max((Math.log10(rho) - G.rho0) / G.drho, 0), G.n_rho - 1.0001);
  const gy = Math.min(Math.max((Math.log10(x) - G.x0) / G.dx, 0), G.n_x - 1.0001);
  const i = Math.floor(gx);
  const j = Math.floor(gy);
  const u = gx - i;
  const v = gy - j;
  const row = byP ? G.row_p : G.row_e;
  const at = (c: number) => {
    const t = (a: number, b: number) => texel(row + i + a, j + b, c);
    return (1 - u) * ((1 - v) * t(0, 0) + v * t(0, 1)) + u * ((1 - v) * t(1, 0) + v * t(1, 1));
  };
  const la = Math.min(Math.max(-Math.log(ia), Math.log(4)), Math.log(56));
  if (la > Math.log(14)) return at(1) + ((la - Math.log(14)) / (Math.log(56) - Math.log(14))) * (at(2) - at(1));
  return at(0) + ((la - Math.log(4)) / (Math.log(14) - Math.log(4))) * (at(1) - at(0));
}

/**
 * ln T at which the thermal energy per gram (byP false) or p_th / rho (byP true) is `target`,
 * as helm_solve, but iterated to convergence. -Infinity for no thermal energy.
 */
export function helmSolve(rho: number, target: number, ye: number, ia: number, byP: boolean): number {
  if (!(target > 0)) return -Infinity;
  const lo = Math.log(HELM_T_MIN);
  const hi = Math.log(HELM_T_MAX);
  let lnT = Math.min(Math.max(helmGuess(rho, target, ia, byP) * LN10, lo), hi);
  for (let it = 0; it < 60; it++) {
    const t = helmThermo(rho, lnT, ye, ia);
    const f = byP ? t.pr : t.e;
    if (lnT <= lo && f > target) return lo + Math.log(target / f);
    if (lnT >= hi && f < target) return hi + 0.25 * Math.log(target / f);
    const step = Math.min(Math.max(-(f - target) / (byP ? t.dpr : t.de), -1), 1);
    lnT = Math.min(Math.max(lnT + step, lo), hi);
    if (Math.abs(step) < 1e-12) break;
  }
  return lnT;
}

/** How many Newton iterations helm_solve takes from its starting point to 2e-6 (for the tests). */
export function helmIterations(rho: number, target: number, ye: number, ia: number, byP: boolean): number {
  let lnT = Math.min(Math.max(helmGuess(rho, target, ia, byP) * LN10, Math.log(HELM_T_MIN)), Math.log(HELM_T_MAX));
  for (let it = 1; it <= 10; it++) {
    const t = helmThermo(rho, lnT, ye, ia);
    const step = -((byP ? t.pr : t.e) - target) / (byP ? t.dpr : t.de);
    lnT += Math.min(Math.max(step, -1), 1);
    if (Math.abs(step) < 2e-6) return it;
  }
  return 10;
}

const deviceTextures = new WeakMap<GPUDevice, GPUTexture>();

/**
 * Ready a device for runs with this equation of state: the table loaded, uploaded once, and
 * registered as binding 22 of every bind group whose shader reaches it (gpu/device.ts).
 */
export async function prepareHelmholtz(device: GPUDevice): Promise<void> {
  await loadHelmholtz();
  if (deviceTextures.has(device)) return;
  const t = helmholtzTexture(device);
  deviceTextures.set(device, t);
  setSharedBinding(device, HELM_BINDING, t.createView());
}

/** The binding of the table in eos/helmholtz.wgsl (HELM_TAB). */
export const HELM_BINDING = 22;
