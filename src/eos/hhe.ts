// The tabulated equation of state of hydrogen and helium (eos/hhe.wgsl): H2, H, H+, He, He+,
// He++ and electrons in chemical equilibrium, from molecular clouds to stellar envelopes. The
// free energy is src/eos/hhe_model.ts; tools/eos/make_hhe_table.mjs puts it on a grid, one
// table per composition (hhe_<name>.bin, layout in hhe_meta.ts). This file loads the tables,
// gives the run's to the GPU, and holds the float64 mirror the initial conditions and the
// tests use. The lookup is eos/table.ts, shared with eos/helmholtz.ts.
//
// With h = F m_u / (k T) per atomic mass unit on log10 rho and log10 T,
//   p / rho = (k T / m_u) dh/dln rho + a T^4 / (3 rho),   e = -(k T / m_u) dh/dln T + a T^4 / rho
// and the radiation term is optional (`rad`, a multiplier). Against the model, between the
// nodes and below 0.1 g/cm^3: p to 6e-5 and e to 4e-4 at worst, c_s^2 to 4e-3 (the worst are
// on the ionization and dissociation fronts at the lowest densities, where they are sharpest).
import { HHE_META as M } from './hhe_meta';
import { HHE } from './hhe_model';
import { bindTable, loadTableData, tableBilinear, tableLookup, type EosTable } from './table';

export type HheTableName = (typeof M.tables)[number]['name'];

/** As HheThermo in eos/hhe.wgsl: per unit mass, cgs. */
export interface HheThermo {
  /** energy per gram, zero at T = 0 */
  e: number;
  /** p / rho */
  pr: number;
  /** de/dln T at fixed density */
  de: number;
  /** d(p / rho)/dln T at fixed density */
  dpr: number;
  /** dp/drho at fixed T */
  dprho: number;
}

const L = M.table;
const KMU = HHE.k / HHE.mu;
const LN10 = Math.LN10;
export const HHE_T_MIN = 10 ** L.t0;
export const HHE_T_MAX = 10 ** (L.t0 + (L.nt - 1) * L.dt);
const LO = Math.log(HHE_T_MIN);
const HI = Math.log(HHE_T_MAX);
/** A Newton step in ln T this small is taken without another evaluation (TAB_LAST_STEP in eos/table.wgsl). */
const LAST_STEP = 3e-4;
/** The sound speed squared is at least this share of p / rho (HH_C2_MIN in eos/hhe.wgsl). */
export const HHE_C2_MIN = 0.01;

/**
 * The table for gas whose hydrogen mass fraction is X, or null if none ships for it: solar
 * (the code's solar gas, its metals counted as helium) and hydrogen.
 */
export function hheTableFor(X: number): HheTableName | null {
  return M.tables.find((t) => Math.abs(t.X - X) < 2e-3)?.name ?? null;
}

/** What the tables are for, for a message. */
export const HHE_TABLES_SAY = M.tables.map((t) => (t.X === 1 ? t.name : `${t.name} (X = ${t.X})`)).join(' and ');

// Each file named in full, for the bundler.
const URLS: Record<HheTableName, () => URL> = {
  solar: () => new URL('./hhe_solar.bin', import.meta.url),
  hydrogen: () => new URL('./hhe_hydrogen.bin', import.meta.url),
};
const tables = new Map<HheTableName, EosTable>();
const loading = new Map<HheTableName, Promise<EosTable>>();

/** Load a composition's table (once): from the page's assets in a browser, from disk under Node. */
export function loadHhe(name: HheTableName): Promise<EosTable> {
  let p = loading.get(name);
  if (!p) {
    p = (async () => {
      const data = await loadTableData(URLS[name](), M.width * M.rows * 4, `hydrogen and helium table (${name})`);
      const t: EosTable = { data, width: M.width, rows: M.rows, x0: L.r0, dx: L.dr, nx: L.nr, y0: L.t0, dy: L.dt, ny: L.nt, diff: true };
      tables.set(name, t);
      return t;
    })();
    loading.set(name, p);
  }
  return p;
}

const loaded = (name: HheTableName): EosTable => {
  const t = tables.get(name);
  if (!t) throw new Error(`the hydrogen and helium table (${name}) is not loaded: await loadHhe() first`);
  return t;
};

/**
 * Ready a device for runs with this equation of state and composition: the table loaded,
 * uploaded once, and made the table of every bind group whose shader reaches it from here on.
 */
export async function prepareHhe(device: GPUDevice, name: HheTableName): Promise<void> {
  bindTable(device, `hhe ${name}`, await loadHhe(name));
}

const radiation = (T: number, rho: number, rad: number) => (rad * HHE.arad * T ** 4) / rho;

/**
 * The state at density rho (g/cm^3) and ln T; `rad` multiplies the radiation term (0: gas
 * alone). Beyond the table in T, the lookup of its nearest end with the true temperature in
 * front, as hhe_thermo.
 */
export function hheThermo(name: HheTableName, rho: number, lnT: number, rad = 0): HheThermo {
  const T = Math.exp(lnT);
  const h = tableLookup(loaded(name), Math.log10(rho), lnT / LN10);
  const sm = KMU * T;
  const rm = radiation(T, rho, rad);
  return {
    e: -sm * h.ht + rm,
    de: -sm * (h.ht + h.htt) + 4 * rm,
    pr: sm * h.hn + rm / 3,
    dpr: sm * (h.hn + h.hnt) + (4 * rm) / 3,
    dprho: sm * (h.hn + h.hnn),
  };
}

/** c_s^2 (cm^2/s^2) of a state, floored as hhe_c2. */
export const hheSound2 = (t: HheThermo): number => Math.max(t.dprho + (t.dpr * t.dpr) / t.de, HHE_C2_MIN * t.pr);

/** Newton's starting point, ln T (hhe_solve), and the most ln T may be on the way. */
function start(t: EosTable, rho: number, target: number, byP: boolean, rad: number): [number, number] {
  const gx = Math.min(Math.max((Math.log10(rho) - L.r0) / L.dr, 0), L.nr - 1.0001);
  const gy = Math.min(Math.max((Math.log10(target) - (byP ? L.p0 : L.e0)) / (byP ? L.dp : L.de), 0), L.nt - 1.0001);
  let lnT = Math.min(Math.max(tableBilinear(t, L.nr, gx, gy, byP ? 2 : 1) * LN10, LO), HI);
  let hi = HI;
  if (rad > 0) {
    hi = Math.min(Math.max(0.25 * Math.log(((byP ? 3 : 1) * target * rho) / (rad * HHE.arad)), LO), HI);
    lnT = Math.min(lnT, hi);
  }
  return [lnT, hi];
}

/** How far past the table's top, in ln T, the goal lies (tab_above in eos/hhe.wgsl). */
function above(f: number, rho: number, target: number, byP: boolean, rad: number): number {
  const r = radiation(HHE_T_MAX, rho, rad) * (byP ? 1 / 3 : 1);
  const g = Math.max(f - r, 1e-30 * f);
  let d = Math.log(target / g);
  if (r > 0) d = Math.min(d, 0.25 * Math.log(target / r));
  for (let it = 0; it < 5; it++) {
    const eg = g * Math.exp(d);
    const er = r * Math.exp(4 * d);
    d = Math.max(d - (eg + er - target) / (eg + 4 * er), 0);
  }
  return d;
}

function solve(name: HheTableName, rho: number, target: number, byP: boolean, rad: number, iters: number, tol: number): { lnT: number; its: number } {
  if (!(target > 0)) return { lnT: -Infinity, its: 0 };
  const t = loaded(name);
  let [lnT, hi] = start(t, rho, target, byP, rad);
  for (let it = 1; it <= iters; it++) {
    const s = hheThermo(name, rho, lnT, rad);
    const f = byP ? s.pr : s.e;
    if (lnT <= LO && f > target) return { lnT: LO + Math.log(target / f), its: it };
    if (lnT >= HI && f < target) return { lnT: HI + above(f, rho, target, byP, rad), its: it };
    const step = Math.min(Math.max(-(f - target) / (byP ? s.dpr : s.de), -1), 1);
    lnT = Math.min(Math.max(lnT + step, LO), hi);
    if (Math.abs(step) < tol && (tol < 1e-6 || (lnT > LO && lnT < HI))) return { lnT, its: it };
  }
  return { lnT, its: iters };
}

/**
 * ln T at which the energy per gram (byP false) or p / rho (byP true) is `target`, as
 * hhe_solve, but iterated to convergence. -Infinity for none.
 */
export function hheSolve(name: HheTableName, rho: number, target: number, byP: boolean, rad = 0): number {
  return solve(name, rho, target, byP, rad, 60, 1e-12).lnT;
}

/** hhe_solve as the shader runs it (for the tests): at most 10 evaluations, the last small step on trust. */
export function hheSolveShader(name: HheTableName, rho: number, target: number, byP: boolean, rad = 0): { lnT: number; its: number } {
  return solve(name, rho, target, byP, rad, 10, LAST_STEP);
}
