// The tabulated equation of state of white dwarf matter that burns (eos/helmholtz.wgsl): the
// electron-positron gas at any degeneracy and temperature, ideal ions and radiation. Its table
// (helmholtz_table.bin, layout in helmholtz_meta.ts) is made by tools/eos/make_table.py from
// the exact gas (tools/eos/electron_gas.py); this file loads it, gives it to the GPU, and holds
// the float64 mirror the initial conditions and the tests use (the lookup itself is
// eos/table.ts, shared with eos/hhe.ts). The ions' Coulomb energy is analytic and needs no
// table: see helmCoulombFns below.
//
// Electrons: the dimensionless thermal free energy h(n, T) = F_th / (n k T), less the pair gas
// at zero net charge, is tabulated in log10(rho Ye) and log10 T with its first derivatives, and
// interpolated by bicubic Hermite polynomials (as Timmes & Swesty's Helmholtz table, with
// one degree less); pressure and energy are its first derivatives, so they are consistent:
//   P_e,th = P_pair + n k T dh/dln n,   E_e,th = P_pair (dln P_pair/dln T - 1) - n k T dh/dln T
// Against the exact gas, in the benchmark's single precision: T to 3e-5 and p to 1e-4 at worst,
// c_s^2 to 5e-4 at the 99th percentile.
import { HELM_META as M } from './helmholtz_meta';
import { bindTable, EOS_TABLE_BINDING, loadTableData, tableBilinear, tableLookup, tableTexture, type EosTable } from './table';

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
/** log10 T below which the pair gas is left out (HP_NONE in eos/helmholtz.wgsl). */
export const HELM_PAIRS_NONE = 8.3;
/** A Newton step in ln T this small is taken without another evaluation (TAB_LAST_STEP in eos/table.wgsl). */
export const HELM_LAST_STEP = 3e-4;
export const HELM_T_MAX = 1e11;

let table: EosTable | null = null;
let loading: Promise<Float32Array> | null = null;

const loaded = (): EosTable => {
  if (!table) throw new Error('the Helmholtz table is not loaded: await loadHelmholtz() first');
  return table;
};

/** The table, once loaded (loadHelmholtz). */
export function helmholtzTable(): Float32Array {
  return loaded().data;
}

/** Load the table (once): from the page's assets in a browser, from disk under Node. */
export function loadHelmholtz(): Promise<Float32Array> {
  if (table) return Promise.resolve(table.data);
  loading ??= (async () => {
    const data = await loadTableData(new URL('./helmholtz_table.bin', import.meta.url), M.width * M.rows * 4, 'Helmholtz table');
    const T = M.table;
    table = { data, width: M.width, rows: M.rows, x0: T.r0, dx: T.d, nx: T.nr, y0: T.t0, dy: T.d, ny: T.nt };
    return data;
  })();
  return loading;
}

/** The table as a texture for binding 22 (eos/table.wgsl, EOS_TAB). */
export function helmholtzTexture(device: GPUDevice): GPUTexture {
  return tableTexture(device, loaded(), 'Helmholtz table');
}

/** h and its derivatives per ln n and ln T at log10(rho Ye) = lrye, log10 T = lt. */
export function helmLookup(lrye: number, lt: number) {
  return tableLookup(loaded(), lrye, lt);
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
  // Below 10^8.3 K, none, as the shader has it (HP_NONE): under 1e-8 of the radiation there.
  if (lt < HELM_PAIRS_NONE) return [0, 0, 0];
  const P = M.pair;
  const lnT = lt * LN10;
  const T = Math.exp(lnT);
  const beta = (K * T) / MEC2;
  const x = Math.min(Math.max((lt - P.mid) / P.half, -1), 1);
  const s = 1 / (P.half * LN10);
  const lnP = Math.log((4 * K) / rho) + lnT + 1.5 * Math.log(NQ * T) - 1 / beta + cheb(P.q, x);
  return [Math.exp(lnP), 2.5 + 1 / beta + cheb(P.dq, x) * s, -1 / beta + cheb(P.d2q, x) * s * s];
}

// ---- The ions' Coulomb energy ----
//
// The one-component plasma's excess free energy per ion over k T, f(Gamma), with the coupling
// Gamma = Zbar^2 e^2 / (a k T) and a the ion-sphere radius: the fits Helmholtz uses (Yakovlev &
// Shalybkov 1989), written as the free energy they are derivatives of. Gamma goes as
// rho^(1/3) / T, so the energy per ion is k T u with u = Gamma f', and the pressure a third of
// the energy density.
//   Gamma >= 1:  u = a1 G + b1 G^(1/4) + c1 G^(-1/4) + d1
//   Gamma <  1:  u = -3 c2 G^(3/2) + a2 G^b2          (c2 = 1 / (2 sqrt 3): Debye-Hueckel)
// a1 Gamma k T does not depend on T (the ion spheres' Madelung energy): that part is cold and
// joins the zero-temperature electrons. The rest (f_th, g below) is thermal, positive, and
// rises with T, as the solve for the temperature needs.
//
// Fully ionized ion spheres are the wrong model for thin cold matter, and there the cold term
// alone would make the pressure negative (below rho_z = 0.18 Zbar^2 / Ye g/cm^3). So the whole
// Coulomb free energy is faded by w = r / (1 + r), r = rho / (64 rho_z): a function of density
// alone, applied to the free energy, so pressure and energy stay consistent (the pressure
// gains the term from w's slope).
const CA1 = -0.898004;
const CB1 = 0.96786;
const CC1 = 0.220703;
const CD1 = -0.86097;
const CA2 = 0.29561;
const CB2 = 1.9885;
const CC2 = 0.288675;
/** f_th continuous at Gamma = 1 */
const CK = -CA1 - 2 * CC2 + CA2 / CB2 - 4 * CB1 + 4 * CC1;
/** e^2, cgs */
const E2 = 2.30707751e-19;
/** Gamma T = HELM_GAMMA_T Zbar^2 (rho / Abar)^(1/3), cgs and kelvin */
export const HELM_GAMMA_T = (E2 / K) * Math.cbrt((4 * Math.PI) / (3 * MU));
/** The fade's density scale is HELM_FADE Zbar^2 / Ye g/cm^3: 64 times the density at which cold carbon-like matter would reach zero pressure. */
export const HELM_FADE = (() => {
  const c = Math.sqrt(MEC2 / M.consts.me);
  const aDeg = (Math.PI * M.consts.me ** 4 * c ** 5) / (3 * M.consts.h ** 3);
  const rhoD = ((MU * 8 * Math.PI) / 3) * ((M.consts.me * c) / M.consts.h) ** 3;
  const cube = ((-CA1 / 3) * E2 * Math.cbrt((4 * Math.PI) / 3) * MU ** (-4 / 3)) / (aDeg * 1.6 * rhoD ** (-5 / 3));
  return 64 * cube ** 3;
})();

/** The thermal Coulomb functions at coupling G: [f_th, g = G f_th', G g']. */
export function helmCoulombFns(G: number): [number, number, number] {
  if (G >= 1) {
    const q = Math.sqrt(Math.sqrt(G));
    return [4 * CB1 * q - (4 * CC1) / q + CD1 * Math.log(G) + CK, CB1 * q + CC1 / q + CD1, 0.25 * (CB1 * q - CC1 / q)];
  }
  const s = G * Math.sqrt(G);
  const y = G ** CB2;
  return [-CA1 * G - 2 * CC2 * s + (CA2 / CB2) * y, -CA1 * G - 3 * CC2 * s + CA2 * y, -CA1 * G - 4.5 * CC2 * s + CA2 * CB2 * y];
}

/** The fade w and the cold Coulomb energy per gram before it (negative), at density rho (g/cm^3). */
function coulombScale(rho: number, ye: number, ia: number): { w: number; e0: number; gT: number } {
  const z = ye / ia;
  const r = (rho * ye) / (HELM_FADE * z * z);
  const gT = HELM_GAMMA_T * z * z * Math.cbrt(rho * ia);
  return { w: r / (1 + r), e0: ((CA1 * ia * K) / MU) * gT, gT };
}

/**
 * The cold part of the Coulomb term per unit mass, cgs, times the multiplier m: energy (erg/g),
 * p / rho, and dp/drho.
 */
export function helmCoulombCold(rho: number, ye: number, ia: number, m: number): { e: number; pr: number; dprho: number } {
  if (!(m > 0) || !(rho > 0)) return { e: 0, pr: 0, dprho: 0 };
  const { w, e0 } = coulombScale(rho, ye, ia);
  const s = m * w * e0;
  return { e: s, pr: s * (4 / 3 - w), dprho: s * ((4 / 3) * (4 / 3 - w) + (1 - w) * (4 / 3 - 2 * w)) };
}

/**
 * The cold Coulomb curve in code units, for what builds stars on it (problems/white_dwarf.ts):
 * the energy per unit mass is c1 rho^(1/3) w, with w = r / (1 + r) and r = rho / rho0. `cr` and
 * `ce` are g/cm^3 and erg/g per code unit (EosParams.helmRho, helmE).
 */
export function helmCoulombColdScales(ye: number, ia: number, m: number, cr: number, ce: number): { c1: number; rho0: number } {
  const z = ye / ia;
  return { c1: (((m * CA1 * ia * K) / MU) * HELM_GAMMA_T * z * z * Math.cbrt(cr * ia)) / ce, rho0: (HELM_FADE * z * z) / ye / cr };
}

/** The thermal part of the Coulomb term, in the terms of HelmThermo, times the multiplier m. */
export function helmCoulombThermal(rho: number, lnT: number, ye: number, ia: number, m: number): HelmThermo {
  if (!(m > 0) || !(rho > 0)) return { e: 0, pr: 0, de: 0, dpr: 0, dprho: 0 };
  const { w, gT } = coulombScale(rho, ye, ia);
  const T = Math.exp(lnT);
  const [f, g, gp] = helmCoulombFns(gT / T);
  const s = (m * w * ia * K * T) / MU;
  const pr = s * (g / 3 + (1 - w) * f);
  return {
    e: s * g,
    de: s * (g - gp),
    pr,
    dpr: s * ((g - gp) / 3 + (1 - w) * (f - g)),
    dprho: pr + s * (gp / 9 + (2 / 3) * (1 - w) * g + (1 - w) * (1 - 2 * w) * f),
  };
}

/** The Coulomb free energy per gram (erg/g), cold and thermal together, times m: what the pieces above are derivatives of (for the tests). */
export function helmCoulombFree(rho: number, lnT: number, ye: number, ia: number, m: number): number {
  const { w, gT } = coulombScale(rho, ye, ia);
  const T = Math.exp(lnT);
  const G = gT / T;
  return ((m * w * ia * K * T) / MU) * (CA1 * G + helmCoulombFns(G)[0]);
}

/**
 * Everything above the zero-temperature matter at density rho (g/cm^3), ln T, Ye and ions per
 * amu ia; with `coulomb` (a multiplier, 0 for none), the thermal part of the ions' Coulomb
 * energy too.
 */
export function helmThermo(rho: number, lnT: number, ye: number, ia: number, coulomb = 0): HelmThermo {
  const lt = lnT / LN10;
  const T = Math.exp(lnT);
  const h = helmLookup(Math.log10(rho * ye), lt);
  const [pp, l1, l2] = helmPairs(lt, rho);
  const sm = (ye * K * T) / MU;
  const im = (ia * K * T) / MU;
  const rm = (ARAD * T ** 4) / rho;
  const c = helmCoulombThermal(rho, lnT, ye, ia, coulomb);
  return {
    e: pp * (l1 - 1) - sm * h.ht + 1.5 * im + rm + c.e,
    de: pp * (l1 * l1 - l1 + l2) - sm * (h.ht + h.htt) + 1.5 * im + 4 * rm + c.de,
    pr: pp + sm * h.hn + im + rm / 3 + c.pr,
    dpr: pp * l1 + sm * (h.hn + h.hnt) + im + (4 * rm) / 3 + c.dpr,
    dprho: sm * (h.hn + h.hnn) + im + c.dprho,
  };
}

/** Newton's starting point, log10 T, from the inverse tables (byP: from p_th / rho, else e_th). */
export function helmGuess(rho: number, x: number, ia: number, byP: boolean): number {
  const G = M.guess;
  const gx = Math.min(Math.max((Math.log10(rho) - G.rho0) / G.drho, 0), G.n_rho - 1.0001);
  const gy = Math.min(Math.max((Math.log10(x) - G.x0) / G.dx, 0), G.n_x - 1.0001);
  const at = (c: number) => tableBilinear(loaded(), byP ? G.row_p : G.row_e, gx, gy, c);
  const la = Math.min(Math.max(-Math.log(ia), Math.log(4)), Math.log(56));
  if (la > Math.log(14)) return at(1) + ((la - Math.log(14)) / (Math.log(56) - Math.log(14))) * (at(2) - at(1));
  return at(0) + ((la - Math.log(4)) / (Math.log(14) - Math.log(4))) * (at(1) - at(0));
}

/**
 * Newton's starting point, ln T. The inverse tables know nothing of the Coulomb heat, which in
 * cold, strongly coupled matter is several times the ideal ions': it is allowed for at the
 * tables' guess, taking both parts as linear in T.
 */
function helmStart(rho: number, target: number, ye: number, ia: number, byP: boolean, coulomb: number): number {
  const lo = Math.log(HELM_T_MIN);
  const hi = Math.log(HELM_T_MAX);
  let lnT = Math.min(Math.max(helmGuess(rho, target, ia, byP) * LN10, lo), hi);
  // Below the inverse tables' densities: the smaller of the ideal gas's and radiation's
  // temperatures for the target, an upper bound (helm_thin_guess in eos/helmholtz.wgsl).
  if (Math.log10(rho) < M.guess.rho0) {
    const lnTi = Math.log(target) + Math.log(MU / ((ia + ye) * K)) - (byP ? 0 : Math.log(1.5));
    const lnTr = 0.25 * (Math.log(target) + Math.log(rho) + (byP ? Math.log(3) : 0) - Math.log(ARAD));
    lnT = Math.min(Math.max(Math.min(lnTi, lnTr), lo), hi);
  }
  if (coulomb > 0) {
    const c = helmCoulombThermal(rho, lnT, ye, ia, coulomb);
    lnT = Math.max(lnT - Math.log(1 + (byP ? c.pr : c.e) / target), lo);
  }
  return lnT;
}

/**
 * ln T at which the thermal energy per gram (byP false) or p_th / rho (byP true) is `target`,
 * as helm_solve, but iterated to convergence. -Infinity for no thermal energy.
 */
export function helmSolve(rho: number, target: number, ye: number, ia: number, byP: boolean, coulomb = 0): number {
  if (!(target > 0)) return -Infinity;
  const lo = Math.log(HELM_T_MIN);
  const hi = Math.log(HELM_T_MAX);
  let lnT = helmStart(rho, target, ye, ia, byP, coulomb);
  for (let it = 0; it < 60; it++) {
    const t = helmThermo(rho, lnT, ye, ia, coulomb);
    const f = byP ? t.pr : t.e;
    if (lnT <= lo && f > target) return lo + Math.log(target / f);
    if (lnT >= hi && f < target) return hi + 0.25 * Math.log(target / f);
    const step = Math.min(Math.max(-(f - target) / (byP ? t.dpr : t.de), -1), 1);
    lnT = Math.min(Math.max(lnT + step, lo), hi);
    if (Math.abs(step) < 1e-12) break;
  }
  return lnT;
}

/**
 * helm_solve as the shader runs it (for the tests): at most 10 iterations, and a step under
 * HELM_LAST_STEP taken on trust, since Newton's method leaves an error of the order of its
 * square. Returns ln T and the evaluations it took.
 */
export function helmSolveShader(rho: number, target: number, ye: number, ia: number, byP: boolean, coulomb = 0): { lnT: number; its: number } {
  const lo = Math.log(HELM_T_MIN);
  const hi = Math.log(HELM_T_MAX);
  let lnT = helmStart(rho, target, ye, ia, byP, coulomb);
  for (let it = 1; it <= 10; it++) {
    const t = helmThermo(rho, lnT, ye, ia, coulomb);
    const f = byP ? t.pr : t.e;
    if (lnT <= lo && f > target) return { lnT: lo + Math.log(target / f), its: it };
    if (lnT >= hi && f < target) return { lnT: hi + 0.25 * Math.log(target / f), its: it };
    const step = Math.min(Math.max(-(f - target) / (byP ? t.dpr : t.de), -1), 1);
    lnT = Math.min(Math.max(lnT + step, lo), hi);
    if (Math.abs(step) < HELM_LAST_STEP && lnT > lo && lnT < hi) return { lnT, its: it };
  }
  return { lnT, its: 10 };
}

/** How many evaluations helm_solve takes from its starting point (for the tests). */
export function helmIterations(rho: number, target: number, ye: number, ia: number, byP: boolean, coulomb = 0): number {
  return helmSolveShader(rho, target, ye, ia, byP, coulomb).its;
}

/**
 * Ready a device for runs with this equation of state: the table loaded, uploaded once, and
 * made the table of every bind group whose shader reaches it from here on (eos/table.ts).
 */
export async function prepareHelmholtz(device: GPUDevice): Promise<void> {
  await loadHelmholtz();
  bindTable(device, 'helmholtz', loaded());
}

/** The binding of the table in eos/table.wgsl (EOS_TAB). */
export const HELM_BINDING = EOS_TABLE_BINDING;
