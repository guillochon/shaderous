// ANEOS materials for planetary impacts (eos/aneos.wgsl): rock, iron and water, each from its
// published input deck run through pyaneos on one shared grid (tools/eos/aneos_table.py, which
// says what is tabulated and why it is the pressure and the energy and not the free energy).
// This file loads a material's table, builds the table of a pair of materials, gives either to
// the GPU, and holds the float64 mirror the initial conditions and the tests use.
//
// A table holds, on a grid uniform in x = log10 rho (g/cm^3) and y = log10 T (K), with first
// and cross derivatives for bicubic Hermite interpolation:
//   psi = asinh(p / P_S)       the pressure: linear below P_S = 1e9 dyn/cm^2, logarithmic above
//   lam = ln(e - E_c(rho))     the energy per gram above the cold curve E_c, the lowest isotherm
// and E_c with its slope. The hydro holds the energy (or the pressure), so the temperature is
// found first: lam (or psi) rises along y, so the node rows on either side are found by
// bisection and the place between them by Newton's method on the cubic there. Then the other
// quantity and the sound speed are read at that place.
//
// Two materials in one cell (an iron core against a rock mantle, and wherever they are stirred)
// share its volume so that both have the cell's temperature and one pressure. That closure is
// solved here, once, when the pair is first asked for: tables of the mixture at mass fractions
// 0, 1/4, 1/2, 3/4 and 1 of the second material, in the same form as a material's own. The
// shader blends the two tables on either side of a cell's mass fraction, texel by texel, and
// reads the blend as it reads one material.
import { ANEOS_META as M } from './aneos_meta';
import { setSharedBinding } from '../gpu/device';

/** The materials there are tables of (src/eos/aneos/<name>.bin), in the order of their ids. */
export const ANEOS_MATERIALS = ['pyrolite', 'fe85si15', 'water', 'forsterite'] as const;
export type AneosMaterial = (typeof ANEOS_MATERIALS)[number];

export const ANEOS_LABELS: Record<AneosMaterial, string> = {
  pyrolite: 'Pyrolite (rock: a mantle)',
  fe85si15: 'Iron alloy Fe85Si15 (a core)',
  water: 'Water and ice',
  forsterite: 'Forsterite (rock)',
};

// Each a literal, so the bundler ships the file and gives its address; fetched only when a run
// asks for the material.
const URLS: Record<AneosMaterial, URL> = {
  pyrolite: new URL('./aneos/pyrolite.bin', import.meta.url),
  fe85si15: new URL('./aneos/fe85si15.bin', import.meta.url),
  water: new URL('./aneos/water.bin', import.meta.url),
  forsterite: new URL('./aneos/forsterite.bin', import.meta.url),
};

const { x0: X0, y0: Y0, d: D, nr: NR, nt: NT, ps: PS } = M;
/** Texels per row: psi's NT, lam's NT, and the cold curve. */
const W = 2 * NT + 1;
const LN10 = Math.LN10;
/** Per ln rho or ln T, from per grid step. */
const SC = 1 / (D * LN10);
/** A pair's tables are at mass fractions k / ANEOS_INTERVALS of its second material. */
export const ANEOS_INTERVALS = 4;
export const ANEOS_T_MIN = 10 ** Y0;
export const ANEOS_T_MAX = 10 ** (Y0 + D * (NT - 1));
export const ANEOS_RHO_MIN = 10 ** X0;
export const ANEOS_RHO_MAX = 10 ** (X0 + D * (NR - 1));
/** The least c^2 the equation of state returns, (cm/s)^2: inside the vapor dome it nearly vanishes. */
export const ANEOS_C2_MIN = 1e7;
/** And the most: 100 km/s. */
export const ANEOS_C2_MAX = 1e14;
/** The most (dp/de)_rho / rho may be (AN_GRUN_MAX in eos/aneos.wgsl, which says why). */
export const ANEOS_GRUN_MAX = 5;

/** A table: one material (levels 1) or a pair at ANEOS_INTERVALS + 1 mass fractions. */
export interface AneosTable {
  key: string;
  levels: number;
  /** levels x NR rows of W rgba texels */
  data: Float32Array;
}

const materials = new Map<AneosMaterial, Float32Array>();
const loadingMaterial = new Map<AneosMaterial, Promise<Float32Array>>();
const tables = new Map<string, AneosTable>();

/** The key of a run's table: its material, or 'first+second'. */
export const aneosKey = (a: AneosMaterial, b?: AneosMaterial | null): string => (b && b !== a ? `${a}+${b}` : a);

/** The materials of a key. */
export function aneosMaterialsOf(key: string): AneosMaterial[] {
  const out = key.split('+') as AneosMaterial[];
  for (const m of out) if (!ANEOS_MATERIALS.includes(m)) throw new Error(`no ANEOS material '${m}' (there are: ${ANEOS_MATERIALS.join(', ')})`);
  return out;
}

async function loadMaterial(name: AneosMaterial): Promise<Float32Array> {
  const have = materials.get(name);
  if (have) return have;
  let p = loadingMaterial.get(name);
  if (!p) {
    p = (async () => {
      const url = URLS[name];
      let bytes: ArrayBuffer;
      if (url.protocol === 'file:') {
        // Under Node (the unit tests): read the file (as eos/helmholtz.ts does).
        const fsName = 'node:fs/promises';
        const { readFile } = (await import(/* @vite-ignore */ fsName)) as { readFile: (u: URL) => Promise<Uint8Array> };
        bytes = new Uint8Array(await readFile(url)).buffer as ArrayBuffer;
      } else {
        bytes = await (await fetch(url)).arrayBuffer();
      }
      const t = new Float32Array(bytes);
      if (t.length !== NR * W * 4) throw new Error(`ANEOS table of ${name}: ${t.length} floats, expected ${NR * W * 4}`);
      materials.set(name, t);
      return t;
    })();
    loadingMaterial.set(name, p);
  }
  return p;
}

/** Load (once) the table of a material or of a pair ('a+b'), building the pair's if need be. */
export async function loadAneos(key: string): Promise<AneosTable> {
  const have = tables.get(key);
  if (have) return have;
  const names = aneosMaterialsOf(key);
  if (names.length > 2) throw new Error('an ANEOS run holds one material or two');
  const data = await Promise.all(names.map(loadMaterial));
  const t: AneosTable = names.length === 1 ? { key, levels: 1, data: data[0] } : { key, levels: ANEOS_INTERVALS + 1, data: buildPair(data[0], data[1]) };
  tables.set(key, t);
  return t;
}

/** The table of a key, once loaded (loadAneos). */
export function aneosTable(key: string): AneosTable {
  const t = tables.get(key);
  if (!t) throw new Error(`the ANEOS table of ${key} is not loaded: await loadAneos() first`);
  return t;
}

// ---- Reading a table ----

const hb = (t: number): [number, number, number, number] => [2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2];
const hb1 = (t: number): [number, number, number, number] => [6 * t ** 2 - 6 * t, 3 * t ** 2 - 4 * t + 1, -6 * t ** 2 + 6 * t, 3 * t ** 2 - 2 * t];

/** A table read at one mass fraction of its second material: the two levels on either side, and the weight of the upper. */
interface View {
  d: Float32Array;
  lo: number;
  hi: number;
  w: number;
}

function view(t: AneosTable, xb: number): View {
  if (t.levels === 1) return { d: t.data, lo: 0, hi: 0, w: 0 };
  const f = Math.min(Math.max(xb, 0), 1) * (t.levels - 1);
  const l = Math.min(Math.floor(f), t.levels - 2);
  return { d: t.data, lo: l * NR, hi: (l + 1) * NR, w: f - l };
}

const tex = (v: View, col: number, row: number, c: number): number => {
  const a = v.d[((v.lo + row) * W + col) * 4 + c];
  return v.w > 0 ? a + v.w * (v.d[((v.hi + row) * W + col) * 4 + c] - a) : a;
};

/** Where a density (g/cm^3) lies along x: the row below it, the place between rows, and rho over the table's least where it is below that. */
interface AtX {
  i: number;
  u: number;
  thin: number;
}

function atX(rho: number): AtX {
  const x = (Math.log10(rho) - X0) / D;
  const thin = x < 0 ? rho / ANEOS_RHO_MIN : 1;
  const xc = Math.min(Math.max(x, 0), NR - 1.0001);
  const i = Math.floor(xc);
  return { i, u: xc - i, thin };
}

/** A column of the table at a place along x: the value there and its slope along y, per grid step. */
function column(v: View, col: number, a: AtX): [number, number] {
  const A = hb(a.u);
  return [
    A[0] * tex(v, col, a.i, 0) + A[1] * tex(v, col, a.i, 1) + A[2] * tex(v, col, a.i + 1, 0) + A[3] * tex(v, col, a.i + 1, 1),
    A[0] * tex(v, col, a.i, 2) + A[1] * tex(v, col, a.i, 3) + A[2] * tex(v, col, a.i + 1, 2) + A[3] * tex(v, col, a.i + 1, 3),
  ];
}

/**
 * The place along y (in grid steps, 0 to NT - 1) at which the block of columns starting at `off`
 * (0: psi, NT: lam) has the value `goal` at `a`. `its` Newton steps inside the cell: the
 * shader takes ANEOS_NEWTON, the mirror runs them out.
 */
export const ANEOS_NEWTON = 8;
function solveY(v: View, off: number, a: AtX, goal: number, its = 40): number {
  let lo = 0;
  let hi = NT - 1;
  let [glo, slo] = column(v, off, a);
  if (!(goal > glo)) return 0;
  let [ghi, shi] = column(v, off + NT - 1, a);
  if (!(goal < ghi)) return NT - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    const [g, s] = column(v, off + mid, a);
    if (g <= goal) [lo, glo, slo] = [mid, g, s];
    else [hi, ghi, shi] = [mid, g, s];
  }
  // The cubic through the two nodes, with their slopes: Newton's method, kept inside the bracket
  // by a step of false position where it would leave (the slopes are limited, and may be zero at
  // a node), which halves the value at an end it has kept twice (the Illinois rule).
  let a0 = 0;
  let a1 = 1;
  let f0 = glo - goal;
  let f1 = ghi - goal;
  let side = 0;
  let t = ghi > glo ? (goal - glo) / (ghi - glo) : 0;
  for (let k = 0; k < its; k++) {
    const B = hb(t);
    const B1 = hb1(t);
    const f = B[0] * glo + B[1] * slo + B[2] * ghi + B[3] * shi - goal;
    if (f > 0) {
      a1 = t;
      f1 = f;
      if (side === 1) f0 *= 0.5;
      side = 1;
    } else {
      a0 = t;
      f0 = f;
      if (side === -1) f1 *= 0.5;
      side = -1;
    }
    const df = B1[0] * glo + B1[1] * slo + B1[2] * ghi + B1[3] * shi;
    let next = t - f / df;
    if (!(next > a0 && next < a1)) next = f1 > f0 ? a0 - (f0 * (a1 - a0)) / (f1 - f0) : 0.5 * (a0 + a1);
    if (next === t) break;
    t = next;
  }
  return lo + t;
}

/** psi and lam at a place, with their derivatives per ln rho and per ln T; and the cold curve there. */
interface At {
  psi: number;
  psiR: number;
  psiT: number;
  lam: number;
  lamR: number;
  lamT: number;
  /** E_c (erg/g) and dE_c/dln rho */
  ec: number;
  ecR: number;
}

function cold(v: View, a: AtX): [number, number] {
  const A = hb(a.u);
  const A1 = hb1(a.u);
  const c = 2 * NT;
  const e0 = tex(v, c, a.i, 0);
  const s0 = tex(v, c, a.i, 1);
  const e1 = tex(v, c, a.i + 1, 0);
  const s1 = tex(v, c, a.i + 1, 1);
  // Thinner than the table: the cold curve of its least density, as a gas's is flat.
  if (a.thin < 1) return [e0, 0];
  return [A[0] * e0 + A[1] * s0 + A[2] * e1 + A[3] * s1, (A1[0] * e0 + A1[1] * s0 + A1[2] * e1 + A1[3] * s1) * SC];
}

function at(v: View, a: AtX, y: number): At {
  const j = Math.min(Math.floor(y), NT - 2);
  const t = y - j;
  const A = hb(a.u);
  const A1 = hb1(a.u);
  const B = hb(t);
  const B1 = hb1(t);
  const comb = (off: number, P: readonly number[], Q: readonly number[]): number => {
    let s = 0;
    for (let p = 0; p < 2; p++)
      for (let q = 0; q < 2; q++) {
        const col = off + j + q;
        const row = a.i + p;
        s += P[2 * p] * Q[2 * q] * tex(v, col, row, 0) + P[2 * p + 1] * Q[2 * q] * tex(v, col, row, 1) + P[2 * p] * Q[2 * q + 1] * tex(v, col, row, 2) + P[2 * p + 1] * Q[2 * q + 1] * tex(v, col, row, 3);
      }
    return s;
  };
  const [ec, ecR] = cold(v, a);
  return {
    psi: comb(0, A, B), psiR: comb(0, A1, B) * SC, psiT: comb(0, A, B1) * SC,
    lam: comb(NT, A, B), lamR: comb(NT, A1, B) * SC, lamT: comb(NT, A, B1) * SC,
    ec, ecR,
  };
}

/** What the hydro needs of a state, cgs. */
export interface AneosState {
  /** pressure, dyn/cm^2 */
  p: number;
  /** energy per gram, erg/g (ANEOS's, plus the table's offset) */
  e: number;
  /** temperature, K */
  T: number;
  /** sound speed squared, (cm/s)^2 */
  c2: number;
}

/**
 * The state at a place of the table. Thinner than the table's least density the gas is taken
 * as ideal: its pressure that of the least density times `thin`, its energy the same.
 */
function stateAt(v: View, a: AtX, y: number, rho: number): AneosState {
  const q = at(v, a, y);
  // The tables' pressures are positive (NOTENSION): so is what is read between their nodes.
  const psi = Math.max(q.psi, 0);
  const sh = Math.sinh(psi);
  const ch = Math.cosh(psi);
  const p = PS * sh * a.thin;
  const eth = Math.exp(q.lam);
  // Derivatives per ln rho and per ln T. Below the table p goes as rho and e does not change.
  const pT = PS * ch * q.psiT * a.thin;
  const pR = a.thin < 1 ? p : PS * ch * q.psiR;
  const eT = eth * q.lamT;
  // c^2 = (dp/drho)_T + (dp/dT)^2 T / (rho^2 c_v), every term positive (an_state in
  // eos/aneos.wgsl says why not the form with the energy's slope along the density).
  let c2 = Math.max(pR, 0) / rho;
  if (eT > 0 && pT > 0) c2 += (Math.min(Math.max(pT / (eT * rho), 0), ANEOS_GRUN_MAX) * pT) / rho;
  c2 = Math.min(Math.max(c2, ANEOS_C2_MIN), ANEOS_C2_MAX);
  return { p, e: q.ec + eth, T: 10 ** (Y0 + D * y), c2 };
}

/** The least energy per gram (erg/g) a state of density rho may have: the table's lowest isotherm's (eos_eint_floor in eos/aneos.wgsl). */
export function aneosEnergyFloor(t: AneosTable, rho: number, xb = 0): number {
  const v = view(t, xb);
  const a = atX(rho);
  return cold(v, a)[0] + Math.exp(column(v, NT, a)[0]);
}

/** The state of density rho (g/cm^3) and energy e (erg/g, with the table's offset), at mass fraction xb of the table's second material. */
export function aneosFromE(t: AneosTable, rho: number, e: number, xb = 0, its?: number): AneosState {
  const v = view(t, xb);
  const a = atX(rho);
  const eth = e - cold(v, a)[0];
  const y = eth > 0 ? solveY(v, NT, a, Math.log(eth), its) : 0;
  return stateAt(v, a, y, rho);
}

/** The state of density rho and pressure p (dyn/cm^2). */
export function aneosFromP(t: AneosTable, rho: number, p: number, xb = 0, its?: number): AneosState {
  const v = view(t, xb);
  const a = atX(rho);
  return stateAt(v, a, solveY(v, 0, a, Math.asinh(p / (PS * a.thin)), its), rho);
}

/** The state of density rho and temperature T (K). */
export function aneosFromT(t: AneosTable, rho: number, T: number, xb = 0): AneosState {
  const v = view(t, xb);
  const y = Math.min(Math.max((Math.log10(T) - Y0) / D, 0), NT - 1);
  return stateAt(v, atX(rho), y, rho);
}

// ---- Two materials ----

/** Slopes m of the values f along a line (per grid step), limited as tools/eos/aneos_table.py limits them. */
function limitLine(f: Float64Array, m: Float64Array, n: number, o: number, step: number) {
  for (let k = 0; k < n; k++) {
    const here = f[o + k * step];
    const dm = k > 0 ? here - f[o + (k - 1) * step] : f[o + step] - here;
    const dp = k < n - 1 ? f[o + (k + 1) * step] - here : dm;
    const i = o + k * step;
    if (!(dm * dp > 0)) m[i] = 0;
    else {
      const cap = 3 * Math.min(Math.abs(dm), Math.abs(dp));
      const s = Math.sign(dp);
      m[i] = s * Math.min(Math.max(s * m[i], 0), cap);
    }
  }
}

/**
 * The tables of a pair: at each node and each mass fraction X of the second material, the two
 * share the volume so that their pressures are equal at the node's temperature,
 *   (1 - X) / rho_a + X / rho_b = 1 / rho,   p_a(rho_a, T) = p_b(rho_b, T),
 * found by bisection on how the volume is shared: u = ln(phi / (1 - phi)) for the first's share
 * phi, so that rho_a = (1 - X) rho / phi and rho_b = X rho / (1 - phi) are both held to rounding
 * however small either share (a thin vapor of one around drops of the other is a share of
 * 1e-9; found by rho_a instead, the drops' density was right to a part in a thousand and their
 * pressure, 1e12 dyn/cm^2 times that, was noise of 1e4 to 1e9 in a mixture whose pressure is
 * nothing). The first's pressure falls as its share grows and the second's rises. The mixture's
 * pressure is the lesser of the two where the bisection ends (at the edge of a condensed phase
 * one of them is a cliff), and its energy the two's by mass. Its node slopes are differences
 * of its own nodes, limited as a material's are, and it has no cross derivative.
 */
export function buildPair(ta: Float32Array, tb: Float32Array): Float32Array {
  const L = ANEOS_INTERVALS + 1;
  const out = new Float32Array(L * NR * W * 4);
  out.set(ta, 0);
  out.set(tb, ANEOS_INTERVALS * NR * W * 4);
  // A material along x at node column `col`: the value by the cubic through the rows.
  const along = (d: Float32Array, col: number, x: number, c: number): number => {
    const xc = Math.min(Math.max(x, 0), NR - 1.0001);
    const i = Math.floor(xc);
    const A = hb(xc - i);
    const o0 = (i * W + col) * 4;
    const o1 = ((i + 1) * W + col) * 4;
    return A[0] * d[o0 + c] + A[1] * d[o0 + c + 1] + A[2] * d[o1 + c] + A[3] * d[o1 + c + 1];
  };
  // Pressure at x (grid steps along log rho, any value) and node column j; as a gas below the table.
  const pOf = (d: Float32Array, j: number, x: number): number => PS * Math.sinh(along(d, j, x, 0)) * (x < 0 ? 10 ** (x * D) : 1);
  const eOf = (d: Float32Array, j: number, x: number): number => along(d, 2 * NT, Math.max(x, 0), 0) + Math.exp(along(d, NT + j, x, 0));
  const ecOf = (d: Float32Array, x: number): number => along(d, 2 * NT, Math.max(x, 0), 0);
  const psi = new Float64Array(NR * NT);
  const lam = new Float64Array(NR * NT);
  const ec = new Float64Array(NR);
  const sx = new Float64Array(NR * NT);
  const sy = new Float64Array(NR * NT);
  const top = NR - 1;
  for (let l = 1; l < ANEOS_INTERVALS; l++) {
    const X = l / ANEOS_INTERVALS;
    const la = Math.log10(1 - X) / D;
    const lb = Math.log10(X) / D;
    for (let j = 0; j < NT; j++) {
      for (let i = 0; i < NR; i++) {
        // In grid steps of log rho: rho_a = (1 - X) rho / phi and rho_b = X rho / (1 - phi), with
        // phi = 1 / (1 + exp(-u)). A density past the table's densest reads as its densest.
        const xaOf = (u: number): number => Math.min(i + la + Math.log10(1 + Math.exp(-u)) / D, top);
        const xbOf = (u: number): number => Math.min(i + lb + Math.log10(1 + Math.exp(u)) / D, top);
        let lo = -60;
        let hi = 60;
        for (let k = 0; k < 52; k++) {
          const u = 0.5 * (lo + hi);
          if (pOf(ta, j, xaOf(u)) > pOf(tb, j, xbOf(u))) lo = u;
          else hi = u;
        }
        const u = 0.5 * (lo + hi);
        const xa = xaOf(u);
        const xb = xbOf(u);
        const p = Math.min(pOf(ta, j, xa), pOf(tb, j, xb));
        if (j === 0) ec[i] = (1 - X) * ecOf(ta, xa) + X * ecOf(tb, xb);
        const e = (1 - X) * eOf(ta, j, xa) + X * eOf(tb, j, xb);
        psi[i * NT + j] = Math.asinh(p / PS);
        lam[i * NT + j] = Math.log(Math.max(e - ec[i], 1e-30));
      }
    }
    const base = l * NR * W * 4;
    for (const [off, f] of [[0, psi], [NT, lam]] as const) {
      // Slopes by central differences of the nodes, then limited, along x and along y.
      for (let i = 0; i < NR; i++)
        for (let j = 0; j < NT; j++) {
          const o = i * NT + j;
          sx[o] = i > 0 && i < top ? 0.5 * (f[o + NT] - f[o - NT]) : i === 0 ? f[o + NT] - f[o] : f[o] - f[o - NT];
          sy[o] = j > 0 && j < NT - 1 ? 0.5 * (f[o + 1] - f[o - 1]) : j === 0 ? f[o + 1] - f[o] : f[o] - f[o - 1];
        }
      for (let j = 0; j < NT; j++) limitLine(f, sx, NR, j, NT);
      for (let i = 0; i < NR; i++) limitLine(f, sy, NT, i * NT, 1);
      for (let i = 0; i < NR; i++)
        for (let j = 0; j < NT; j++) {
          const o = i * NT + j;
          const q = base + (i * W + off + j) * 4;
          out[q] = f[o];
          out[q + 1] = sx[o];
          out[q + 2] = sy[o];
          out[q + 3] = 0;
        }
    }
    for (let i = 0; i < NR; i++) {
      const q = base + (i * W + 2 * NT) * 4;
      out[q] = ec[i];
      out[q + 1] = i > 0 && i < top ? 0.5 * (ec[i + 1] - ec[i - 1]) : i === 0 ? ec[1] - ec[0] : ec[top] - ec[top - 1];
    }
  }
  return out;
}

/** How a pair shares a cell at density rho, temperature T and mass fraction X of the second: the two densities and pressures (for the tests). */
export function aneosSplit(a: AneosTable, b: AneosTable, rho: number, T: number, X: number): { rhoA: number; rhoB: number; pA: number; pB: number } {
  // By the share of the volume, as buildPair does (which says why).
  const ra = (u: number): number => Math.min((1 - X) * rho * (1 + Math.exp(-u)), ANEOS_RHO_MAX);
  const rb = (u: number): number => Math.min(X * rho * (1 + Math.exp(u)), ANEOS_RHO_MAX);
  let lo = -60;
  let hi = 60;
  for (let k = 0; k < 80; k++) {
    const u = 0.5 * (lo + hi);
    if (aneosFromT(a, ra(u), T).p > aneosFromT(b, rb(u), T).p) lo = u;
    else hi = u;
  }
  const u = 0.5 * (lo + hi);
  return { rhoA: ra(u), rhoB: rb(u), pA: aneosFromT(a, ra(u), T).p, pB: aneosFromT(b, rb(u), T).p };
}

// ---- The GPU ----

/** The binding of the table in eos/aneos.wgsl (AN_TAB). */
export const ANEOS_BINDING = 41;

const deviceTextures = new WeakMap<GPUDevice, Map<string, GPUTexture>>();

/**
 * Ready a device for runs with the table of `key`: loaded (and built, for a pair), uploaded
 * once, and registered as binding 41 of every bind group whose shader reaches it
 * (gpu/device.ts). One table at a time per device: the last one asked for.
 */
export async function prepareAneos(device: GPUDevice, key: string): Promise<void> {
  const t = await loadAneos(key);
  let m = deviceTextures.get(device);
  if (!m) deviceTextures.set(device, (m = new Map()));
  let tex = m.get(key);
  if (!tex) {
    const rows = t.levels * NR;
    tex = device.createTexture({ label: `ANEOS table ${key}`, size: [W, rows], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: tex }, t.data as Float32Array<ArrayBuffer>, { bytesPerRow: W * 16 }, [W, rows]);
    m.set(key, tex);
  }
  setSharedBinding(device, ANEOS_BINDING, tex.createView());
}
