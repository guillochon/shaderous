// Nuclear burning by a trained network: the 13-isotope alpha chain (he4, c12, o16, ..., ni56;
// aprox13's reactions), stepped by a small neural network that learned the reference burner's
// steps (tools/burn: burner.py integrates the network of reactions at fixed density, the
// energy released heating the gas; train.py fits the network to its trajectories). The weights
// and the data its inputs are computed from are generated (aprox13_net.ts, by
// tools/burn/export_net.py); this file holds a float64 mirror of one step for the tests, the
// WGSL of its inputs, and the pass that applies it on either grid (hydro/burn_net.wgsl).
//
// One step of the network, over dt at density rho, from mass fractions X and temperature T:
//
//   inputs (61)  log rho, log T, log dt scaled to about [-1, 1]; X^(1/4) and log X of each
//                species; and the log of every term of the reference's dY/dt (each reaction,
//                forward and reverse, screened as the reference screens it: REACLIB fits,
//                pynucastro's screen5), which rise far more steeply with T than the network
//                could resolve from log T alone.
//   network      4 hidden layers of SiLU units (130 wide: 128 trained, two gated rate units added by tools/burn/weightprog/twounit.py), 54 outputs: two stages of 27.
//   each stage   D = (1 - exp(-k dt)) (M g - X s): every species loses a share s_i = sigmoid(u_i)
//                of itself, and what is lost, M = sum X_i s_i, goes to the species by a
//                softmax g. k = 10^(5 + 5 z_0) per second. Mass is conserved exactly and no
//                species goes below zero, whatever the weights. The second stage starts from
//                what the first left.
//
// The energy released is the change in nuclear binding, Q . D, never predicted: energy and
// composition always agree. It goes into the gas's internal energy at fixed density, and the
// equation of state sets the new temperature, as the reference's self-heating does.
import { NET_DATA, NET_WEIGHTS_F16 } from './aprox13_net';
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { assemble } from '../gpu/shaders';
import { speciesConst } from '../species/species';
import { COMPOSITIONS, meanMolecularWeight, type Composition } from '../units/composition';
import { kelvinPerCodeT, unitOf, type UnitSystem } from '../units/units';

/** The species of the network, in its order: a problem that burns with it carries exactly these. */
export const NET_SPECIES: readonly string[] = NET_DATA.names;

/** What a problem burns with the network (Problem.burn). */
export interface NetworkBurnSpec {
  network: 'aprox13';
  /**
   * Burn in cells the shock detector flags (default off): a captured shock is a few cells of
   * numerical smearing whose states no real gas visits (burn/burn.ts, BurnSpec.inShocks).
   */
  inShocks?: boolean;
  /** Coolest gas that burns, in kelvin (default 1e8, the coolest the network was trained on). */
  tMin?: number;
  /** Least density that burns, g/cm^3 (default 1e4, the least the network was trained on). */
  rhoMin?: number;
  /**
   * Burning limiter (Kushnir et al. 2013): no cell takes in more than f e / t_s of energy per
   * unit time, e its internal energy density and t_s = dx / c_s its sound-crossing time. They
   * use f = 0.1, and find it leaves the result alone for f up to about 0.4. 0 or none: no limit.
   */
  limiter?: number;
}

/** The <burn_dx> slot on a uniform grid: every cell is P.dx across. */
export const BURN_DX_UNIFORM = 'fn burn_cell_dx(c: vec3<i32>) -> f32 { return P.dx; }\n';
/** The <burn_dx> slot on the block atlas: the size of the cells of the block a texel belongs to. */
export const BURN_DX_ATLAS = `#include "amr/grid.wgsl"
fn burn_cell_dx(c: vec3<i32>) -> f32 {
  let a = i32(P.amr.x);
  var s = vec3<i32>(c.x / PADW, c.y / PADW, c.z / PADW);
  if (is2d()) { s.z = c.z; }
  return blk_dx(s.x + a * (s.y + a * s.z));
}
`;

/** Longest step one application of the network takes, in seconds: longer steps are split. */
export const NET_MAX_DT = 0.01;
/** Most network steps per hydro step. */
export const NET_MAX_SUBSTEPS = 16;
/** Hottest temperature the network is given (kelvin), as train.roll clamps it. */
export const NET_T_MAX = 3e10;

const N = 13;
const LN10 = Math.LN10;

// ---- The weights ----

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

let weights: Float32Array | null = null;

/** Every layer's weights (rows of outputs) and then its biases, one after another. */
export function netWeights(): Float32Array {
  if (weights) return weights;
  const bin = atob(NET_WEIGHTS_F16);
  const out = new Float32Array(bin.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
  weights = out;
  return out;
}

const LAYERS = NET_DATA.layers as readonly (readonly [number, number])[];
const N_IN = LAYERS[0][0];
const N_OUT = LAYERS[LAYERS.length - 1][1];

// ---- Float64 mirror ----

type Sets = readonly (readonly number[])[];

const lnReaclib = (sets: Sets, basis: number[]): number => {
  const v = sets.map((a) => a.reduce((s, c, i) => s + c * basis[i], 0));
  const m = Math.max(...v);
  return m + Math.log(v.reduce((s, x) => s + Math.exp(x - m), 0));
};
const logaddexp = (a: number, b: number): number => {
  const m = Math.max(a, b);
  return m === -Infinity ? m : m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

/** Constants of one screening pair, as screen5 has them. */
function pairConsts([z1, a1, z2, a2]: readonly number[]) {
  return {
    bb: z1 * z2,
    zs13: Math.cbrt(z1 + z2),
    zhat: (z1 + z2) ** (5 / 3) - z1 ** (5 / 3) - z2 ** (5 / 3),
    zhat2: (z1 + z2) ** (5 / 12) - z1 ** (5 / 12) - z2 ** (5 / 12),
    lzav: (5 / 3) * Math.log((z1 * z2) / (z1 + z2)),
    aznut: Math.cbrt((z1 ** 2 * z2 ** 2 * a1 * a2) / (a1 + a2)),
  };
}
const PAIRS = Object.fromEntries(Object.entries(NET_DATA.screenPairs).map(([k, v]) => [k, pairConsts(v)]));
const FACT = Math.cbrt(2);

/** ln of the factor screening multiplies a pair's rate by (pynucastro's screen5), float64. */
function screen5(T: number, rho: number, Y: number[], pair: ReturnType<typeof pairConsts>): number {
  let ytot = 0;
  let zy = 0;
  let z2y = 0;
  for (let i = 0; i < N; i++) {
    const y = Math.max(Y[i], 1e-30);
    ytot += y;
    zy += NET_DATA.Z[i] * y;
    z2y += NET_DATA.Z[i] ** 2 * y;
  }
  const zbar = zy / ytot;
  const z2bar = z2y / ytot;
  const rr = rho * ytot;
  const qlam0z = (1.88e8 / T) * Math.sqrt((rr / T) * (z2bar + zbar));
  const taufac = NET_DATA.co2 / Math.cbrt(T);
  let gamp = (NET_DATA.aaFac / T) * Math.cbrt(rr * zbar);
  const { bb, zs13, zhat, zhat2, lzav, aznut } = pair;
  let gamef = ((FACT * bb) / zs13) * gamp;
  const tau12 = taufac * aznut;
  let alph12 = gamef / tau12;
  if (alph12 > 1.6) {
    alph12 = 1.6;
    gamef = 1.6 * tau12;
    gamp = (gamef * zs13) / (FACT * bb);
  }
  const h12w = bb * qlam0z;
  const cc = 0.896434 * gamp * zhat - 3.4474 * gamp ** 0.25 * zhat2 - 0.5551 * (Math.log(gamp) + lzav) - 2.996;
  const a3 = alph12 ** 3;
  const ss = tau12 * (5 / 32 - alph12 * (0.014 + 0.0128 * alph12));
  const vv = gamef * alph12 * (0.0055 + alph12 * (-0.0098 + 0.0048 * alph12));
  const h12 = cc - a3 * (ss + vv) + Math.log(Math.max(1 - 0.0562 * a3, 0.77));
  const [gx, gs] = [0.3, 0.8];
  const h = gamef > gx ? (gamef <= gs ? (h12w * (gs - gamef) + h12 * (gamef - gx)) / (gs - gx) : h12) : h12w;
  return Math.max(h, 0);
}

const scale = (v: number, [lo, hi]: readonly number[]): number => (2 * (v - lo)) / (hi - lo) - 1;
const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/** The network's 61 inputs (train.features, 'full'), float64. */
export function netFeatures(logRho: number, logT: number, logDt: number, Xin: readonly number[]): number[] {
  const X = Xin.map((x) => Math.max(x, 0));
  const f = [scale(logRho, NET_DATA.logRho), scale(logT, NET_DATA.logT), scale(logDt, NET_DATA.logDt)];
  for (const x of X) f.push(x ** 0.25);
  for (const x of X) f.push((clamp(Math.log10(x + 1e-30), -30, 0) + 15) / 15);
  const T = 10 ** logT;
  const rho = 10 ** logRho;
  const T9 = clamp(T * 1e-9, 0.01, 10);
  const basis = [1, 1 / T9, T9 ** (-1 / 3), Math.cbrt(T9), T9, T9 ** (5 / 3), Math.log(T9)];
  const Y = X.map((x, i) => x / NET_DATA.A[i]);
  const lnY = Y.map((y) => Math.log(y + 1e-30));
  const scr: Record<string, number> = {};
  for (const k of Object.keys(PAIRS)) scr[k] = screen5(T, rho, Y, PAIRS[k]);
  for (const term of NET_DATA.terms) {
    const r = term.rate as { sets: Sets | Record<string, Sets>; screen?: readonly string[]; approx?: string; alpha?: string; proton?: string };
    let lnLam: number;
    if (r.approx) {
      const s = r.sets as Record<string, Sets>;
      const ag = lnReaclib(s.ag, basis) + scr[r.alpha!];
      const ap = lnReaclib(s.ap, basis) + scr[r.alpha!];
      const pg = lnReaclib(s.pg, basis) + scr[r.proton!];
      const pa = lnReaclib(s.pa, basis) + scr[r.proton!];
      const both = logaddexp(pg, pa);
      lnLam = r.approx === 'fwd' ? logaddexp(ag, ap + pg - both) : logaddexp(lnReaclib(s.ga, basis), pa + lnReaclib(s.gp, basis) - both);
    } else {
      lnLam = lnReaclib(r.sets as Sets, basis);
      for (const k of r.screen!) lnLam += scr[k];
    }
    let ln = lnLam + Math.log(term.coef) + term.nrho * LN10 * logRho;
    for (const [i, n] of term.powers) ln += n * lnY[i];
    f.push((clamp(ln / LN10, -30, 10) + 10) / 20);
  }
  return f;
}

/** The network's outputs for inputs f, float64. */
function netForward(f: readonly number[]): number[] {
  const w = netWeights();
  let x = [...f];
  let o = 0;
  LAYERS.forEach(([nin, nout], l) => {
    const y = new Array<number>(nout);
    for (let j = 0; j < nout; j++) {
      let s = w[o + nin * nout + j];
      for (let i = 0; i < nin; i++) s += w[o + j * nin + i] * x[i];
      y[j] = l < LAYERS.length - 1 ? s / (1 + Math.exp(-s)) : s;
    }
    o += nin * nout + nout;
    x = y;
  });
  return x;
}

/** One stage's change in composition (Burner.flow_step) from its 27 outputs. */
function flowStage(z: readonly number[], X: readonly number[], logDt: number): number[] {
  const u = z.slice(1, 27).map((v) => 40 * Math.tanh(v / 40));
  const loss = X.map((x, i) => x / (1 + Math.exp(-u[i])));
  const M = loss.reduce((a, b) => a + b, 0);
  const g = u.slice(13);
  const gm = Math.max(...g);
  const e = g.map((v) => Math.exp(v - gm));
  const es = e.reduce((a, b) => a + b, 0);
  const kdt = 10 ** Math.min(5 + 5 * z[0] + logDt, 6);
  const c = -Math.expm1(-kdt);
  return X.map((_x, i) => c * ((M * e[i]) / es - loss[i]));
}

/** One step of the network: the change in mass fractions over dt seconds, float64. */
export function netStep(logRho: number, logT: number, logDt: number, Xin: readonly number[]): number[] {
  const X = Xin.map((x) => Math.max(x, 0));
  const z = netForward(netFeatures(logRho, logT, logDt, X));
  const D1 = flowStage(z.slice(0, 27), X, logDt);
  const X1 = X.map((x, i) => Math.max(x + D1[i], 0));
  const D2 = flowStage(z.slice(27, 54), X1, logDt);
  return D1.map((d, i) => d + D2[i]);
}

/** Energy released per gram (erg/g) by a change D in mass fractions. */
export const netEnergy = (D: readonly number[]): number => 1e18 * D.reduce((s, d, i) => s + d * NET_DATA.q[i], 0);

// ---- WGSL ----

const wf = (x: number): string => {
  if (!Number.isFinite(x)) throw new Error(`not a finite constant: ${x}`);
  const s = Math.fround(x).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/**
 * The <burn_net> slot of hydro/burn_net.wgsl: net_features(), the network's inputs, with every
 * rate fit and screening pair written in, and the layout of the weights.
 */
export function netSlot(): string {
  const lines: string[] = [];
  const rates = new Map<string, string>();
  // ln of one rate's REACLIB fit, its sets summed (by logsumexp), once per distinct rate.
  const rate = (sets: Sets): string => {
    const key = JSON.stringify(sets);
    let v = rates.get(key);
    if (!v) {
      v = `r${rates.size}`;
      const terms = sets.map((a) => `dot(vec4<f32>(${wf(a[0])}, ${wf(a[1])}, ${wf(a[2])}, ${wf(a[3])}), b0) + dot(vec3<f32>(${wf(a[4])}, ${wf(a[5])}, ${wf(a[6])}), b1)`);
      if (terms.length === 1) lines.push(`  let ${v} = ${terms[0]};`);
      else {
        lines.push(`  var ${v} = ${terms[0]};`);
        for (const t of terms.slice(1)) lines.push(`  ${v} = logaddexp(${v}, ${t});`);
      }
      rates.set(key, v);
    }
    return v;
  };
  const screens = new Map<string, string>();
  const screen = (k: string): string => {
    let v = screens.get(k);
    if (!v) {
      const p = PAIRS[k];
      v = `s_${k}`;
      lines.push(`  let ${v} = screen5(sc, ${wf(p.bb)}, ${wf(p.zs13)}, ${wf(p.zhat)}, ${wf(p.zhat2)}, ${wf(p.lzav)}, ${wf(p.aznut)});`);
      screens.set(k, v);
    }
    return v;
  };
  NET_DATA.terms.forEach((term, j) => {
    const r = term.rate as { sets: Sets | Record<string, Sets>; screen?: readonly string[]; approx?: string; alpha?: string; proton?: string };
    let lam: string;
    if (r.approx) {
      const s = r.sets as Record<string, Sets>;
      const sa = screen(r.alpha!);
      const sp = screen(r.proton!);
      const pg = `(${rate(s.pg)} + ${sp})`;
      const pa = `(${rate(s.pa)} + ${sp})`;
      const both = `logaddexp(${pg}, ${pa})`;
      lam =
        r.approx === 'fwd'
          ? `logaddexp(${rate(s.ag)} + ${sa}, ${rate(s.ap)} + ${sa} + ${pg} - ${both})`
          : `logaddexp(${rate(s.ga)}, ${pa} + ${rate(s.gp)} - ${both})`;
    } else {
      lam = [rate(r.sets as Sets), ...r.screen!.map(screen)].join(' + ');
    }
    const ys = term.powers.map(([i, n]) => (n === 1 ? ` + lny[${i}]` : ` + ${wf(n)} * lny[${i}]`)).join('');
    const rho = term.nrho ? ` + ${wf(term.nrho)} * ln_rho` : '';
    lines.push(`  f[${29 + j}] = (clamp((${lam} + ${wf(Math.log(term.coef))}${rho}${ys}) / LN10, -30.0, 10.0) + 10.0) / 20.0;`);
  });
  const offs: number[] = [];
  let o = 0;
  for (const [nin, nout] of LAYERS) {
    offs.push(o);
    o += nin * nout + nout;
  }
  const A = NET_DATA.A.map(wf).join(', ');
  const Z = NET_DATA.Z.map(wf).join(', ');
  const Q = NET_DATA.q.map(wf).join(', ');
  return `// Generated by src/burn/network.ts (netSlot) from aprox13_net.ts.
const NET_IN: u32 = ${N_IN}u;
const NET_OUT: u32 = ${N_OUT}u;
const NET_HIDDEN: u32 = ${LAYERS[0][1]}u;
const NET_DEPTH: u32 = ${LAYERS.length - 1}u;
// Where each layer's weights start (then its biases, after nin * nout).
const NET_OFFSETS = array<u32, ${LAYERS.length}>(${offs.map((x) => `${x}u`).join(', ')});
const NET_A = array<f32, 13>(${A});
const NET_Z = array<f32, 13>(${Z});
// Energy per unit mass fraction moved into each species, relative to C12, in 1e18 erg/g.
const NET_Q = array<f32, 13>(${Q});
const LN10: f32 = 2.30258509;
const SCREEN_CO2: f32 = ${wf(NET_DATA.co2)};
const SCREEN_AA: f32 = ${wf(NET_DATA.aaFac)};

fn net_features(log_rho: f32, log_T: f32, log_dt: f32, X: array<f32, 13>) -> array<f32, ${N_IN}> {
  var f: array<f32, ${N_IN}>;
  f[0] = 2.0 * (log_rho - ${wf(NET_DATA.logRho[0])}) / ${wf(NET_DATA.logRho[1] - NET_DATA.logRho[0])} - 1.0;
  f[1] = 2.0 * (log_T - ${wf(NET_DATA.logT[0])}) / ${wf(NET_DATA.logT[1] - NET_DATA.logT[0])} - 1.0;
  f[2] = 2.0 * (log_dt - ${wf(NET_DATA.logDt[0])}) / ${wf(NET_DATA.logDt[1] - NET_DATA.logDt[0])} - 1.0;
  var lny: array<f32, 13>;
  var Y: array<f32, 13>;
  for (var i = 0u; i < 13u; i++) {
    let x = max(X[i], 0.0);
    f[3u + i] = sqrt(sqrt(x));
    f[16u + i] = (clamp(log(x + 1e-30) / LN10, -30.0, 0.0) + 15.0) / 15.0;
    Y[i] = x / NET_A[i];
    lny[i] = log(Y[i] + 1e-30);
  }
  let T = exp(LN10 * log_T);
  let ln_rho = LN10 * log_rho;
  let sc = screen_common(T, exp(ln_rho), Y);
  let T9 = clamp(T * 1e-9, 0.01, 10.0);
  let t13 = pow(T9, 1.0 / 3.0);
  let b0 = vec4<f32>(1.0, 1.0 / T9, 1.0 / t13, t13);
  let b1 = vec3<f32>(T9, T9 * t13 * t13, log(T9));
${lines.join('\n')}
  return f;
}
`;
}

/** Bytes of the NetBurn uniform of hydro/burn_net.wgsl. */
const NET_UNIFORM_BYTES = 64;

/** Conversions between the run's code units and the network's (cgs, kelvin, seconds). */
export interface NetUnits {
  /** g/cm^3 per code density unit */
  rho: number;
  /** kelvin per code temperature unit */
  T: number;
  /** seconds per code time unit */
  time: number;
  /** erg/g per code specific energy unit */
  specificEnergy: number;
}

/**
 * The conversions for a run's units and reference composition (whose mean molecular weight
 * fixes the kelvin scale: units/units.ts).
 */
export function netUnits(u: UnitSystem | undefined, comp: Composition = COMPOSITIONS.solar): NetUnits {
  if (!u) throw new Error('burning with the network needs a problem with physical units');
  return {
    rho: unitOf(u, 'density') * 1e-3,
    T: kelvinPerCodeT(u, meanMolecularWeight(comp)),
    time: u.time,
    specificEnergy: unitOf(u, 'specificEnergy') * 1e4,
  };
}

/** What the network's pass works on: one grid's textures (as burn/burn.ts's BurnTargets). */
export interface NetBurnTargets {
  dims: [number, number, number];
  stride: number;
  groups: number;
  slots: Record<string, string>;
  params: GPUBuffer;
  clock: GPUBuffer;
  state: GPUTexture[][];
  species: GPUTexture[];
  pb: GPUTexture;
  /** each texel's heating rate over the last step, written (r32float) */
  heat: GPUTexture;
  /** the composition moments' texture, where the species set the equation of state */
  comp?: GPUTexture;
  /** on the block atlas: the block table (binding 30), from which each cell's size is read */
  blockTable?: GPUBuffer;
}

/**
 * The <burn_heat> slot of the burn passes: each texel's heating rate to the heat texture
 * (binding 11), if the device allows one more storage texture than the pass's `writes` others;
 * otherwise nowhere, and the heating reads 0. WebGPU guarantees 4, the network's pass with the
 * composition's moments writes 4 without it, and most adapters allow 8 (gpu/device.ts asks).
 */
export function burnHeatSlot(d: GPUDevice, writes: number): { slot: string; bound: boolean } {
  if (d.limits.maxStorageTexturesPerShaderStage > writes) {
    return {
      slot: '@group(0) @binding(11) var HO: texture_storage_3d<r32float, write>;\nfn store_heat(c: vec3<i32>, h: f32) { textureStore(HO, c, vec4<f32>(h, 0.0, 0.0, 0.0)); }\n',
      bound: true,
    };
  }
  return { slot: 'fn store_heat(c: vec3<i32>, h: f32) {}\n', bound: false };
}

/** The species a problem must carry to burn with the network, checked. */
export function checkNetSpecies(species: readonly string[] | undefined) {
  if (!species || species.length !== NET_SPECIES.length || species.some((s, i) => s !== NET_SPECIES[i])) {
    throw new Error(`burning with the network needs the species ${NET_SPECIES.join(', ')}, in that order`);
  }
}

/**
 * The network's burn pass on one grid: reads the state and species of one parity and writes
 * them, burned, to the other (and the composition moments, where they set the equation of
 * state). The caller flips both parities after encode().
 */
export class NetBurnPass {
  private constructor(
    private device: GPUDevice,
    private pipe: GPUComputePipeline,
    private groupsBy: GPUBindGroup[][],
    private dims: [number, number, number],
    private buffers: GPUBuffer[],
  ) {}

  static async create(d: GPUDevice, spec: NetworkBurnSpec, species: readonly string[] | undefined, units: NetUnits, t: NetBurnTargets): Promise<NetBurnPass> {
    checkNetSpecies(species);
    // Writes O1, O2, the species and, where the species set the equation of state, their moments.
    const heat = burnHeatSlot(d, t.comp ? 4 : 3);
    const code = assemble('hydro/burn_net.wgsl', {
      eos: t.slots.eos, comp: t.slots.comp, species_n: speciesConst(t.groups), burn_net: netSlot(), burn_heat: heat.slot,
      burn_dx: t.blockTable ? BURN_DX_ATLAS : BURN_DX_UNIFORM,
    });
    const pipe = await computePipeline(d, code, 'burn network');
    const uniform = d.createBuffer({ label: 'burn network', size: NET_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const buf = new ArrayBuffer(NET_UNIFORM_BYTES);
    new Float32Array(buf, 0, 8).set([
      units.rho, units.T, units.time, 1 / units.specificEnergy,
      spec.tMin ?? 1e8, spec.rhoMin ?? 1e4, NET_MAX_DT, NET_T_MAX,
    ]);
    new Uint32Array(buf, 32, 4).set([t.stride, spec.inShocks ? 1 : 0, NET_MAX_SUBSTEPS, 0]);
    new Float32Array(buf, 48, 4).set([spec.limiter ?? 0, 0, 0, 0]);
    d.queue.writeBuffer(uniform, 0, buf);
    const w = netWeights();
    const wbuf = d.createBuffer({ label: 'burn network weights', size: w.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(wbuf, 0, w as Float32Array<ArrayBuffer>);
    const v = (x: GPUTexture) => x.createView();
    const groupsBy = [0, 1].map((s) =>
      [0, 1].map((q) => {
        const r: Record<number, Bindable> = {
          0: t.params, 1: t.clock, 2: v(t.state[s][0]), 3: v(t.state[s][1]), 4: v(t.pb), 5: v(t.species[q]),
          6: v(t.state[1 - s][0]), 7: v(t.state[1 - s][1]), 8: v(t.species[1 - q]), 9: uniform, 10: wbuf,
        };
        if (t.comp) r[26] = v(t.comp);
        if (heat.bound) r[11] = v(t.heat);
        // The limiter reads each cell's size from it (burn_cell_dx), on or off: the code is there either way.
        if (t.blockTable) r[30] = t.blockTable;
        return bindGroup(d, pipe, r, 'burn network');
      }),
    );
    return new NetBurnPass(d, pipe, groupsBy, t.dims, [uniform, wbuf]);
  }

  /** Burn the state of parity s with the species of parity q, into the other parities. */
  encode(pass: GPUComputePassEncoder, s: number, q: number) {
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.groupsBy[s][q]);
    const [x, y, z] = this.dims;
    pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    void this.device;
  }
}
