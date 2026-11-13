// Nuclear burning, fuel to ash: the settings, the pass that applies it (hydro/burn.wgsl) on
// either grid, float64 mirrors for the tests, and the steady detonation the model makes (ZND)
// for initial conditions and as the validation's reference.
//
// The reaction is single-step Arrhenius, the standard model of detonation numerics (Fickett &
// Davis, "Detonation"): the fuel fraction f obeys df/dt = -K f exp(-Ea / T), and each unit of
// mass burned releases q into the gas. With an ideal gas in code units T = p / rho.
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { assemble } from '../gpu/shaders';
import { speciesConst } from '../species/species';
import type { Eos } from '../eos/eos';

/** What a problem burns, from its parameters (Problem.burn). */
export interface BurnSpec {
  /** names of the fuel and ash species (Problem.species) */
  fuel: string;
  ash: string;
  /** energy released per unit mass burned */
  q: number;
  /** activation temperature, in the equation of state's temperature units */
  ea: number;
  /** rate constant (per unit time) */
  k: number;
  /**
   * Burn in cells the shock detector flags. Off by default: a captured shock is a few cells of
   * numerical smearing whose states no real gas visits, and burning there lets the scheme set
   * the speed of an unresolved front (Colella, Majda & Roytburd 1986).
   */
  inShocks?: boolean;
}

/** Substeps of hydro/burn.wgsl. */
export const BURN_SUBSTEPS = 8;

const oneMinusExp = (x: number): number => -Math.expm1(-x);

/**
 * One cell's burn over dt at density rho, from fuel fraction f and internal energy density e
 * (temperature from `temperature(rho, e)`): as hydro/burn.wgsl. Returns the new [f, e].
 */
export function burnCell(spec: BurnSpec, temperature: (rho: number, e: number) => number, rho: number, f: number, e: number, dt: number): [number, number] {
  let left = dt;
  for (let i = 0; i < BURN_SUBSTEPS && left > 0 && f > 0; i++) {
    const T = temperature(rho, e);
    if (!(T > 0)) break;
    const k = spec.k * Math.exp(-spec.ea / T);
    if (k <= 0) break;
    let h = left;
    const dmax = (0.2 * e) / (rho * spec.q);
    if (i < BURN_SUBSTEPS - 1 && f * oneMinusExp(k * left) > dmax && dmax < f) h = -Math.log(1 - dmax / f) / k;
    const df = f * oneMinusExp(k * h);
    f -= df;
    e += rho * spec.q * df;
    left -= h;
  }
  return [f, e];
}

/**
 * Refinement on the burning (amr/flag.wgsl, <flag_burn>). The burning length is how far sound
 * travels while the burn releases the cell's internal energy at its present rate,
 * l = c_s e / (rho q f k), with e the internal energy density and k = K exp(-Ea / T). The
 * indicator is BURN_REFINE_SCALE dx / l: a block is refined while l is under 80 of its cells
 * (the default threshold, 0.8) and may merge once l is over 320. In the steady detonation
 * l bottoms out at about 2.5 half-reaction lengths, a little behind the half-burned point, so
 * the reaction zone is refined down to about 30 cells per half-reaction length (or to the
 * finest level). Cold fuel (exp(-Ea / T) tiny) and burned gas (f near 0) ask for nothing, but
 * fuel heating toward a runaway does, before the front reaches it.
 */
export const BURN_REFINE_SCALE = 64;

/** The burning's refinement indicator at one cell of cell size dx, as amr/flag.wgsl. */
export function burnIndicator(spec: BurnSpec, eos: Pick<Eos, 'temperature' | 'soundSpeed' | 'eintFromP'>, rho: number, p: number, f: number, dx: number): number {
  const e = eos.eintFromP(rho, p);
  const cs = eos.soundSpeed(rho, p);
  if (!(f > 0 && e > 0 && cs > 0)) return 0;
  const T = eos.temperature(rho, e);
  if (!(T > 0)) return 0;
  return (BURN_REFINE_SCALE * dx * spec.q * f * spec.k * Math.exp(-spec.ea / T) * rho) / (e * cs);
}

/**
 * The <flag_burn> slot of amr/flag.wgsl: burn_indicator() from the primitives (W1, W2: rho;
 * p, c_s, scalar, e_int) and the fuel's mass fraction in the species with ghosts (SXF, from
 * <flag_species>), with the burn's constants written in.
 */
export function flagBurnSlot(spec: BurnSpec, species: readonly string[] | undefined): string {
  const [fi] = burnIndices(spec, species);
  const n = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
  return `#include <eos>
fn burn_indicator(c: vec3<i32>, dx: f32) -> f32 {
  let rho = textureLoad(W1, c, 0).x;
  let w2 = textureLoad(W2, c, 0);
  let f = textureLoad(SXF, c + vec3<i32>(0, 0, ${fi >> 2} * FLAG_SP_DEPTH), 0)[${fi & 3}];
  let e = w2.w;
  let cs = w2.y;
  if (!(f > 0.0 && e > 0.0 && cs > 0.0 && rho > 0.0)) { return 0.0; }
  let T = eos_temperature(rho, e);
  if (!(T > 0.0)) { return 0.0; }
  return ${n(BURN_REFINE_SCALE)} * dx * ${n(spec.q)} * f * ${n(spec.k)} * exp(-${n(spec.ea)} / T) * rho / (e * cs);
}
`;
}

/** Bytes of the Burn uniform of hydro/burn.wgsl. */
const BURN_BYTES = 32;

/** What the burn pass works on: one grid's textures, laid out alike (uniform grid or atlas). */
export interface BurnTargets {
  /** cells (texels) along x, y, z of the state textures */
  dims: [number, number, number];
  /** texels along z from one group of four species to the next */
  stride: number;
  groups: number;
  /** the hydro shaders' slots (eos, comp) */
  slots: Record<string, string>;
  params: GPUBuffer;
  clock: GPUBuffer;
  /** state[parity][0 or 1] */
  state: GPUTexture[][];
  /** rho X, by parity */
  species: GPUTexture[];
  /** the predictor's second texture: (p, scalar, shock flag, first order) */
  pb: GPUTexture;
}

/** The species indices of a burn, checked against the problem's species. */
export function burnIndices(spec: BurnSpec, species: readonly string[] | undefined): [number, number] {
  const f = species?.indexOf(spec.fuel) ?? -1;
  const a = species?.indexOf(spec.ash) ?? -1;
  if (f < 0 || a < 0) throw new Error(`burning needs species '${spec.fuel}' and '${spec.ash}'`);
  return [f, a];
}

/**
 * The burn pass on one grid: reads the state and species of one parity and writes them,
 * burned, to the other. The caller flips both parities after encode().
 */
export class BurnPass {
  private constructor(
    private pipe: GPUComputePipeline,
    private groupsBy: GPUBindGroup[][],
    private dims: [number, number, number],
    readonly uniform: GPUBuffer,
  ) {}

  static async create(d: GPUDevice, spec: BurnSpec, species: readonly string[] | undefined, t: BurnTargets): Promise<BurnPass> {
    const [fi, ai] = burnIndices(spec, species);
    const code = assemble('hydro/burn.wgsl', { eos: t.slots.eos, comp: t.slots.comp, species_n: speciesConst(t.groups) });
    const pipe = await computePipeline(d, code, 'burn');
    const uniform = d.createBuffer({ label: 'burn', size: BURN_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const buf = new ArrayBuffer(BURN_BYTES);
    new Float32Array(buf, 0, 4).set([spec.q, spec.ea, spec.k, spec.inShocks ? 1 : 0]);
    new Uint32Array(buf, 16, 4).set([fi, ai, t.stride, 0]);
    d.queue.writeBuffer(uniform, 0, buf);
    const v = (x: GPUTexture) => x.createView();
    const groupsBy = [0, 1].map((s) =>
      [0, 1].map((q) => {
        const r: Record<number, Bindable> = {
          0: t.params, 1: t.clock, 2: v(t.state[s][0]), 3: v(t.state[s][1]), 4: v(t.pb), 5: v(t.species[q]),
          6: v(t.state[1 - s][0]), 7: v(t.state[1 - s][1]), 8: v(t.species[1 - q]), 9: uniform,
        };
        return bindGroup(d, pipe, r, 'burn');
      }),
    );
    return new BurnPass(pipe, groupsBy, t.dims, uniform);
  }

  /** Burn the state of parity s with the species of parity q, into the other parities. */
  encode(pass: GPUComputePassEncoder, s: number, q: number) {
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.groupsBy[s][q]);
    const [x, y, z] = this.dims;
    pass.dispatchWorkgroups(Math.ceil(x / 8), Math.ceil(y / 8), Math.ceil(z / 4));
  }

  destroy() {
    this.uniform.destroy();
  }
}

/** Chapman-Jouguet speed into gas of density rho0 and pressure p0 at rest. */
export function cjSpeed(gamma: number, q: number, rho0 = 1, p0 = 1): number {
  const a = ((gamma * gamma - 1) * q) / 2;
  return Math.sqrt(a + (gamma * p0) / rho0) + Math.sqrt(a);
}

/** The gas at one point of a steady detonation: lab frame, the shock moving toward +x. */
export interface ZndState {
  rho: number;
  u: number;
  p: number;
  /** fuel fraction left */
  f: number;
}

/**
 * The steady (ZND) detonation of the Arrhenius model at speed D into gas at rest (rho0, p0),
 * ideal gas: the state at each distance behind the shock, and the rate constant K that puts
 * half of the fuel's burning within `halfLength` of the shock. The flow behind the shock lies
 * on the Rayleigh line and on the Hugoniot of the fraction burned so far, so each fraction has
 * one state; distance follows from dx = (D - u) dlambda / (rate), integrated in
 * s = -ln(1 - lambda), where the integrand stays finite as lambda goes to 1.
 */
export function znd(gamma: number, q: number, ea: number, D: number, halfLength = 1, rho0 = 1, p0 = 1) {
  const m = rho0 * D;
  const v0 = 1 / rho0;
  const g1 = gamma / (gamma - 1);
  const a = (-m * m * (gamma + 1)) / (2 * (gamma - 1));
  const b = g1 * (p0 + m * m * v0);
  const state = (lam: number) => {
    const c = -(g1 * p0 * v0 + 0.5 * m * m * v0 * v0 + q * lam);
    const disc = Math.max(b * b - 4 * a * c, 0);
    const v = (b - Math.sqrt(disc)) / (-2 * a);
    const p = p0 + m * m * (v0 - v);
    return { v, p, w: m * v };
  };
  // x(s) for K = 1 on a grid in s, by Simpson's rule on each interval.
  const S_MAX = 30;
  const N = 6000;
  const ds = S_MAX / N;
  const integrand = (s: number) => {
    const { v, p, w } = state(-Math.expm1(-s));
    return w / Math.exp(-ea / (p * v));
  };
  const xs = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) {
    const s = i * ds;
    xs[i + 1] = xs[i] + (ds / 6) * (integrand(s) + 4 * integrand(s + ds / 2) + integrand(s + ds));
  }
  // K so that lambda = 1/2 (s = ln 2) is at halfLength.
  const sHalf = Math.LN2 / ds;
  const i0 = Math.floor(sHalf);
  const xHalf = xs[i0] + (xs[i0 + 1] - xs[i0]) * (sHalf - i0);
  const K = xHalf / halfLength;
  for (let i = 0; i <= N; i++) xs[i] /= K;
  /** The state a distance x >= 0 behind the shock (beyond the table: the burned end state). */
  const at = (x: number): ZndState => {
    let s: number;
    if (x >= xs[N]) s = S_MAX;
    else {
      let lo = 0;
      let hi = N;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
      }
      s = (lo + (x - xs[lo]) / (xs[hi] - xs[lo])) * ds;
    }
    const lam = -Math.expm1(-s);
    const { v, p, w } = state(lam);
    return { rho: 1 / v, u: D - w, p, f: 1 - lam };
  };
  return { K, at, vonNeumann: at(0) };
}
