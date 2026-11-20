// The turbulence closure: it tells the flux update, cell by cell, how much of its own numerical
// dissipation to leave out, so that a coarse run keeps more of what a
// fine one resolves (docs/numerics.md section 26, docs/closure.md for how it was trained and
// what it does and does not do).
//
// A coarse run of this scheme is short of small-scale motion, not long on it: the Riemann
// solver damps the velocity jump it is shown at every face, and on a grid that only just
// resolves an eddy those jumps are large. So the closure is not an eddy viscosity, which would
// only add to that. Each cell gets two numbers from its own state and its four neighbors':
//
//   a  in [0, 0.8]  the share of the velocity's jump at a face that the Riemann solver is not shown
//   b  in [0, 0.9]  the same for the passive scalar's jump, each face state then held to where
//                   a limited slope could have put it (hydro/flux.wgsl closure_bound)
//
// and a face takes the smaller of its two cells' values (hydro/flux.wgsl, hydro/closure_on.wgsl).
// The solver's dissipation of shear and its diffusion of the scalar are proportional to those
// jumps, so they fall by those shares and never by all of it. The flux is still one number per
// face, computed alike by both of its cells: mass, momentum and energy are conserved to
// rounding, as without the closure.
//
// Where the gates are open, a and b are either their limits (`constant`) or what a small trained
// network returns (`network`): trained, it returns 99% of the limits everywhere, so the two
// give the same results and the constants cost far less (kinds.ts, docs/closure.md).
//
// Gates (closureCell below) shut the closure, exactly, where the flow is compressive (any
// planar shock, rarefaction or blast), across a pressure jump, and where the gas is neither
// sheared nor turning; there the fluxes are the solver's own to the bit.
//
// The weights and constants are generated (closure_net.ts, by tools/closure/export.py); this
// file holds a float64 mirror of one cell for the tests, the WGSL of its constants, and the
// pass that runs it on the uniform grid (hydro/closure.wgsl).
import { CLOSURE_DATA, CLOSURE_WEIGHTS_F32 } from './closure_net';
import type { ClosureKind } from './kinds';
import { bindGroup, computePipeline, type Bindable } from '../gpu/device';
import { assemble } from '../gpu/shaders';

/** What the closure reads of a cell: its primitives. */
export interface ClosureCell {
  rho: number;
  vx: number;
  vy: number;
  p: number;
  s: number;
}

let weights: Float32Array | undefined;
/** The network's weights: every layer's matrix (rows of outputs), then its bias. */
export function closureWeights(): Float32Array {
  if (!weights) {
    const s = atob(CLOSURE_WEIGHTS_F32);
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    weights = new Float32Array(bytes.buffer);
  }
  return weights;
}

/**
 * The weights as the shader holds them (hydro/closure_net.wgsl cl_net): four to a vec4, each
 * row of a layer's matrix padded with zeros to whole vec4s, then the layer's biases likewise.
 */
export function packedWeights(): Float32Array {
  const W = closureWeights();
  const { features: nin, hidden: H, layers } = CLOSURE_DATA;
  if (H % 4) throw new Error('the closure\u2019s hidden layers must be a multiple of 4 wide');
  const dims = [nin, ...Array(layers).fill(H), 4];
  const out: number[] = [];
  let o = 0;
  for (let l = 0; l + 1 < dims.length; l++) {
    const [a, b] = [dims[l], dims[l + 1]];
    const k = Math.ceil(a / 4) * 4;
    for (let j = 0; j < b; j++) for (let i = 0; i < k; i++) out.push(i < a ? W[o + j * a + i] : 0);
    for (let j = 0; j < Math.ceil(b / 4) * 4; j++) out.push(j < b ? W[o + a * b + j] : 0);
    o += a * b + b;
  }
  return Float32Array.from(out);
}

const smoothstep = ([lo, hi]: readonly number[], x: number): number => {
  const t = Math.min(Math.max((x - lo) / (hi - lo), 0), 1);
  return t * t * (3 - 2 * t);
};

/** hydro/closure_net.wgsl cl_net: the network's four outputs, each through a sigmoid. */
export function closureNet(f: readonly number[]): number[] {
  const W = closureWeights();
  const { features: nin, hidden: H, layers } = CLOSURE_DATA;
  let x = [...f];
  let o = 0;
  const dims = [nin, ...Array(layers).fill(H), 4];
  for (let l = 0; l + 1 < dims.length; l++) {
    const [a, b] = [dims[l], dims[l + 1]];
    const y: number[] = [];
    for (let j = 0; j < b; j++) {
      let acc = W[o + a * b + j];
      for (let i = 0; i < a; i++) acc += W[o + j * a + i] * x[i];
      y.push(l + 2 < dims.length ? Math.tanh(acc) : 1 / (1 + Math.exp(-acc)));
    }
    x = y;
    o += a * b + b;
  }
  return x;
}

/**
 * One cell of the closure in double precision, as hydro/closure_net.wgsl computes it: the
 * features, the gate and the coefficients a and b, from the cell, its neighbors below and
 * above along x and y, and its sound speed.
 */
export function closureCell(c: ClosureCell, xm: ClosureCell, xp: ClosureCell, ym: ClosureCell, yp: ClosureCell, cs: number, kind: Exclude<ClosureKind, 'off'> = 'network'): { features: number[]; gate: number; a: number; b: number } {
  const K = CLOSURE_DATA;
  const gx = [0.5 * (xp.vx - xm.vx), 0.5 * (xp.vy - xm.vy)];
  const gy = [0.5 * (yp.vx - ym.vx), 0.5 * (yp.vy - ym.vy)];
  const D = gx[0] + gy[1];
  const Wz = gx[1] - gy[0];
  const sxx = gx[0] - D / 3;
  const syy = gy[1] - D / 3;
  const szz = -D / 3;
  const sxy = 0.5 * (gy[0] + gx[1]);
  const A2 = 2 * (sxx * sxx + syy * syy + szz * szz + 2 * sxy * sxy);
  const A = Math.sqrt(A2 + 1e-30);
  const e2 = (K.epsV * cs) ** 2;
  const tot = D * D + Wz * Wz + A2 + e2;
  const kink = (dm: number, dp: number, floor: number) => Math.abs(dp - dm) / (Math.abs(dp) + Math.abs(dm) + floor);
  const kink2 = (m: ClosureCell, p: ClosureCell) => {
    const dm = [c.vx - m.vx, c.vy - m.vy];
    const dp = [p.vx - c.vx, p.vy - c.vy];
    const num = Math.sqrt((dp[0] - dm[0]) ** 2 + (dp[1] - dm[1]) ** 2 + 1e-30);
    return num / (Math.sqrt(dp[0] ** 2 + dp[1] ** 2 + 1e-30) + Math.sqrt(dm[0] ** 2 + dm[1] ** 2 + 1e-30) + 1e-4 * cs);
  };
  const kv = Math.max(kink2(xm, xp), kink2(ym, yp));
  const kr = Math.max(kink(c.rho - xm.rho, xp.rho - c.rho, 1e-5 * c.rho), kink(c.rho - ym.rho, yp.rho - c.rho, 1e-5 * c.rho));
  const sf = 1e-4 * (Math.abs(c.s) + Math.abs(xp.s) + Math.abs(xm.s)) + 1e-6;
  const ks = Math.max(kink(c.s - xm.s, xp.s - c.s, sf), kink(c.s - ym.s, yp.s - c.s, sf));
  const speed = Math.sqrt(c.vx * c.vx + c.vy * c.vy + 1e-30);
  const M = speed / cs;
  const grho = Math.sqrt((0.5 * (xp.rho - xm.rho)) ** 2 + (0.5 * (yp.rho - ym.rho)) ** 2 + 1e-30) / c.rho;
  const gp = Math.sqrt((0.5 * (xp.p - xm.p)) ** 2 + (0.5 * (yp.p - ym.p)) ** 2 + 1e-30) / c.p;
  const features = [
    0.5 * Math.log10(A / cs + 1e-4) + 1,
    (Wz * Wz - A2) / (Wz * Wz + A2 + e2),
    D / Math.sqrt(tot),
    kv,
    M / (1 + M),
    grho / (grho + 0.05),
    kr,
    ks,
    gp / (gp + 0.05),
    A / (A + speed + 1e-4 * cs),
  ];
  const jump = Math.max(Math.abs(xp.p - xm.p) / Math.min(xp.p, xm.p), Math.abs(yp.p - ym.p) / Math.min(yp.p, ym.p));
  const active = Math.sqrt(Wz * Wz + A2 + 1e-30) / cs;
  const gate = (1 - smoothstep(K.ducros, (D * D) / tot)) * (1 - smoothstep(K.jump, jump)) * smoothstep(K.active, active);
  if (gate <= 0) return { features, gate: 0, a: 0, b: 0 };
  if (kind === 'constant') return { features, gate, a: K.aMax * gate, b: K.bMax * gate };
  const o = closureNet(features);
  return { features, gate, a: K.aMax * o[0] * gate * K.heads[0], b: K.bMax * o[1] * gate * K.heads[1] };
}

const f = (x: number): string => {
  const s = Math.fround(x).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/** The <closure_net> slot of hydro/closure_net.wgsl: the weights, and the constants exported with them. */
export function closureNetSlot(kind: Exclude<ClosureKind, 'off'> = 'network'): string {
  const K = CLOSURE_DATA;
  const v2 = (a: readonly number[]) => `vec2<f32>(${f(a[0])}, ${f(a[1])})`;
  const w = packedWeights();
  return [
    `const CL_NET = ${kind === 'network'};`,
    `const CL_IN: u32 = ${K.features}u;`,
    `const CL_H: u32 = ${K.hidden}u;`,
    `const CL_K0: u32 = ${Math.ceil(K.features / 4)}u;`,
    `const CL_KH: u32 = ${K.hidden / 4}u;`,
    `const CL_LAYERS: u32 = ${K.layers}u;`,
    `const CL_A_MAX: f32 = ${f(K.aMax)};`,
    `const CL_B_MAX: f32 = ${f(K.bMax)};`,
    `const CL_HEADS = ${v2(K.heads)};`,
    `const CL_DUCROS = ${v2(K.ducros)};`,
    `const CL_JUMP = ${v2(K.jump)};`,
    `const CL_ACTIVE = ${v2(K.active)};`,
    `const CL_EPS_V: f32 = ${f(K.epsV)};`,
    // The weights, written into the shader: read from a buffer, the pass took a fifth longer.
    `const W = array<vec4<f32>, ${w.length / 4}>(${Array.from({ length: w.length / 4 }, (_, i) => `vec4<f32>(${f(w[4 * i])}, ${f(w[4 * i + 1])}, ${f(w[4 * i + 2])}, ${f(w[4 * i + 3])})`).join(', ')});`,
    '',
  ].join('\n');
}

/** The closure's pass on a uniform 2D grid, and the texture of coefficients it writes. */
export class ClosurePass {
  private constructor(
    private pipeline: GPUComputePipeline,
    private groups: GPUBindGroup[],
    /** each cell's (a, b, gate, 0), as of the last step's start */
    readonly coefficients: GPUTexture,
    private dims: number[],
  ) {}

  /**
   * `state` is the two parities of the state's two textures; `slots` the hydro shaders' (the
   * equation of state and the composition); `comp` the composition's binding where it varies.
   */
  static async create(device: GPUDevice, kind: Exclude<ClosureKind, 'off'>, dims: number[], slots: Record<string, string>, params: GPUBuffer, state: GPUTexture[][], comp: Record<number, Bindable>): Promise<ClosurePass> {
    const [nx, ny, nz] = dims;
    const coefficients = device.createTexture({
      label: 'closure coefficients',
      size: [nx, ny, nz],
      dimension: '3d',
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const pipeline = await computePipeline(device, assemble('hydro/closure.wgsl', { ...slots, closure_net: closureNetSlot(kind) }), 'closure');
    const out = coefficients.createView();
    const groups = [0, 1].map((p) =>
      bindGroup(device, pipeline, { 0: params, 1: state[p][0].createView(), 2: state[p][1].createView(), 3: out, ...comp }),
    );
    return new ClosurePass(pipeline, groups, coefficients, dims);
  }

  /** Encode the pass for the state at `parity`. */
  encode(pass: GPUComputePassEncoder, parity: number) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.groups[parity]);
    pass.dispatchWorkgroups(Math.ceil(this.dims[0] / 16), Math.ceil(this.dims[1] / 16), 1);
  }

  destroy() {
    this.coefficients.destroy();
  }
}
