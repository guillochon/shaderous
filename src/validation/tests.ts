// Validation suite (plan section 7). Runs in the page against the real GPU; driven by
// validate.html interactively and by Playwright in tests/gpu.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { configFor, PARAM_BYTES, Sim, type Diagnostics, type RiemannKind, type SimConfig } from '../sim';
import { AmrSim, type LeafCells } from '../amr/amr_sim';
import { defaultEosParams, EOS_MODULES, eosConstants2, makeEos, type EosKind, type EosParams } from '../eos/eos';
import { hydrogenParticles, mixMoments, problemCoefficients } from '../species/mixture';
import { COMPOSITIONS, composition } from '../units/composition';
import { bubbleDensity, shockBubble } from '../problems/shock_bubble';
import { detonationStable, DETONATION_CJ_SPEED, zndOf } from '../problems/detonation';
import { exactRiemann } from '../reference/riemann_exact';
import { RefHydro } from '../reference/hydro';
import { sodX, sodY, sodZ, sodDiag, strongShock, shockTube } from '../problems/shocktube';
import { sedov, SEDOV_XI0 } from '../problems/sedov';
import { advection } from '../problems/advection';
import { kelvinHelmholtz } from '../problems/kelvin_helmholtz';
import { rayleighTaylor, RT } from '../problems/rayleigh_taylor';
import { jeans, JEANS } from '../problems/jeans';
import { POLY, POLY_K, polytrope, polytropeDipole, polytropeRandom } from '../problems/polytrope';
import { whiteDwarfStar } from '../problems/white_dwarf_star';
import { evrard } from '../problems/evrard';
import { tidalDisruption, tdeOrbit, TDE, TDE_STAR } from '../problems/tidal_disruption';
/** The tidal disruption in the box its tests were calibrated in (the presets' is twice as wide). */
const tdeTest: Problem = { ...tidalDisruption, box: TDE.box };
import { sphere, spherePotential } from '../problems/sphere';
import { all, PERIODIC, REFLECT, type Problem } from '../problems/types';
import { binary } from '../problems/binary';
import { thermalInstability } from '../problems/thermal_instability';
import { capture, restartFrom } from '../io/capture';
import { readAthdf, writeAthdf } from '../io/athdf';
import { alfvenWave, brioWu, fieldLoop, fieldLoopDense, mhdBlast, mhdRotor, orszagTang } from '../problems/mhd';
import { initialFaces, setMhdSlabBytes } from '../mhd/mhd';
import { RefMhd } from '../reference/mhd';
import { setMhdScratchBytes } from '../mhd/amr_mhd';
import { BS, PAD as PAD_BLOCK } from '../amr/tree';

export interface TestResult {
  id: string;
  name: string;
  criterion: string;
  pass: boolean;
  metrics: Record<string, number | string>;
  seconds: number;
}

interface TestDef {
  id: string;
  name: string;
  criterion: string;
  run(gpu: Gpu): Promise<{ pass: boolean; metrics: Record<string, number | string> }>;
}

/** Uniform gas at rest in a periodic unit box: rho = 1, p = 1 (so T = 1), gamma = 5/3. */
const stillGas = (over: Partial<Problem>): Problem => ({
  id: 'still-gas', label: 'Still gas', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
  init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }),
  ...over,
});

/**
 * Five species (two groups of four) on a density bump carried diagonally around the box: a
 * smooth one (and the passive scalar with the same profile), one with sharp edges of height
 * `sharp`, one at 0.2 everywhere, and two sharing the rest.
 */
const speciesCarry = (sharp: number, tEnd = 2): Problem =>
  stillGas({
    id: 'species-advection', tEnd, species: ['a', 'b', 'c', 'd', 'e'],
    init: () => (x, y) => {
      const a = 0.3 + 0.2 * Math.sin(2 * Math.PI * x) * Math.sin(2 * Math.PI * y);
      const b = Math.abs(x) < 0.2 && Math.abs(y) < 0.2 ? sharp : 0;
      const rest = 1 - a - b - 0.2;
      return {
        rho: 1 + 0.5 * Math.exp(-((x - 0.1) ** 2 + y ** 2) / 0.02), vx: 1, vy: 0.5, vz: 0, p: 1, s: a,
        X: [a, b, 0.2, rest * 0.75, rest * 0.25],
      };
    },
  });


/**
 * Two gases side by side in pressure balance, with species that set the equation of state
 * (species/mixture.ts): a disc of helium, twice as dense, in hydrogen, p = 1 everywhere, at
 * rest or carried diagonally across the periodic box.
 */
const twoGases = (eos: EosKind, eosParams: Partial<EosParams>, moving: boolean, tEnd: number): Problem =>
  stillGas({
    id: 'two-gases', eos, eosParams, tEnd, species: ['h1', 'he4'],
    init: () => (x, y) => {
      const he = Math.hypot(x, y) < 0.25;
      return { rho: he ? 2 : 1, vx: moving ? 1 : 0, vy: moving ? 0.5 : 0, vz: 0, p: 1, s: he ? 1 : 0, X: he ? [0, 1] : [1, 0] };
    },
  });

/**
 * Two gases with their own gamma: a disc of helium (5/3) in air (1.4), ideal gases at p = 1,
 * the helium at the air's temperature (`hot` = 1) or `hot` times it.
 */
const heliumInAir = (moving: boolean, tEnd: number, hot: number): Problem =>
  stillGas({
    id: 'helium-in-air', eos: 'ideal', eosParams: { gamma: 1.4 }, tEnd, species: ['he4', 'air'],
    composition: composition('air', { air: 1 }, 'neutral'), speciesGamma: { air: 1.4, he4: 5 / 3 },
    init: () => (x, y) => {
      const he = Math.hypot(x, y) < 0.25;
      return { rho: he ? bubbleDensity(1) / hot : 1, vx: moving ? 1 : 0, vy: moving ? 0.5 : 0, vz: 0, p: 1, s: he ? 1 : 0, X: he ? [1, 0] : [0, 1] };
    },
  });

/** The equations of state the two-gas tests cover, each with constants that make the composition matter. */
const MIXED_EOS: { kind: EosKind; params: Partial<EosParams> }[] = [
  { kind: 'ideal', params: { gamma: 5 / 3 } },
  { kind: 'gas_radiation', params: { gamma: 5 / 3, R: 1, arad: 1 } },
  { kind: 'degenerate', params: { gamma: 5 / 3, R: 1, arad: 1e-2, degRho: 1, degP: 0.2 } },
  { kind: 'ionizing', params: { gamma: 5 / 3 } },
];

/**
 * How a two-gas run looks: the largest speed off the carrying velocity v0 and the largest
 * pressure error, both from the state as the CPU's float64 equations of state read it with each
 * cell's own composition; and the GPU's own temperature and pressure ranges against the CPU's.
 */
function twoGasErrors(cfg: SimConfig, cells: { u1: Float32Array; u2: Float32Array; X: Float32Array[]; count: number }, v0: [number, number], diag: Diagnostics) {
  const coeffs = problemCoefficients(cfg.problem, cfg.eosParams.gamma)!;
  let dv = 0;
  let dp = 0;
  let tLo = Infinity;
  let tHi = -Infinity;
  for (let q = 0; q < cells.count; q++) {
    const rho = cells.u1[4 * q];
    dv = Math.max(dv, Math.hypot(cells.u1[4 * q + 1] / rho - v0[0], cells.u1[4 * q + 2] / rho - v0[1]));
    const eos = makeEos(cfg.eos, cfg.eosParams, mixMoments(coeffs, cells.X.map((x) => x[q])), hydrogenParticles(cfg.problem.composition));
    const eint = cells.u2[4 * q + 1];
    dp = Math.max(dp, Math.abs(eos.pressure(rho, eint) - 1));
    const T = eos.temperature(rho, eint);
    tLo = Math.min(tLo, T);
    tHi = Math.max(tHi, T);
  }
  const [gLo, gHi] = diag.ranges[1];
  const [pLo, pHi] = diag.ranges[3];
  const tErr = Math.max(Math.abs(gLo - tLo) / tLo, Math.abs(gHi - tHi) / tHi);
  return { dv, dp, tErr, gpuDp: Math.max(Math.abs(pLo - 1), Math.abs(pHi - 1)), tRatio: tHi / tLo };
}

/**
 * Gresho vortex: a steady vortex in a periodic unit box, the pressure balancing the centrifugal
 * force, with the background pressure setting the Mach number of its fastest gas (speed 1 at
 * r = 0.2). Nothing should change; what does is the solver's own dissipation.
 */
const gresho = (mach: number, tEnd = 1): Problem => {
  const gamma = 1.4;
  const p0 = 1 / (gamma * mach * mach);
  return stillGas({
    id: 'gresho', eosParams: { gamma }, tEnd,
    init: () => (x, y) => {
      const r = Math.hypot(x, y);
      let u = 0;
      let p = p0 - 2 + 4 * Math.log(2);
      if (r < 0.2) {
        u = 5 * r;
        p = p0 + 12.5 * r * r;
      } else if (r < 0.4) {
        u = 2 - 5 * r;
        p = p0 + 12.5 * r * r + 4 * (1 - 5 * r - Math.log(0.2) + Math.log(r));
      }
      const c = r > 0 ? [-y / r, x / r] : [0, 0];
      return { rho: 1, vx: u * c[0], vy: u * c[1], vz: 0, p, s: 0 };
    },
  });
};

/**
 * An isothermal column at rest in a uniform field, between two walls: p = rho (so the
 * temperature is 1) falling by e every `cells` cells. The cell values obey the trapezoid rule
 * p[j+1] - p[j] = -g dx (rho[j] + rho[j+1]) / 2 exactly, which is the balance the well-balanced
 * reconstruction holds.
 */
export const column = (cells: number): Problem =>
  stillGas({
    id: 'column', eosParams: { gamma: 1.4 }, tEnd: 1,
    bcLo: [0, 2, 0], bcHi: [0, 2, 0],
    gext: () => [0, -64 / cells, 0],
    init: (n) => {
      const a = (64 / cells) / n / 2; // g dx / 2, with p = rho
      const q = (1 - a) / (1 + a);
      return (_x, y) => {
        const p = Math.pow(q, Math.round((y + 0.5) * n - 0.5));
        return { rho: p, vx: 0, vy: 0, vz: 0, p, s: 0 };
      };
    },
  });

/** The binary on a 2D adaptive mesh: largest deviation of the separation from 1 over one period. */
async function adaptiveBinary(gpu: Gpu, period: number): Promise<number> {
  return withAmr(gpu, binary, 256, { ndim: 2, amrLevels: 2, tEnd: 0 }, async (sim) => {
    let worst = 0;
    for (let i = 1; i <= 8; i++) {
      sim.cfg.tEnd = (period * i) / 8;
      sim.writeParams();
      await sim.runToEnd();
      const [a, b] = (await sim.diagnostics()).masses;
      worst = Math.max(worst, Math.abs(Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) - 1));
    }
    return worst;
  });
}

/**
 * The faces of every leaf, in the order of readLeaves(): (BS + 1)^3 texels a leaf (2D:
 * (BS + 1)^2), texel l holding the lower faces of cell l. Read leaf by leaf, not as the atlas
 * lies: two runs of one setup may get atlases of different sizes (AmrSim.create makes a smaller
 * one when the GPU refuses the first, as Firefox does for some of the 3D runs, 9^3 blocks
 * falling to 8^3).
 */
async function leafFaces(sim: AmrSim): Promise<Float32Array> {
  const s = sim as unknown as { slotsPerSide: number; layers: number; origin(slot: number): number[]; tree: { leaves(): { slot: number }[] } };
  const three = sim.cfg.ndim === 3;
  const side = s.slotsPerSide * PAD_BLOCK;
  const atlas = await sim.mhd!.readAtlas(sim.cur, side, three ? side : s.layers);
  const W = BS + 1;
  const D = three ? W : 1;
  const leaves = s.tree.leaves();
  const out = new Float32Array(leaves.length * W * W * D * 4);
  let q = 0;
  for (const blk of leaves) {
    const [ox, oy, oz] = s.origin(blk.slot);
    for (let k = 0; k < D; k++)
      for (let j = 0; j < W; j++, q += W * 4) {
        const src = (((oz + k) * side + oy + j) * side + ox) * 4;
        out.set(atlas.subarray(src, src + W * 4), q);
      }
  }
  return out;
}

export async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
  const sim = await Sim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

/** Radius of a blast wave on the adaptive mesh: the half-height point of the outer edge of the shell-averaged density. */
function leafShockRadius(c: LeafCells): number {
  const nb = 128;
  const bw = 0.5 / nb;
  const sum = new Float64Array(nb);
  const vol = new Float64Array(nb);
  for (let q = 0; q < c.count; q++) {
    const b = Math.floor(Math.hypot(c.x[q], c.y[q], c.z[q]) / bw);
    if (b >= nb) continue;
    const v = c.dx[q] ** 3;
    sum[b] += c.u1[q * 4] * v;
    vol[b] += v;
  }
  const prof = Array.from(sum, (v, i) => (vol[i] ? v / vol[i] : 1));
  const half = 0.5 * (Math.max(...prof) + 1);
  for (let b = nb - 1; b > 0; b--) {
    if (prof[b - 1] >= half) return (b - 0.5 + (prof[b - 1] - half) / (prof[b - 1] - prof[b])) * bw;
  }
  return NaN;
}

export async function withAmr<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: AmrSim) => Promise<T>) {
  const sim = await AmrSim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

// Detonations (problems/detonation.ts): where the front is, as the farthest point along the line
// y = 0 at which the pressure falls through 10, between the fuel's 1 and the spike's 42. With
// gamma 1.2 the pressure is 0.2 e_int.
const DET_P = 10;

function frontOf(xs: number[], ps: number[]): number {
  for (let i = xs.length - 2; i >= 0; i--) {
    if (ps[i] > DET_P && ps[i + 1] <= DET_P) return xs[i] + ((ps[i] - DET_P) / (ps[i] - ps[i + 1])) * (xs[i + 1] - xs[i]);
  }
  return NaN;
}

/** The detonation front and the fuel left 8 to 15 units behind it, on a uniform 2D grid. */
function detonationRow(u1: Float32Array, u2: Float32Array, X: Float32Array[], n: number, box: number) {
  const j = n >> 1;
  const dx = box / n;
  const xs: number[] = [];
  const ps: number[] = [];
  for (let i = 0; i < n; i++) {
    xs.push(-box / 2 + (i + 0.5) * dx);
    ps.push(0.2 * u2[4 * (j * n + i) + 1]);
  }
  const front = frontOf(xs, ps);
  let left = 0;
  let cells = 0;
  let spike = 0;
  for (let i = 0; i < n; i++) {
    const behind = front - xs[i];
    if (behind > 8 && behind < 15) {
      left += X[0][j * n + i];
      cells++;
    }
    if (behind > -1 && behind < 3) spike = Math.max(spike, ps[i]);
  }
  return { front, fuelLeft: left / Math.max(cells, 1), spike };
}

/** The same on the adaptive mesh, from the leaf cells along y = 0. */
function detonationLeaves(c: LeafCells, X: Float32Array[]) {
  const row: { x: number; p: number; f: number }[] = [];
  for (let q = 0; q < c.count; q++) if (Math.abs(c.y[q]) < c.dx[q]) row.push({ x: c.x[q], p: 0.2 * c.u2[4 * q + 1], f: X[0][q] });
  row.sort((a, b) => a.x - b.x);
  // Two leaves may straddle y = 0 at each x: keep one per x.
  const one = row.filter((r, i) => i === 0 || r.x !== row[i - 1].x);
  const front = frontOf(one.map((r) => r.x), one.map((r) => r.p));
  const behind = one.filter((r) => front - r.x > 8 && front - r.x < 15);
  // Cells of the reaction zone (1% to 99% of the fuel burned) that are not at the finest level.
  const finest = c.dx.reduce((a, v) => Math.min(a, v), Infinity);
  let zone = 0;
  let zoneCoarse = 0;
  for (let q = 0; q < c.count; q++)
    if (X[0][q] > 0.01 && X[0][q] < 0.99) {
      zone++;
      if (c.dx[q] > 1.5 * finest) zoneCoarse++;
    }
  return { front, fuelLeft: behind.reduce((a, r) => a + r.f, 0) / Math.max(behind.length, 1), zoneCoarse: zoneCoarse / Math.max(zone, 1) };
}

/** Uniform-grid index of a leaf cell (for meshes whose leaves all sit at the finest level). */
const cellIndex = (c: LeafCells, q: number, n: number, box: number, ndim: number) => {
  const at = (v: number) => Math.round((v + box / 2) / (box / n) - 0.5);
  return ((ndim === 3 ? at(c.z[q]) : 0) * n + at(c.y[q])) * n + at(c.x[q]);
};

const rhoAt = (u1: Float32Array, n: number, i: number, j: number, k: number) => u1[((k * n + j) * n + i) * 4];

/** Density along a grid axis through the box center, plus the largest deviation across the transverse plane. */
function axisProfile(u1: Float32Array, n: number, axis: number) {
  const prof = new Float64Array(n);
  let spread = 0;
  for (let a = 0; a < n; a++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let b = 0; b < n; b++)
      for (let c = 0; c < n; c++) {
        const ijk = axis === 0 ? [a, b, c] : axis === 1 ? [c, a, b] : [b, c, a];
        const r = rhoAt(u1, n, ijk[0], ijk[1], ijk[2]);
        lo = Math.min(lo, r);
        hi = Math.max(hi, r);
      }
    prof[a] = rhoAt(u1, n, ...((axis === 0 ? [a, n / 2, n / 2] : axis === 1 ? [n / 2, a, n / 2] : [n / 2, n / 2, a]) as [number, number, number]));
    spread = Math.max(spread, hi - lo);
  }
  return { prof, spread };
}

async function tubeProfile(gpu: Gpu, problem: Problem, n: number, axis: number) {
  return withSim(gpu, problem, n, {}, async (sim) => {
    const steps = await sim.runToEnd();
    const { u1 } = await sim.readState();
    const d = await sim.diagnostics();
    return { ...axisProfile(u1, n, axis), steps, diag: d };
  });
}

function l1VsExact(prof: Float64Array, tube: ReturnType<typeof shockTube>, t: number, ds: number, relative = false) {
  const exact = exactRiemann(tube.left, tube.right, tube.eosParams.gamma!);
  const n = prof.length;
  let err = 0;
  for (let i = 0; i < n; i++) {
    const s = (i + 0.5 - n / 2) * ds;
    const e = exact(s / t).rho;
    err += Math.abs(prof[i] - e) / (relative ? e : 1);
  }
  return err / n;
}

const maxDiff = (a: Float64Array, b: Float64Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

/** Shell-averaged density and the half-height radius of its outer edge. */
function shockRadius(u1: Float32Array, n: number, box: number, dir?: [number, number, number]) {
  const dx = box / n;
  if (dir) {
    // Along a ray from the center: last sample (outside-in) above the half-way density.
    const norm = Math.hypot(...dir);
    let peak = 0;
    const vals: [number, number][] = [];
    for (let s = 0; s < n * 0.5 * 0.98; s += 0.25) {
      const p = dir.map((c) => n / 2 + (c / norm) * s - 0.5);
      // trilinear sample
      const i0 = p.map(Math.floor);
      const f = p.map((v, d) => v - i0[d]);
      let v = 0;
      for (let c = 0; c < 8; c++) {
        const o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
        const w = o.reduce((acc, bit, d) => acc * (bit ? f[d] : 1 - f[d]), 1);
        const idx = o.map((bit, d) => Math.min(n - 1, Math.max(0, i0[d] + bit)));
        v += w * rhoAt(u1, n, idx[0], idx[1], idx[2]);
      }
      vals.push([s * dx, v]);
      peak = Math.max(peak, v);
    }
    const half = 0.5 * (peak + 1);
    for (let i = vals.length - 1; i > 0; i--) {
      if (vals[i - 1][1] >= half) {
        const [r0, v0] = vals[i - 1];
        const [r1, v1] = vals[i];
        return r0 + ((v0 - half) / (v0 - v1)) * (r1 - r0);
      }
    }
    return NaN;
  }
  const nb = n;
  const sum = new Float64Array(nb);
  const cnt = new Float64Array(nb);
  const bw = (0.5 * box) / nb;
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const r = Math.hypot(i + 0.5 - n / 2, j + 0.5 - n / 2, k + 0.5 - n / 2) * dx;
        const b = Math.floor(r / bw);
        if (b < nb) {
          sum[b] += rhoAt(u1, n, i, j, k);
          cnt[b]++;
        }
      }
  const prof = Array.from(sum, (s, i) => (cnt[i] ? s / cnt[i] : 1));
  const peak = Math.max(...prof);
  const half = 0.5 * (peak + 1);
  for (let b = nb - 1; b > 0; b--) {
    if (prof[b - 1] >= half) {
      return (b - 0.5 + (prof[b - 1] - half) / (prof[b - 1] - prof[b])) * bw;
    }
  }
  return NaN;
}

const N_TUBE = 64;

export const TESTS: TestDef[] = [
  {
    id: 'sod-axes',
    name: 'Sod shock tube along x, y, z',
    criterion:
      'L1 density error < 0.01 against the exact solution; the three runs agree under axis permutation to 1e-6 (bit-identical in Chrome, last-bit differences in Firefox); planar to round-off',
    async run(gpu) {
      const runs = [];
      for (const [p, axis] of [[sodX, 0], [sodY, 1], [sodZ, 2]] as const) runs.push(await tubeProfile(gpu, p, N_TUBE, axis));
      const l1 = runs.map((r) => l1VsExact(r.prof, sodX, sodX.tEnd, 1 / N_TUBE));
      const perm = Math.max(maxDiff(runs[0].prof, runs[1].prof), maxDiff(runs[0].prof, runs[2].prof));
      const spread = Math.max(...runs.map((r) => r.spread));
      return {
        pass: Math.max(...l1) < 0.01 && perm < 1e-6 && spread < 1e-5,
        metrics: { l1_x: l1[0], l1_y: l1[1], l1_z: l1[2], permutation_max_diff: perm, transverse_spread: spread, steps: runs[0].steps },
      };
    },
  },
  {
    id: 'sod-diagonal',
    name: 'Sod shock tube along the (1,1,1) diagonal',
    criterion: 'L1 density error < 0.015 against the exact solution along the diagonal',
    async run(gpu) {
      const n = N_TUBE;
      return withSim(gpu, sodDiag, n, {}, async (sim) => {
        const steps = await sim.runToEnd();
        const { u1 } = await sim.readState();
        const exact = exactRiemann(sodDiag.left, sodDiag.right, 1.4);
        let err = 0;
        let cnt = 0;
        for (let i = 0; i < n; i++) {
          const s = Math.sqrt(3) * (i + 0.5 - n / 2) / n;
          if (Math.abs(s) > 0.3) continue;
          err += Math.abs(rhoAt(u1, n, i, i, i) - exact(s / sodDiag.tEnd).rho);
          cnt++;
        }
        const l1 = err / cnt;
        return { pass: l1 < 0.015, metrics: { l1, steps } };
      });
    },
  },
  {
    id: 'strong-shock',
    name: 'Strong shock tube (pressure ratio 1e5)',
    criterion: 'No floored cells anywhere; relative L1 density error < 0.1 at 128 cells (the dense shell is 6 cells wide)',
    async run(gpu) {
      const r = await tubeProfile(gpu, strongShock, 128, 0);
      const l1 = l1VsExact(r.prof, strongShock, strongShock.tEnd, 1 / 128, true);
      return {
        pass: r.diag.floored === 0 && l1 < 0.1,
        metrics: { l1_relative: l1, floored_cells: r.diag.floored, max_mach: r.diag.maxMach, steps: r.steps },
      };
    },
  },
  {
    id: 'sedov',
    name: 'Sedov-Taylor blast',
    criterion: 'Shell-averaged shock radius within 2% of the similarity solution; radii along axes and diagonals agree to 1.5 cells',
    async run(gpu) {
      const n = 128;
      return withSim(gpu, sedov, n, {}, async (sim) => {
        const steps = await sim.runToEnd();
        const { u1 } = await sim.readState();
        const d = await sim.diagnostics();
        const rExact = SEDOV_XI0['1.4'] * Math.pow(sedov.tEnd ** 2, 0.2);
        const r = shockRadius(u1, n, 1);
        const dirs: [number, number, number][] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 1]];
        const rays = dirs.map((v) => shockRadius(u1, n, 1, v));
        const mean = rays.reduce((a, b) => a + b, 0) / rays.length;
        const aniso = (Math.max(...rays) - Math.min(...rays)) / mean;
        const anisoCells = (Math.max(...rays) - Math.min(...rays)) * n;
        const err = Math.abs(r - rExact) / rExact;
        return {
          pass: err < 0.02 && anisoCells < 1.5,
          metrics: {
            shock_radius: r, similarity_radius: rExact, radius_error: err, anisotropy: aniso, anisotropy_cells: anisoCells,
            radius_along_axes: (rays[0] + rays[1] + rays[2]) / 3,
            radius_along_face_diagonals: (rays[3] + rays[4] + rays[5]) / 3,
            radius_along_body_diagonals: (rays[6] + rays[7]) / 2,
            cell_size: 1 / n, total_energy: d.eKin + d.eInt, steps,
          },
        };
      });
    },
  },
  {
    id: 'kelvin-helmholtz',
    name: 'Kelvin-Helmholtz instability',
    criterion: 'Qualitative: the seeded mode grows (transverse kinetic energy up > 5x by t = 1) and the run stays finite',
    async run(gpu) {
      const n = 64;
      const eKy = (u1: Float32Array) => {
        let e = 0;
        for (let o = 0; o < u1.length; o += 4) e += (0.5 * u1[o + 2] * u1[o + 2]) / u1[o];
        return e / (n * n * n);
      };
      return withSim(gpu, kelvinHelmholtz, n, { tEnd: 1 }, async (sim) => {
        const e0 = eKy((await sim.readState()).u1);
        const steps = await sim.runToEnd();
        const e1 = eKy((await sim.readState()).u1);
        const d = await sim.diagnostics();
        return { pass: e1 / e0 > 5 && Number.isFinite(d.eTot), metrics: { transverse_ke_growth: e1 / e0, max_mach: d.maxMach, steps } };
      });
    },
  },
  {
    id: 'low-mach',
    name: 'HLLC with the low-Mach fix: Gresho vortex at Mach 0.01, and a Sod shock tube',
    criterion:
      'After t = 2 (1.6 turns) on a 64 x 64 grid the vortex loses less than half the kinetic energy it loses with plain HLLC, and gains none; Sod L1 density error < 0.01, as with plain HLLC',
    async run(gpu) {
      const n = 64;
      const kept = (riemann: RiemannKind) =>
        withSim(gpu, gresho(0.01, 2), n, { ndim: 2, riemann }, async (sim) => {
          const ke = async () => {
            const { u1 } = await sim.readState();
            let e = 0;
            for (let o = 0; o < u1.length; o += 4) e += (0.5 * (u1[o + 1] ** 2 + u1[o + 2] ** 2)) / u1[o];
            return e;
          };
          const e0 = await ke();
          const steps = await sim.runToEnd();
          return { kept: (await ke()) / e0, steps };
        });
      const plain = await kept('hllc');
      const fixed = await kept('hllclm');
      const tube = await withSim(gpu, sodX, N_TUBE, { riemann: 'hllclm' }, async (sim) => {
        await sim.runToEnd();
        return l1VsExact(axisProfile((await sim.readState()).u1, N_TUBE, 0).prof, sodX, sodX.tEnd, 1 / N_TUBE);
      });
      return {
        pass: fixed.kept < 1 && 1 - fixed.kept < 0.5 * (1 - plain.kept) && tube < 0.01,
        metrics: { kinetic_energy_kept_hllc: plain.kept, kinetic_energy_kept_low_mach: fixed.kept, sod_l1: tube, steps: fixed.steps },
      };
    },
  },
  {
    id: 'well-balanced',
    name: 'Well-balanced reconstruction: a stratified column at rest between walls',
    criterion:
      'A 64-cell isothermal column whose density falls by e every 4 cells (a factor of 9e6 in all), in discrete hydrostatic balance, after one sound crossing: with the well-balanced reconstruction the largest Mach number stays below 1e-3 on the uniform grid and on the block grid, and is at least 100 times smaller than without it',
    // What is left is rounding: sound waves seeded at a Mach number of 1e-8 where the density is
    // 1, growing as 1 / sqrt(rho) on their way up, to 3e-5 to 2e-4 at the top. It stays there
    // (measured to four sound crossings) and is a different draw on each grid, adiabatic index
    // and shader compiler: Firefox gave 1.1e-4 where Chrome gave 8.6e-5.
    async run(gpu) {
      const over = { ndim: 2 as const, tEnd: 1 };
      const mach = async (wellBalanced: boolean, blocks: boolean) => {
        const go = async (sim: Sim | AmrSim) => {
          await sim.runToEnd();
          return (await sim.diagnostics()).maxMach;
        };
        return blocks ? withAmr(gpu, column(4), 64, { ...over, amrLevels: 0, wellBalanced }, go) : withSim(gpu, column(4), 64, { ...over, wellBalanced }, go);
      };
      const plain = await mach(false, false);
      const balanced = await mach(true, false);
      const plainBlocks = await mach(false, true);
      const balancedBlocks = await mach(true, true);
      return {
        pass: balanced < 1e-3 && balancedBlocks < 1e-3 && balanced < 0.01 * plain && balancedBlocks < 0.01 * plainBlocks,
        metrics: { max_mach_plain: plain, max_mach_well_balanced: balanced, max_mach_plain_blocks: plainBlocks, max_mach_well_balanced_blocks: balancedBlocks },
      };
    },
  },
  {
    id: 'advection',
    name: 'Uniform advection, 10 box crossings',
    criterion: 'Relative drift of mass and of each momentum component < 2e-6 (float32 round-off over ~4000 steps)',
    async run(gpu) {
      return withSim(gpu, advection, 64, {}, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64);
        const d1 = await sim.diagnostics();
        const dm = Math.abs(d1.mass - d0.mass) / d0.mass;
        const dp = Math.max(...d0.momentum.map((p, i) => Math.abs(d1.momentum[i] - p) / Math.abs(p)));
        const de = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        return { pass: dm < 2e-6 && dp < 2e-6, metrics: { mass_drift: dm, momentum_drift: dp, energy_drift: de, steps } };
      });
    },
  },
  {
    id: 'species',
    name: 'Species carried with the gas (2D)',
    criterion:
      'On a density bump carried twice around the box, five species (two groups of four) with a sharp-edged one among them: mass fractions stay in [0, 1] and sum to 1 within 1e-5, the mass of each species is kept within 1e-5 of the total, and one at 0.2 everywhere stays 0.2 within 1e-5. A smooth species returns at least as accurately as the passive scalar carrying the same profile (mean error 3.5e-4 at 128^2), and beside the sharp one (each species limited on its own) its mean error stays below 5e-4.',
    async run(gpu) {
      const n = 128;
      const mass = (X: Float32Array[], u1: Float32Array) => X.map((x) => x.reduce((s, v, q) => s + v * u1[4 * q], 0));
      const run = (sharp: number) =>
        withSim(gpu, speciesCarry(sharp), n, { ndim: 2 }, async (sim) => {
          const X0 = await sim.readSpecies();
          const m0 = mass(X0, (await sim.readState()).u1);
          await sim.runToEnd(64);
          const X1 = await sim.readSpecies();
          const st = await sim.readState();
          const m1 = mass(X1, st.u1);
          let sumErr = 0;
          let lo = Infinity;
          let hi = -Infinity;
          let flatErr = 0;
          let l1 = 0;
          let l1Scalar = 0;
          for (let q = 0; q < n * n; q++) {
            let s = 0;
            for (const x of X1) {
              s += x[q];
              lo = Math.min(lo, x[q]);
              hi = Math.max(hi, x[q]);
            }
            sumErr = Math.max(sumErr, Math.abs(s - 1));
            flatErr = Math.max(flatErr, Math.abs(X1[2][q] - 0.2));
            l1 += Math.abs(X1[0][q] - X0[0][q]) / (n * n);
            l1Scalar += Math.abs(st.u2[4 * q + 2] / st.u1[4 * q] - X0[0][q]) / (n * n);
          }
          return { sumErr, lo, hi, flatErr, l1, l1Scalar, massErr: Math.max(...m0.map((m, i) => Math.abs(m1[i] - m))) / m0.reduce((a, b) => a + b, 0) };
        });
      const mixed = await run(0.3);
      const plain = await run(0);
      const ok = (r: typeof mixed) => r.sumErr < 1e-5 && r.lo >= 0 && r.hi <= 1 + 1e-6 && r.massErr < 1e-5 && r.flatErr < 1e-5;
      return {
        pass: ok(mixed) && ok(plain) && plain.l1 <= plain.l1Scalar && mixed.l1 < 5e-4,
        metrics: {
          sum_error: Math.max(mixed.sumErr, plain.sumErr), min_fraction: Math.min(mixed.lo, plain.lo), max_fraction: Math.max(mixed.hi, plain.hi),
          species_mass_drift: Math.max(mixed.massErr, plain.massErr), constant_species_error: Math.max(mixed.flatErr, plain.flatErr),
          smooth_species_l1: plain.l1, passive_scalar_l1: plain.l1Scalar, smooth_species_l1_beside_sharp_one: mixed.l1,
        },
      };
    },
  },
  {
    id: 'species-eos',
    name: 'Species that set the equation of state (2D)',
    criterion:
      "A disc of helium twice as dense as the hydrogen around it, in pressure balance, under an ideal gas, gas plus radiation, degenerate electrons and ionizing hydrogen. At rest it stays at rest (speeds under 1e-5) with p = 1 to 1e-5, and the GPU's temperature and pressure ranges match the float64 equations of state of each cell's composition to 1e-5. Carried once across the box the pressure stays within each equation of state's bound (balanced to 1e-5 for the ideal gas, whose pressure does not depend on the composition; within 0.05 for the others, which err where the two gases mix: 0.3% to 1.3% at 64^2) and the faces match bit for bit. Two ideal gases with their own gammas (helium in air, 5/3 and 1.4) do the same: at one temperature the pressure stays within 0.01 of 1 (0.2%), with the helium twice as hot within 0.05 (2.8%). With a uniform composition, carrying species that set the equation of state changes a Sod shock tube under gas plus radiation by under 1e-5 of its range",
    async run(gpu) {
      const n = 64;
      const metrics: Record<string, number> = {};
      let pass = true;
      // Moving contacts: what each equation of state is allowed (the mixing cells).
      const moveBound: Record<string, number> = { ideal: 1e-5, gas_radiation: 0.05, degenerate: 0.05, ionizing: 0.05, two_gammas: 0.01, two_gammas_hot: 0.05 };
      const cases = [
        ...MIXED_EOS.map((c) => ({ tag: c.kind as string, make: (moving: boolean, t: number) => twoGases(c.kind, c.params, moving, t) })),
        { tag: 'two_gammas', make: (moving: boolean, t: number) => heliumInAir(moving, t, 1) },
        { tag: 'two_gammas_hot', make: (moving: boolean, t: number) => heliumInAir(moving, t, 2) },
      ];
      for (const c of cases) {
        for (const moving of [false, true]) {
          const r = await withSim(gpu, c.make(moving, moving ? 1 : 0.25), n, { ndim: 2 }, async (sim) => {
            await sim.runToEnd();
            const st = await sim.readState();
            const X = await sim.readSpecies();
            const e = twoGasErrors(sim.cfg, { ...st, X, count: n * n }, moving ? [1, 0.5] : [0, 0], await sim.diagnostics());
            const flux = moving ? await sim.fluxMismatch() : { mismatched: 0 };
            return { ...e, mismatched: flux.mismatched };
          });
          const tag = `${c.tag}_${moving ? 'moving' : 'static'}`;
          metrics[`${tag}_speed`] = r.dv;
          metrics[`${tag}_pressure_error`] = r.dp;
          metrics[`${tag}_gpu_temperature_error`] = r.tErr;
          if (!moving) metrics[`${c.tag}_temperature_ratio`] = r.tRatio;
          if (moving) metrics[`${tag}_mismatched_faces`] = r.mismatched;
          if (moving) pass &&= r.dp < moveBound[c.tag] && r.mismatched === 0;
          else pass &&= r.dv < 1e-5 && r.dp < 1e-5 && r.gpuDp < 1e-5 && r.tErr < 1e-5;
        }
      }
      // A uniform composition carried as species gives the run it would have had without them.
      const solar = COMPOSITIONS.solar;
      const sod = (species: boolean): Problem =>
        stillGas({
          id: 'sod-gasrad', eos: 'gas_radiation', eosParams: { gamma: 5 / 3, R: 1, arad: 0.3 }, tEnd: 0.15,
          bcLo: [1, 0, 0], bcHi: [1, 0, 0],
          ...(species ? { species: solar.species.map((x) => x.name) } : {}),
          init: () => (x) => ({ rho: x < 0 ? 1 : 0.125, vx: 0, vy: 0, vz: 0, p: x < 0 ? 1 : 0.1, s: 0, X: solar.X }),
        });
      const runSod = (species: boolean) => withSim(gpu, sod(species), n, { ndim: 2 }, async (sim) => (await sim.runToEnd(), sim.readState()));
      const [a, b] = [await runSod(false), await runSod(true)];
      let diff = 0;
      let scale = 0;
      for (let i = 0; i < a.u1.length; i++) {
        diff = Math.max(diff, Math.abs(a.u1[i] - b.u1[i]), Math.abs(a.u2[i] - b.u2[i]));
        scale = Math.max(scale, Math.abs(a.u1[i]), Math.abs(a.u2[i]));
      }
      metrics.uniform_composition_difference = diff / scale;
      pass &&= diff / scale < 1e-5;
      return { pass, metrics };
    },
  },
  {
    id: 'shock-bubble',
    name: 'Shock hitting a helium bubble (2D, two gammas)',
    criterion:
      'The shock-bubble preset (Mach 1.22 in air, gamma 1.4, into helium with 28% air, gamma 5/3) to t = 0.6, the bubble flattened and rolling up, on 256^2 and on an adaptive mesh 256 finest with 2 levels: the helium is kept to 1e-5 of its mass, mass fractions stay in [0, 1], the pressure stays positive and finite, faces match bit for bit (across refinement boundaries too), and the helium moves downstream by at least 0.2. On the adaptive mesh over 95% of the helium is at the finest level when it ends, which the mass fractions in the refinement indicator see to (the density and pressure alone let the bubble drift onto coarse blocks). On the uniform grid the helium stays mirror-symmetric about the axis to 1e-5',
    async run(gpu) {
      const n = 256;
      const problem: Problem = { ...shockBubble, tEnd: 0.6 };
      const metrics: Record<string, number> = {};
      let pass = true;
      // Helium mass, its centroid along x, and its fractions' range, from cell fractions, densities and volumes.
      const helium = (X: Float32Array[], u1: Float32Array, x: (q: number) => number, vol: (q: number) => number, count: number) => {
        let m = 0;
        let mx = 0;
        let lo = Infinity;
        let hi = -Infinity;
        for (let q = 0; q < count; q++) {
          const w = X[0][q] * u1[4 * q] * vol(q);
          m += w;
          mx += w * x(q);
          lo = Math.min(lo, X[0][q], X[1][q]);
          hi = Math.max(hi, X[0][q], X[1][q]);
        }
        return { m, x: mx / m, lo, hi };
      };
      const judge = (tag: string, a: ReturnType<typeof helium>, b: ReturnType<typeof helium>, d: Diagnostics, mismatched: number) => {
        metrics[`${tag}_helium_mass_drift`] = Math.abs(b.m - a.m) / a.m;
        metrics[`${tag}_min_fraction`] = b.lo;
        metrics[`${tag}_max_fraction`] = b.hi;
        metrics[`${tag}_min_pressure`] = d.ranges[3][0];
        metrics[`${tag}_max_pressure`] = d.ranges[3][1];
        metrics[`${tag}_helium_centroid_shift`] = b.x - a.x;
        metrics[`${tag}_mismatched_faces`] = mismatched;
        pass &&= metrics[`${tag}_helium_mass_drift`] < 1e-5 && b.lo >= 0 && b.hi <= 1 + 1e-6 && d.ranges[3][0] > 0 && Number.isFinite(d.ranges[3][1]) &&
          mismatched === 0 && b.x - a.x > 0.2;
      };
      await withSim(gpu, problem, n, { ndim: 2 }, async (sim) => {
        const cellX = (q: number) => ((q % n) + 0.5) / n - 0.5;
        const read = async () => {
          const { u1 } = await sim.readState();
          const X = await sim.readSpecies();
          return { X, h: helium(X, u1, cellX, () => 1, n * n) };
        };
        const before = await read();
        await sim.runToEnd();
        const after = await read();
        let asym = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) asym = Math.max(asym, Math.abs(after.X[0][j * n + i] - after.X[0][(n - 1 - j) * n + i]));
        judge('uniform', before.h, after.h, await sim.diagnostics(), (await sim.fluxMismatch()).mismatched);
        metrics.uniform_asymmetry = asym;
        pass &&= asym < 1e-5;
      });
      await withAmr(gpu, problem, n, { ndim: 2, amrLevels: 2 }, async (sim) => {
        let finest = 0;
        const read = async () => {
          const c = await sim.readLeaves();
          const X = await sim.readLeafSpecies();
          // The share of the helium on the finest level: the bubble is small enough that the
          // blocks its interface asks for cover it.
          let all = 0;
          let fine = 0;
          for (let q = 0; q < c.count; q++) {
            const m = X[0][q] * c.u1[4 * q] * c.dx[q] ** 2;
            all += m;
            if (c.dx[q] < 1.5 / n) fine += m;
          }
          finest = fine / all;
          return helium(X, c.u1, (q) => c.x[q], (q) => c.dx[q] ** 2, c.count);
        };
        const before = await read();
        await sim.runToEnd();
        const after = await read();
        metrics.amr_helium_at_finest_level = finest;
        pass &&= finest > 0.95;
        const flux = await sim.fluxMismatch();
        judge('amr', before, after, await sim.diagnostics(), flux.mismatched);
        metrics.amr_regrids = sim.regrids;
        metrics.amr_leaf_blocks = sim.tree.stats().leaves;
        pass &&= sim.regrids >= 1;
      });
      return { pass, metrics };
    },
  },
  {
    id: 'detonation',
    name: 'Detonation at the Chapman-Jouguet speed (2D, 512^2)',
    criterion:
      'The stable detonation of the single-step Arrhenius model (gamma 1.2, q = 50, Ea = 25; 12.8 cells per half-reaction length), started from its exact ZND profile: from t = 1 to t = 3 the front runs within 1% of the Chapman-Jouguet speed 6.809, and the gas 8 to 15 units behind it has burned all but 1e-3 of its fuel. Reported beside it: the speed with burning suppressed in cells flagged as shocks, which falls short of CJ by an error first order in the cell size (about 5% here, half that at twice the resolution)',
    async run(gpu) {
      const n = 512;
      const box = detonationStable.box;
      const run = (shockburn: number) =>
        withSim(gpu, detonationStable, n, { ndim: 2, params: { ea: 25, overdrive: 1, shockburn } }, async (sim) => {
          const at = async (t: number) => {
            sim.cfg.tEnd = t;
            sim.writeParams();
            await sim.runToEnd();
            const st = await sim.readState();
            return detonationRow(st.u1, st.u2, await sim.readSpecies(), n, box);
          };
          const a = await at(1);
          const b = await at(3);
          return { speed: (b.front - a.front) / 2, fuelLeft: b.fuelLeft, spike: b.spike };
        });
      const plain = await run(1);
      const suppressed = await run(0);
      const err = plain.speed / DETONATION_CJ_SPEED - 1;
      return {
        pass: Math.abs(err) < 0.01 && plain.fuelLeft < 1e-3,
        metrics: {
          front_speed: plain.speed, cj_speed: DETONATION_CJ_SPEED, relative_error: err, fuel_left_behind: plain.fuelLeft,
          peak_pressure: plain.spike, znd_spike: zndOf({ ea: 25, overdrive: 1 }).vonNeumann.p,
          speed_no_burning_in_shocks: suppressed.speed, relative_error_no_burning_in_shocks: suppressed.speed / DETONATION_CJ_SPEED - 1,
        },
      };
    },
  },
  {
    id: 'advection-accuracy',
    name: 'Smooth advection: error and order of convergence (2D)',
    criterion:
      'A Gaussian carried once around the box diagonally: mean density error at 256^2 below 2e-4 and largest error below 5e-3 (the peak is not flattened), converging at second order or better from 128^2',
    async run(gpu) {
      const err: Record<number, { l1: number; max: number; peak: number }> = {};
      for (const n of [128, 256]) {
        await withSim(gpu, advection, n, { ndim: 2, tEnd: 2 }, async (sim) => {
          const before = (await sim.readState()).u1.slice();
          await sim.runToEnd(64);
          const after = (await sim.readState()).u1;
          let l1 = 0;
          let max = 0;
          let peak = 0;
          for (let i = 0; i < n * n; i++) {
            const e = Math.abs(after[4 * i] - before[4 * i]);
            l1 += e;
            max = Math.max(max, e);
            peak = Math.max(peak, after[4 * i]);
          }
          err[n] = { l1: l1 / (n * n), max, peak: peak - 1.5 };
        });
      }
      const order = Math.log2(err[128].l1 / err[256].l1);
      return {
        pass: err[256].l1 < 2e-4 && err[256].max < 5e-3 && order > 1.9,
        metrics: {
          l1_128: err[128].l1, l1_256: err[256].l1, max_error_128: err[128].max, max_error_256: err[256].max,
          peak_height_128: err[128].peak, peak_height_256: err[256].peak, order,
        },
      };
    },
  },
  {
    id: 'flux-determinism',
    name: 'Flux determinism',
    criterion: 'Both cells sharing a face compute bit-identical mass fluxes, on every face, for three flows',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let bad = 0;
      for (const prob of [advection, sedov, kelvinHelmholtz]) {
        await withSim(gpu, prob, 32, { tEnd: Infinity }, async (sim) => {
          sim.step(30);
          const r = await sim.fluxMismatch();
          metrics[`${prob.id}_faces`] = r.faces;
          metrics[`${prob.id}_mismatched`] = r.mismatched;
          bad += r.mismatched;
        });
      }
      return { pass: bad === 0, metrics };
    },
  },
  {
    id: 'reflecting-box',
    name: 'Blast in a closed reflecting box',
    criterion: 'Mass conserved to 1e-5 and total energy to 1e-3 after the shock has reflected off the walls',
    async run(gpu) {
      return withSim(gpu, sedov, 64, { bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.5 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const dm = Math.abs(d1.mass - d0.mass) / d0.mass;
        const de = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        const mom = Math.max(...d1.momentum.map(Math.abs));
        return { pass: dm < 1e-5 && de < 1e-3, metrics: { mass_drift: dm, energy_drift: de, net_momentum: mom, steps } };
      });
    },
  },
  {
    id: 'two-dimensional',
    name: '2D grids (one periodic layer in z)',
    criterion:
      'Sod along x at 256^2 has L1 density error < 0.003 and is planar; 2D advection conserves mass and momentum to 2e-6; fluxes are bit-identical on every face',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      const n = 256;
      const sod = await withSim(gpu, sodX, n, { ndim: 2 }, async (sim) => {
        const steps = await sim.runToEnd();
        const { u1 } = await sim.readState();
        const exact = exactRiemann(sodX.left, sodX.right, 1.4);
        let err = 0;
        let spread = 0;
        for (let i = 0; i < n; i++) {
          const ref = u1[i * 4];
          err += Math.abs(ref - exact(((i + 0.5) / n - 0.5) / sodX.tEnd).rho);
          for (let j = 0; j < n; j++) spread = Math.max(spread, Math.abs(u1[(j * n + i) * 4] - ref));
        }
        return { l1: err / n, spread, steps };
      });
      metrics.sod_l1 = sod.l1;
      metrics.sod_transverse_spread = sod.spread;
      metrics.sod_steps = sod.steps;
      const adv = await withSim(gpu, advection, 256, { ndim: 2, tEnd: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        sim.step(30);
        const mm = await sim.fluxMismatch();
        const steps = await sim.runToEnd();
        const d1 = await sim.diagnostics();
        return {
          mass: Math.abs(d1.mass - d0.mass) / d0.mass,
          mom: Math.max(Math.abs(d1.momentum[0] - d0.momentum[0]) / d0.momentum[0], Math.abs(d1.momentum[1] - d0.momentum[1]) / d0.momentum[1]),
          mismatched: mm.mismatched,
          steps,
        };
      });
      metrics.advection_mass_drift = adv.mass;
      metrics.advection_momentum_drift = adv.mom;
      metrics.flux_mismatched_faces = adv.mismatched;
      metrics.advection_steps = adv.steps;
      return { pass: sod.l1 < 0.003 && sod.spread < 1e-5 && adv.mass < 2e-6 && adv.mom < 2e-6 && adv.mismatched === 0, metrics };
    },
  },
  {
    id: 'rayleigh-taylor',
    name: 'Rayleigh-Taylor in a uniform field (2D)',
    criterion:
      'Unperturbed, the stratified atmosphere stays at rest (max Mach < 1e-3 at t = 2); perturbed, the mode grows and total energy, including potential, is conserved to 1e-3 by t = 3',
    async run(gpu) {
      const n = 256;
      const still: Problem = {
        ...rayleighTaylor,
        init: (m, e, ndim) => {
          const f = rayleighTaylor.init(m, e, ndim, {});
          return (x, y, z) => ({ ...f(x, y, z), vx: 0, vy: 0, vz: 0 });
        },
      };
      const rest = await withSim(gpu, still, n, { ndim: 2, tEnd: 2 }, async (sim) => {
        await sim.runToEnd();
        return sim.diagnostics();
      });
      const run = await withSim(gpu, rayleighTaylor, n, { ndim: 2, tEnd: 3 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd();
        const d1 = await sim.diagnostics();
        return { d0, d1, steps };
      });
      const drift = Math.abs(run.d1.eTot - run.d0.eTot) / Math.abs(run.d0.eTot);
      const growth = run.d1.eKin / run.d0.eKin;
      return {
        pass: rest.maxMach < 1e-3 && growth > 10 && drift < 1e-3,
        metrics: {
          hydrostatic_max_mach: rest.maxMach, hydrostatic_ke: rest.eKin, kinetic_energy_growth: growth,
          energy_drift: drift, mass_drift: Math.abs(run.d1.mass - run.d0.mass) / run.d0.mass,
          linear_growth_rate_theory: Math.sqrt((RT.g * 4 * Math.PI) / 3), steps: run.steps,
        },
      };
    },
  },
  {
    id: 'amr-unrefined',
    name: 'Block grid without refinement against the uniform grid',
    criterion:
      'Density in every cell within 1e-5 of the uniform grid\'s, relative, and the same number of steps, for a 2D shock tube and a 3D blast. (Not to the bit: the two grids run different shaders, and WGSL lets a compiler fuse and reassociate floating-point arithmetic differently in each; on an Apple M4 Pro in Chrome the 3D blast differs by a few float32 units.)',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [prob, ndim, n] of [[sodX, 2, 128], [sedov, 3, 64]] as const) {
        const tEnd = prob === sedov ? 0.02 : prob.tEnd;
        const uni = await withSim(gpu, prob, n, { ndim, tEnd }, async (sim) => ({ steps: await sim.runToEnd(), u1: (await sim.readState()).u1 }));
        const blk = await withAmr(gpu, prob, n, { ndim, tEnd, amrLevels: 0 }, async (sim) => ({ steps: await sim.runToEnd(), cells: await sim.readLeaves() }));
        let worst = 0;
        let rel = 0;
        for (let q = 0; q < blk.cells.count; q++) {
          const u = uni.u1[cellIndex(blk.cells, q, n, prob.box, ndim) * 4];
          const d = Math.abs(blk.cells.u1[q * 4] - u);
          worst = Math.max(worst, d);
          rel = Math.max(rel, d / Math.abs(u));
        }
        metrics[`${prob.id}_${ndim}d_max_density_difference`] = worst;
        metrics[`${prob.id}_${ndim}d_max_relative_difference`] = rel;
        metrics[`${prob.id}_${ndim}d_steps`] = blk.steps;
        pass &&= rel < 1e-5 && blk.steps === uni.steps && blk.cells.count === n ** ndim;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'amr-shock-tube',
    name: 'Adaptive mesh: Sod shock tube (2D, 256 finest, 2 levels)',
    criterion:
      'L1 density error < 0.003 (the uniform 256 grid gives 0.002) on under 60% of the cells; mass and energy drift < 1e-6; fluxes bit-identical between same-level blocks and equal to 1e-6 across refinement boundaries; the mesh was rebuilt at least once',
    async run(gpu) {
      return withAmr(gpu, sodX, 256, { ndim: 2, amrLevels: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const c = await sim.readLeaves();
        const exact = exactRiemann(sodX.left, sodX.right, 1.4);
        let err = 0;
        let area = 0;
        for (let q = 0; q < c.count; q++) {
          const a = c.dx[q] ** 2;
          err += Math.abs(c.u1[q * 4] - exact(c.x[q] / sodX.tEnd).rho) * a;
          area += a;
        }
        const l1 = err / area;
        const flux = await sim.fluxMismatch();
        const frac = c.count / 256 ** 2;
        const dm = Math.abs(d1.mass - d0.mass) / d0.mass;
        const de = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        return {
          pass: l1 < 0.003 && frac < 0.6 && dm < 1e-6 && de < 1e-6 && flux.mismatched === 0 && flux.worstCoarseFine < 1e-6 && sim.regrids >= 1,
          metrics: {
            l1, fraction_of_uniform_cells: frac, leaf_blocks: sim.tree.stats().leaves, regrids: sim.regrids, mass_drift: dm, energy_drift: de,
            same_level_faces: flux.faces, same_level_mismatched: flux.mismatched, coarse_fine_faces: flux.coarseFine, coarse_fine_worst: flux.worstCoarseFine, steps,
          },
        };
      });
    },
  },
  {
    id: 'amr-species',
    name: 'Adaptive mesh: species carried with the gas (2D, 256 finest, 2 levels)',
    criterion:
      'The species test on a mesh whose finest level follows the density bump: mass fractions stay in [0, 1] and sum to 1 within 1e-5, the mass of each species is kept within 1e-5 of the total, one at 0.2 everywhere stays 0.2 within 1e-5, and the mesh was rebuilt; a smooth species ends within 1.3e-3 (mean) of the passive scalar carrying the same profile, and within 1.5e-3 beside the sharp-edged one (ghosts next to coarser blocks interpolated, each species limited on its own)',
    async run(gpu) {
      const run = (sharp: number) =>
        withAmr(gpu, speciesCarry(sharp), 256, { ndim: 2, amrLevels: 2, refineDense: 1.2 }, async (sim) => {
          const mass = async () => {
            const c = await sim.readLeaves();
            const X = await sim.readLeafSpecies();
            return { c, X, m: X.map((x) => x.reduce((s, v, q) => s + v * c.u1[4 * q] * c.dx[q] ** 2, 0)) };
          };
          const before = await mass();
          await sim.runToEnd();
          const { c, X, m } = await mass();
          let sumErr = 0;
          let lo = Infinity;
          let hi = -Infinity;
          let flatErr = 0;
          let vsScalar = 0;
          let area = 0;
          for (let q = 0; q < c.count; q++) {
            let s = 0;
            for (const x of X) {
              s += x[q];
              lo = Math.min(lo, x[q]);
              hi = Math.max(hi, x[q]);
            }
            sumErr = Math.max(sumErr, Math.abs(s - 1));
            flatErr = Math.max(flatErr, Math.abs(X[2][q] - 0.2));
            const a = c.dx[q] ** 2;
            vsScalar += Math.abs(X[0][q] - c.u2[4 * q + 2] / c.u1[4 * q]) * a;
            area += a;
          }
          const total = before.m.reduce((a, b) => a + b, 0);
          const massErr = Math.max(...before.m.map((v, i) => Math.abs(m[i] - v))) / total;
          return { sumErr, lo, hi, flatErr, massErr, vsScalar: vsScalar / area, regrids: sim.regrids, leaves: sim.tree.stats().leaves };
        });
      const mixed = await run(0.3);
      const plain = await run(0);
      const ok = (r: typeof mixed) => r.sumErr < 1e-5 && r.lo >= 0 && r.hi <= 1 + 1e-6 && r.massErr < 1e-5 && r.flatErr < 1e-5 && r.regrids >= 1;
      return {
        pass: ok(mixed) && ok(plain) && plain.vsScalar < 1.3e-3 && mixed.vsScalar < 1.5e-3,
        metrics: {
          sum_error: Math.max(mixed.sumErr, plain.sumErr), min_fraction: Math.min(mixed.lo, plain.lo), max_fraction: Math.max(mixed.hi, plain.hi),
          species_mass_drift: Math.max(mixed.massErr, plain.massErr), constant_species_error: Math.max(mixed.flatErr, plain.flatErr),
          smooth_species_vs_passive_scalar: plain.vsScalar, beside_sharp_one: mixed.vsScalar, regrids: mixed.regrids, leaf_blocks: mixed.leaves,
        },
      };
    },
  },
  {
    id: 'amr-species-eos',
    name: 'Adaptive mesh: species that set the equation of state (2D, 256 finest, 2 levels)',
    criterion:
      'The two-gas test of species-eos on a mesh whose finest level holds the helium disc, under gas plus radiation: at rest it stays at rest (speeds under 1e-5) with p = 1 to 1e-5, and the GPU temperature range matches the float64 equation of state to 1e-5; carried once across the box (the mesh rebuilt as the disc moves) the pressure stays within 0.05 of 1 and the faces match bit for bit, also across refinement boundaries',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const c = MIXED_EOS[1];
      for (const moving of [false, true]) {
        const r = await withAmr(gpu, twoGases(c.kind, c.params, moving, moving ? 1 : 0.25), 256, { ndim: 2, amrLevels: 2, refineDense: 1.5 }, async (sim) => {
          await sim.runToEnd();
          const cells = await sim.readLeaves();
          const X = await sim.readLeafSpecies();
          const e = twoGasErrors(sim.cfg, { ...cells, X }, moving ? [1, 0.5] : [0, 0], await sim.diagnostics());
          const flux = await sim.fluxMismatch();
          return { ...e, flux, regrids: sim.regrids, leaves: sim.tree.stats().leaves };
        });
        const tag = moving ? 'moving' : 'static';
        metrics[`${tag}_speed`] = r.dv;
        metrics[`${tag}_pressure_error`] = r.dp;
        metrics[`${tag}_gpu_temperature_error`] = r.tErr;
        metrics[`${tag}_mismatched_faces`] = r.flux.mismatched;
        metrics[`${tag}_coarse_fine_worst`] = r.flux.worstCoarseFine;
        metrics[`${tag}_leaf_blocks`] = r.leaves;
        if (moving) metrics.regrids = r.regrids;
        pass &&= r.flux.mismatched === 0;
        if (moving) pass &&= r.dp < 0.05 && r.regrids >= 1;
        else pass &&= r.dv < 1e-5 && r.dp < 1e-5 && r.gpuDp < 1e-5 && r.tErr < 1e-5;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'amr-detonation',
    name: 'Adaptive mesh: detonation at the Chapman-Jouguet speed (2D, 512 finest, 2 levels)',
    criterion:
      'The detonation test on a mesh three-quarters coarser away from the front: from t = 1 to t = 3 the front runs within 1% of the Chapman-Jouguet speed 6.809, the gas 8 to 15 units behind it has burned all but 1e-3 of its fuel, the whole reaction zone (1% to 99% burned) is at the finest level, and the mesh was rebuilt as the front moved',
    async run(gpu) {
      return withAmr(gpu, detonationStable, 512, { ndim: 2, amrLevels: 2, params: { ea: 25, overdrive: 1, shockburn: 1 } }, async (sim) => {
        const at = async (t: number) => {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd();
          return detonationLeaves(await sim.readLeaves(), await sim.readLeafSpecies());
        };
        const a = await at(1);
        const b = await at(3);
        const speed = (b.front - a.front) / 2;
        const err = speed / DETONATION_CJ_SPEED - 1;
        return {
          pass: Math.abs(err) < 0.01 && b.fuelLeft < 1e-3 && sim.regrids >= 1 && a.zoneCoarse === 0 && b.zoneCoarse === 0,
          metrics: {
            front_speed: speed, cj_speed: DETONATION_CJ_SPEED, relative_error: err, fuel_left_behind: b.fuelLeft, regrids: sim.regrids, leaf_blocks: sim.tree.stats().leaves,
            reaction_zone_coarse_t1: a.zoneCoarse, reaction_zone_coarse_t3: b.zoneCoarse,
          },
        };
      });
    },
  },
  {
    id: 'amr-sedov',
    name: 'Adaptive mesh: Sedov-Taylor blast (3D, 128 finest, 2 levels)',
    criterion: 'Shell-averaged shock radius within 2% of the similarity solution; mass and energy conserved to 1e-6; fluxes consistent across levels',
    async run(gpu) {
      return withAmr(gpu, sedov, 128, { amrLevels: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const c = await sim.readLeaves();
        const r = leafShockRadius(c);
        const rExact = SEDOV_XI0['1.4'] * Math.pow(sedov.tEnd ** 2, 0.2);
        const err = Math.abs(r - rExact) / rExact;
        const flux = await sim.fluxMismatch();
        const dm = Math.abs(d1.mass - d0.mass) / d0.mass;
        const de = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        return {
          pass: err < 0.02 && dm < 1e-6 && de < 1e-6 && flux.mismatched === 0 && flux.worstCoarseFine < 1e-6,
          metrics: {
            shock_radius: r, similarity_radius: rExact, radius_error: err, fraction_of_uniform_cells: c.count / 128 ** 3,
            leaf_blocks: sim.tree.stats().leaves, regrids: sim.regrids, mass_drift: dm, energy_drift: de,
            same_level_mismatched: flux.mismatched, coarse_fine_faces: flux.coarseFine, coarse_fine_worst: flux.worstCoarseFine, steps,
          },
        };
      });
    },
  },
  {
    id: 'amr-instabilities',
    name: 'Adaptive mesh: Kelvin-Helmholtz and Rayleigh-Taylor (2D, 512 finest, 2 levels)',
    criterion:
      'The mesh follows the flow (rebuilt at least 5 times) while mass is conserved to 1e-6 and total energy to 1e-5; the Rayleigh-Taylor mode grows; fluxes consistent across levels',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [prob, tEnd] of [[kelvinHelmholtz, 1], [rayleighTaylor, 3]] as const) {
        await withAmr(gpu, prob, 512, { ndim: 2, amrLevels: 2, tEnd }, async (sim) => {
          const d0 = await sim.diagnostics();
          const steps = await sim.runToEnd();
          const d1 = await sim.diagnostics();
          const flux = await sim.fluxMismatch();
          const dm = Math.abs(d1.mass - d0.mass) / d0.mass;
          const de = Math.abs(d1.eTot - d0.eTot) / Math.abs(d0.eTot);
          const st = sim.tree.stats();
          metrics[`${prob.id}_mass_drift`] = dm;
          metrics[`${prob.id}_energy_drift`] = de;
          metrics[`${prob.id}_regrids`] = sim.regrids;
          metrics[`${prob.id}_fraction_of_uniform_cells`] = st.cells / 512 ** 2;
          metrics[`${prob.id}_kinetic_energy_growth`] = d1.eKin / d0.eKin;
          metrics[`${prob.id}_coarse_fine_worst`] = flux.worstCoarseFine;
          metrics[`${prob.id}_steps`] = steps;
          pass &&= dm < 1e-6 && de < 1e-5 && sim.regrids >= 5 && flux.mismatched === 0 && flux.worstCoarseFine < 1e-6;
          if (prob === rayleighTaylor) pass &&= d1.eKin / d0.eKin > 10;
        });
      }
      return { pass, metrics };
    },
  },
  {
    id: 'amr-gravity-static',
    name: 'Adaptive mesh: Poisson solve and its coupling',
    criterion:
      'Potential of a uniform sphere within 1% of the analytic solution everywhere, on meshes with 0, 2 and 3 levels of refinement; Jeans growth rate on the block grid within 5% of linear theory',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [n, levels] of [[64, 0], [128, 2], [256, 3]] as const) {
        await withAmr(gpu, sphere, n, { amrLevels: levels, tEnd: Infinity }, async (sim) => {
          const c = await sim.readLeaves();
          const phi = await sim.readLeafPotential();
          let worst = 0;
          for (let q = 0; q < c.count; q++) {
            const e = spherePotential(Math.hypot(c.x[q], c.y[q], c.z[q]));
            worst = Math.max(worst, Math.abs(phi[q] - e) / Math.abs(e));
          }
          const st = sim.tree.stats();
          metrics[`sphere_${n}_L${levels}_max_error`] = worst;
          metrics[`sphere_${n}_L${levels}_leaf_blocks`] = st.leaves;
          pass &&= worst < 0.01 && st.deepest === levels;
        });
      }
      await withAmr(gpu, jeans, 64, { amrLevels: 1, refineHi: 1e-6, refineLo: 1e-7 }, async (sim) => {
        const steps = await sim.runToEnd();
        const c = await sim.readLeaves();
        let a = 0;
        let v = 0;
        for (let q = 0; q < c.count; q++) {
          const w = c.dx[q] ** 3;
          a += (c.u1[q * 4] - 1) * Math.cos(JEANS.k * c.x[q]) * w;
          v += w;
        }
        const rate = Math.acosh((2 * a) / v / JEANS.amp) / jeans.tEnd;
        const err = Math.abs(rate - JEANS.growth) / JEANS.growth;
        metrics.jeans_growth_rate = rate;
        metrics.jeans_error = err;
        metrics.jeans_steps = steps;
        pass &&= err < 0.05;
      });
      return { pass, metrics };
    },
  },
  {
    id: 'amr-gravity-dynamic',
    name: 'Adaptive mesh: Evrard collapse and a hydrostatic polytrope (128 finest, 2 levels)',
    criterion:
      'Evrard: total energy conserved to 2% through collapse and bounce, thermal energy peaking between t = 0.9 and 1.3. Polytrope: central density within 10% of its initial value over 5 dynamical times',
    async run(gpu) {
      const metrics: Record<string, number | string> = {};
      let pass = true;
      await withAmr(gpu, evrard, 128, { amrLevels: 2, tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        let worst = 0;
        let peakT = 0;
        let peakU = 0;
        let last = d0;
        let steps = 0;
        for (let i = 1; i <= 12; i++) {
          sim.cfg.tEnd = 0.125 * i;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot));
          if (last.eInt > peakU) [peakU, peakT] = [last.eInt, last.t];
        }
        const st = sim.tree.stats();
        Object.assign(metrics, {
          evrard_max_energy_drift: worst, evrard_thermal_peak_time: peakT, evrard_thermal_peak: peakU,
          evrard_fraction_of_uniform_cells: st.cells / 128 ** 3, evrard_regrids: sim.regrids, evrard_steps: steps,
        });
        pass &&= worst < 0.02 && peakT >= 0.9 && peakT <= 1.3;
      });
      await withAmr(gpu, polytrope, 128, { amrLevels: 2, tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const rho0 = d0.ranges[0][1];
        let worst = 0;
        let last = d0;
        let steps = 0;
        for (let t = 1; t <= 5; t++) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.ranges[0][1] - rho0) / rho0);
        }
        Object.assign(metrics, {
          polytrope_central_density_max_deviation: worst, polytrope_energy_drift: Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot),
          polytrope_fraction_of_uniform_cells: sim.tree.stats().cells / 128 ** 3, polytrope_steps: steps,
        });
        pass &&= worst < 0.1;
      });
      return { pass, metrics };
    },
  },
  {
    id: 'amr-tidal-disruption',
    name: 'Adaptive mesh: tidal disruption (128 finest, 2 levels)',
    criterion:
      'The perturber reaches pericenter within 2% of r_t / beta; while all the gas is in the box its center of mass stays within 0.005 stellar radii of the box center; the star is disrupted',
    async run(gpu) {
      const orbit = tdeOrbit({ q: 100, beta: 1 });
      return withAmr(gpu, tdeTest, 128, { amrLevels: 2, tEnd: 0, params: { q: 100, beta: 1 } }, async (sim) => {
        const d0 = await sim.diagnostics();
        let rMin = Infinity;
        let comMax = 0;
        let last = d0;
        let steps = 0;
        for (let t = 0.25; t <= orbit.tPeri + 3 + 1e-9; t += 0.25) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          rMin = Math.min(rMin, Math.hypot(...last.perturber!));
          if (last.mass >= 0.999 * d0.mass) comMax = Math.max(comMax, Math.hypot(...last.com));
        }
        // The orbit is sampled every 0.25, so the closest sample can sit a little outside pericenter.
        const rErr = (rMin - orbit.rp) / orbit.rp;
        const drop = last.ranges[0][1] / d0.ranges[0][1];
        const st = sim.tree.stats();
        return {
          pass: rErr > -0.02 && rErr < 0.03 && comMax < 0.005 && drop < 0.5,
          metrics: {
            closest_sampled_separation: rMin, pericenter_expected: orbit.rp, max_com_offset_while_contained: comMax,
            peak_density_final_ratio: drop, leaf_blocks: st.leaves, fraction_of_uniform_cells: st.cells / 128 ** 3, regrids: sim.regrids, steps,
          },
        };
      });
    },
  },
  {
    id: 'amr-time-bins',
    name: 'Adaptive mesh: per-block timesteps (Sod 2D, Sedov 3D, Evrard collapse)',
    criterion:
      'With blocks stepping up to 8 times less often than the fastest: Sod L1 density error < 0.003 and Sedov shock radius within 2%, as with one timestep; mass and energy drift < 1e-6 in both; Sedov makes under 85% of the block updates (a small box: the blast fills much of it); Evrard collapse to t = 0.8 (64 steps) loses no more than 3 times the energy it does with one timestep, both at two V-cycles a step (about 2 here, 1.1 at twice the resolution: the half-step potential of a block that steps less often is extrapolated further)',
    async run(gpu) {
      const bins = { timeBins: 3, timeBinsAlways: true };
      const drift = (a: { mass: number; eTot: number }, b: { mass: number; eTot: number }) => [Math.abs(b.mass - a.mass) / a.mass, Math.abs(b.eTot - a.eTot) / Math.abs(a.eTot)];
      const sod = await withAmr(gpu, sodX, 256, { ndim: 2, amrLevels: 2, ...bins }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const c = await sim.readLeaves();
        const exact = exactRiemann(sodX.left, sodX.right, 1.4);
        let err = 0;
        let area = 0;
        for (let q = 0; q < c.count; q++) {
          const a = c.dx[q] ** 2;
          err += Math.abs(c.u1[q * 4] - exact(c.x[q] / sodX.tEnd).rho) * a;
          area += a;
        }
        return { l1: err / area, drift: drift(d0, d1) };
      });
      const blast = await withAmr(gpu, sedov, 128, { amrLevels: 2, ...bins }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const r = leafShockRadius(await sim.readLeaves());
        const rExact = SEDOV_XI0['1.4'] * Math.pow(sedov.tEnd ** 2, 0.2);
        const work = await sim.binWork();
        return { err: Math.abs(r - rExact) / rExact, drift: drift(d0, d1), share: work.updates / work.single };
      });
      const collapse = async (over: Partial<SimConfig>) =>
        withAmr(gpu, evrard, 128, { amrLevels: 2, tEnd: 0.8, vcycles: 2, ...over }, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd();
          return drift(d0, await sim.diagnostics())[1];
        });
      const eOne = await collapse({});
      const eBins = await collapse(bins);
      return {
        pass:
          sod.l1 < 0.003 && Math.max(...sod.drift) < 1e-6 && blast.err < 0.02 && Math.max(...blast.drift) < 1e-6 && blast.share < 0.85 && eBins < 3 * eOne,
        metrics: {
          sod_l1: sod.l1, sod_mass_drift: sod.drift[0], sod_energy_drift: sod.drift[1],
          sedov_radius_error: blast.err, sedov_mass_drift: blast.drift[0], sedov_energy_drift: blast.drift[1], sedov_share_of_block_updates: blast.share,
          evrard_energy_drift_one_timestep: eOne, evrard_energy_drift_per_block: eBins,
        },
      };
    },
  },
  {
    id: 'snapshot-roundtrip',
    name: 'Export to HDF5 and import again',
    criterion:
      'A run saved to an .athdf file and started again from it has the same time, the same mesh, and a state equal to round-off (1e-5 relative); continued for the same number of steps, the two runs still agree to 1e-4. Checked on a uniform grid, an adaptive mesh, and a run with a perturber; with species, on both grids, every mass fraction comes back to 1e-6',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const worst = (a: Float32Array, b: Float32Array) => {
        let scale = 0;
        let diff = 0;
        for (let i = 0; i < a.length; i += 4) {
          scale = Math.max(scale, Math.abs(a[i]));
          diff = Math.max(diff, Math.abs(a[i] - b[i]));
        }
        return diff / scale;
      };
      const roundtrip = async (s: Sim | AmrSim) => {
        const bytes = await writeAthdf(await capture(s, { program: 'shaderous', build: 'test', query: '' }));
        return { bytes: bytes.length, snap: await readAthdf(bytes) };
      };

      // Uniform grid.
      await withSim(gpu, sedov, 64, { tEnd: 0.02 }, async (a) => {
        await a.runToEnd();
        const { bytes, snap } = await roundtrip(a);
        const cfg = { ...configFor(sedov, 64, { tEnd: 0.03 }), ...restartFrom(snap, sedov, false, 64) };
        const b = await Sim.create(gpu, cfg);
        try {
          const [da, db] = [await a.diagnostics(), await b.diagnostics()];
          const state = worst((await a.readState()).u1, (await b.readState()).u1);
          a.cfg.tEnd = 0.03;
          a.writeParams();
          const [sa, sb] = [await a.runToEnd(), await b.runToEnd()];
          const later = worst((await a.readState()).u1, (await b.readState()).u1);
          Object.assign(metrics, { uniform_file_bytes: bytes, uniform_time_difference: Math.abs(da.t - db.t), uniform_state_difference: state, uniform_difference_after_continuing: later, uniform_steps_continued: sb });
          pass &&= da.t === db.t && db.steps === da.steps && state < 1e-5 && later < 1e-4 && sa === sb;
        } finally {
          b.destroy();
        }
      });

      // Adaptive mesh: the same blocks at the same levels, and the same cells in them.
      await withAmr(gpu, sedov, 128, { amrLevels: 2, tEnd: 0.02 }, async (a) => {
        await a.runToEnd();
        a.flush();
        const { bytes, snap } = await roundtrip(a);
        const cfg = { ...configFor(sedov, 128, { amrLevels: 2, tEnd: 0.03 }), ...restartFrom(snap, sedov, true, 32) };
        const b = await AmrSim.create(gpu, cfg);
        try {
          const key = (s: AmrSim) => s.tree.leaves().map((l) => `${l.level}:${l.c.join(',')}`).sort().join(' ');
          const sameMesh = key(a) === key(b);
          const [ca, cb] = [await a.readLeaves(), await b.readLeaves()];
          // Leaves may be stored in a different order; compare cell by cell through positions.
          const at = new Map<string, number>();
          for (let q = 0; q < ca.count; q++) at.set(`${ca.x[q]},${ca.y[q]},${ca.z[q]}`, ca.u1[4 * q]);
          let scale = 0;
          let diff = 0;
          let missing = 0;
          for (let q = 0; q < cb.count; q++) {
            const v = at.get(`${cb.x[q]},${cb.y[q]},${cb.z[q]}`);
            if (v === undefined) missing++;
            else {
              scale = Math.max(scale, Math.abs(v));
              diff = Math.max(diff, Math.abs(v - cb.u1[4 * q]));
            }
          }
          const [da, db] = [await a.diagnostics(), await b.diagnostics()];
          Object.assign(metrics, {
            adaptive_file_bytes: bytes, adaptive_leaf_blocks: a.tree.stats().leaves, adaptive_same_mesh: sameMesh ? 1 : 0,
            adaptive_cells_without_a_match: missing, adaptive_state_difference: diff / scale, adaptive_mass_difference: Math.abs(da.mass - db.mass) / da.mass,
          });
          pass &&= sameMesh && missing === 0 && cb.count === ca.count && diff / scale < 1e-5 && da.t === db.t;
        } finally {
          b.destroy();
        }
      });

      // Species, on both grids: every cell's mass fractions come back.
      const carried = speciesCarry(0.3, 0.2);
      await withSim(gpu, carried, 64, { ndim: 2 }, async (a) => {
        await a.runToEnd();
        const { snap } = await roundtrip(a);
        const b = await Sim.create(gpu, { ...configFor(carried, 64, { ndim: 2 }), ...restartFrom(snap, carried, false, 64) });
        try {
          const [xa, xb] = [await a.readSpecies(), await b.readSpecies()];
          metrics.uniform_species_difference = Math.max(...xa.map((x, i) => x.reduce((m, v, q) => Math.max(m, Math.abs(v - xb[i][q])), 0)));
          pass &&= xa.length === 5 && metrics.uniform_species_difference < 1e-6;
        } finally {
          b.destroy();
        }
      });
      await withAmr(gpu, carried, 128, { ndim: 2, amrLevels: 2, refineDense: 1.2 }, async (a) => {
        await a.runToEnd();
        a.flush();
        const { snap } = await roundtrip(a);
        const b = await AmrSim.create(gpu, { ...configFor(carried, 128, { ndim: 2, amrLevels: 2, refineDense: 1.2 }), ...restartFrom(snap, carried, true, 32) });
        try {
          const species = async (s: AmrSim) => {
            const c = await s.readLeaves();
            const X = await s.readLeafSpecies();
            return new Map(Array.from({ length: c.count }, (_, q) => [`${c.x[q]},${c.y[q]}`, X.map((x) => x[q])]));
          };
          const [ma, mb] = [await species(a), await species(b)];
          let diff = 0;
          let missing = 0;
          for (const [k, xa] of ma) {
            const xb = mb.get(k);
            if (!xb) missing++;
            else diff = Math.max(diff, ...xa.map((v, i) => Math.abs(v - xb[i])));
          }
          Object.assign(metrics, { adaptive_species_difference: diff, adaptive_species_cells_without_a_match: missing });
          pass &&= missing === 0 && ma.size === mb.size && diff < 1e-6;
        } finally {
          b.destroy();
        }
      });

      // A run with a perturber: its position and velocity come back too.
      await withSim(gpu, tdeTest, 32, { tEnd: 0.5, params: { q: 100, beta: 1 } }, async (a) => {
        await a.runToEnd();
        const before = await a.readRestart();
        const { snap } = await roundtrip(a);
        const cfg = { ...configFor(tdeTest, 32, { tEnd: 1, params: { q: 100, beta: 1 } }), ...restartFrom(snap, tdeTest, false, 32) };
        const b = await Sim.create(gpu, cfg);
        try {
          const after = await b.readRestart();
          const d = Math.max(...[0, 1, 2].map((i) => Math.max(Math.abs(before.masses![0].pos[i] - after.masses![0].pos[i]), Math.abs(before.masses![0].vel[i] - after.masses![0].vel[i]))));
          metrics.perturber_difference = d;
          pass &&= d === 0 && after.time === before.time && after.masses![0].gm === before.masses![0].gm && after.masses![0].accreted === before.masses![0].accreted;
        } finally {
          b.destroy();
        }
      });
      return { pass, metrics };
    },
  },
  {
    id: 'cooling',
    name: 'Optically thin cooling: power-law decay and thermal balance',
    criterion:
      'Uniform gas with Lambda ~ T^(1/2) follows T^(1/2) = 1 - t / 3 to 0.1% (uniform and adaptive grids); with heating and a cooling time 100 times shorter than the timestep it settles at the balance temperature to 1%, without overshoot',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      // T = p / rho, e = rho T / (gamma - 1): dT/dt = -(gamma - 1) rho Lambda0 T^alpha.
      const still = stillGas({ cooling: 'power', coolParams: { rate: 1, alpha: 0.5, heat: 0, tFloor: 0 } });
      const expected = 1.5 * (1 - 1.2 / 3) ** 2;
      await withSim(gpu, still, 32, { ndim: 2, tEnd: 1.2 }, async (sim) => {
        metrics.decay_steps = await sim.runToEnd();
        const d = await sim.diagnostics();
        metrics.decay_error = Math.abs(d.eInt - expected) / expected;
        pass &&= metrics.decay_error < 1e-3;
      });
      await withAmr(gpu, still, 32, { ndim: 2, amrLevels: 1, tEnd: 1.2 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        metrics.decay_error_adaptive = Math.abs(d.eInt - expected) / expected;
        pass &&= metrics.decay_error_adaptive < 1e-3;
      });
      // Stiff: balance at T = 1/4 (heat = rho Lambda0 T^alpha), cooling time about 1e-4.
      const stiff = stillGas({ cooling: 'power', coolParams: { rate: 1e4, alpha: 0.5, heat: 5e3, tFloor: 0 } });
      await withSim(gpu, stiff, 32, { ndim: 2, tEnd: 0 }, async (sim) => {
        let low = Infinity;
        for (const t of [0.01, 0.02, 0.05, 0.2]) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd();
          low = Math.min(low, (await sim.diagnostics()).eInt / 1.5);
        }
        const T = (await sim.diagnostics()).eInt / 1.5;
        metrics.balance_temperature = T;
        metrics.balance_lowest_temperature = low;
        pass &&= Math.abs(T - 0.25) < 0.0025 && low > 0.2475;
      });
      // A temperature floor stops the cooling there.
      const floored = stillGas({ cooling: 'power', coolParams: { rate: 100, alpha: 0.5, heat: 0, tFloor: 0.3 } });
      await withSim(gpu, floored, 32, { ndim: 2, tEnd: 0.5 }, async (sim) => {
        await sim.runToEnd();
        metrics.floor_temperature = (await sim.diagnostics()).eInt / 1.5;
        pass &&= Math.abs(metrics.floor_temperature - 0.3) < 0.003;
      });
      return { pass, metrics };
    },
  },
  {
    id: 'viscosity',
    name: 'Viscosity: shear waves decay at nu k^2, walls hold the gas, energy is conserved',
    criterion:
      'The kinetic energy of a shear wave falls as exp(-2 nu k^2 t) to 1% along an axis and along the diagonal (2D, 3D and on a block grid) and to 3% between two no-slip walls (uniform grid, and refined at the walls); total energy is conserved to 1e-6 throughout (1e-5 where refined); Kelvin-Helmholtz with viscosity and conduction on a mesh that refines and coarsens, with per-block timesteps, conserves mass to 1e-6 and energy to 1e-5',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const nu = 0.01;
      const A = 0.05;
      const k0 = 2 * Math.PI;
      // A wave of wavevector k0 (kx, ky, 0) whose velocity, along `dir`, is perpendicular to it.
      const shear = (kx: number, ky: number, dir: number[]) =>
        stillGas({
          id: 'shear-wave',
          init: () => (x, y) => {
            const s = A * Math.sin(k0 * (kx * x + ky * y));
            return { rho: 1, vx: s * dir[0], vy: s * dir[1], vz: s * dir[2], p: 1, s: 0 };
          },
        });
      const decay = async (tag: string, sim: Sim | AmrSim, k2: number, tol: number, etol = 1e-6) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const expected = Math.exp(-2 * nu * k2 * sim.cfg.tEnd);
        metrics[`${tag}_decay_error`] = Math.abs(d1.eKin / d0.eKin / expected - 1);
        metrics[`${tag}_energy_drift`] = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        pass &&= metrics[`${tag}_decay_error`] < tol && metrics[`${tag}_energy_drift`] < etol;
      };
      const r = Math.SQRT1_2;
      const on = { viscosity: nu };
      await withSim(gpu, shear(1, 0, [0, 1, 0]), 64, { ndim: 2, tEnd: 1.2, ...on }, (sim) => decay('axis', sim, k0 * k0, 0.01));
      await withSim(gpu, shear(1, 1, [-r, r, 0]), 64, { ndim: 2, tEnd: 0.6, ...on }, (sim) => decay('diagonal', sim, 2 * k0 * k0, 0.01));
      await withSim(gpu, shear(1, 1, [0, 0, 1]), 32, { tEnd: 0.6, ...on }, (sim) => decay('out_of_plane_3d', sim, 2 * k0 * k0, 0.01));
      // Blocks of 16 cells, none refined: every block face has the rim rows of amr/update.wgsl.
      await withAmr(gpu, shear(1, 1, [-r, r, 0]), 128, { ndim: 2, amrLevels: 1, tEnd: 0.6, ...on }, (sim) => decay('diagonal_blocks', sim, 2 * k0 * k0, 0.01));
      await withAmr(gpu, shear(1, 1, [-r, r, 0]), 64, { amrLevels: 1, tEnd: 0.3, ...on }, (sim) => decay('diagonal_blocks_3d', sim, 2 * k0 * k0, 0.01));

      // Between two walls half a box apart the mode that vanishes on both has the same k0.
      // (Density and pressure off a power of two: on one, float32 round-off is biased, and
      // over the thousands of steps of the refined run the drift would show.)
      const channel = stillGas({
        id: 'channel',
        init: () => (_x, y) => (Math.abs(y) > 0.25 ? { rho: 1.3, vx: 0, vy: 0, vz: 0, p: 1.3, s: 0, solid: true } : { rho: 1.3, vx: A * Math.cos(k0 * y), vy: 0, vz: 0, p: 1.3, s: 0 }),
      });
      await withSim(gpu, channel, 64, { ndim: 2, tEnd: 1.2, ...on }, (sim) => decay('walls', sim, k0 * k0, 0.03));
      // Blocks an eighth of the box tall: those along the walls are refined, the rest are not.
      await withAmr(gpu, channel, 256, { ndim: 2, amrLevels: 1, tEnd: 0.6, ...on }, async (sim) => {
        await decay('walls_refined', sim, k0 * k0, 0.03, 1e-5);
        metrics.walls_refined_fraction_of_uniform_cells = sim.tree.stats().cells / 256 ** 2;
        pass &&= metrics.walls_refined_fraction_of_uniform_cells < 1;
      });

      // Conservation where fluxes are matched between levels and between timesteps.
      await withAmr(gpu, kelvinHelmholtz, 256, { ndim: 2, amrLevels: 2, tEnd: 1.5, viscosity: 2e-4, conductivity: 2e-4, timeBins: 2, timeBinsAlways: true }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        metrics.refined_mass_drift = Math.abs(d1.mass - d0.mass) / d0.mass;
        metrics.refined_energy_drift = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        metrics.refined_momentum_drift = Math.hypot(d1.momentum[0] - d0.momentum[0], d1.momentum[1] - d0.momentum[1]);
        metrics.refined_regrids = sim.regrids;
        metrics.refined_fraction_of_uniform_cells = sim.tree.stats().cells / 256 ** 2;
        pass &&= metrics.refined_mass_drift < 1e-6 && metrics.refined_energy_drift < 1e-5 && metrics.refined_momentum_drift < 1e-5 && sim.regrids >= 2 && metrics.refined_fraction_of_uniform_cells < 1;
      });
      return { pass, metrics };
    },
  },
  {
    id: 'conduction',
    name: 'Thermal conduction: a temperature wave at constant pressure decays at kappa k^2 (gamma - 1) / gamma',
    criterion:
      'A 1% temperature wave in pressure balance keeps its shape and falls as exp(-kappa k^2 (gamma - 1) t / gamma) to 1% (uniform grid and block grid), with total energy conserved to 1e-6; in the thermal instability at t = 0.6 the scale of the structure changes by less than 12% from 512^2 to 1024^2 cells with the preset\'s conduction, and by more than 15% without',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const kappa = 0.02;
      const gamma = 5 / 3;
      const a = 0.01;
      const k = 2 * Math.PI;
      const rate = (kappa * k * k * (gamma - 1)) / gamma;
      // T = 1 + a cos(k x) at p = 1. As the wave fades the gas flows toward where it was hot;
      // starting with that velocity leaves no sound wave behind.
      const wave = stillGas({
        id: 'temperature-wave', tEnd: 2,
        init: () => (x) => ({ rho: 1 / (1 + a * Math.cos(k * x)), vx: (-rate * a * Math.sin(k * x)) / k, vy: 0, vz: 0, p: 1, s: 0 }),
      });
      // Amplitude of the wave in 1 / rho, which is T at constant pressure.
      const amplitude = (rho: (q: number) => number, x: (q: number) => number, count: number) => {
        let s = 0;
        for (let q = 0; q < count; q++) s += (1 / rho(q) - 1) * Math.cos(k * x(q));
        return (2 * s) / count;
      };
      const check = async (tag: string, sim: Sim | AmrSim, amp: () => Promise<number>) => {
        const d0 = await sim.diagnostics();
        const a0 = await amp();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        metrics[`${tag}_decay_error`] = Math.abs((await amp()) / a0 / Math.exp(-rate * sim.cfg.tEnd) - 1);
        metrics[`${tag}_energy_drift`] = Math.abs(d1.eTot - d0.eTot) / d0.eTot;
        pass &&= metrics[`${tag}_decay_error`] < 0.01 && metrics[`${tag}_energy_drift`] < 1e-6;
      };
      await withSim(gpu, wave, 64, { ndim: 2, conductivity: kappa }, (sim) =>
        check('uniform', sim, async () => {
          const { u1 } = await sim.readState();
          return amplitude((q) => u1[4 * q], (q) => ((q % 64) + 0.5) / 64 - 0.5, 64 * 64);
        }));
      await withAmr(gpu, wave, 128, { ndim: 2, amrLevels: 1, conductivity: kappa }, (sim) =>
        check('blocks', sim, async () => {
          const c = await sim.readLeaves();
          return amplitude((q) => c.u1[4 * q], (q) => c.x[q], c.count);
        }));

      // Thermal instability: conduction, not the grid, sets the size of the clouds. The scale
      // is the wavelength 2 pi sqrt(<d^2> / <|grad d|^2>) of d = ln(rho) less its mean, as a
      // fraction of the box.
      const scale = (n: number, conductivity: number) =>
        withSim(gpu, thermalInstability, n, { ndim: 2, tEnd: 0.6, conductivity }, async (sim) => {
          await sim.runToEnd(64);
          const { u1 } = await sim.readState();
          const ln = new Float64Array(n * n);
          let mean = 0;
          for (let q = 0; q < n * n; q++) mean += ln[q] = Math.log(u1[4 * q]);
          mean /= n * n;
          let s2 = 0;
          let g2 = 0;
          for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
              const d = ln[j * n + i];
              s2 += (d - mean) ** 2;
              g2 += ((ln[j * n + ((i + 1) % n)] - d) ** 2 + (ln[((j + 1) % n) * n + i] - d) ** 2) * n * n;
            }
          }
          return 2 * Math.PI * Math.sqrt(s2 / g2);
        });
      const kappaTi = thermalInstability.conductivity ?? 0;
      const change = async (tag: string, conductivity: number) => {
        const coarse = await scale(512, conductivity);
        const fine = await scale(1024, conductivity);
        metrics[`clouds_${tag}_scale_512`] = coarse;
        metrics[`clouds_${tag}_scale_1024`] = fine;
        return (metrics[`clouds_${tag}_change`] = Math.abs(fine / coarse - 1));
      };
      pass &&= kappaTi > 0 && (await change('conducting', kappaTi)) < 0.12 && (await change('plain', 0)) > 0.15;
      return { pass, metrics };
    },
  },
  {
    id: 'point-mass-sink',
    name: 'Point masses: the sink takes exactly what the gas loses',
    criterion:
      'A softened mass at the center of a closed box of gas accretes; gas mass plus accreted mass stays constant to 1e-5, G M grows by G times the accreted mass, and the mass stays at the center (2D and 3D, uniform and adaptive); the same holds for a mass added in mid-run, and for a sink so slow that each step takes in under half float32 resolution of G M (G M growth within 1e-3 of the accreted mass)',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const box = stillGas({
        bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.5,
        masses: () => [{ gm: 0.05, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.08, sinkRadius: 0.15 }],
      });
      const check = async (tag: string, sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const m = d.masses[0];
        metrics[`${tag}_accreted_fraction`] = m.accreted / d0.mass;
        metrics[`${tag}_mass_error`] = Math.abs(d.mass + m.accreted - d0.mass) / d0.mass;
        metrics[`${tag}_gm_error`] = Math.abs(m.gm - 0.05 - m.accreted) / 0.05;
        metrics[`${tag}_offset`] = Math.hypot(...m.pos);
        pass &&= m.accreted > 0.003 * d0.mass && metrics[`${tag}_mass_error`] < 1e-5 && metrics[`${tag}_gm_error`] < 1e-4 && metrics[`${tag}_offset`] < 1e-3;
      };
      await withSim(gpu, box, 64, { ndim: 2 }, (sim) => check('flat', sim));
      await withSim(gpu, box, 32, {}, (sim) => check('uniform', sim));
      await withAmr(gpu, box, 64, { amrLevels: 1 }, (sim) => check('adaptive', sim));
      // The same mass placed in mid-run (as the sink tool does), then taken away again.
      const empty = stillGas({ bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.1 });
      const placed = async (tag: string, sim: Sim | AmrSim) => {
        await sim.runToEnd();
        const added = sim.addMass({ gm: 0.05, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.08, sinkRadius: 0.15 });
        sim.cfg.tEnd = 0.6;
        sim.writeParams();
        await check(tag, sim);
        sim.removeAddedMasses();
        pass &&= added && (await sim.diagnostics()).masses.length === 0;
      };
      await withSim(gpu, empty, 64, { ndim: 2 }, (sim) => placed('placed_flat', sim));
      await withAmr(gpu, empty, 64, { amrLevels: 1 }, (sim) => placed('placed_adaptive', sim));
      await withAmr(gpu, empty, 64, { ndim: 2, amrLevels: 1 }, (sim) => placed('placed_flat_adaptive', sim));
      await withAmr(gpu, box, 64, { ndim: 2, amrLevels: 1 }, (sim) => check('flat_adaptive', sim));
      // A sink so slow that each step takes far less than float32 resolution of G M (as a
      // binary's sinks do): G M must still grow by G times what was taken in.
      const slow = stillGas({
        bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.5,
        masses: () => [{ gm: 0.05, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.08, sinkRadius: 0.15, sinkRate: 1e-7 }],
      });
      const growth = async (tag: string, sim: Sim | AmrSim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const m = d.masses[0];
        metrics[`${tag}_steps`] = d.steps;
        metrics[`${tag}_accreted_per_step_over_ulp`] = m.accreted / d.steps / (2 ** -24 * 0.05);
        metrics[`${tag}_gm_growth_error`] = Math.abs(m.gm - 0.05 - m.accreted) / m.accreted;
        pass &&= m.accreted > 0 && metrics[`${tag}_accreted_per_step_over_ulp`] < 0.5 && metrics[`${tag}_gm_growth_error`] < 1e-3;
      };
      await withSim(gpu, slow, 64, { ndim: 2 }, (sim) => growth('slow_flat', sim));
      await withAmr(gpu, slow, 64, { ndim: 2, amrLevels: 1 }, (sim) => growth('slow_flat_adaptive', sim));
      return { pass, metrics };
    },
  },
  {
    id: 'binary-orbit',
    name: 'Point masses: a binary in a disk keeps its orbit (256^2)',
    criterion:
      'Over one period of the softened circular orbit the separation stays within 0.5% of 1, the masses return to within 0.03 of where they started, their total momentum stays below 1e-4, and both accrete',
    async run(gpu) {
      return withSim(gpu, binary, 256, { ndim: 2, tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const soft = d0.masses[0].soft;
        const period = 2 * Math.PI * Math.pow(1 + soft * soft, 0.75);
        let worst = 0;
        let last = d0;
        let steps = 0;
        for (let i = 1; i <= 16; i++) {
          sim.cfg.tEnd = (period * i) / 16;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          const [a, b] = last.masses;
          worst = Math.max(worst, Math.abs(Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) - 1));
        }
        const back = Math.max(...[0, 1].map((i) => Math.hypot(last.masses[i].pos[0] - d0.masses[i].pos[0], last.masses[i].pos[1] - d0.masses[i].pos[1])));
        const mom = Math.hypot(...[0, 1].map((a) => last.masses[0].gm * last.masses[0].vel[a] + last.masses[1].gm * last.masses[1].vel[a]));
        const accreted = Math.min(last.masses[0].accreted, last.masses[1].accreted);
        return {
          pass: worst < 0.005 && back < 0.03 && mom < 1e-4 && accreted > 0 && (await adaptiveBinary(gpu, period)) < 0.005,
          metrics: { separation_max_deviation: worst, return_offset: back, binary_momentum: mom, least_accreted: accreted, softening: soft, steps },
        };
      });
    },
  },
  {
    id: 'point-mass-encounter',
    name: 'Point masses: the timestep follows a close encounter',
    criterion:
      'Two masses on an orbit of eccentricity 0.9 in near-empty gas, whose own timestep is several times the pericenter passage: over two periods the mean step of each 96th of a period stays within a sixteenth of the CFL number times the lesser of r / v and sqrt(r^3 / G M) (to 10%), and the energy of the orbit is conserved to 1%',
    async run(gpu) {
      const ecc = 0.9;
      const soft = 0.02;
      const far = 1 + ecc;
      // Apocenter of a Kepler orbit with G M = 1 and semi-major axis 1.
      const v = Math.sqrt((1 - ecc) / (1 + ecc));
      const pair = stillGas({
        id: 'encounter', box: 8, tEnd: 0,
        bcLo: all(1), bcHi: all(1),
        init: () => () => ({ rho: 1e-6, vx: 0, vy: 0, vz: 0, p: 1e-6, s: 0 }),
        masses: () => [
          { gm: 0.5, pos: [-far / 2, 0, 0], vel: [0, -v / 2, 0], soft, sinkRadius: 0 },
          { gm: 0.5, pos: [far / 2, 0, 0], vel: [0, v / 2, 0], soft, sinkRadius: 0 },
        ],
      });
      return withSim(gpu, pair, 32, { ndim: 2 }, async (sim) => {
        const orbit = (m: { pos: number[]; vel: number[] }[]) => {
          const r = Math.hypot(m[0].pos[0] - m[1].pos[0], m[0].pos[1] - m[1].pos[1], soft);
          const u = Math.hypot(m[0].vel[0] - m[1].vel[0], m[0].vel[1] - m[1].vel[1]);
          return { r, u, e: 0.5 * u * u - 1 / r };
        };
        const e0 = orbit((await sim.diagnostics()).masses).e;
        let worst = 0;
        let over = 0;
        let closest = Infinity;
        let least = Infinity;
        let steps = 0;
        let before = orbit((await sim.diagnostics()).masses);
        const chunks = 96;
        for (let i = 1; i <= 2 * chunks; i++) {
          sim.cfg.tEnd = (2 * Math.PI * i) / chunks;
          sim.writeParams();
          const taken = await sim.runToEnd();
          const now = orbit((await sim.diagnostics()).masses);
          worst = Math.max(worst, Math.abs(now.e / e0 - 1));
          // The mean step of the chunk against the limit at its ends, whichever is the looser.
          const limit = (o: { r: number; u: number }) => 0.0625 * sim.cfg.cfl * Math.min(o.r / o.u, Math.sqrt(o.r ** 3));
          over = Math.max(over, (2 * Math.PI) / chunks / (taken - steps) / Math.max(limit(before), limit(now)));
          steps = taken;
          closest = Math.min(closest, now.r);
          least = Math.min(least, limit(now));
          before = now;
        }
        return {
          pass: worst < 0.01 && over < 1.1 && closest < 0.25,
          metrics: { orbit_energy_error: worst, mean_step_over_limit: over, closest_separation_seen: closest, least_limit: least, steps },
        };
      });
    },
  },
  {
    id: 'gpu-vs-cpu',
    name: 'GPU against the float64 CPU reference (32^3)',
    criterion: 'After 20 steps, L1 difference in density and energy < 1e-4 of the mean for a diagonal Sod tube and a blast',
    async run(gpu) {
      const n = 32;
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const prob of [sodDiag, sedov]) {
        const cfg = configFor(prob, n, { tEnd: Infinity });
        const ref = new RefHydro({
          nx: n, ny: n, nz: n, dx: prob.box / n, eos: makeEos(cfg.eos, cfg.eosParams), bcLo: cfg.bcLo, bcHi: cfg.bcHi,
          cfl: cfg.cfl, rhoFloor: cfg.rhoFloor, pFloor: cfg.pFloor, etaDual: cfg.etaDual, shockThresh: cfg.shockThresh,
          riemann: cfg.riemann === 'hll' ? 'hll' : 'hllc', limiter: cfg.limiter,
        });
        ref.init(prob.init(n, ref.cfg.eos, 3, {}));
        for (let i = 0; i < 20; i++) ref.step();
        await withSim(gpu, prob, n, { tEnd: Infinity }, async (sim) => {
          sim.step(20);
          const { u1, u2 } = await sim.readState();
          const d = await sim.diagnostics();
          let dr = 0, sr = 0, de = 0, se = 0, mr = 0;
          for (let o = 0; o < n * n * n; o++) {
            dr += Math.abs(u1[o * 4] - ref.U[0][o]);
            mr = Math.max(mr, Math.abs(u1[o * 4] - ref.U[0][o]));
            sr += Math.abs(ref.U[0][o]);
            de += Math.abs(u2[o * 4] - ref.U[4][o]);
            se += Math.abs(ref.U[4][o]);
          }
          metrics[`${prob.id}_rho_l1`] = dr / sr;
          metrics[`${prob.id}_rho_max`] = mr;
          metrics[`${prob.id}_energy_l1`] = de / se;
          metrics[`${prob.id}_time_diff`] = Math.abs(d.t - ref.t) / ref.t;
          pass &&= dr / sr < 1e-4 && de / se < 1e-4;
        });
      }
      return { pass, metrics };
    },
  },
  {
    id: 'eos',
    name: 'Equations of state on the GPU',
    criterion: 'p -> e_int -> p round trip to 1e-5; WGSL matches the float64 mirrors to 1e-4; gas+radiation reduces to its ideal-gas and pure-radiation limits. The degenerate case puts each pressure on top of the degenerate one; the ionizing case spans neutral to fully ionized gas',
    async run(gpu) {
      const d = gpu.device;
      const metrics: Record<string, number> = {};
      let pass = true;
      const pairs: [number, number][] = [];
      for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) pairs.push([10 ** (-4 + (8 * i) / 15), 10 ** (-4 + (8 * j) / 15)]);
      const cases: { kind: EosKind; label: string; params: Partial<ReturnType<typeof defaultEosParams>> }[] = [
        { kind: 'ideal', label: 'ideal', params: { gamma: 1.4 } },
        { kind: 'barotropic', label: 'isothermal', params: { gamma: 1, K: 0.7 } },
        { kind: 'barotropic', label: 'polytropic', params: { gamma: 2, K: 0.3 } },
        { kind: 'gas_radiation', label: 'gasrad', params: { gamma: 5 / 3, R: 1, arad: 1 } },
        { kind: 'gas_radiation', label: 'gasrad_no_radiation', params: { gamma: 5 / 3, R: 1, arad: 0 } },
        { kind: 'gas_radiation', label: 'gasrad_radiation_only', params: { gamma: 5 / 3, R: 1e-12, arad: 1 } },
        { kind: 'degenerate', label: 'degenerate', params: { gamma: 5 / 3, R: 1, arad: 1e-2, degRho: 1, degP: 1 } },
        { kind: 'ionizing', label: 'ionizing', params: { gamma: 5 / 3 } },
      ];
      const inData = new Float32Array(pairs.length * 4);
      const inBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(inBuf, 0, inData);
      const outBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      for (const c of cases) {
        const ep = { ...defaultEosParams(), ...c.params };
        const eos = makeEos(c.kind, ep);
        // Degenerate gas cannot be below its zero-temperature pressure: test the thermal part on top.
        const cp = c.kind === 'degenerate' ? pairs.map(([rho, p]) => [rho, eos.pressure(rho, 0) + p]) : pairs;
        cp.forEach(([rho, p], i) => inData.set([rho, p, 0, 0], i * 4));
        d.queue.writeBuffer(inBuf, 0, inData);
        const pb = new Float32Array(PARAM_BYTES / 4);
        pb.set([ep.gamma, ep.K, ep.R, ep.arad], 8);
        pb.set(eosConstants2(c.kind, ep), 64);
        d.queue.writeBuffer(params, 0, pb);
        const pipe = await computePipeline(d, assemble('eos/eos_test.wgsl', { eos: EOS_MODULES[c.kind] }), `eos test ${c.label}`);
        const enc = d.createCommandEncoder();
        const pass_ = enc.beginComputePass();
        pass_.setPipeline(pipe);
        pass_.setBindGroup(0, bindGroup(d, pipe, { 0: params, 1: inBuf, 2: outBuf }));
        pass_.dispatchWorkgroups(Math.ceil(pairs.length / 64));
        pass_.end();
        enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, inData.byteLength);
        d.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();

        let round = 0, mirror = 0, limit = 0;
        cp.forEach(([rho, p], i) => {
          const [eint, pBack, cs] = out.subarray(i * 4, i * 4 + 3);
          const pIn = eos.hasEnergy ? p : eos.pressure(rho, 0);
          const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
          round = Math.max(round, rel(pBack, pIn));
          if (eos.hasEnergy) {
            mirror = Math.max(mirror, rel(eint, eos.eintFromP(rho, p)), rel(cs, eos.soundSpeed(rho, p)));
          } else {
            mirror = Math.max(mirror, rel(pBack, pIn));
          }
          if (c.label === 'gasrad_no_radiation') limit = Math.max(limit, rel(eint, p / (ep.gamma - 1)), rel(cs, Math.sqrt((ep.gamma * p) / rho)));
          if (c.label === 'gasrad_radiation_only') limit = Math.max(limit, rel(eint, 3 * p), rel(cs, Math.sqrt(((4 / 3) * p) / rho)));
        });
        metrics[`${c.label}_roundtrip`] = round;
        metrics[`${c.label}_vs_float64`] = mirror;
        if (c.label.startsWith('gasrad_')) metrics[`${c.label}_limit`] = limit;
        pass &&= round < 1e-5 && mirror < 1e-4 && limit < 1e-4;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'sphere-potential',
    name: 'Potential of a uniform-density sphere (isolated boundaries)',
    criterion: 'Max relative error of the potential < 1% over the whole grid',
    async run(gpu) {
      const n = 64;
      return withSim(gpu, sphere, n, {}, async (sim) => {
        sim.solvePotential(10);
        const phi = await sim.readPotential();
        const res = await sim.poissonResidual();
        let err = 0;
        for (let k = 0; k < n; k++)
          for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) {
              const r = Math.hypot(i + 0.5 - n / 2, j + 0.5 - n / 2, k + 0.5 - n / 2) / n;
              const e = spherePotential(r);
              err = Math.max(err, Math.abs(phi[(k * n + j) * n + i] - e) / Math.abs(e));
            }
        return { pass: err < 0.01, metrics: { max_relative_error: err, max_residual: res, source_scale: 4 * Math.PI } };
      });
    },
  },
  {
    id: 'jeans',
    name: 'Jeans instability growth rate (periodic gravity)',
    criterion: 'Growth rate within 5% of linear theory',
    async run(gpu) {
      const n = 64;
      return withSim(gpu, jeans, n, {}, async (sim) => {
        const steps = await sim.runToEnd();
        const { u1 } = await sim.readState();
        let a = 0;
        for (let k = 0; k < n; k++)
          for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) a += (rhoAt(u1, n, i, j, k) - 1) * Math.cos(JEANS.k * ((i + 0.5) / n - 0.5));
        const amp = (2 * a) / n ** 3;
        // delta(t) = A cosh(w t)
        const rate = Math.acosh(amp / JEANS.amp) / jeans.tEnd;
        const err = Math.abs(rate - JEANS.growth) / JEANS.growth;
        return { pass: err < 0.05, metrics: { growth_rate: rate, linear_theory: JEANS.growth, error: err, amplitude: amp, steps } };
      });
    },
  },
  {
    id: 'polytrope',
    name: 'n = 1 polytrope in hydrostatic equilibrium',
    criterion: 'Central density stays within 10% of its initial value over 10 dynamical times',
    async run(gpu) {
      return withSim(gpu, polytrope, 64, { tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const rho0 = d0.ranges[0][1];
        let worst = 0;
        let last = d0;
        let steps = 0;
        for (let t = 1; t <= 10; t++) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.ranges[0][1] - rho0) / rho0);
        }
        return {
          pass: worst < 0.1,
          metrics: {
            central_density_initial: rho0, central_density_final: last.ranges[0][1], max_deviation: worst,
            energy_drift: Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot), max_mach: last.maxMach, steps,
          },
        };
      });
    },
  },
  {
    id: 'white-dwarf',
    name: 'Cold white dwarf in hydrostatic equilibrium (degenerate electrons, settled)',
    criterion: 'A 0.6 M_sun C/O white dwarf, settled into the grid’s balance, keeps its central density within 5% over about three dynamical times (t <= 3), with total energy conserved to 1e-3',
    async run(gpu) {
      return withSim(gpu, whiteDwarfStar, 64, { tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const rho0 = d0.ranges[0][1];
        let worst = 0;
        let last = d0;
        let steps = 0;
        for (let t = 0.5; t <= 3; t += 0.5) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.ranges[0][1] - rho0) / rho0);
        }
        const drift = Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot);
        return {
          pass: worst < 0.05 && drift < 1e-3,
          metrics: {
            central_density_initial: rho0, central_density_final: last.ranges[0][1], max_deviation: worst,
            energy_drift: drift, settle_rounds: sim.settleLog.length, settle_ms: sim.settleMs, steps,
          },
        };
      });
    },
  },
  {
    id: 'evrard',
    name: 'Evrard collapse',
    criterion:
      'Total energy conserved to 2% through collapse and bounce (t <= 1.5, before the outgoing shock reaches the box edge); thermal energy peaks between t = 0.9 and 1.3',
    async run(gpu) {
      return withSim(gpu, evrard, 64, { tEnd: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        let worst = 0;
        let peakT = 0;
        let peakU = 0;
        let minW = 0;
        let last = d0;
        let steps = 0;
        const curve: string[] = [];
        for (let i = 1; i <= 12; i++) {
          sim.cfg.tEnd = 0.125 * i;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot));
          if (last.eInt > peakU) [peakU, peakT] = [last.eInt, last.t];
          minW = Math.min(minW, last.eGrav);
          if (i % 2 === 0) curve.push(`${last.t.toFixed(2)}:${last.eKin.toFixed(3)}/${last.eInt.toFixed(3)}/${last.eGrav.toFixed(3)}`);
        }
        return {
          pass: worst < 0.02 && peakT >= 0.9 && peakT <= 1.3,
          metrics: {
            e_total_initial: d0.eTot, e_total_final: last.eTot, max_energy_drift: worst, thermal_peak_time: peakT,
            thermal_peak: peakU, potential_minimum: minW, mass_change: (last.mass - d0.mass) / d0.mass, steps,
            't:kin/int/grav': curve.join(' '),
          },
        };
      });
    },
  },
  {
    id: 'tidal-disruption',
    name: 'Tidal disruption (mass ratio 100, beta = 1)',
    criterion:
      'The perturber reaches pericenter within 2% of r_t / beta and of the Kepler time; while all the gas is still in the box its center of mass stays within 0.005 stellar radii of the box center; the star is disrupted (peak density falls by more than half)',
    async run(gpu) {
      const orbit = tdeOrbit({ q: 100, beta: 1 });
      return withSim(gpu, tdeTest, 64, { tEnd: 0, params: { q: 100, beta: 1 } }, async (sim) => {
        const d0 = await sim.diagnostics();
        let rMin = Infinity;
        let tMin = 0;
        let comMax = 0;
        let tExit = Infinity;
        let last = d0;
        let steps = 0;
        const track: string[] = [];
        for (let t = 0.125; t <= orbit.tPeri + 5 + 1e-9; t += 0.125) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          const r = Math.hypot(...last.perturber!);
          if (r < rMin) [rMin, tMin] = [r, last.t];
          // Ambient gas trickles out from the start; 0.1% of the mass gone means debris is leaving.
          if (last.mass < 0.999 * d0.mass) tExit = Math.min(tExit, last.t);
          else comMax = Math.max(comMax, Math.hypot(...last.com));
          if (Math.round(t * 8) % 8 === 0) track.push(`${last.t.toFixed(0)}:${r.toFixed(2)}/${last.ranges[0][1].toFixed(2)}`);
        }
        const rErr = Math.abs(rMin - orbit.rp) / orbit.rp;
        const tErr = Math.abs(tMin - orbit.tPeri) / orbit.tPeri;
        const drop = last.ranges[0][1] / d0.ranges[0][1];
        return {
          pass: rErr < 0.02 && tErr < 0.02 && comMax < 0.005 && tExit > orbit.tPeri && drop < 0.5,
          metrics: {
            pericenter: rMin, pericenter_expected: orbit.rp, pericenter_error: rErr,
            pericenter_time: tMin, pericenter_time_kepler: orbit.tPeri,
            max_com_offset_while_contained: comMax, debris_starts_leaving_at: tExit,
            final_com_offset_of_gas_in_box: Math.hypot(...last.com),
            central_density_initial: d0.ranges[0][1], central_density_lane_emden: TDE_STAR.rhoc,
            peak_density_final_ratio: drop, mass_left_in_box: last.mass / d0.mass, steps,
            't:r_perturber/rho_max': track.join(' '),
          },
        };
      });
    },
  },
  {
    id: 'mhd-brio-wu',
    name: 'MHD: Brio-Wu shock tube (2D, 256 cells, HLLD and HLLE)',
    criterion:
      'Total energy and mass conserved to 1e-6 and momentum equal to what the boundaries let in (0.09, -0.15) to 1e-5; the compound wave peak at x in [-0.05, -0.01], the slow shock at x in [0.13, 0.16] and min v_y in [-1.62, -1.55] (Brio & Wu 1988); HLLD and HLLE agree to a mean density difference below 0.01',
    async run(gpu) {
      const n = 256;
      const rows: Float64Array[] = [];
      const m: Record<string, number> = {};
      let ok = true;
      for (const riemann of ['hllc', 'hll'] as RiemannKind[]) {
        await withSim(gpu, brioWu, n, { ndim: 2, riemann }, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          const { u1 } = await sim.readState();
          const j = n / 2;
          const rho = new Float64Array(n);
          let vyMin = Infinity;
          for (let i = 0; i < n; i++) {
            const o = 4 * (j * n + i);
            rho[i] = u1[o];
            vyMin = Math.min(vyMin, u1[o + 2] / u1[o]);
          }
          rows.push(rho);
          const x = (i: number) => (i + 0.5) / n - 0.5;
          // Compound wave: the density peak just left of the contact; slow shock: the last drop below 0.18.
          let peak = -1;
          for (let i = 0; i < n; i++) if (x(i) > -0.1 && x(i) < 0.03 && (peak < 0 || rho[i] > rho[peak])) peak = i;
          let slow = 0;
          for (let i = n - 1; i > 0; i--) {
            if (rho[i - 1] > 0.18 && rho[i] <= 0.18) {
              slow = i;
              break;
            }
          }
          const k = riemann === 'hll' ? 'hlle' : 'hlld';
          const eErr = Math.abs(d.eTot / d0.eTot - 1);
          const mErr = Math.abs(d.mass / d0.mass - 1);
          const pErr = Math.max(Math.abs(d.momentum[0] - 0.09), Math.abs(d.momentum[1] + 0.15));
          Object.assign(m, {
            [`${k}_energy_error`]: eErr, [`${k}_mass_error`]: mErr, [`${k}_momentum_error`]: pErr,
            [`${k}_compound_x`]: x(peak), [`${k}_slow_shock_x`]: x(slow), [`${k}_vy_min`]: vyMin, [`${k}_divB`]: d.magnetic!.divB,
          });
          ok &&= eErr < 1e-6 && mErr < 1e-6 && pErr < 1e-5 && x(peak) > -0.05 && x(peak) < -0.01 && x(slow) > 0.13 && x(slow) < 0.16 && vyMin > -1.62 && vyMin < -1.55;
        });
      }
      let diff = 0;
      for (let i = 0; i < n; i++) diff += Math.abs(rows[0][i] - rows[1][i]) / n;
      m.hlld_vs_hlle_mean_density_difference = diff;
      return { pass: ok && diff < 0.01, metrics: m };
    },
  },
  {
    id: 'mhd-orszag-tang',
    name: 'MHD: Orszag-Tang vortex (2D, 256 cells, to t = 0.5)',
    criterion:
      'Mass and total energy conserved to 1e-6 (periodic box); max |div B| dx / max |B| below 1e-6 (the compensated face sums keep it at rounding, however many steps); magnetic energy 0.059 to 0.064 and peak pressure 0.48 to 0.54 (this code, HLLD; regression ranges)',
    async run(gpu) {
      return withSim(gpu, orszagTang, 256, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const mg = d.magnetic!;
        const eErr = Math.abs(d.eTot / d0.eTot - 1);
        const mErr = Math.abs(d.mass / d0.mass - 1);
        const pmax = d.ranges[3][1];
        return {
          pass: eErr < 1e-6 && mErr < 1e-6 && mg.divB < 1e-6 && mg.eMag > 0.059 && mg.eMag < 0.064 && pmax > 0.48 && pmax < 0.54,
          metrics: { energy_error: eErr, mass_error: mErr, divB: mg.divB, magnetic_energy: mg.eMag, magnetic_energy_initial: d0.magnetic!.eMag, peak_pressure: pmax, steps },
        };
      });
    },
  },
  {
    id: 'mhd-field-loop',
    name: 'MHD: field loop carried once around the box (2D, 128 cells)',
    criterion:
      'After t = 1 (back at its start): at least 90% of the magnetic energy left, peak |B| within 10% of the initial 1e-3, the field-energy centroid within one cell of the box center, and max |div B| dx / max |B| below 1e-6 (Gardiner & Stone 2005)',
    async run(gpu) {
      const n = 128;
      return withSim(gpu, fieldLoop, n, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const f = await sim.mhd!.readFaces(sim.cur);
        // Centroid of B^2 at the cell centers, the periodic way (angles), in cells from the center.
        const fx = n + 1;
        const face = (a: number, i: number, j: number) => f[4 * ((j % n) * fx + (i % n)) + a];
        let cx = 0;
        let sx = 0;
        let cy = 0;
        let sy = 0;
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < n; i++) {
            const bx = 0.5 * (face(0, i, j) + face(0, i + 1, j));
            const by = 0.5 * (face(1, i, j) + face(1, i, j + 1));
            const w = bx * bx + by * by;
            const ti = (2 * Math.PI * (i + 0.5)) / n;
            const tj = (2 * Math.PI * (j + 0.5)) / n;
            cx += w * Math.cos(ti);
            sx += w * Math.sin(ti);
            cy += w * Math.cos(tj);
            sy += w * Math.sin(tj);
          }
        }
        const pos = (c: number, s: number) => (((Math.atan2(s, c) / (2 * Math.PI)) * n + n) % n) - n / 2;
        const off = Math.hypot(pos(cx, sx), pos(cy, sy));
        const kept = d.magnetic!.eMag / d0.magnetic!.eMag;
        const peak = d.magnetic!.bMax / 1e-3;
        return {
          pass: kept > 0.9 && Math.abs(peak - 1) < 0.1 && off < 1 && d.magnetic!.divB < 1e-6,
          metrics: { magnetic_energy_kept: kept, peak_field_ratio: peak, centroid_offset_cells: off, divB: d.magnetic!.divB, steps },
        };
      });
    },
  },
  {
    id: 'mhd-alfven-wave',
    name: 'MHD: circularly polarized Alfven wave, one period (2D, 128 and 256 cells)',
    criterion: 'Mean error of v_z against the exact solution below 3e-5 at 256^2, converging at second order or better from 128^2 (Toth 2000); max |div B| dx / max |B| below 1e-6',
    async run(gpu) {
      const err: Record<number, number> = {};
      let div = 0;
      for (const n of [128, 256]) {
        await withSim(gpu, alfvenWave, n, { ndim: 2 }, async (sim) => {
          await sim.runToEnd(64);
          const { u1 } = await sim.readState();
          let e = 0;
          for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
              const x = (i + 0.5) / n - 0.5;
              const y = (j + 0.5) / n - 0.5;
              const o = 4 * (j * n + i);
              e += Math.abs(u1[o + 3] / u1[o] - 0.1 * Math.cos(2 * Math.PI * (x + y)));
            }
          }
          err[n] = e / (n * n);
          div = Math.max(div, (await sim.diagnostics()).magnetic!.divB);
        });
      }
      const order = Math.log2(err[128] / err[256]);
      return { pass: err[256] < 3e-5 && order > 1.9 && div < 1e-6, metrics: { l1_128: err[128], l1_256: err[256], order, divB: div } };
    },
  },
  {
    id: 'mhd-blast-3d',
    name: 'MHD: blast wave in an oblique field (3D, 64 cells)',
    criterion:
      'Mass conserved to 1e-6 and total energy to 3e-6 (periodic box: a few roundings of the float32 totals, near 0.7); max |div B| dx / max |B| below 1e-6; the blast stretched along the field: the second moment of the excess pressure along (1, 1, 0) at least 1.3 times that across it, in the mid-plane',
    async run(gpu) {
      const n = 64;
      return withSim(gpu, mhdBlast, n, { ndim: 3 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const { u2 } = await sim.readState();
        const k = n / 2;
        let along = 0;
        let across = 0;
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < n; i++) {
            const o = 4 * ((k * n + j) * n + i);
            const x = (i + 0.5) / n - 0.5;
            const y = (j + 0.5) / n - 0.5;
            // Pressure above the ambient 0.1, from the internal energy (gamma = 5/3).
            const w = Math.max(u2[o + 1] * (2 / 3) - 0.1, 0);
            along += (w * (x + y) ** 2) / 2;
            across += (w * (x - y) ** 2) / 2;
          }
        }
        const eErr = Math.abs(d.eTot / d0.eTot - 1);
        const mErr = Math.abs(d.mass / d0.mass - 1);
        const ratio = along / across;
        return {
          pass: eErr < 3e-6 && mErr < 1e-6 && d.magnetic!.divB < 1e-6 && ratio > 1.3,
          metrics: { energy_error: eErr, mass_error: mErr, divB: d.magnetic!.divB, elongation: ratio, steps },
        };
      });
    },
  },
  {
    id: 'mhd-slabs',
    name: 'MHD: stepping in slabs changes nothing (3D, 64 cells, outflow box)',
    criterion:
      'A blast in an oblique field stepped 40 times with the flux and edge buffers covering 4 layers at a time (16 slabs) and with one slab for the whole grid: the states and face fields agree bit for bit, and |div B| dx / max |B| stays below 1e-6 in both',
    async run(gpu) {
      const n = 64;
      const blast: Problem = { ...mhdBlast, bcLo: all(1), bcHi: all(1) };
      const runs: { u1: Float32Array; u2: Float32Array; f: Float32Array; div: number }[] = [];
      for (const bytes of [null, 1]) {
        // 1 byte: the fewest layers a slab may have (4).
        setMhdSlabBytes(bytes);
        try {
          await withSim(gpu, blast, n, { ndim: 3 }, async (sim) => {
            sim.step(40);
            const { u1, u2 } = await sim.readState();
            const f = await sim.mhd!.readFaces(sim.cur);
            runs.push({ u1, u2, f, div: (await sim.diagnostics()).magnetic!.divB });
          });
        } finally {
          setMhdSlabBytes(null);
        }
      }
      let diff = 0;
      for (const k of ['u1', 'u2', 'f'] as const) {
        const a = runs[0][k];
        const b = runs[1][k];
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
      }
      const div = Math.max(runs[0].div, runs[1].div);
      return { pass: diff === 0 && div < 1e-6, metrics: { differing_values: diff, divB: div } };
    },
  },
  {
    id: 'mhd-gpu-vs-cpu',
    name: 'MHD: GPU against the float64 CPU reference (Orszag-Tang 2D at 64, blast 3D at 32)',
    criterion:
      'After 20 steps (Orszag-Tang) and 6 (blast, outflow box): L1 difference in density, total energy and face field below 1e-5 of the mean, and the clocks agree to 1e-5 (src/reference/mhd.ts, the same scheme in float64)',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const cases: [Problem, number, 2 | 3, number][] = [
        [orszagTang, 64, 2, 20],
        [{ ...mhdBlast, bcLo: all(1), bcHi: all(1) }, 32, 3, 6],
      ];
      for (const [prob, n, ndim, steps] of cases) {
        const cfg = configFor(prob, n, { ndim, tEnd: Infinity });
        const nz = ndim === 2 ? 1 : n;
        const faces = initialFaces(cfg, n, nz, prob.box / n);
        const ref = new RefMhd({
          n, nz, dx: prob.box / n, eos: makeEos(cfg.eos, cfg.eosParams), bcLo: cfg.bcLo, bcHi: cfg.bcHi, cfl: cfg.cfl,
          rhoFloor: cfg.rhoFloor, pFloor: cfg.pFloor, etaDual: cfg.etaDual, shockThresh: cfg.shockThresh,
          riemann: cfg.riemann === 'hll' ? 'hlle' : 'hlld', limiter: cfg.limiter,
        });
        ref.init(prob.init(n, ref.cfg.eos, ndim, {}), faces);
        for (let i = 0; i < steps; i++) ref.step();
        await withSim(gpu, prob, n, { ndim, tEnd: Infinity }, async (sim) => {
          sim.step(steps);
          const { u1, u2 } = await sim.readState();
          const f = await sim.mhd!.readFaces(sim.cur);
          const d = await sim.diagnostics();
          let dr = 0, sr = 0, de = 0, se = 0, db = 0, sb = 0;
          for (let o = 0; o < n * n * nz; o++) {
            dr += Math.abs(u1[o * 4] - ref.U[0][o]);
            sr += Math.abs(ref.U[0][o]);
            de += Math.abs(u2[o * 4] - ref.U[4][o]);
            se += Math.abs(ref.U[4][o]);
          }
          // Faces held by in-range texels (every texel of the grid of faces holds valid ones or zeros on both).
          for (let q = 0; q < ref.B[0].length; q++) {
            for (let a = 0; a < 3; a++) {
              db += Math.abs(f[4 * q + a] - ref.B[a][q]);
              sb += Math.abs(ref.B[a][q]);
            }
          }
          const k = prob.id;
          metrics[`${k}_rho_l1`] = dr / sr;
          metrics[`${k}_energy_l1`] = de / se;
          metrics[`${k}_field_l1`] = db / sb;
          metrics[`${k}_time_diff`] = Math.abs(d.t - ref.t) / ref.t;
          pass &&= dr / sr < 1e-5 && de / se < 1e-5 && db / sb < 1e-5 && Math.abs(d.t - ref.t) / ref.t < 1e-5;
        });
      }
      return { pass, metrics };
    },
  },
  {
    id: 'mhd-restart',
    name: 'MHD: a magnetized run saved to .athdf and restarted (rotor 2D at 128, blast 3D at 32, outflow boxes; on blocks, the dense field loop 2D at 256 and the blast 3D at 128, 1 level)',
    criterion:
      'The restarted run starts with the very face field it was saved with (bit for bit, the upper boundary faces included) and |div B| dx / max |B| below 1e-6; after 10 more steps of each, densities agree to 1e-5 (the file holds pressure, not internal energy, so the state comes back to rounding). On blocks: the restart has the same leaves with the same faces bit for bit, shared faces agree, |div B| below 1e-6, and 3 more steps of each (before either re-meshes) give densities that agree to 1e-5',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      const cases: [Problem, number, 2 | 3][] = [
        [mhdRotor, 128, 2],
        [{ ...mhdBlast, bcLo: all(1), bcHi: all(1) }, 32, 3],
      ];
      for (const [prob, n, ndim] of cases) {
        const k = prob.id;
        await withSim(gpu, prob, n, { ndim, tEnd: Infinity }, async (a) => {
          a.step(20);
          const bytes = await writeAthdf(await capture(a, { program: 'shaderous', build: 'test', query: '' }));
          const snap = await readAthdf(bytes);
          const saved = await a.mhd!.readFaces(a.cur);
          const b = await Sim.create(gpu, { ...configFor(prob, n, { ndim, tEnd: Infinity }), ...restartFrom(snap, prob, false, n) });
          try {
            const back = await b.mhd!.readFaces(b.cur);
            let differ = 0;
            for (let i = 0; i < saved.length; i++) if (i % 4 !== 3 && saved[i] !== back[i]) differ++;
            const div = (await b.diagnostics()).magnetic!.divB;
            a.step(10);
            b.step(10);
            const ra = (await a.readState()).u1;
            const rb = (await b.readState()).u1;
            let drho = 0;
            for (let i = 0; i < ra.length; i += 4) drho = Math.max(drho, Math.abs(ra[i] - rb[i]) / ra[i]);
            metrics[`${k}_faces_differing`] = differ;
            metrics[`${k}_divB_restarted`] = div;
            metrics[`${k}_max_rel_density_difference`] = drho;
            pass &&= differ === 0 && div < 1e-6 && drho < 1e-5;
          } finally {
            b.destroy();
          }
        });
      }
      // On blocks: the file holds every leaf's faces, and the restart rebuilds the same mesh.
      const blocks: [Problem, number, 2 | 3, string][] = [
        [fieldLoopDense, 256, 2, 'loop_blocks'],
        [{ ...mhdBlast, bcLo: all(1), bcHi: all(1) }, 128, 3, 'blast_blocks'],
      ];
      const key = (l: { level: number; c: number[] }) => `${l.level}:${l.c}`;
      for (const [prob, n, ndim, k] of blocks) {
        const over = { ndim, amrLevels: 1, tEnd: Infinity };
        await withAmr(gpu, prob, n, over, async (a) => {
          a.step(20);
          const snap = await readAthdf(await writeAthdf(await capture(a, { program: 'shaderous', build: 'test', query: '' })));
          const leavesA = a.tree.leaves();
          const saved = new Map((await a.leafFaces(leavesA)).map((f, i) => [key(leavesA[i]), f]));
          const b = await AmrSim.create(gpu, { ...configFor(prob, n, over), ...restartFrom(snap, prob, true, n >> 1) });
          try {
            const leavesB = b.tree.leaves();
            const back = await b.leafFaces(leavesB);
            let differ = 0;
            let missing = 0;
            leavesB.forEach((l, i) => {
              const f = saved.get(key(l));
              if (!f) return void missing++;
              for (let t = 0; t < f.length; t++) if (t % 4 !== 3 && f[t] !== back[i][t]) differ++;
            });
            const div = (await b.diagnostics()).magnetic!.divB;
            const faces = await b.faceConsistency();
            // A few steps, before either run checks its refinement: the same mesh, cell for cell.
            const ra = a.regrids;
            a.step(3);
            b.step(3);
            const db = await b.diagnostics();
            const ca = await a.readLeaves();
            const cb = await b.readLeaves();
            const cells = BS * BS * (ndim === 2 ? 1 : BS);
            const at = new Map(a.tree.leaves().map((l, i) => [key(l), i]));
            let drho = 0;
            b.tree.leaves().forEach((l, i) => {
              const j = at.get(key(l));
              if (j === undefined) return;
              for (let q = 0; q < cells; q++) drho = Math.max(drho, Math.abs(ca.u1[4 * (j * cells + q)] - cb.u1[4 * (i * cells + q)]) / ca.u1[4 * (j * cells + q)]);
            });
            metrics[`${k}_leaves`] = leavesB.length;
            metrics[`${k}_leaves_not_in_file`] = missing + Math.abs(leavesA.length - leavesB.length);
            metrics[`${k}_faces_differing`] = differ;
            metrics[`${k}_shared_mismatched`] = faces.mismatched;
            metrics[`${k}_divB_restarted`] = div;
            metrics[`${k}_regrids_meanwhile`] = a.regrids - ra + b.regrids;
            metrics[`${k}_max_rel_density_difference`] = drho;
            metrics[`${k}_divB_later`] = db.magnetic!.divB;
            pass &&=
              metrics[`${k}_leaves_not_in_file`] === 0 && differ === 0 && faces.mismatched === 0 && div < 1e-6 && db.magnetic!.divB < 1e-6 &&
              metrics[`${k}_regrids_meanwhile`] === 0 && drho < 1e-5;
          } finally {
            b.destroy();
          }
        });
      }
      return { pass, metrics };
    },
  },
  {
    id: 'mhd-amr-unrefined',
    name: 'MHD on blocks without refinement against the uniform grid (Orszag-Tang 2D, 128 root cells)',
    criterion:
      'A block grid that the initial conditions do not refine takes the uniform grid\'s steps: after 40 steps the kinetic, internal and magnetic energies, time and peak density agree to 1e-6, every face two blocks share agrees bit for bit, and |div B| dx / max |B| stays below 1e-6',
    async run(gpu) {
      const m: Record<string, number> = {};
      const uni = await withSim(gpu, orszagTang, 128, { ndim: 2 }, async (sim) => {
        sim.step(40);
        return sim.diagnostics();
      });
      // A refinement threshold the indicator (at most 1) never reaches: the mesh stays as it starts.
      // Run to the uniform run's time (a block grid's step() stops early at refinement checks).
      return withAmr(gpu, orszagTang, 256, { ndim: 2, amrLevels: 1, refineHi: 2, tEnd: uni.t }, async (sim) => {
        const levels = new Set(sim.tree.leaves().map((b) => b.level));
        await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const faces = await sim.faceConsistency();
        const rel = (x: number, y: number) => Math.abs(x - y) / Math.max(Math.abs(y), 1e-30);
        m.leaf_levels = levels.size;
        m.kinetic = rel(d.eKin, uni.eKin);
        m.internal = rel(d.eInt, uni.eInt);
        m.magnetic = rel(d.magnetic!.eMag, uni.magnetic!.eMag);
        m.time = rel(d.t, uni.t);
        m.peak_density = rel(d.ranges[0][1], uni.ranges[0][1]);
        m.shared_faces = faces.shared;
        m.shared_mismatched = faces.mismatched;
        m.divB = d.magnetic!.divB;
        const worst = Math.max(m.kinetic, m.internal, m.magnetic, m.time, m.peak_density);
        return { pass: levels.size === 1 && worst < 1e-6 && faces.mismatched === 0 && faces.shared > 0 && m.divB < 1e-6, metrics: m };
      });
    },
  },
  {
    id: 'mhd-amr-refined',
    name: 'MHD on a refined mesh, re-meshed as it goes: rotor (2D, 256 finest, 1 level) and blast (3D, 128 finest, 1 level)',
    criterion:
      'Mass conserved to 3e-6 and total energy to 1e-4 (re-meshing keeps the gas energy, not the field\'s); |div B| dx / max |B| below 1e-6; faces shared by two leaves at one level agree bit for bit, and a coarse face on a refinement boundary is the mean of the fine faces on it to 1e-6 of the largest face; magnetic energy within 1% (rotor) and 0.5% (blast) of the uniform grid at the finest resolution, and the rotor\'s peak density within 3%',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const cases: [Problem, number, 2 | 3, number, number][] = [
        [mhdRotor, 256, 2, 0.01, 0.03],
        [mhdBlast, 128, 3, 0.005, 1],
      ];
      for (const [prob, n, ndim, magTol, rhoTol] of cases) {
        const k = prob.id;
        const uni = await withSim(gpu, prob, n, { ndim }, async (sim) => {
          await sim.runToEnd(64);
          return sim.diagnostics();
        });
        await withAmr(gpu, prob, n, { ndim, amrLevels: 1 }, async (sim) => {
          const d0 = await sim.diagnostics();
          const levels = new Set(sim.tree.leaves().map((b) => b.level));
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          const faces = await sim.faceConsistency();
          m[`${k}_levels`] = levels.size;
          m[`${k}_energy_error`] = Math.abs(d.eTot / d0.eTot - 1);
          m[`${k}_mass_error`] = Math.abs(d.mass / d0.mass - 1);
          m[`${k}_divB`] = d.magnetic!.divB;
          m[`${k}_shared_mismatched`] = faces.mismatched;
          m[`${k}_coarse_fine_faces`] = faces.cfFaces;
          m[`${k}_coarse_fine_difference`] = faces.coarseFine;
          m[`${k}_magnetic_vs_uniform`] = Math.abs(d.magnetic!.eMag / uni.magnetic!.eMag - 1);
          m[`${k}_peak_density_vs_uniform`] = Math.abs(d.ranges[0][1] / uni.ranges[0][1] - 1);
          pass &&=
            levels.size === 2 && m[`${k}_energy_error`] < 1e-4 && m[`${k}_mass_error`] < 3e-6 && m[`${k}_divB`] < 1e-6 &&
            faces.mismatched === 0 && faces.cfFaces > 0 && faces.coarseFine < 1e-6 &&
            m[`${k}_magnetic_vs_uniform`] < magTol && m[`${k}_peak_density_vs_uniform`] < rhoTol;
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-amr-chunks',
    name: 'MHD on blocks: stepping the leaves in chunks changes nothing (blast 3D, 128 finest, 1 level; 2D, 512 finest, 1 level, with per-block timesteps)',
    criterion:
      'With the flux scratch holding one block at a time (3D; every leaf its own chunk, fluxes computed again for the update) or 16 (2D, with 2 bins of per-block timesteps, the chunks of the list of each substep sized on the GPU) and with all leaves in one chunk, 10 steps (2D: cycles) across a refinement boundary give the same state and faces bit for bit',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      const cases: [string, number, number, Partial<SimConfig>, number][] = [
        ['blast_3d', 128, 3, {}, 1],
        // 2D: 2 (fields) x 18^2 faces x 48 bytes a block.
        ['blast_2d_bins', 512, 2, { timeBins: 2, timeBinsAlways: true }, 16 * 2 * 18 * 18 * 48],
      ];
      for (const [k, n, ndim, over, small] of cases) {
        const runs: Float32Array[][] = [];
        for (const bytes of [null, small]) {
          setMhdScratchBytes(bytes);
          try {
            await withAmr(gpu, mhdBlast, n, { ndim: ndim as 2 | 3, amrLevels: 1, ...over }, async (sim) => {
              sim.step(10);
              const c = await sim.readLeaves();
              runs.push([c.u1, c.u2, await leafFaces(sim)]);
            });
          } finally {
            setMhdScratchBytes(null);
          }
        }
        let diff = 0;
        for (let j = 0; j < 3; j++) for (let i = 0; i < runs[0][j].length; i++) if (runs[0][j][i] !== runs[1][j][i]) diff++;
        metrics[`${k}_differing_values`] = diff;
      }
      return { pass: Object.values(metrics).every((d) => d === 0), metrics };
    },
  },
  {
    id: 'mhd-amr-moving',
    name: 'MHD on a moving mesh: the dense field loop around the box (2D, 256 finest, 1 level) and a blast (3D, 128 finest, 1 level)',
    criterion:
      'Re-meshing at least 20 times (refining ahead of the loop, merging behind it): |div B| dx / max |B| below 1e-6 throughout and at the end; faces two leaves share agree (bit for bit at one level, coarse the mean of fine to 1e-6); mass conserved to 1e-6; the loop keeps at least 94% of its magnetic energy and is back at the center (density center within 1e-3); the 3D blast re-meshes with |div B| below 1e-6 and its total energy within 1e-3 (re-meshing keeps the gas energy, not the field\'s)',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      await withAmr(gpu, fieldLoopDense, 256, { ndim: 2, amrLevels: 1 }, async (sim) => {
        const d0 = await sim.diagnostics();
        let worstDiv = 0;
        for (const t of [0.25, 0.5, 0.75, 1]) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd(64);
          worstDiv = Math.max(worstDiv, (await sim.diagnostics()).magnetic!.divB);
        }
        const d = await sim.diagnostics();
        const faces = await sim.faceConsistency();
        const regrids = (sim as unknown as { regrids: number }).regrids;
        m.loop_regrids = regrids;
        m.loop_divB_worst = worstDiv;
        m.loop_mass_error = Math.abs(d.mass / d0.mass - 1);
        m.loop_magnetic_kept = d.magnetic!.eMag / d0.magnetic!.eMag;
        m.loop_center_offset = Math.hypot(d.com[0], d.com[1]);
        m.loop_shared_mismatched = faces.mismatched;
        m.loop_coarse_fine = faces.coarseFine;
        pass &&= regrids >= 20 && worstDiv < 1e-6 && m.loop_mass_error < 1e-6 && m.loop_magnetic_kept > 0.94 && m.loop_center_offset < 1e-3 &&
          faces.mismatched === 0 && faces.coarseFine < 1e-6;
      });
      await withAmr(gpu, mhdBlast, 128, { ndim: 3, amrLevels: 1 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const faces = await sim.faceConsistency();
        m.blast_regrids = (sim as unknown as { regrids: number }).regrids;
        m.blast_divB = d.magnetic!.divB;
        m.blast_energy_error = Math.abs(d.eTot / d0.eTot - 1);
        m.blast_mass_error = Math.abs(d.mass / d0.mass - 1);
        m.blast_shared_mismatched = faces.mismatched;
        m.blast_coarse_fine = faces.coarseFine;
        pass &&= m.blast_regrids >= 1 && m.blast_divB < 1e-6 && m.blast_energy_error < 1e-3 && m.blast_mass_error < 1e-6 && faces.mismatched === 0 && faces.coarseFine < 1e-6;
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-amr-bins',
    name: 'MHD with per-block timesteps: rotor, dense field loop (2D, 256 finest, 1 level), blast (2D, 512 finest, to t = 0.05; 3D, 128 finest; 1 level)',
    criterion:
      'With blocks stepping up to 4 times less often than the fastest (2 bins, kept on): |div B| dx / max |B| below 1e-6; faces two leaves share agree bit for bit (coarse the mean of fine to 1e-6); mass conserved to 3e-6 and total energy within 1e-4 of its drift with one timestep; magnetic energy within 0.5% of the run with one timestep and peak density within 1%; the loop keeps at least 94% of its magnetic energy; never more block updates than with one timestep (deposits included), and under 90% of them for the early 2D blast, where the gas far from it is slow (in the others, and the 2D blast later on, the signal speeds differ by less than the factor of 2 between bins almost everywhere)',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      // The blast in 2D, early on, has the contrast in signal speed that bins save work on; in the others
      // it is less than the factor of 2 a bin takes, and they check the stepping alone.
      const cases: [Problem, number, 2 | 3, number, string][] = [
        [mhdRotor, 256, 2, Infinity, 'rotor'],
        [fieldLoopDense, 256, 2, 0.5, 'loop'],
        [mhdBlast, 512, 2, 0.05, 'blast_2d'],
        [mhdBlast, 128, 3, Infinity, 'blast_3d'],
      ];
      for (const [prob, n, ndim, tEnd, k] of cases) {
        const over = { ndim, amrLevels: 1, ...(Number.isFinite(tEnd) ? { tEnd } : {}) };
        const one = await withAmr(gpu, prob, n, over, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64);
          return { d0, d: await sim.diagnostics() };
        });
        await withAmr(gpu, prob, n, { ...over, timeBins: 2, timeBinsAlways: true }, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          const faces = await sim.faceConsistency();
          const work = await sim.binWork();
          m[`${k}_divB`] = d.magnetic!.divB;
          m[`${k}_mass_error`] = Math.abs(d.mass / d0.mass - 1);
          m[`${k}_energy_error`] = Math.abs(d.eTot / d0.eTot - 1);
          m[`${k}_energy_error_one_timestep`] = Math.abs(one.d.eTot / one.d0.eTot - 1);
          m[`${k}_magnetic_vs_one_timestep`] = Math.abs(d.magnetic!.eMag / one.d.magnetic!.eMag - 1);
          m[`${k}_peak_density_vs_one_timestep`] = Math.abs(d.ranges[0][1] / one.d.ranges[0][1] - 1);
          m[`${k}_magnetic_kept`] = d.magnetic!.eMag / d0.magnetic!.eMag;
          m[`${k}_shared_mismatched`] = faces.mismatched;
          m[`${k}_coarse_fine_difference`] = faces.coarseFine;
          m[`${k}_cycles`] = work.cycles;
          m[`${k}_share_of_block_updates`] = work.updates / work.single;
          faces.worst.forEach((w, i) => console.log(`${k} face ${i}: ${w}`));
          pass &&=
            m[`${k}_divB`] < 1e-6 && faces.mismatched === 0 && faces.coarseFine < 1e-6 && m[`${k}_mass_error`] < 3e-6 &&
            m[`${k}_energy_error`] < m[`${k}_energy_error_one_timestep`] + 1e-4 && m[`${k}_magnetic_vs_one_timestep`] < 0.005 &&
            m[`${k}_peak_density_vs_one_timestep`] < 0.01 && work.cycles > 0 && work.updates <= work.single &&
            (k !== 'blast_2d' || m[`${k}_share_of_block_updates`] < 0.9) &&
            (prob !== fieldLoopDense || m[`${k}_magnetic_kept`] > 0.94);
        });
      }
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-gravity',
    name: 'MHD with gravity, point masses and cooling (uniform grid)',
    criterion:
      'A uniform field exerts no force, so with one the gas must do what it does without: an n = 1 polytrope threaded by a field at beta 1000 at its center (64^3, self-gravity, to t = 2) keeps its central density within 5% and to 0.03 of the unmagnetized star (the field is dragged by the thin hot gas around the star, which settles onto it either way and is the fastest gas on the grid: the Mach number is reported, bounded at 3); a stratified column with a horizontal field (2D, uniform gravity, beta 1e4 at its foot, one sound crossing) stays as still as without one (Mach at most twice plus 1e-3); gas with a field cooling as Lambda ~ T^(1/2) follows the analytic decay to 1e-3; a sink in magnetized gas keeps gas plus accreted mass to 1e-5 and leaves the field (its energy within 5%). Throughout |div B| dx / max |B| stays below 1e-6',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const divOk = (d: Diagnostics) => d.magnetic!.divB < 1e-6;
      // A field along z, B = sqrt(2 p_c / beta) with beta 1000 at the star's center.
      const bStar = Math.sqrt((2 * POLY_K * POLY.rhoc ** 2) / 1000);
      const magStar: Problem = { ...polytrope, id: 'polytrope-uniform-field', mhd: { face: () => (axis) => (axis === 2 ? bStar : 0) } };
      const star = async (prob: Problem) =>
        withSim(gpu, prob, 64, { tEnd: 2 }, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd();
          const d = await sim.diagnostics();
          return { dev: Math.abs(d.ranges[0][1] / d0.ranges[0][1] - 1), mach: d.maxMach, d0, d };
        });
      const plain = await star(polytrope);
      const mag = await star(magStar);
      m.star_central_density_change = mag.dev;
      m.star_central_density_change_unmagnetized = plain.dev;
      m.star_max_mach = mag.mach;
      m.star_max_mach_unmagnetized = plain.mach;
      m.star_kinetic_energy = mag.d.eKin;
      m.star_kinetic_energy_unmagnetized = plain.d.eKin;
      m.star_magnetic_energy_change = Math.abs(mag.d.magnetic!.eMag / mag.d0.magnetic!.eMag - 1);
      m.star_divB = mag.d.magnetic!.divB;
      pass &&= mag.dev < 0.05 && Math.abs(mag.dev - plain.dev) < 0.03 && mag.mach < 3 && divOk(mag.d);
      // The column, falling by e every 8 cells, without the well-balanced reconstruction (MHD's
      // reconstruction does not have it), with and without a field along x.
      const col = column(8);
      const bCol = Math.sqrt(2 / 1e4);
      const magCol: Problem = { ...col, id: 'column-field', mhd: { face: () => (axis) => (axis === 0 ? bCol : 0) } };
      const still = async (prob: Problem) =>
        withSim(gpu, prob, 64, { ndim: 2, tEnd: 1, wellBalanced: false }, async (sim) => {
          await sim.runToEnd();
          return sim.diagnostics();
        });
      const colPlain = await still(col);
      const colMag = await still(magCol);
      m.column_max_mach = colMag.maxMach;
      m.column_max_mach_unmagnetized = colPlain.maxMach;
      m.column_divB = colMag.magnetic!.divB;
      pass &&= colMag.maxMach < 2 * colPlain.maxMach + 1e-3 && divOk(colMag);
      // Cooling, as in the cooling test, with a field: e_int (without B^2/2) decays the same.
      const cool = stillGas({
        cooling: 'power', coolParams: { rate: 1, alpha: 0.5, heat: 0, tFloor: 0 },
        mhd: { face: () => (axis) => (axis === 0 ? 0.5 : 0) },
      });
      await withSim(gpu, cool, 32, { ndim: 2, tEnd: 1.2 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const expected = 1.5 * (1 - 1.2 / 3) ** 2;
        m.cooling_decay_error = Math.abs(d.eInt - expected) / expected;
        pass &&= m.cooling_decay_error < 1e-3 && divOk(d);
      });
      // A sink in magnetized gas (2D, closed box).
      const sinkBox = stillGas({
        bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.5,
        masses: () => [{ gm: 0.05, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.08, sinkRadius: 0.15 }],
        mhd: { face: () => (axis) => (axis === 0 ? 0.3 : 0) },
      });
      await withSim(gpu, sinkBox, 64, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const mass = d.masses[0];
        m.sink_accreted_fraction = mass.accreted / d0.mass;
        m.sink_mass_error = Math.abs(d.mass + mass.accreted - d0.mass) / d0.mass;
        m.sink_magnetic_energy_change = Math.abs(d.magnetic!.eMag / d0.magnetic!.eMag - 1);
        m.sink_divB = d.magnetic!.divB;
        pass &&= mass.accreted > 0.003 * d0.mass && m.sink_mass_error < 1e-5 && m.sink_magnetic_energy_change < 0.05 && divOk(d);
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-gravity-blocks',
    name: 'MHD with gravity, point masses and cooling on the adaptive mesh',
    criterion:
      'As mhd-gravity, on blocks: the polytrope threaded by a field (128 finest, 1 level, self-gravity, to t = 1) keeps its central density within 5% and to 0.03 of the unmagnetized star, its Mach number bounded at 3; the column with a horizontal field (2D, uniform gravity, block grid at 64, with per-block timesteps) stays as still as without one (Mach at most twice plus 1e-3); cooling with a field follows the analytic decay to 1e-3; a sink keeps gas plus accreted mass to 1e-5 and leaves the field (its energy within 5%). Throughout |div B| dx / max |B| stays below 1e-6 and shared faces agree',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const ok = async (sim: AmrSim, d: Diagnostics, tag: string) => {
        const faces = await sim.faceConsistency();
        m[`${tag}_divB`] = d.magnetic!.divB;
        m[`${tag}_shared_mismatched`] = faces.mismatched;
        return d.magnetic!.divB < 1e-6 && faces.mismatched === 0;
      };
      const bStar = Math.sqrt((2 * POLY_K * POLY.rhoc ** 2) / 1000);
      const magStar: Problem = { ...polytrope, id: 'polytrope-uniform-field', mhd: { face: () => (axis) => (axis === 2 ? bStar : 0) } };
      const star = async (prob: Problem, tag: string | null) =>
        withAmr(gpu, prob, 128, { amrLevels: 1, tEnd: 1 }, async (sim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          const good = tag ? await ok(sim, d, tag) : true;
          return { dev: Math.abs(d.ranges[0][1] / d0.ranges[0][1] - 1), mach: d.maxMach, good, eKin: d.eKin };
        });
      const plain = await star(polytrope, null);
      const mag = await star(magStar, 'star');
      m.star_central_density_change = mag.dev;
      m.star_central_density_change_unmagnetized = plain.dev;
      m.star_max_mach = mag.mach;
      m.star_max_mach_unmagnetized = plain.mach;
      m.star_kinetic_energy = mag.eKin;
      m.star_kinetic_energy_unmagnetized = plain.eKin;
      pass &&= mag.good && mag.dev < 0.05 && Math.abs(mag.dev - plain.dev) < 0.03 && mag.mach < 3;
      const col = column(8);
      const magCol: Problem = { ...col, id: 'column-field', mhd: { face: () => (axis) => (axis === 0 ? Math.sqrt(2 / 1e4) : 0) } };
      const still = async (prob: Problem, tag: string | null) =>
        withAmr(gpu, prob, 64, { ndim: 2, amrLevels: 0, tEnd: 1, wellBalanced: false, timeBins: 2, timeBinsAlways: true }, async (sim) => {
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          return { mach: d.maxMach, good: tag ? await ok(sim, d, tag) : true };
        });
      const colPlain = await still(col, null);
      const colMag = await still(magCol, 'column');
      m.column_max_mach = colMag.mach;
      m.column_max_mach_unmagnetized = colPlain.mach;
      pass &&= colMag.good && colMag.mach < 2 * colPlain.mach + 1e-3;
      const cool = stillGas({
        cooling: 'power', coolParams: { rate: 1, alpha: 0.5, heat: 0, tFloor: 0 },
        mhd: { face: () => (axis) => (axis === 0 ? 0.5 : 0) },
      });
      await withAmr(gpu, cool, 32, { ndim: 2, amrLevels: 1, tEnd: 1.2 }, async (sim) => {
        await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const expected = 1.5 * (1 - 1.2 / 3) ** 2;
        m.cooling_decay_error = Math.abs(d.eInt - expected) / expected;
        pass &&= m.cooling_decay_error < 1e-3 && (await ok(sim, d, 'cooling'));
      });
      const sinkBox = stillGas({
        bcLo: all(REFLECT), bcHi: all(REFLECT), tEnd: 0.5,
        masses: () => [{ gm: 0.05, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.08, sinkRadius: 0.15 }],
        mhd: { face: () => (axis) => (axis === 0 ? 0.3 : 0) },
      });
      await withAmr(gpu, sinkBox, 64, { ndim: 2, amrLevels: 1 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd(64);
        const d = await sim.diagnostics();
        const mass = d.masses[0];
        m.sink_accreted_fraction = mass.accreted / d0.mass;
        m.sink_mass_error = Math.abs(d.mass + mass.accreted - d0.mass) / d0.mass;
        m.sink_magnetic_energy_change = Math.abs(d.magnetic!.eMag / d0.magnetic!.eMag - 1);
        pass &&= mass.accreted > 0.003 * d0.mass && m.sink_mass_error < 1e-5 && m.sink_magnetic_energy_change < 0.05 && (await ok(sim, d, 'sink'));
      });
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-star-fields',
    name: 'Magnetized polytropes: a dipole and a random field inside the star (64^3; blocks 128 finest, 1 level)',
    criterion:
      'A dipole confined to the n = 1 polytrope (beta 1000 at its center) for two dynamical times: the least beta starts within 10% of 1000, the central density stays within 5% and to 0.03 of the unmagnetized star\'s, the magnetic energy within 5% (the field, closed inside the star, stays where it was put), on the uniform grid and on blocks (shared faces agreeing); a random Kolmogorov field (beta 1000 at the center on average) for one: the central density within 5%. Throughout |div B| dx / max |B| below 1e-6',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const run = async (prob: Problem, blocks: boolean, tEnd: number, tag: string) => {
        const go = async (sim: Sim | AmrSim) => {
          const d0 = await sim.diagnostics();
          await sim.runToEnd(64);
          const d = await sim.diagnostics();
          if (prob.mhd) {
            m[`${tag}_beta_min_initial`] = d0.magnetic!.betaMin;
            m[`${tag}_magnetic_energy_change`] = Math.abs(d.magnetic!.eMag / d0.magnetic!.eMag - 1);
            m[`${tag}_divB`] = d.magnetic!.divB;
            if (sim instanceof AmrSim) m[`${tag}_shared_mismatched`] = (await sim.faceConsistency()).mismatched;
          }
          m[`${tag}_central_density_change`] = Math.abs(d.ranges[0][1] / d0.ranges[0][1] - 1);
          m[`${tag}_max_mach`] = d.maxMach;
          return m[`${tag}_central_density_change`];
        };
        return blocks ? withAmr(gpu, prob, 128, { amrLevels: 1, tEnd }, go) : withSim(gpu, prob, 64, { tEnd }, go);
      };
      const plain = await run(polytrope, false, 2, 'plain');
      const dipole = await run(polytropeDipole, false, 2, 'dipole');
      const plainBlocks = await run(polytrope, true, 2, 'plain_blocks');
      const dipoleBlocks = await run(polytropeDipole, true, 2, 'dipole_blocks');
      const random = await run(polytropeRandom, false, 1, 'random');
      for (const [tag, dev, ref] of [['dipole', dipole, plain], ['dipole_blocks', dipoleBlocks, plainBlocks]] as const) {
        pass &&=
          Math.abs(m[`${tag}_beta_min_initial`] / 1000 - 1) < 0.1 && dev < 0.05 && Math.abs(dev - ref) < 0.03 &&
          m[`${tag}_magnetic_energy_change`] < 0.05 && m[`${tag}_divB`] < 1e-6 && (m[`${tag}_shared_mismatched`] ?? 0) === 0;
      }
      pass &&= random < 0.05 && m.random_divB < 1e-6;
      return { pass, metrics: m };
    },
  },
  {
    id: 'mhd-species',
    name: 'Species carried with the gas in a magnetic field (2D, uniform 128^2 and blocks 128 finest, 1 level)',
    criterion:
      'The species test\'s bump carried twice around the box, in a uniform field along x (beta 4): mass fractions stay in [0, 1] and sum to 1 within 1e-5, each species keeps its mass within 1e-5 of the total, and the smooth species returns at least as accurately as the passive scalar carrying the same profile, which the MHD fluxes carry; on the uniform grid and on blocks. |div B| dx / max |B| below 1e-6',
    async run(gpu) {
      const m: Record<string, number> = {};
      let pass = true;
      const prob: Problem = { ...speciesCarry(0), id: 'species-field', mhd: { face: () => (axis) => (axis === 0 ? Math.sqrt(2 / 4) : 0) } };
      const mass = (X: Float32Array[], u1: Float32Array) => X.map((x) => x.reduce((s, v, q) => s + v * u1[4 * q], 0));
      const measure = async (tag: string, read: () => Promise<{ X: Float32Array[]; u1: Float32Array; u2: Float32Array }>, go: () => Promise<unknown>, d: () => Promise<Diagnostics>) => {
        const s0 = await read();
        const m0 = mass(s0.X, s0.u1);
        await go();
        const s1 = await read();
        const m1 = mass(s1.X, s1.u1);
        const cells = s1.u1.length / 4;
        let sumErr = 0;
        let lo = Infinity;
        let hi = -Infinity;
        let l1 = 0;
        let l1s = 0;
        for (let q = 0; q < cells; q++) {
          let s = 0;
          for (const x of s1.X) {
            s += x[q];
            lo = Math.min(lo, x[q]);
            hi = Math.max(hi, x[q]);
          }
          sumErr = Math.max(sumErr, Math.abs(s - 1));
          l1 += Math.abs(s1.X[0][q] - s0.X[0][q]) / cells;
          l1s += Math.abs(s1.u2[4 * q + 2] / s1.u1[4 * q] - s0.X[0][q]) / cells;
        }
        const massErr = Math.max(...m0.map((v, i) => Math.abs(m1[i] - v))) / m0.reduce((a, b) => a + b, 0);
        m[`${tag}_sum_error`] = sumErr;
        m[`${tag}_min_fraction`] = lo;
        m[`${tag}_max_fraction`] = hi;
        m[`${tag}_species_mass_drift`] = massErr;
        m[`${tag}_smooth_species_l1`] = l1;
        m[`${tag}_passive_scalar_l1`] = l1s;
        m[`${tag}_divB`] = (await d()).magnetic!.divB;
        pass &&= sumErr < 1e-5 && lo >= 0 && hi <= 1 + 1e-6 && massErr < 1e-5 && l1 <= l1s * 1.05 && m[`${tag}_divB`] < 1e-6;
      };
      await withSim(gpu, prob, 128, { ndim: 2 }, (sim) =>
        measure('uniform', async () => ({ X: await sim.readSpecies(), ...(await sim.readState()) }), () => sim.runToEnd(64), () => sim.diagnostics()),
      );
      // On blocks, the leaves in the tree's order, the same for every read (the mesh may change).
      await withAmr(gpu, prob, 128, { ndim: 2, amrLevels: 1, refineHi: 2 }, (sim) =>
        measure('blocks', async () => {
          const c = await sim.readLeaves();
          return { X: await sim.readLeafSpecies(), u1: c.u1, u2: c.u2 };
        }, () => sim.runToEnd(64), () => sim.diagnostics()),
      );
      return { pass, metrics: m };
    },
  },
];

export async function runTest(gpu: Gpu, id: string): Promise<TestResult> {
  const t = TESTS.find((x) => x.id === id);
  if (!t) throw new Error(`unknown test: ${id}`);
  const start = performance.now();
  try {
    const r = await t.run(gpu);
    return { id: t.id, name: t.name, criterion: t.criterion, ...r, seconds: (performance.now() - start) / 1000 };
  } catch (e) {
    return {
      id: t.id, name: t.name, criterion: t.criterion, pass: false,
      metrics: { error: e instanceof Error ? e.message : String(e) },
      seconds: (performance.now() - start) / 1000,
    };
  }
}
