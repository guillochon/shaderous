// Validation suite (plan section 7). Runs in the page against the real GPU; driven by
// validate.html interactively and by Playwright in tests/gpu.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { configFor, PARAM_BYTES, Sim, type Diagnostics, type RiemannKind, type SimConfig } from '../sim';
import { AmrSim, type LeafCells } from '../amr/amr_sim';
import { defaultEosParams, EOS_MODULES, eosConstants2, makeEos, type EosKind, type EosParams } from '../eos/eos';
import { mixMoments, problemCoefficients } from '../species/mixture';
import { COMPOSITIONS } from '../units/composition';
import { exactRiemann } from '../reference/riemann_exact';
import { RefHydro } from '../reference/hydro';
import { sodX, sodY, sodZ, sodDiag, strongShock, shockTube } from '../problems/shocktube';
import { sedov, SEDOV_XI0 } from '../problems/sedov';
import { advection } from '../problems/advection';
import { kelvinHelmholtz } from '../problems/kelvin_helmholtz';
import { rayleighTaylor, RT } from '../problems/rayleigh_taylor';
import { jeans, JEANS } from '../problems/jeans';
import { polytrope } from '../problems/polytrope';
import { whiteDwarfStar } from '../problems/white_dwarf_star';
import { evrard } from '../problems/evrard';
import { tidalDisruption, tdeOrbit, TDE_STAR } from '../problems/tidal_disruption';
import { sphere, spherePotential } from '../problems/sphere';
import { all, PERIODIC, REFLECT, type Problem } from '../problems/types';
import { binary } from '../problems/binary';
import { thermalInstability } from '../problems/thermal_instability';
import { capture, restartFrom } from '../io/capture';
import { readAthdf, writeAthdf } from '../io/athdf';

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
  const coeffs = problemCoefficients(cfg.problem)!;
  let dv = 0;
  let dp = 0;
  let tLo = Infinity;
  let tHi = -Infinity;
  for (let q = 0; q < cells.count; q++) {
    const rho = cells.u1[4 * q];
    dv = Math.max(dv, Math.hypot(cells.u1[4 * q + 1] / rho - v0[0], cells.u1[4 * q + 2] / rho - v0[1]));
    const eos = makeEos(cfg.eos, cfg.eosParams, mixMoments(coeffs, cells.X.map((x) => x[q])));
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
const column = (cells: number): Problem =>
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

async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
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

async function withAmr<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: AmrSim) => Promise<T>) {
  const sim = await AmrSim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
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
      'A 64-cell isothermal column whose density falls by e every 4 cells (a factor of 9e6 in all), in discrete hydrostatic balance, after one sound crossing: with the well-balanced reconstruction the largest Mach number stays below 1e-4 on the uniform grid and on the block grid, and is at least 100 times smaller than without it',
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
        pass: balanced < 1e-4 && balancedBlocks < 1e-4 && balanced < 0.01 * plain && balancedBlocks < 0.01 * plainBlocks,
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
      "A disc of helium twice as dense as the hydrogen around it, in pressure balance, under an ideal gas, gas plus radiation, degenerate electrons and ionizing hydrogen. At rest it stays at rest (speeds under 1e-5) with p = 1 to 1e-5, and the GPU's temperature and pressure ranges match the float64 equations of state of each cell's composition to 1e-5. Carried once across the box the pressure stays within each equation of state's bound (balanced to 1e-5 for the ideal gas, whose pressure does not depend on the composition; within 0.05 for the others, which err where the two gases mix: 0.3% to 1.3% at 64^2) and the faces match bit for bit. With a uniform composition, carrying species that set the equation of state changes a Sod shock tube under gas plus radiation by under 1e-5 of its range",
    async run(gpu) {
      const n = 64;
      const metrics: Record<string, number> = {};
      let pass = true;
      // Moving contacts: what each equation of state is allowed (the mixing cells).
      const moveBound: Record<string, number> = { ideal: 1e-5, gas_radiation: 0.05, degenerate: 0.05, ionizing: 0.05 };
      for (const c of MIXED_EOS) {
        for (const moving of [false, true]) {
          const r = await withSim(gpu, twoGases(c.kind, c.params, moving, moving ? 1 : 0.25), n, { ndim: 2 }, async (sim) => {
            await sim.runToEnd();
            const st = await sim.readState();
            const X = await sim.readSpecies();
            const e = twoGasErrors(sim.cfg, { ...st, X, count: n * n }, moving ? [1, 0.5] : [0, 0], await sim.diagnostics());
            const flux = moving ? await sim.fluxMismatch() : { mismatched: 0 };
            return { ...e, mismatched: flux.mismatched };
          });
          const tag = `${c.kind}_${moving ? 'moving' : 'static'}`;
          metrics[`${tag}_speed`] = r.dv;
          metrics[`${tag}_pressure_error`] = r.dp;
          metrics[`${tag}_gpu_temperature_error`] = r.tErr;
          if (!moving) metrics[`${c.kind}_temperature_ratio`] = r.tRatio;
          if (moving) metrics[`${tag}_mismatched_faces`] = r.mismatched;
          if (moving) pass &&= r.dp < moveBound[c.kind] && r.mismatched === 0;
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
    criterion: 'Identical density in every cell and the same number of steps, for a 2D shock tube and a 3D blast',
    async run(gpu) {
      const metrics: Record<string, number> = {};
      let pass = true;
      for (const [prob, ndim, n] of [[sodX, 2, 128], [sedov, 3, 64]] as const) {
        const tEnd = prob === sedov ? 0.02 : prob.tEnd;
        const uni = await withSim(gpu, prob, n, { ndim, tEnd }, async (sim) => ({ steps: await sim.runToEnd(), u1: (await sim.readState()).u1 }));
        const blk = await withAmr(gpu, prob, n, { ndim, tEnd, amrLevels: 0 }, async (sim) => ({ steps: await sim.runToEnd(), cells: await sim.readLeaves() }));
        let worst = 0;
        for (let q = 0; q < blk.cells.count; q++) worst = Math.max(worst, Math.abs(blk.cells.u1[q * 4] - uni.u1[cellIndex(blk.cells, q, n, prob.box, ndim) * 4]));
        metrics[`${prob.id}_${ndim}d_max_density_difference`] = worst;
        metrics[`${prob.id}_${ndim}d_steps`] = blk.steps;
        pass &&= worst === 0 && blk.steps === uni.steps && blk.cells.count === n ** ndim;
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
      return withAmr(gpu, tidalDisruption, 128, { amrLevels: 2, tEnd: 0 }, async (sim) => {
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
      await withSim(gpu, tidalDisruption, 32, { tEnd: 0.5 }, async (a) => {
        await a.runToEnd();
        const before = await a.readRestart();
        const { snap } = await roundtrip(a);
        const cfg = { ...configFor(tidalDisruption, 32, { tEnd: 1 }), ...restartFrom(snap, tidalDisruption, false, 32) };
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
      'A softened mass at the center of a closed box of gas accretes; gas mass plus accreted mass stays constant to 1e-5, G M grows by G times the accreted mass, and the mass stays at the center (2D and 3D, uniform and adaptive); the same holds for a mass added in mid-run',
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
      return withSim(gpu, tidalDisruption, 64, { tEnd: 0 }, async (sim) => {
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
