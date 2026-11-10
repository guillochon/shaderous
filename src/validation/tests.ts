// Validation suite (plan section 7). Runs in the page against the real GPU; driven by
// validate.html interactively and by Playwright in tests/gpu.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { configFor, Sim, type SimConfig } from '../sim';
import { defaultEosParams, EOS_MODULES, makeEos, type EosKind } from '../eos/eos';
import { exactRiemann } from '../reference/riemann_exact';
import { RefHydro } from '../reference/hydro';
import { sodX, sodY, sodZ, sodDiag, strongShock, shockTube } from '../problems/shocktube';
import { sedov, SEDOV_XI0 } from '../problems/sedov';
import { advection } from '../problems/advection';
import { kelvinHelmholtz } from '../problems/kelvin_helmholtz';
import { jeans, JEANS } from '../problems/jeans';
import { polytrope } from '../problems/polytrope';
import { evrard } from '../problems/evrard';
import { sphere, spherePotential } from '../problems/sphere';
import { all, REFLECT, type Problem } from '../problems/types';

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

async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
  const sim = await Sim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

const rhoAt = (u1: Float32Array, n: number, i: number, j: number, k: number) => u1[((k * n + j) * n + i) * 4];

/** Density along a grid axis through the box centre, plus the largest deviation across the transverse plane. */
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
    // Along a ray from the centre: last sample (outside-in) above the half-way density.
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
          riemann: cfg.riemann, limiter: cfg.limiter,
        });
        ref.init(prob.init(n, ref.cfg.eos));
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
    criterion: 'p -> e_int -> p round trip to 1e-5; WGSL matches the float64 mirrors to 1e-4; gas+radiation reduces to its ideal-gas and pure-radiation limits',
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
      ];
      const inData = new Float32Array(pairs.length * 4);
      pairs.forEach(([rho, p], i) => inData.set([rho, p, 0, 0], i * 4));
      const inBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(inBuf, 0, inData);
      const outBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = d.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const params = d.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      for (const c of cases) {
        const ep = { ...defaultEosParams(), ...c.params };
        const pb = new Float32Array(28);
        pb.set([ep.gamma, ep.K, ep.R, ep.arad], 8);
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

        const eos = makeEos(c.kind, ep);
        let round = 0, mirror = 0, limit = 0;
        pairs.forEach(([rho, p], i) => {
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
