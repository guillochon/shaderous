// Validation tests of the weakly compressible liquid (plans/incompressible.md, stage 1; eos/liquid.wgsl).
// Registered in tests.ts after the others.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline, type Gpu } from '../gpu/device';
import { configFor, PARAM_BYTES, Sim, type SimConfig } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { defaultEosParams, EOS_MODULES, eosConstants1, eosConstants2, makeEos } from '../eos/eos';
import { problemById } from '../presets/index';
import { compiledFrom } from '../setup/compile';
import type { Problem } from '../problems/types';
import type { TestDef } from './tests';

// (tests.ts imports this file, so its helpers are not imported back: these are their few lines.)
async function withSim<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: Sim) => Promise<T>) {
  const sim = await Sim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}
async function withAmr<T>(gpu: Gpu, problem: Problem, n: number, over: Partial<SimConfig>, fn: (s: AmrSim) => Promise<T>) {
  const sim = await AmrSim.create(gpu, configFor(problem, n, over));
  try {
    return await fn(sim);
  } finally {
    sim.destroy();
  }
}

/**
 * Ghia, Ghia & Shin (1982), Table I, Re = 100: u along the vertical line through the center of
 * the lid-driven cavity, as (height, u over the lid speed).
 */
export const GHIA_RE100: [number, number][] = [
  [0.9766, 0.84123], [0.9688, 0.78871], [0.9609, 0.73722], [0.9531, 0.68717], [0.8516, 0.23151], [0.7344, 0.00332], [0.6172, -0.13641],
  [0.5, -0.20581], [0.4531, -0.2109], [0.2813, -0.15662], [0.1719, -0.1015], [0.1016, -0.06434], [0.0703, -0.04775], [0.0625, -0.04192], [0.0547, -0.03717],
];

/**
 * A cavity run's u on the vertical centerline (between columns n/2 - 1 and n/2, with the walls'
 * values at the ends) against a table of (height, u): the largest difference, and the least u.
 */
export function cavityError(u1: Float32Array, n: number, table: [number, number][]): { error: number; minU: number } {
  const ys = [0];
  const us = [0];
  for (let j = 0; j < n; j++) {
    const at = (i: number) => u1[4 * (j * n + i) + 1] / u1[4 * (j * n + i)];
    ys.push((j + 0.5) / n);
    us.push(0.5 * (at(n / 2 - 1) + at(n / 2)));
  }
  ys.push(1);
  us.push(1);
  const interp = (y: number) => {
    const k = ys.findIndex((v) => v >= y);
    return us[k - 1] + ((us[k] - us[k - 1]) * (y - ys[k - 1])) / (ys[k] - ys[k - 1]);
  };
  return { error: Math.max(...table.map(([y, u]) => Math.abs(interp(y) - u))), minU: Math.min(...us) };
}

const gresho = problemById('gresho');
/** The Gresho preset's flow in an ideal gas at Mach `mach` (the flow sets the pressure). */
const greshoGas = (mach: number): Problem =>
  compiledFrom({
    ...gresho.setup!,
    id: 'gresho-gas',
    physics: { ...gresho.setup!.physics, eos: 'ideal', eosParams: { gamma: 1.4 } },
    parts: [{ kind: 'flow', name: 'gresho', mach }],
  }).problem;

/** The liquid's sound speed set to `c0` on a preset's configuration. */
const withC0 = (c0: number): Partial<SimConfig> => ({ eosParams: { ...defaultEosParams(), liqRho: 1, liqC: c0 } });

export const LIQUID_TESTS: TestDef[] = [
  {
    id: 'liquid-eos',
    name: 'Liquid equation of state on the GPU',
    criterion:
      'p = c0^2 (rho - rho0) for densities within 1e-6 to 0.1 of rho0 on either side, against the float64 mirror at the same single-precision density: the pressure to 1e-6 of rho0 c0^2 times the departure (it is one exact subtraction and one product), the sound speed exactly c0, the elastic energy to 1e-5; for c0 = 10 and 1000 and rho0 = 1 and 1.3',
    async run(gpu) {
      const d = gpu.device;
      const metrics: Record<string, number> = {};
      let pass = true;
      const pipe = await computePipeline(d, assemble('eos/eos_test.wgsl', { eos: EOS_MODULES.liquid }), 'eos test liquid');
      for (const [r0, c0] of [[1, 10], [1.3, 10], [1, 1000]]) {
        // The shader holds its two constants in single precision: so does the mirror here.
        const ep = { ...defaultEosParams(), liqRho: Math.fround(r0), liqC: Math.fround(c0) };
        const eos = makeEos('liquid', ep);
        const xs: number[] = [];
        for (let i = 0; i <= 20; i++) for (const sg of [-1, 1]) xs.push(sg * 10 ** (-6 + (5 * i) / 20));
        const inData = new Float32Array(xs.length * 4);
        xs.forEach((x, i) => (inData[i * 4] = r0 * (1 + x)));
        // The module takes (rho, p): give it the pressure it would itself form.
        xs.forEach((_, i) => (inData[i * 4 + 1] = Math.fround(c0 * c0) * (inData[i * 4] - Math.fround(r0))));
        const mk = (usage: number) => d.createBuffer({ size: inData.byteLength, usage });
        const inBuf = mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const outBuf = mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const readBuf = mk(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(inBuf, 0, inData);
        const pb = new Float32Array(PARAM_BYTES / 4);
        pb.set(eosConstants1('liquid', ep), 8);
        pb.set(eosConstants2('liquid', ep), 64);
        d.queue.writeBuffer(params, 0, pb);
        const enc = d.createCommandEncoder();
        const cp = enc.beginComputePass();
        cp.setPipeline(pipe);
        cp.setBindGroup(0, bindGroup(d, pipe, { 0: params, 1: inBuf, 2: outBuf }));
        cp.dispatchWorkgroups(Math.ceil(xs.length / 64));
        cp.end();
        enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, inData.byteLength);
        d.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        for (const b of [inBuf, outBuf, readBuf, params]) b.destroy();
        let dp = 0, de = 0, dc = 0;
        xs.forEach((_, i) => {
          const rho = inData[i * 4];
          const [eint, p, cs] = out.subarray(i * 4, i * 4 + 3);
          const want = eos.pressure(rho, 0);
          dp = Math.max(dp, Math.abs(p - want) / Math.abs(want));
          de = Math.max(de, Math.abs(eint - eos.eintFromP(rho, want)) / eos.eintFromP(rho, want));
          dc = Math.max(dc, Math.abs(cs - Math.fround(c0)));
        });
        const tag = `rho0_${r0}_c0_${c0}`;
        metrics[`${tag}_pressure`] = dp;
        metrics[`${tag}_elastic_energy`] = de;
        metrics[`${tag}_sound_speed`] = dc;
        pass &&= dp < 1e-6 && de < 1e-5 && dc === 0;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'liquid-gresho',
    name: 'Liquid: Gresho vortex at Mach 0.1, 0.03 and 0.01 (2D 64^2, uniform and on blocks)',
    criterion:
      'After t = 2 (1.6 turns) the liquid keeps as much kinetic energy as the ideal gas at the same Mach number with the low-Mach HLLC, to 0.002, and gains none; its density stays within 2 M^2 of the reference; a block grid without refinement keeps the uniform grid\'s kinetic energy to 1e-4 (Mach 0.1)',
    async run(gpu) {
      const n = 64;
      const m: Record<string, number> = {};
      let pass = true;
      const kept = async (sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        const steps = await sim.runToEnd(64, 1e7);
        const d1 = await sim.diagnostics();
        return { kept: d1.eKin / d0.eKin, steps, dev: Math.max(Math.abs(d1.ranges[0][0] - 1), Math.abs(d1.ranges[0][1] - 1)) };
      };
      for (const mach of [0.1, 0.03, 0.01]) {
        const liquid = await withSim(gpu, gresho, n, { ndim: 2, tEnd: 2, ...withC0(1 / mach) }, kept);
        const gas = await withSim(gpu, greshoGas(mach), n, { ndim: 2, tEnd: 2, riemann: 'hllclm' }, kept);
        m[`mach_${mach}_kept_liquid`] = liquid.kept;
        m[`mach_${mach}_kept_ideal_gas`] = gas.kept;
        m[`mach_${mach}_density_deviation_over_mach2`] = liquid.dev / (mach * mach);
        m[`mach_${mach}_steps`] = liquid.steps;
        pass &&= liquid.kept < 1 && liquid.kept > gas.kept - 0.002 && liquid.dev < 2 * mach * mach;
      }
      const blocks = await withAmr(gpu, gresho, n, { ndim: 2, tEnd: 2, amrLevels: 0 }, kept);
      m.blocks_kept_difference = Math.abs(blocks.kept - m['mach_0.1_kept_liquid']);
      pass &&= m.blocks_kept_difference < 1e-4;
      return { pass, metrics: m };
    },
  },
  {
    id: 'liquid-taylor-green',
    name: 'Liquid: viscous Taylor-Green vortex (2D, 64 cells per wavelength, Mach 0.1)',
    criterion:
      'The kinetic energy of the vortex array falls as exp(-4 nu k^2 t), nu = 0.005, k = 2 pi: its decay rate over 0 < t < 1 is within 1% of that, on the uniform grid and on a block grid without refinement, and the density stays within 2 M^2 of the reference',
    async run(gpu) {
      const tg = problemById('taylor-green');
      const m: Record<string, number> = {};
      let pass = true;
      const rate = async (sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd(64);
        const d1 = await sim.diagnostics();
        const k = 2 * Math.PI;
        return { ratio: -Math.log(d1.eKin / d0.eKin) / (4 * sim.cfg.viscosity * k * k * sim.cfg.tEnd), dev: Math.max(Math.abs(d1.ranges[0][0] - 1), Math.abs(d1.ranges[0][1] - 1)) };
      };
      const uni = await withSim(gpu, tg, 64, { ndim: 2, tEnd: 1 }, rate);
      const blk = await withAmr(gpu, tg, 64, { ndim: 2, tEnd: 1, amrLevels: 0 }, rate);
      m.decay_rate_over_exact = uni.ratio;
      m.decay_rate_over_exact_blocks = blk.ratio;
      m.density_deviation = uni.dev;
      pass &&= Math.abs(uni.ratio - 1) < 0.01 && Math.abs(blk.ratio - 1) < 0.01 && uni.dev < 0.02;
      return { pass, metrics: m };
    },
  },
  {
    id: 'liquid-cavity',
    name: 'Liquid: lid-driven cavity at Reynolds number 100 (2D 128^2, Mach 0.1)',
    criterion:
      'The liquid cavity preset with viscosity 0.01 settles by t = 12 to a vertical-centerline u within 0.015 of the lid speed of Ghia, Ghia & Shin (1982) at every tabulated height (the ideal gas preset\'s criterion), with the density within 5% of the reference (the lid\'s corners are singular)',
    async run(gpu) {
      const n = 128;
      return withSim(gpu, problemById('lid-driven-cavity-liquid'), n, { ndim: 2, tEnd: 12, viscosity: 0.01 }, async (sim) => {
        await sim.runToEnd(64, 1e7);
        const { u1 } = await sim.readState();
        const d = await sim.diagnostics();
        const c = cavityError(u1, n, GHIA_RE100);
        const dev = Math.max(Math.abs(d.ranges[0][0] - 1), Math.abs(d.ranges[0][1] - 1));
        return { pass: c.error < 0.015 && dev < 0.05, metrics: { cavity_max_error: c.error, cavity_min_u: c.minU, density_deviation: dev, steps: d.steps } };
      });
    },
  },
  {
    id: 'liquid-amr',
    name: 'Liquid on the adaptive mesh: a Gresho vortex across a refinement boundary (2D, 256 finest, 2 levels)',
    criterion:
      'The mesh starts with three levels, the boundaries between them running through the vortex, and no block is merged (the refinement may only grow). After t = 1 (6800 steps): mass conserved to 2e-6, each momentum component to 1e-6 of the sum of |momentum|, and the vortex keeps more than 99% of its kinetic energy and gains none',
    async run(gpu) {
      return withAmr(gpu, gresho, 256, { ndim: 2, tEnd: 1, amrLevels: 2, refineLo: 0 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const levels = new Set(sim.tree.leaves().map((b) => b.level));
        const cells = await sim.readLeaves();
        let scale = 0;
        for (let q = 0; q < cells.count; q++) scale += Math.hypot(cells.u1[q * 4 + 1], cells.u1[q * 4 + 2]) * cells.dx[q] ** 2;
        await sim.runToEnd(64, 1e7);
        const d1 = await sim.diagnostics();
        const m = {
          leaf_levels: levels.size,
          mass_drift: Math.abs(d1.mass / d0.mass - 1),
          momentum_drift: Math.max(Math.abs(d1.momentum[0] - d0.momentum[0]), Math.abs(d1.momentum[1] - d0.momentum[1])) / scale,
          kinetic_energy_kept: d1.eKin / d0.eKin,
          steps: d1.steps,
          leaf_levels_at_end: new Set(sim.tree.leaves().map((b) => b.level)).size,
        };
        return { pass: m.leaf_levels === 3 && m.leaf_levels_at_end === 3 && m.mass_drift < 2e-6 && m.momentum_drift < 1e-6 && m.kinetic_energy_kept > 0.99 && m.kinetic_energy_kept < 1, metrics: m };
      });
    },
  },
];
