// Validation tests for comoving coordinates (plans/coordinate-systems-plan.md, Part C).
// Registered through validation/frames.ts.
import { all, PERIODIC, type Problem } from '../problems/types';
import { cosmoG, edsScaleFactor, hubble, rk4, scaleFactor, timeOf, type Cosmology } from '../cosmo';
import { coolRate, defaultCoolingParams } from '../cooling/cooling';
import { defaultEosParams, makeEos } from '../eos/eos';
import type { Sim, SimConfig } from '../sim';
import type { AmrSim } from '../amr/amr_sim';
import { withAmr, withSim, type TestDef } from './tests';

const EDS: Cosmology = { omegaM: 1, omegaL: 0, aStart: 0.01 };
const LCDM: Cosmology = { omegaM: 0.3, omegaL: 0.7, aStart: 0.01 };

/** A periodic box of ideal gas in an expanding background, from `init`, run to the scale factor aEnd. */
function universe(k: Cosmology, aEnd: number, init: Problem['init'], over: Partial<Problem> = {}): Problem {
  return {
    id: 'universe', label: 'Universe', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
    bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: cosmoG(k), box: 1, tEnd: timeOf(k, aEnd), supports2d: true,
    cosmology: k, rhoFloor: 1e-12, pFloor: 1e-20, init, ...over,
  };
}

/** Steps to the scale factors in `as`, one stop each, calling `at` at each. */
async function through(sim: Sim | AmrSim, k: Cosmology, as: number[], at: (a: number) => Promise<void>) {
  for (const a of as) {
    sim.cfg.tEnd = timeOf(k, a);
    sim.writeParams();
    await sim.runToEnd();
    await at(a);
  }
}

/**
 * The Zel'dovich pancake in Einstein-de Sitter (Bryan et al. 1995, Zel'dovich 1970): one plane
 * wave of wavenumber k along x whose Lagrangian displacement reaches a caustic at a = aCaustic.
 * With D = a / aCaustic, gas from comoving q sits at x = q - D sin(kq) / k, with comoving
 * density 1 / (1 - D cos(kq)) and proper peculiar velocity -a H D sin(kq) / k.
 */
const pancake = (aCaustic: number) => {
  const k = (2 * Math.PI) / 1;
  const q = (x: number, D: number) => {
    // Solve x = q - D sin(kq) / k for q by Newton's method (D < 1 keeps it monotonic).
    let s = x;
    for (let i = 0; i < 50; i++) s -= (s - (D * Math.sin(k * s)) / k - x) / (1 - D * Math.cos(k * s));
    return s;
  };
  const exact = (x: number, a: number) => {
    const D = a / aCaustic;
    const s = q(x, D);
    const H = Math.pow(a, -1.5);
    return { rho: 1 / (1 - D * Math.cos(k * s)), v: (-a * H * D * Math.sin(k * s)) / k };
  };
  return { exact };
};

export const COSMO_TESTS: TestDef[] = [
  {
    id: 'cosmo-a',
    name: 'Comoving coordinates: the scale factor on the GPU',
    criterion: 'From a = 0.01 to 1, the scale factor the clock pass advances in float32 (one RK4 step a step, at most 1% each) is within 1e-5 of float64, for Einstein-de Sitter (against its closed form) and for a flat Lambda model',
    async run(gpu) {
      const m: Record<string, number> = {};
      for (const [name, k] of [['eds', EDS], ['lcdm', LCDM]] as const) {
        const p = universe(k, 1, () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1e-6, s: 0 }));
        await withSim(gpu, p, 16, { ndim: 2 }, async (sim) => {
          let worst = 0;
          await through(sim, k, [0.03, 0.1, 0.3, 1], async () => {
            const d = await sim.diagnostics();
            const want = name === 'eds' ? edsScaleFactor(k.aStart, d.t) : scaleFactor(k, d.t);
            worst = Math.max(worst, Math.abs(d.scale! / want - 1));
            m[`${name}_final_a`] = d.scale!;
            m[`${name}_steps`] = d.steps;
          });
          m[`${name}_max_error`] = worst;
        });
      }
      return { pass: m.eds_max_error < 1e-5 && m.lcdm_max_error < 1e-5 && Math.abs(m.eds_final_a - 1) < 1e-4, metrics: m };
    },
  },
  {
    id: 'cosmo-drift',
    name: 'Comoving coordinates: the expansion drag on uniform gas',
    criterion: 'Uniform gas moving at 0.01 in a periodic box (Einstein-de Sitter, a from 0.01 to 0.1): the peculiar velocity goes as 1/a and the comoving pressure as a^(3 - 3 gamma) to 1e-5, on the uniform grid and on blocks. (Its internal energy stays comparable to its kinetic energy: in much colder gas the dual-energy scheme takes it from E - KE, which float32 holds to some 1e-4 after a few hundred steps.)',
    async run(gpu) {
      const p0 = 1e-4;
      const v0 = 0.01;
      const p = universe(EDS, 0.1, () => () => ({ rho: 1, vx: v0, vy: 0, vz: 0, p: p0, s: 0 }));
      const m: Record<string, number> = {};
      const check = async (sim: Sim | AmrSim, tag: string) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const a = d.scale!;
        const v = d.momentum[0] / d.mass;
        m[`${tag}_velocity_error`] = Math.abs(v / ((v0 * EDS.aStart) / a) - 1);
        // The internal energy density rho e the state holds (E - KE, as the diagnostics sum it,
        // has lost the digits a gas this much colder than its motion needs): p = (gamma - 1) rho e.
        const u2 = tag === 'uniform' ? (await (sim as Sim).readState()).u2 : (await (sim as AmrSim).readLeaves()).u2;
        let ei = 0;
        for (let q = 0; q < u2.length; q += 4) ei += u2[q + 1];
        const pNow = (ei / (u2.length / 4)) * (2 / 3);
        m[`${tag}_pressure_error`] = Math.abs(pNow / (p0 * Math.pow(a / EDS.aStart, 3 - 5)) - 1);
        m[`${tag}_a`] = a;
      };
      await withSim(gpu, p, 16, { ndim: 2 }, (sim) => check(sim, 'uniform'));
      await withAmr(gpu, p, 32, { ndim: 2, amrLevels: 0 }, (sim) => check(sim, 'blocks'));
      const pass = ['uniform', 'blocks'].every((t) => m[`${t}_velocity_error`] < 1e-5 && m[`${t}_pressure_error`] < 1e-5);
      return { pass, metrics: m };
    },
  },
  {
    id: 'cosmo-cool',
    name: 'Comoving coordinates: uniform gas cooling as it expands',
    criterion: 'Uniform gas at rest (Einstein-de Sitter, a from 0.1 to 0.3) with power-law cooling at the proper density (Lambda T^0.5) and the expansion’s adiabatic cooling: its internal energy at the end is within 1e-3 of a float64 integration of d(rho e)/dt = -3 (gamma - 1) H rho e + a^3 Lambda(rho / a^3, T), on the uniform grid and on blocks',
    async run(gpu) {
      const k: Cosmology = { omegaM: 1, omegaL: 0, aStart: 0.1 };
      const coolParams = { ...defaultCoolingParams(), rate: 5e-4, alpha: 0.5, heat: 0, tFloor: 0 };
      const gamma = 5 / 3;
      const p0 = 1e-3;
      const p = universe(k, 0.3, () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: p0, s: 0 }), { cooling: 'power', coolParams, eosParams: { gamma } });
      // The float64 reference, in steps of a 1e-4 change in a.
      const eos = makeEos('ideal', { ...defaultEosParams(), gamma });
      let a = k.aStart;
      let e = p0 / (gamma - 1);
      let t = 0;
      const tEnd = timeOf(k, 0.3);
      const rate = (aa: number, ee: number) => -3 * (gamma - 1) * hubble(k, aa) * ee + aa ** 3 * coolRate('power', coolParams, 1 / aa ** 3, eos.temperature(1, ee));
      while (t < tEnd) {
        const h = Math.min(tEnd - t, 1e-4 / hubble(k, a));
        const ah = rk4(k, a, 0.5 * h);
        const a1 = rk4(k, a, h);
        const k1 = rate(a, e);
        const k2 = rate(ah, e + 0.5 * h * k1);
        const k3 = rate(ah, e + 0.5 * h * k2);
        const k4 = rate(a1, e + h * k3);
        e += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        a = a1;
        t += h;
      }
      const m: Record<string, number> = { reference: e, adiabatic_only: (p0 / (gamma - 1)) * (k.aStart / 0.3) ** 2 };
      const check = async (sim: Sim | AmrSim, tag: string) => {
        await sim.runToEnd();
        const u2 = tag === 'uniform' ? (await (sim as Sim).readState()).u2 : (await (sim as AmrSim).readLeaves()).u2;
        let s = 0;
        for (let q = 0; q < u2.length; q += 4) s += u2[q + 1];
        m[tag] = s / (u2.length / 4);
        m[`${tag}_error`] = Math.abs(m[tag] / e - 1);
      };
      await withSim(gpu, p, 16, { ndim: 2 }, (sim) => check(sim, 'uniform'));
      await withAmr(gpu, p, 32, { ndim: 2, amrLevels: 0 }, (sim) => check(sim, 'blocks'));
      return { pass: m.uniform_error < 1e-3 && m.blocks_error < 1e-3, metrics: m };
    },
  },
  {
    id: 'cosmo-growth',
    name: 'Comoving coordinates: linear growth of a density wave',
    criterion: 'A cold plane wave of 0.2% in density, on its growing mode (Einstein-de Sitter, self-gravity, 32^3): from a = 0.01 to 0.1 its amplitude grows as a, within 1%',
    async run(gpu) {
      const d0 = 0.002;
      const kx = 2 * Math.PI;
      const a0 = EDS.aStart;
      const H0 = Math.pow(a0, -1.5);
      const p = universe(EDS, 0.1, () => (x) => ({
        rho: 1 + d0 * Math.cos(kx * x), vx: (-a0 * H0 * d0 * Math.sin(kx * x)) / kx, vy: 0, vz: 0, p: 1e-10, s: 0,
      }), { gravity: 'periodic', supports2d: false });
      return withSim(gpu, p, 32, {}, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const [lo, hi] = d.ranges[0];
        const amp = (hi - lo) / 2;
        const growth = amp / d0 / (d.scale! / a0);
        return { pass: Math.abs(growth - 1) < 0.01, metrics: { amplitude: amp, growth_over_linear: growth, a: d.scale!, steps: d.steps } };
      });
    },
  },
  {
    id: 'cosmo-pancake',
    name: 'Comoving coordinates: the Zel’dovich pancake',
    criterion: 'A plane wave that reaches its caustic at a = 0.1 (Einstein-de Sitter, self-gravity, cold gas, 64^3), started at a = 0.01 on the exact solution: at a = 0.05 the density is within 2% and the velocity within 2% of the largest of the exact solution at every cell; on blocks (64^3 finest, one level, the rest at 32^3) within 5% and 2%',
    async run(gpu) {
      const aC = 0.1;
      const pc = pancake(aC);
      const p = universe(EDS, 0.05, () => (x) => {
        const e = pc.exact(x, EDS.aStart);
        return { rho: e.rho, vx: e.v, vy: 0, vz: 0, p: 1e-12, s: 0 };
      }, { gravity: 'periodic', supports2d: false });
      const m: Record<string, number> = {};
      const check = async (sim: Sim | AmrSim, tag: string, n: number) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const a = d.scale!;
        const vmax = Math.abs(pc.exact(0.25, a).v) * 1.5;
        let er = 0;
        let ev = 0;
        if (tag === 'uniform') {
          const { u1 } = await (sim as Sim).readState();
          for (let i = 0; i < n; i++) {
            const x = (i + 0.5) / n - 0.5;
            const e = pc.exact(x, a);
            const o = 4 * i;
            er = Math.max(er, Math.abs(u1[o] / e.rho - 1));
            ev = Math.max(ev, Math.abs(u1[o + 1] / u1[o] - e.v) / vmax);
          }
        } else {
          const c = await (sim as AmrSim).readLeaves();
          for (let q = 0; q < c.count; q++) {
            const e = pc.exact(c.x[q], a);
            er = Math.max(er, Math.abs(c.u1[4 * q] / e.rho - 1));
            ev = Math.max(ev, Math.abs(c.u1[4 * q + 1] / c.u1[4 * q] - e.v) / vmax);
          }
        }
        m[`${tag}_density_error`] = er;
        m[`${tag}_velocity_error`] = ev;
        m[`${tag}_a`] = a;
      };
      await withSim(gpu, p, 64, {}, (sim) => check(sim, 'uniform', 64));
      await withAmr(gpu, p, 64, { amrLevels: 1, jeansRefine: 0 } as Partial<SimConfig>, (sim) => check(sim, 'blocks', 64));
      const pass = m.uniform_density_error < 0.02 && m.uniform_velocity_error < 0.02 && m.blocks_density_error < 0.05 && m.blocks_velocity_error < 0.02;
      return { pass, metrics: m };
    },
  },
];
