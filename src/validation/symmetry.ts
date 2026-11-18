// Validation tests for mirror symmetry (plans/coordinate-systems-plan.md, Part B): a run that
// computes one octant (or a 2D quadrant) of a mirror-symmetric problem against the whole of it
// at the same cell size. Registered through validation/frames.ts.
import type { Gpu } from '../gpu/device';
import { all, OUTFLOW, type Problem } from '../problems/types';
import { evrard, polytrope, sedov, sphere, spherePotential } from '../presets/names';
import type { Sim, SimConfig } from '../sim';
import type { AmrSim } from '../amr/amr_sim';
import { withAmr, withSim, type TestDef } from './tests';
import { compileSetup } from '../setup/compile';

const MIRRORS: [boolean, boolean, boolean] = [true, true, true];

/**
 * The problem, allowed every mirror, in a box `scale` times its own. A preset paints its initial
 * state from its setup document, whose box is scaled with it.
 */
function mirrored(p: Problem, scale = 1): Problem {
  const box = p.box * scale;
  const q = p.setup ? compileSetup({ ...p.setup, domain: { ...p.setup.domain, box } }).problem : { ...p, box };
  return { ...q, symmetry: { axes: MIRRORS } };
}

/** Largest relative difference between an octant's (quadrant's) cells and the same cells of the whole. */
function compareCells(part: Float32Array, whole: Float32Array, n: number, ndim: 2 | 3, stride: number, channel: number): number {
  const N = 2 * n;
  const nz = ndim === 3 ? n : 1;
  const NZ = ndim === 3 ? N : 1;
  const kz = ndim === 3 ? n : 0;
  let worst = 0;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const a = part[((k * n + j) * n + i) * stride + channel];
        const b = whole[(((k + kz) * N + (j + n)) * N + (i + n)) * stride + channel];
        worst = Math.max(worst, Math.abs(a - b) / Math.max(Math.abs(b), 1e-30));
      }
  void NZ;
  return worst;
}

/** Potential of the whole box and of its octant after a solve, and their worst errors against `exact(r)`. */
async function potentials(gpu: Gpu, p: Problem, n: number, exact?: (r: number) => number) {
  const solve = async (sim: Sim) => {
    sim.solvePotential(10);
    return sim.readPotential();
  };
  const whole = await withSim(gpu, p, 2 * n, {}, solve);
  const part = await withSim(gpu, mirrored(p, 0.5), n, { symmetry: MIRRORS }, solve);
  const m: Record<string, number> = { octant_vs_whole: compareCells(part, whole, n, 3, 1, 0) };
  if (exact) {
    const dx = p.box / (2 * n);
    let eo = 0;
    let ew = 0;
    for (let k = 0; k < n; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const e = exact(Math.hypot(i + 0.5, j + 0.5, k + 0.5) * dx);
          eo = Math.max(eo, Math.abs(part[(k * n + j) * n + i] - e) / Math.abs(e));
          ew = Math.max(ew, Math.abs(whole[((k + n) * 2 * n + j + n) * 2 * n + i + n] - e) / Math.abs(e));
        }
    m.octant_error = eo;
    m.whole_error = ew;
  }
  return m;
}

/** A uniform triaxial ellipsoid of semi-axes (0.3, 0.2, 0.12), density 1, in near-empty space. */
const ellipsoid: Problem = {
  ...sphere,
  setup: undefined,
  id: 'ellipsoid',
  init: () => (x, y, z) => {
    const inside = (x / 0.3) ** 2 + (y / 0.2) ** 2 + (z / 0.12) ** 2 < 1;
    return { rho: inside ? 1 : 1e-6, vx: 0, vy: 0, vz: 0, p: 1e-4, s: 0 };
  },
};

/** Energy history of a self-gravitating run, in `chunks` stops to `tEnd`. */
async function energyDrift(sim: Sim | AmrSim, tEnd: number, chunks: number) {
  const d0 = await sim.diagnostics();
  let worst = 0;
  let peakT = 0;
  let peakU = 0;
  for (let i = 1; i <= chunks; i++) {
    sim.cfg.tEnd = (tEnd * i) / chunks;
    sim.writeParams();
    await sim.runToEnd();
    const d = await sim.diagnostics();
    worst = Math.max(worst, Math.abs(d.eTot - d0.eTot) / Math.abs(d0.eTot));
    if (d.eInt > peakU) [peakU, peakT] = [d.eInt, d.t];
  }
  return { worst, peakT, e0: d0.eTot, mass: d0.mass };
}

export const SYM_TESTS: TestDef[] = [
  {
    id: 'sym-flux',
    name: 'Mirror symmetry: nothing crosses a plane of symmetry',
    criterion: 'A Sedov blast in an octant (32^3): after some steps the mass flux through each plane of symmetry is zero to the bit, and both cells of every face compute the same flux',
    async run(gpu) {
      return withSim(gpu, mirrored(sedov, 0.5), 32, { symmetry: MIRRORS, tEnd: 0.005 }, async (sim) => {
        await sim.runToEnd();
        const f = await sim.fluxMismatch();
        return { pass: f.mirror === 0 && f.mismatched === 0, metrics: { mirror_flux_max: f.mirror, faces: f.faces, mismatched: f.mismatched } };
      });
    },
  },
  {
    id: 'sym-sedov',
    name: 'Mirror symmetry: a Sedov blast in an octant and a 2D quadrant against the whole box',
    criterion: 'At the same cell size, the octant (32^3, box 0.5) and the whole box (64^3, box 1) agree in density to 1e-5 at every cell of the octant; the same for a 2D quadrant (64^2 against 128^2)',
    async run(gpu) {
      const run = (p: Problem, n: number, over: Partial<SimConfig>) =>
        withSim(gpu, p, n, over, async (sim) => {
          await sim.runToEnd();
          const s = await sim.readState();
          return { u1: s.u1, mass: (await sim.diagnostics()).mass };
        });
      const w3 = await run(sedov, 64, {});
      const o3 = await run(mirrored(sedov, 0.5), 32, { symmetry: MIRRORS });
      const w2 = await run(sedov, 128, { ndim: 2 });
      const o2 = await run(mirrored(sedov, 0.5), 64, { ndim: 2, symmetry: MIRRORS });
      const m = {
        octant_density_difference: compareCells(o3.u1, w3.u1, 32, 3, 4, 0),
        quadrant_density_difference: compareCells(o2.u1, w2.u1, 64, 2, 4, 0),
        octant_mass_unfolded: o3.mass, whole_mass: w3.mass,
      };
      return { pass: m.octant_density_difference < 1e-5 && m.quadrant_density_difference < 1e-5 && Math.abs(o3.mass / w3.mass - 1) < 1e-5, metrics: m };
    },
  },
  {
    id: 'sym-sphere',
    name: 'Mirror symmetry: potential of a uniform sphere in an octant',
    criterion: 'Isolated boundaries, the octant at 32^3 against the whole box at 64^3: the octant potential is within 1% of the analytic one everywhere (as the whole box is) and within 1e-4 of the whole box',
    async run(gpu) {
      const m = await potentials(gpu, sphere, 32, spherePotential);
      return { pass: m.octant_error < 0.01 && m.octant_vs_whole < 1e-4, metrics: m };
    },
  },
  {
    id: 'sym-ellipsoid',
    name: 'Mirror symmetry: potential of a triaxial ellipsoid in an octant',
    criterion: 'A uniform ellipsoid of semi-axes 0.3, 0.2 and 0.12 (isolated boundaries, its quadrupole unfolded from the octant): the octant at 32^3 is within 1e-4 of the whole box at 64^3',
    async run(gpu) {
      const m = await potentials(gpu, ellipsoid, 32);
      return { pass: m.octant_vs_whole < 1e-4, metrics: m };
    },
  },
  {
    id: 'sym-polytrope',
    name: 'Mirror symmetry: a settled polytrope in an octant',
    criterion: 'The n = 1 polytrope, settled, in an octant (32^3, box 0.5) and in the whole box (64^3): over 3 dynamical times the central density stays within 5% in both, and the two deviations agree to 1%',
    async run(gpu) {
      const run = (p: Problem, n: number, over: Partial<SimConfig>) =>
        withSim(gpu, p, n, { ...over, tEnd: 0 }, async (sim) => {
          const rho0 = (await sim.diagnostics()).ranges[0][1];
          let worst = 0;
          for (let t = 1; t <= 3; t++) {
            sim.cfg.tEnd = t;
            sim.writeParams();
            await sim.runToEnd();
            worst = Math.max(worst, Math.abs((await sim.diagnostics()).ranges[0][1] - rho0) / rho0);
          }
          return { rho0, worst };
        });
      const w = await run(polytrope, 64, {});
      const o = await run(mirrored(polytrope, 0.5), 32, { symmetry: MIRRORS });
      return {
        pass: o.worst < 0.05 && w.worst < 0.05 && Math.abs(o.worst - w.worst) < 0.01,
        metrics: { octant_central_density: o.rho0, whole_central_density: w.rho0, octant_max_deviation: o.worst, whole_max_deviation: w.worst },
      };
    },
  },
  {
    id: 'sym-evrard',
    name: 'Mirror symmetry: the Evrard collapse in an octant',
    criterion: 'Through collapse and bounce (t <= 1.5) the octant (32^3, box 2) conserves total energy to 2% and its thermal energy peaks between t = 0.9 and 1.3, as the whole box does (64^3); on blocks with per-block timesteps (octant, 64^3 finest, one level) too',
    async run(gpu) {
      const w = await withSim(gpu, evrard, 64, { tEnd: 0 }, (sim) => energyDrift(sim, 1.5, 6));
      const o = await withSim(gpu, mirrored(evrard, 0.5), 32, { symmetry: MIRRORS, tEnd: 0 }, (sim) => energyDrift(sim, 1.5, 6));
      const b = await withAmr(gpu, mirrored(evrard, 0.5), 64, { symmetry: MIRRORS, tEnd: 0, amrLevels: 1, timeBins: 2, timeBinsAlways: true }, (sim) => energyDrift(sim, 1.5, 6));
      const ok = (r: { worst: number; peakT: number }) => r.worst < 0.02 && r.peakT >= 0.9 && r.peakT <= 1.3;
      return {
        pass: ok(w) && ok(o) && ok(b) && Math.abs(o.e0 / w.e0 - 1) < 0.01,
        metrics: {
          whole_drift: w.worst, octant_drift: o.worst, blocks_drift: b.worst, whole_peak: w.peakT, octant_peak: o.peakT, blocks_peak: b.peakT,
          whole_energy: w.e0, octant_energy_unfolded: o.e0, octant_mass_unfolded: o.mass, whole_mass: w.mass,
        },
      };
    },
  },
  {
    id: 'sym-mass',
    name: 'Mirror symmetry: a sink at the origin of an octant',
    criterion: 'A point mass with a sink at the center of uniform gas at rest: what it takes in by t = 0.1, computed in an octant (32^3, box 0.5), equals what the whole box (64^3) gives to 1e-5',
    async run(gpu) {
      const p: Problem = {
        ...sedov, setup: undefined, id: 'sink', box: 1, tEnd: 0.1, bcLo: all(OUTFLOW), bcHi: all(OUTFLOW), gravity: 'off', G: 1,
        init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }),
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.05, sinkRadius: 0.06 }],
      };
      const take = (sim: Sim) => sim.runToEnd().then(() => sim.diagnostics()).then((d) => ({ acc: d.masses[0].accreted, gm: d.masses[0].gm }));
      const w = await withSim(gpu, p, 64, {}, take);
      const o = await withSim(gpu, mirrored(p, 0.5), 32, { symmetry: MIRRORS }, take);
      const diff = Math.abs(o.acc / w.acc - 1);
      return { pass: diff < 1e-5 && w.acc > 0, metrics: { whole_accreted: w.acc, octant_accreted: o.acc, relative_difference: diff, whole_gm: w.gm, octant_gm: o.gm } };
    },
  },
];
