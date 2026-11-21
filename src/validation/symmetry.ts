// Validation tests for mirror symmetry (docs/numerics.md section 22): a run that
// computes one octant (or a 2D quadrant) of a mirror-symmetric problem against the whole of it
// at the same cell size. Registered through validation/frames.ts.
import type { Gpu } from '../gpu/device';
import { all, OUTFLOW, type Problem } from '../problems/types';
import { evrard, polytrope, sedov, sphere, spherePotential } from '../presets/names';
import type { Sim, SimConfig } from '../sim';
import type { AmrSim } from '../amr/amr_sim';
import { withAmr, withSim, type TestDef } from './tests';
import { compileSetup } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';

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

/**
 * A magnetized blob symmetric under all three mirrors: the field of the vector potential
 * A = (x y^2 f(r), 0, 0), f a Gaussian, which is odd in its normal component and even in the
 * others across every plane (a vector's parity, so that B = curl A has a pseudovector's).
 */
const magBlob: Problem = {
  id: 'mag-blob', label: 'Magnetized blob', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW), bcHi: all(OUTFLOW), gravity: 'off', G: 0, box: 1, tEnd: 0.05,
  init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }),
  mhd: { potential: () => (x, y, z) => [x * y * y * 40 * Math.exp(-(x * x + y * y + z * z) / 0.02), 0, 0] },
};

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
    id: 'sym-half',
    name: 'Mirror symmetry: half and quadrant runs on rectangular grids',
    criterion: 'At the same box and cell size as the whole (64^3), a run mirrored about z alone computes 64 x 64 x 32 cells and one mirrored about x and y 32 x 32 x 64 (docs/numerics.md section 22): a Sedov blast agrees with the whole box in density to 1e-5 at every computed cell, with the same mass unfolded; and a uniform sphere’s potential (isolated boundaries, the rectangular multigrid) within 1e-4 of the whole box’s; and on blocks (two levels), the half has half the leaves and the same unfolded mass (1e-5), total energy (1e-4) and kinetic energy (1e-3) as the whole',
    async run(gpu) {
      const N = 64;
      const sedovAt = (sym: [boolean, boolean, boolean]) => ({ ...(sedov.setup ? compileSetup(sedov.setup).problem : sedov), symmetry: { axes: sym } });
      const compare = (part: Float32Array, whole: Float32Array, sym: boolean[], stride: number) => {
        const d = sym.map((m) => (m ? N / 2 : N));
        let worst = 0;
        for (let k = 0; k < d[2]; k++)
          for (let j = 0; j < d[1]; j++)
            for (let i = 0; i < d[0]; i++) {
              const a = part[((k * d[1] + j) * d[0] + i) * stride];
              const q = [i, j, k].map((c, x) => c + (sym[x] ? N / 2 : 0));
              const b = whole[((q[2] * N + q[1]) * N + q[0]) * stride];
              worst = Math.max(worst, Math.abs(a - b) / Math.max(Math.abs(b), 1e-30));
            }
        return worst;
      };
      const m: Record<string, number> = {};
      const blast = (sym: [boolean, boolean, boolean]) =>
        withSim(gpu, sedovAt(sym), N, sym.some(Boolean) ? { symmetry: sym } : {}, async (sim) => {
          await sim.runToEnd();
          return { u1: (await sim.readState()).u1, mass: (await sim.diagnostics()).mass, dims: sim.displayDims };
        });
      const whole = await blast([false, false, false]);
      for (const [name, sym] of [['half', [false, false, true]], ['quadrant', [true, true, false]]] as const) {
        const r = await blast([...sym] as [boolean, boolean, boolean]);
        m[`${name}_cells`] = r.dims[0] * r.dims[1] * r.dims[2];
        m[`${name}_density_difference`] = compare(r.u1, whole.u1, [...sym], 4);
        m[`${name}_mass_difference`] = Math.abs(r.mass / whole.mass - 1);
      }
      const solve = async (sim: Sim) => {
        sim.solvePotential(10);
        return sim.readPotential();
      };
      const wp = await withSim(gpu, sphere, N, {}, solve);
      for (const [name, sym] of [['half', [false, false, true]], ['quadrant', [true, true, false]]] as const) {
        const ax = [...sym] as [boolean, boolean, boolean];
        const pp = await withSim(gpu, { ...sphere, symmetry: { axes: ax } }, N, { symmetry: ax }, solve);
        m[`${name}_potential_difference`] = compare(pp, wp, [...sym], 1);
      }
      // On blocks (two levels, 2^3 root blocks of the whole): the half has half the root blocks
      // along z; its unfolded totals match the whole's.
      const onBlocks = (sym: [boolean, boolean, boolean]) =>
        withAmr(gpu, sedovAt(sym), N, { amrLevels: 1, ...(sym.some(Boolean) ? { symmetry: sym } : {}) }, async (sim) => {
          await sim.runToEnd();
          const d = await sim.diagnostics();
          return { mass: d.mass, e: d.eTot, ek: d.eKin, leaves: sim.tree.leaves().length };
        });
      const bw = await onBlocks([false, false, false]);
      const bh = await onBlocks([false, false, true]);
      m.blocks_mass_difference = Math.abs(bh.mass / bw.mass - 1);
      m.blocks_energy_difference = Math.abs(bh.e / bw.e - 1);
      m.blocks_kinetic_difference = Math.abs(bh.ek / bw.ek - 1);
      m.blocks_leaves_whole = bw.leaves;
      m.blocks_leaves_half = bh.leaves;
      const pass = ['half', 'quadrant'].every((k) => m[`${k}_density_difference`] < 1e-5 && m[`${k}_mass_difference`] < 1e-5 && m[`${k}_potential_difference`] < 1e-4)
        && m.half_cells === N * N * N / 2 && m.quadrant_cells === N * N * N / 4
        && m.blocks_mass_difference < 1e-5 && m.blocks_energy_difference < 1e-4 && m.blocks_kinetic_difference < 1e-3 && 2 * bh.leaves === bw.leaves;
      return { pass, metrics: m };
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
    criterion: 'Through collapse and bounce (t <= 1.5) the octant (32^3, box 2) conserves total energy to 0.6% and its thermal energy peaks between t = 0.9 and 1.3, as the whole box does (64^3); on blocks with per-block timesteps (octant, 64^3 finest, one level) too',
    async run(gpu) {
      const w = await withSim(gpu, evrard, 64, { tEnd: 0 }, (sim) => energyDrift(sim, 1.5, 6));
      const o = await withSim(gpu, mirrored(evrard, 0.5), 32, { symmetry: MIRRORS, tEnd: 0 }, (sim) => energyDrift(sim, 1.5, 6));
      const b = await withAmr(gpu, mirrored(evrard, 0.5), 64, { symmetry: MIRRORS, tEnd: 0, amrLevels: 1, timeBins: 2, timeBinsAlways: true }, (sim) => energyDrift(sim, 1.5, 6));
      const ok = (r: { worst: number; peakT: number }) => r.worst < 0.006 && r.peakT >= 0.9 && r.peakT <= 1.3;
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
    id: 'sym-mhd',
    name: 'Mirror symmetry: a magnetic field symmetric under the mirrors',
    criterion: 'The field of A = (x y^2 f(r), 0, 0) pushing gas about: the octant (32^3, box 0.5) against the whole box (64^3) at t = 0.05, density within 1e-4 at every cell of the octant, magnetic energy within 1e-5 (1e-4 for the octant on unrefined blocks), and div B at round-off in all three',
    async run(gpu) {
      const run = (p: Problem, n: number, over: Partial<SimConfig>) =>
        withSim(gpu, p, n, over, async (sim) => {
          await sim.runToEnd();
          const d = await sim.diagnostics();
          return { u1: (await sim.readState()).u1, eMag: d.magnetic!.eMag, divB: d.magnetic!.divB };
        });
      const w = await run(magBlob, 64, {});
      const o = await run({ ...mirrored(magBlob, 0.5), symmetry: { axes: MIRRORS, mhd: true } }, 32, { symmetry: MIRRORS });
      // The octant on blocks too (unrefined, so its cells are the uniform octant's).
      const b = await withAmr(gpu, { ...mirrored(magBlob, 0.5), symmetry: { axes: MIRRORS, mhd: true } }, 32, { symmetry: MIRRORS, amrLevels: 0 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        return { eMag: d.magnetic!.eMag, divB: d.magnetic!.divB };
      });
      const m = {
        density_difference: compareCells(o.u1, w.u1, 32, 3, 4, 0),
        magnetic_energy_difference: Math.abs(o.eMag / w.eMag - 1),
        blocks_magnetic_energy_difference: Math.abs(b.eMag / w.eMag - 1),
        div_b_octant: o.divB, div_b_whole: w.divB, div_b_blocks: b.divB, magnetic_energy: w.eMag,
      };
      return {
        pass: m.density_difference < 1e-4 && m.magnetic_energy_difference < 1e-5 && m.blocks_magnetic_energy_difference < 1e-4 && o.divB < 1e-5 && w.divB < 1e-5 && b.divB < 1e-5,
        metrics: m,
      };
    },
  },
  {
    id: 'sym-stars',
    name: 'Mirror symmetry: two stars in the plane z = 0, settled and falling together, against the whole box',
    criterion: 'Two polytropes (n = 1.5, masses 0.1 and 0.06) at rest on the x axis with isolated self-gravity, settled and run to t = 0.05: computed above z = 0 only (64 x 64 x 32) and whole (64^3) at the same cell size, the settled stars have the same mass to 1e-5 (each settled with the whole star’s pull on the other) and the same peak density to 1e-4, and at the end the density agrees to 2e-3 (mass-weighted) and the total energy to 1e-4; on blocks of one level (128^3 finest), mass and energy agree to 1e-4 and the peak density to 1e-3',
    async run(gpu) {
      const MZ: [boolean, boolean, boolean] = [false, false, true];
      const doc = (symmetric: boolean): SetupSpec => ({
        format: 'shaderous-setup', version: 1,
        domain: { ndim: 3, box: 1, bc: { lo: [1, 1, 1], hi: [1, 1, 1] }, ...(symmetric ? { symmetric: 'z' } : {}) },
        physics: { eos: 'ideal', eosParams: { gamma: 5 / 3 }, gravity: 'isolated', G: 1 },
        background: { rho: 1e-6, p: 1e-9 },
        parts: [
          { kind: 'star', id: 'a', type: 'polytrope', n: 1.5, mass: 0.1, radius: 0.12, pos: [-0.2, 0, 0] },
          { kind: 'star', id: 'b', type: 'polytrope', n: 1.5, mass: 0.06, radius: 0.1, pos: [0.22, 0.05, 0] },
        ],
        run: { tEnd: 0.05 },
      });
      const m: Record<string, number> = {};
      const n = 64;
      const uniform = (mirror: boolean) =>
        withSim(gpu, compileSetup(doc(mirror)).problem, n, mirror ? { symmetry: MZ } : {}, async (sim) => {
          await sim.settled;
          const d0 = await sim.diagnostics();
          await sim.runToEnd();
          const d1 = await sim.diagnostics();
          return { mass: d0.mass, peak0: d0.ranges[0][1], e: d1.eTot, u1: (await sim.readState()).u1 };
        });
      const w = await uniform(false);
      const h = await uniform(true);
      // The mirrored grid's cells (i, j, k) are the whole's (i, j, k + n/2).
      let diff = 0;
      let sum = 0;
      for (let k = 0; k < n / 2; k++)
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const a = h.u1[4 * ((k * n + j) * n + i)];
            const b = w.u1[4 * (((k + n / 2) * n + j) * n + i)];
            diff += Math.abs(a - b);
            sum += b;
          }
      m.settled_mass = Math.abs(h.mass / w.mass - 1);
      m.settled_peak = Math.abs(h.peak0 / w.peak0 - 1);
      m.density_difference = diff / sum;
      m.energy = Math.abs(h.e / w.e - 1);
      const blocks = (mirror: boolean) =>
        withAmr(gpu, compileSetup(doc(mirror)).problem, 128, { amrLevels: 1, ...(mirror ? { symmetry: MZ } : {}) }, async (sim) => {
          await sim.settled;
          await sim.runToEnd();
          const d = await sim.diagnostics();
          return { mass: d.mass, e: d.eTot, peak: d.ranges[0][1] };
        });
      const bw = await blocks(false);
      const bh = await blocks(true);
      m.blocks_mass = Math.abs(bh.mass / bw.mass - 1);
      m.blocks_energy = Math.abs(bh.e / bw.e - 1);
      m.blocks_peak = Math.abs(bh.peak / bw.peak - 1);
      const pass =
        m.settled_mass < 1e-5 && m.settled_peak < 1e-4 && m.density_difference < 2e-3 && m.energy < 1e-4 &&
        m.blocks_mass < 1e-4 && m.blocks_energy < 1e-4 && m.blocks_peak < 1e-3;
      return { pass, metrics: m };
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
