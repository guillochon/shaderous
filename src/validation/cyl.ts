// Validation tests for axisymmetric (R, z) runs (plans/coordinate-systems-plan.md, Part E).
// Registered through validation/frames.ts.
import { all, OUTFLOW, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
import { SEDOV_XI0, SPHERE, spherePotential } from '../presets/names';
import type { Sim } from '../sim';
import { withAmr, withSim, type TestDef } from './tests';
import type { AmrSim } from '../amr/amr_sim';

/** An axisymmetric box of side `box` (R from 0 to box, z across it), ideal gas, from `init`. */
function ring(init: (R: number, z: number) => Prim, over: Partial<Problem> = {}): Problem {
  return {
    id: 'axisymmetric', label: 'Axisymmetric', blurb: '', eos: 'ideal', eosParams: { gamma: 1.4 },
    bcLo: [REFLECT, PERIODIC, PERIODIC], bcHi: [OUTFLOW, PERIODIC, PERIODIC], gravity: 'off', G: 0, box: 1, tEnd: 0.1, supports2d: true,
    geometry: 'cylindrical', init: () => (R, z) => init(R, z), ...over,
  };
}

/** Totals over the rings of a run: mass, z momentum, angular momentum R rho v_phi, total energy. */
async function totals(sim: Sim) {
  const { u1, u2 } = await sim.readState();
  const n = sim.cfg.n;
  const dx = sim.cfg.problem.box / n;
  let m = 0;
  let pz = 0;
  let l = 0;
  let e = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const o = 4 * (j * n + i);
      const R = (i + 0.5) * dx;
      const dv = 2 * Math.PI * R * dx * dx;
      m += u1[o] * dv;
      pz += u1[o + 2] * dv;
      l += R * u1[o + 3] * dv;
      e += u2[o] * dv;
    }
  return { m, pz, l, e };
}

export const CYL_TESTS: TestDef[] = [
  {
    id: 'cyl-rest',
    name: 'Axisymmetric: uniform gas at rest stays at rest',
    criterion: 'Uniform gas in (R, z), 64^2, between the axis and a reflecting wall: after a sound crossing the largest Mach number is below 1e-5 (the radial faces’ pressure difference and the geometric source cancel to rounding)',
    async run(gpu) {
      const p = ring(() => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }), { bcHi: [REFLECT, PERIODIC, PERIODIC], tEnd: 1 });
      return withSim(gpu, p, 64, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        return { pass: d.maxMach < 1e-5, metrics: { max_mach: d.maxMach, mass: d.mass, mass_expected: Math.PI } };
      });
    },
  },
  {
    id: 'cyl-conserve',
    name: 'Axisymmetric: a swirling ring carried along z',
    criterion: 'A dense ring with swirl, carried along the periodic z axis (64^2, walls at the axis and at R = 1): mass, total energy, z momentum and angular momentum R rho v_phi are kept to 1e-5 of their scales over a box crossing',
    async run(gpu) {
      const p = ring((R, z) => {
        const r = Math.hypot(R - 0.5, z);
        const inside = r < 0.2;
        return { rho: inside ? 3 : 1, vx: 0, vy: 1, vz: inside ? 0.5 * (1 - r / 0.2) : 0, p: 1, s: inside ? 1 : 0 };
      }, { bcHi: [REFLECT, PERIODIC, PERIODIC], tEnd: 1 });
      return withSim(gpu, p, 64, { ndim: 2 }, async (sim) => {
        const a = await totals(sim);
        await sim.runToEnd();
        const b = await totals(sim);
        const m = {
          mass: Math.abs(b.m / a.m - 1),
          energy: Math.abs(b.e / a.e - 1),
          z_momentum: Math.abs(b.pz - a.pz) / a.m,
          angular_momentum: Math.abs(b.l / a.l - 1),
          angular_momentum_initial: a.l,
        };
        return { pass: m.mass < 1e-5 && m.energy < 1e-5 && m.z_momentum < 1e-5 && m.angular_momentum < 1e-5, metrics: m };
      });
    },
  },
  {
    id: 'cyl-sedov',
    name: 'Axisymmetric: a point blast is spherical',
    criterion: 'A Sedov blast on the axis (gamma 1.4, 128^2 in (R, z), E = 1 in a sphere of 3 cells): at t = 0.05 the shock radius along R and along z are each within 2% of the 3D similarity solution',
    async run(gpu) {
      const n = 128;
      const dx = 1 / n;
      const r0 = 3 * dx;
      // The energy's density from the rings it is in, so that their total is exactly E.
      let vol = 0;
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const R = (i + 0.5) * dx;
          const z = (j + 0.5) * dx - 0.5;
          if (Math.hypot(R, z) < r0) vol += 2 * Math.PI * R * dx * dx;
        }
      const pIn = (0.4 * 1) / vol;
      const p = ring((R, z) => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: Math.hypot(R, z) < r0 ? pIn : 1e-5, s: 0 }), {
        bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC], tEnd: 0.05,
      });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const { u1 } = await sim.readState();
        const want = SEDOV_XI0['1.4'] * Math.pow(0.05 * 0.05, 0.2);
        // The shock: the outermost cell, along each ray, whose density is above 2.
        let rR = 0;
        for (let i = 0; i < n; i++) if (u1[4 * ((n / 2) * n + i)] > 2) rR = (i + 0.5) * dx;
        let rz = 0;
        for (let j = n / 2; j < n; j++) if (u1[4 * (j * n)] > 2) rz = (j + 0.5) * dx - 0.5;
        const m = { radius_R: rR, radius_z: rz, similarity: want, error_R: Math.abs(rR / want - 1), error_z: Math.abs(rz / want - 1) };
        return { pass: m.error_R < 0.02 && m.error_z < 0.02, metrics: m };
      });
    },
  },
  {
    id: 'cyl-sink',
    name: 'Axisymmetric: a sink on the axis against the same sink in 3D',
    criterion: 'A point mass with a sink in uniform gas at rest: what it takes in by t = 0.1 in (R, z) (32^2, R to 0.5) is within 5% of what it takes in a 3D Cartesian box of the same cell size (64^3), and the mass stays on the axis',
    async run(gpu) {
      const base: Problem = {
        id: 'sink', label: 'Sink', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
        bcLo: [OUTFLOW, OUTFLOW, OUTFLOW], bcHi: [OUTFLOW, OUTFLOW, OUTFLOW], gravity: 'off', G: 1, box: 1, tEnd: 0.1,
        init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }),
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.05, sinkRadius: 0.06 }],
      };
      const take = async (sim: Sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        return { acc: d.masses[0].accreted, x: d.masses[0].pos[0], z: d.masses[0].pos[2] };
      };
      const cart = await withSim(gpu, base, 64, {}, take);
      const cyl = await withSim(gpu, { ...base, box: 0.5, geometry: 'cylindrical', bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC] }, 32, { ndim: 2 }, take);
      const m = { accreted_3d: cart.acc, accreted_axisymmetric: cyl.acc, relative_difference: Math.abs(cyl.acc / cart.acc - 1), off_axis: Math.hypot(cyl.x, cyl.z) };
      return { pass: m.relative_difference < 0.05 && m.off_axis === 0 && cart.acc > 0, metrics: m };
    },
  },
  {
    id: 'cyl-blocks',
    name: 'Axisymmetric: the swirling ring and the blast on blocks',
    criterion: 'On an adaptive mesh of two levels (128^2 finest), the swirling ring carried along z keeps mass, energy and angular momentum to 1e-5 through its re-meshing, and the point blast’s radius along R and along z are within 2% of the similarity solution',
    async run(gpu) {
      const ringTotals = async (sim: AmrSim) => {
        const c = await sim.readLeaves();
        let m = 0;
        let l = 0;
        let e = 0;
        for (let q = 0; q < c.count; q++) {
          const dv = 2 * Math.PI * c.x[q] * c.dx[q] * c.dx[q];
          m += c.u1[4 * q] * dv;
          l += c.x[q] * c.u1[4 * q + 3] * dv;
          e += c.u2[4 * q] * dv;
        }
        return { m, l, e };
      };
      const swirl = ring((R, z) => {
        const r = Math.hypot(R - 0.5, z);
        const inside = r < 0.2;
        return { rho: inside ? 3 : 1, vx: 0, vy: 1, vz: inside ? 0.5 * (1 - r / 0.2) : 0, p: 1, s: inside ? 1 : 0 };
      }, { bcHi: [REFLECT, PERIODIC, PERIODIC], tEnd: 0.5 });
      const cons = await withAmr(gpu, swirl, 128, { ndim: 2, amrLevels: 2 }, async (sim) => {
        const a = await ringTotals(sim);
        await sim.runToEnd();
        const b = await ringTotals(sim);
        return { mass: Math.abs(b.m / a.m - 1), energy: Math.abs(b.e / a.e - 1), angular_momentum: Math.abs(b.l / a.l - 1) };
      });
      const n = 128;
      const dx = 1 / n;
      const r0 = 3 * dx;
      let vol = 0;
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const R = (i + 0.5) * dx;
          const z = (j + 0.5) * dx - 0.5;
          if (Math.hypot(R, z) < r0) vol += 2 * Math.PI * R * dx * dx;
        }
      const blast = ring((R, z) => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: Math.hypot(R, z) < r0 ? 0.4 / vol : 1e-5, s: 0 }), {
        bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC], tEnd: 0.05,
      });
      const radii = await withAmr(gpu, blast, n, { ndim: 2, amrLevels: 2 }, async (sim) => {
        await sim.runToEnd();
        const c = await sim.readLeaves();
        let rR = 0;
        let rz = 0;
        for (let q = 0; q < c.count; q++) {
          if (!(c.u1[4 * q] > 2)) continue;
          if (Math.abs(c.y[q]) < c.dx[q]) rR = Math.max(rR, c.x[q]);
          if (c.x[q] < c.dx[q]) rz = Math.max(rz, c.y[q]);
        }
        return { rR, rz };
      });
      const want = SEDOV_XI0['1.4'] * Math.pow(0.05 * 0.05, 0.2);
      const m = { ...cons, radius_R: radii.rR, radius_z: radii.rz, similarity: want, error_R: Math.abs(radii.rR / want - 1), error_z: Math.abs(radii.rz / want - 1) };
      return { pass: m.mass < 1e-5 && m.energy < 1e-5 && m.angular_momentum < 1e-5 && m.error_R < 0.02 && m.error_z < 0.02, metrics: m };
    },
  },
  {
    id: 'cyl-sphere',
    name: 'Axisymmetric: the potential of a uniform sphere (self-gravity in (R, z))',
    criterion: 'A uniform sphere on the axis, solved for in (R, z) with isolated boundaries (64^2, the 2D multigrid weighing its radial neighbors by R): the potential is within 1% of the analytic one everywhere, as the 3D solve is',
    async run(gpu) {
      const n = 64;
      // The sphere's share of each ring cell, from 8 x 8 samples weighed by their radii (as the
      // 3D test's sphere is supersampled), so that its mass is the sphere's.
      const dx = 1 / n;
      const share = (R: number, z: number) => {
        let inside = 0;
        let all = 0;
        for (let b = 0; b < 8; b++)
          for (let a = 0; a < 8; a++) {
            const r = R + ((a + 0.5) / 8 - 0.5) * dx;
            const zz = z + ((b + 0.5) / 8 - 0.5) * dx;
            all += r;
            if (Math.hypot(r, zz) < SPHERE.R) inside += r;
          }
        return inside / all;
      };
      const p = ring((R, z) => ({ rho: 1e-6 + SPHERE.rho * share(R, z), vx: 0, vy: 0, vz: 0, p: 1e-4, s: 0 }), {
        bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC], gravity: 'isolated', G: SPHERE.G, tEnd: 0,
      });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        sim.solvePotential(10);
        const phi = await sim.readPotential();
        let err = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const e = spherePotential(Math.hypot((i + 0.5) / n, (j + 0.5) / n - 0.5));
            err = Math.max(err, Math.abs(phi[j * n + i] - e) / Math.abs(e));
          }
        const d = await sim.diagnostics();
        return { pass: err < 0.01, metrics: { max_relative_error: err, mass: d.mass, mass_expected: (4 / 3) * Math.PI * SPHERE.R ** 3 * SPHERE.rho } };
      });
    },
  },
  {
    id: 'cyl-evrard',
    name: 'Axisymmetric: the Evrard collapse with self-gravity in (R, z)',
    criterion: 'The Evrard sphere (G = M = R = 1, u = 0.05) in (R, z) at 128^2 (R to 4, z across 4): through collapse and bounce (t <= 1.5) the total energy keeps to 2% and the thermal energy peaks between t = 0.9 and 1.3, as in 3D',
    async run(gpu) {
      const n = 128;
      // R to 4 and z across +-2, as the 3D test's box: the sphere and its bounce well inside.
      const box = 4;
      const dx = box / n;
      const p = ring((R, z) => {
        const r = Math.max(Math.hypot(R, z), 0.5 * dx);
        const rho = r < 1 ? 1 / (2 * Math.PI * r) : 1e-4;
        return { rho, vx: 0, vy: 0, vz: 0, p: (2 / 3) * rho * 0.05, s: r < 1 ? 1 : 0 };
      }, {
        box, eosParams: { gamma: 5 / 3 }, bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC],
        gravity: 'isolated', G: 1, tEnd: 0, rhoFloor: 1e-7, pFloor: 1e-12,
      });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        let worst = 0;
        let peakT = 0;
        let peakU = 0;
        for (let i = 1; i <= 12; i++) {
          sim.cfg.tEnd = 0.125 * i;
          sim.writeParams();
          await sim.runToEnd();
          const d = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(d.eTot - d0.eTot) / Math.abs(d0.eTot));
          if (d.eInt > peakU) [peakU, peakT] = [d.eInt, d.t];
        }
        return { pass: worst < 0.02 && peakT >= 0.9 && peakT <= 1.3, metrics: { energy_drift: worst, thermal_peak_time: peakT, e_total_initial: d0.eTot, mass: d0.mass } };
      });
    },
  },
  {
    id: 'cyl-column',
    name: 'Axisymmetric: a stratified column at rest under gravity along the axis',
    criterion: 'An isothermal column in discrete balance under a uniform field along z (64^2 in (R, z), walls all round): after a sound crossing the largest Mach number stays below 1e-4, as the Cartesian column does',
    async run(gpu) {
      const n = 64;
      const g = 16;
      const a = g / n / 2;
      const q = (1 - a) / (1 + a);
      const p = ring((_R, z) => {
        const pr = Math.pow(q, Math.round((z + 0.5) * n - 0.5));
        return { rho: pr, vx: 0, vy: 0, vz: 0, p: pr, s: 0 };
      }, { bcLo: [REFLECT, REFLECT, PERIODIC], bcHi: [REFLECT, REFLECT, PERIODIC], gext: () => [0, -g, 0], tEnd: 1 });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        return { pass: d.maxMach < 1e-4, metrics: { max_mach: d.maxMach } };
      });
    },
  },
];
