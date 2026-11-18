// Validation tests for axisymmetric (R, z) runs (plans/coordinate-systems-plan.md, Part E).
// Registered through validation/frames.ts.
import { all, OUTFLOW, OUTFLOW_ONLY, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
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
    id: 'cyl-noh',
    name: 'Axisymmetric: the spherical Noh implosion',
    criterion: 'Cold gas (gamma = 5/3) falling at unit speed toward the origin from every direction, in (R, z) at 512^2 (R to 2, so that no gas measured came in through a boundary): at t = 0.6 the shock is at r = 0.2 and the post-shock density, 64 in the exact solution, averages within 5% of it over 0.06 < r < 0.17 away from the axis (R > 4 cells; the origin keeps the wall heating every code has there), and the pre-shock density at r = 0.3 is within 5% of (1 + t / r)^2 = 9',
    async run(gpu) {
      const n = 512;
      const box = 2;
      // The gas at r at time t started at r + t: within 1 of the origin, the box's nearest face.
      const p = ring((R, z) => {
        const r = Math.max(Math.hypot(R, z), 1e-9);
        return { rho: 1, vx: -R / r, vy: -z / r, vz: 0, p: 1e-6, s: 0 };
      }, { box, eosParams: { gamma: 5 / 3 }, bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC], tEnd: 0.6, pFloor: 1e-10 });
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const { u1 } = await sim.readState();
        const dx = box / n;
        let post = 0;
        let np = 0;
        let pre = 0;
        let nq = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const R = (i + 0.5) * dx;
            const z = (j + 0.5) * dx - 0.5 * box;
            const r = Math.hypot(R, z);
            const rho = u1[4 * (j * n + i)];
            if (r > 0.06 && r < 0.17 && R > 4 * dx) {
              post += rho;
              np++;
            }
            if (Math.abs(r - 0.3) < 0.5 * dx) {
              pre += rho;
              nq++;
            }
          }
        const at = (R: number, z: number) => u1[4 * (Math.floor((z + 0.5 * box) / dx) * n + Math.floor(R / dx))];
        const m: Record<string, number> = { post_shock_density: post / np, post_expected: 64, pre_shock_density: pre / nq, pre_expected: 9, steps: (await sim.diagnostics()).steps };
        for (const r of [0.1, 0.18, 0.22, 0.26, 0.3, 0.4]) {
          m[`rho_diag_${r}`] = at(r / Math.SQRT2, r / Math.SQRT2);
          m[`rho_R_${r}`] = at(r, 0);
          m[`rho_z_${r}`] = at(dx, r);
        }
        return { pass: Math.abs(m.post_shock_density / 64 - 1) < 0.05 && Math.abs(m.pre_shock_density / 9 - 1) < 0.05, metrics: m };
      });
    },
  },
  {
    id: 'cyl-bondi',
    name: 'Axisymmetric: Bondi accretion onto a point mass on the axis',
    criterion: 'Isothermal gas (c = 1, density 1e-5 far away, so the mass barely grows) around a fixed point mass GM = 1 with a sink well inside the sonic point, in (R, z) at 256^2 (R to 16): the accretion rate from t = 10 to 20 is within 5% of Bondi’s, 4 pi lambda (GM)^2 rho / c^3 with lambda = e^1.5 / 4',
    async run(gpu) {
      const n = 256;
      const box = 16;
      const rho0 = 1e-5;
      const p = ring(() => ({ rho: rho0, vx: 0, vy: 0, vz: 0, p: rho0, s: 0 }), {
        box, eos: 'barotropic', eosParams: { gamma: 1, K: 1 }, G: 1, bcLo: [REFLECT, OUTFLOW, PERIODIC], bcHi: [OUTFLOW, OUTFLOW, PERIODIC], tEnd: 10,
        rhoFloor: 1e-12,
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft: 0.125, sinkRadius: 0.25, fixed: true }],
      });
      const bondi = 4 * Math.PI * (Math.exp(1.5) / 4) * rho0;
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const a = await sim.diagnostics();
        sim.cfg.tEnd = 20;
        sim.writeParams();
        await sim.runToEnd();
        const b = await sim.diagnostics();
        const rate = (b.masses[0].accreted - a.masses[0].accreted) / (b.t - a.t);
        const m = { rate, bondi, relative_difference: Math.abs(rate / bondi - 1), mass_growth: b.masses[0].accreted, steps: b.steps };
        return { pass: m.relative_difference < 0.05, metrics: m };
      });
    },
  },
  {
    id: 'cyl-torus',
    name: 'Axisymmetric: a pressure-supported torus around a point mass',
    criterion: 'A polytropic torus of constant specific angular momentum (gamma = 5/3, pressure maximum at R = 1, inner edge 0.7) in equilibrium around a softened point mass on the axis, in (R, z) at 256^2 (R to 2.5, faces open for outflow only, so that gas the mass pulls in from outside the box cannot feed it): over five orbits at its center (t = 10 pi) its mass and the radius of its center of mass stay within 1%, and its peak density, which oscillates by a few percent as the 3D torus’s does, within 5%',
    async run(gpu) {
      const n = 256;
      const box = 2.5;
      const eps = 0.05;
      const gamma = 5 / 3;
      const l2 = 1 / (1 + eps * eps) ** 1.5;
      const phi = (R: number, z: number) => -1 / Math.sqrt(R * R + z * z + eps * eps);
      const C = phi(0.7, 0) + l2 / (2 * 0.49);
      const h = (R: number, z: number) => C - phi(R, z) - l2 / (2 * R * R);
      const K = (h(1, 0) * (gamma - 1)) / gamma;
      const rhoAmb = 1e-5;
      const p = ring((R, z) => {
        const hh = h(R, z);
        if (hh <= 0) return { rho: rhoAmb, vx: 0, vy: 0, vz: 0, p: 1e-9, s: 0 };
        const rho = Math.max(rhoAmb, ((hh * (gamma - 1)) / (gamma * K)) ** (1 / (gamma - 1)));
        return { rho, vx: 0, vy: 0, vz: Math.sqrt(l2) / R, p: K * rho ** gamma, s: 1 };
      }, {
        box, eosParams: { gamma }, G: 1, bcLo: [REFLECT, OUTFLOW_ONLY, PERIODIC], bcHi: [OUTFLOW_ONLY, OUTFLOW_ONLY, PERIODIC], tEnd: 10 * Math.PI,
        rhoFloor: 1e-8, pFloor: 1e-12,
        masses: () => [{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], soft: eps, sinkRadius: 0.15, fixed: true }],
      });
      const torus = async (sim: Sim) => {
        const { u1 } = await sim.readState();
        const dx = box / n;
        let peak = 0;
        let mass = 0;
        let mr = 0;
        for (let j = 0; j < n; j++)
          for (let i = 0; i < n; i++) {
            const rho = u1[4 * (j * n + i)];
            if (rho < 1e-3) continue;
            const R = (i + 0.5) * dx;
            const dm = rho * 2 * Math.PI * R * dx * dx;
            peak = Math.max(peak, rho);
            mass += dm;
            mr += dm * R;
          }
        return { peak, mass, R: mr / mass };
      };
      return withSim(gpu, p, n, { ndim: 2 }, async (sim) => {
        const a = await torus(sim);
        const track: Record<string, number> = {};
        for (const t of [0.5, 2, 2 * Math.PI, 3 * Math.PI, 4 * Math.PI, 5 * Math.PI, 6 * Math.PI, 8 * Math.PI, 10 * Math.PI]) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          await sim.runToEnd();
          const x = await torus(sim);
          track[`peak_t${t.toFixed(1)}`] = x.peak;
          track[`mass_t${t.toFixed(1)}`] = x.mass;
          // A torus gone wrong (gas piling up at the mass takes ever shorter steps): stop.
          if (!(Math.abs(x.mass / a.mass - 1) < 0.1)) break;
        }
        const b = await torus(sim);
        const m = {
          peak_change: Math.abs(b.peak / a.peak - 1),
          mass_change: Math.abs(b.mass / a.mass - 1),
          center_change: Math.abs(b.R / a.R - 1),
          peak_initial: a.peak, center_initial: a.R, mass_initial: a.mass, steps: (await sim.diagnostics()).steps, ...track,
        };
        return { pass: m.peak_change < 0.05 && m.mass_change < 0.01 && m.center_change < 0.01, metrics: m };
      });
    },
  },
  {
    id: 'cyl-mhd',
    name: 'Axisymmetric: a magnetic field in (R, z)',
    criterion: 'A poloidal field loop (A_phi = 1e-3 (0.3 - d) within 0.3 of R = 0.5) carried once along the periodic z axis at unit speed (128^2): the divergence (weighed by the faces’ radii) stays at round-off (below 1e-5), the mass and total energy keep to 1e-5, and the magnetic energy it keeps is within 1% of what the same loop keeps carried across a Cartesian box. And uniform gas threaded by a uniform field along the axis, at rest between the axis and a wall, stays at rest (Mach below 1e-5)',
    async run(gpu) {
      const n = 128;
      const loop = (d: number) => (d < 0.3 ? 1e-3 * (0.3 - d) : 0);
      const m: Record<string, number> = {};
      // The loop in (R, z), centered at R = 0.5.
      const cylP = ring(() => ({ rho: 1, vx: 0, vy: 1, vz: 0, p: 1, s: 0 }), {
        eosParams: { gamma: 5 / 3 }, bcHi: [REFLECT, PERIODIC, PERIODIC], tEnd: 1,
        mhd: { potential: () => (R, z) => [0, 0, loop(Math.hypot(R - 0.5, z))] },
      });
      await withSim(gpu, cylP, n, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        m.div_b = Math.max(d0.magnetic!.divB, d1.magnetic!.divB);
        m.mass_change = Math.abs(d1.mass / d0.mass - 1);
        m.energy_change = Math.abs(d1.eTot / d0.eTot - 1);
        m.kept_axisymmetric = d1.magnetic!.eMag / d0.magnetic!.eMag;
      });
      // The same loop across a Cartesian box, carried along y.
      const cart: Problem = {
        id: 'loop', label: 'Loop', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 }, bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 0, box: 1, tEnd: 1, supports2d: true,
        init: () => () => ({ rho: 1, vx: 0, vy: 1, vz: 0, p: 1, s: 0 }),
        mhd: { potential: () => (x, y) => [0, 0, loop(Math.hypot(x, y))] },
      };
      await withSim(gpu, cart, n, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        m.kept_cartesian = (await sim.diagnostics()).magnetic!.eMag / d0.magnetic!.eMag;
      });
      m.kept_difference = Math.abs(m.kept_axisymmetric / m.kept_cartesian - 1);
      // A uniform axial field: its pressure uniform, nothing to push the gas.
      const rest = ring(() => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }), {
        eosParams: { gamma: 5 / 3 }, bcHi: [REFLECT, PERIODIC, PERIODIC], tEnd: 1,
        mhd: { face: () => (axis) => (axis === 1 ? 0.5 : 0) },
      });
      await withSim(gpu, rest, 64, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        m.rest_mach = (await sim.diagnostics()).maxMach;
      });
      const pass = m.div_b < 1e-5 && m.mass_change < 1e-5 && m.energy_change < 1e-5 && m.kept_difference < 0.01 && m.rest_mach < 1e-5;
      return { pass, metrics: m };
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
