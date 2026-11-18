// Validation tests for axisymmetric (R, z) runs (plans/coordinate-systems-plan.md, Part E).
// Registered through validation/frames.ts.
import { all, OUTFLOW, PERIODIC, REFLECT, type Prim, type Problem } from '../problems/types';
import { SEDOV_XI0 } from '../presets/names';
import type { Sim } from '../sim';
import { withSim, type TestDef } from './tests';

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
