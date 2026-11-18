// Validation tests for the non-inertial frames and coordinate systems (plans/coordinate-systems-plan.md):
// the rotating frame (Part A) and mirror symmetry (Part B). Registered in tests.ts after the others.
import type { Gpu } from '../gpu/device';
import { all, OUTFLOW, PERIODIC, REFLECT, type Problem } from '../problems/types';
import { toRotating } from '../problems/rotating';
import { domainLo } from '../domain';
import type { Sim } from '../sim';
import type { AmrSim } from '../amr/amr_sim';
import { withAmr, withSim, type TestDef } from './tests';
import { SYM_TESTS } from './symmetry';
import { COSMO_TESTS } from './cosmo';
import { SHEAR_TESTS } from './shear';
import { CYL_TESTS } from './cyl';
import { SPH_TESTS } from './sph';

const gas = (over: Partial<Problem>): Problem => ({
  id: 'frame-test', label: 'Frame test', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 1, box: 1, tEnd: 1, supports2d: true,
  init: () => () => ({ rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 0 }),
  ...over,
});

/** Angle from a to b, in (-pi, pi]. */
const turn = (a: number, b: number) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

/** Uniform gas moving at v0 along x in a periodic box, in an f-plane turning at omega about z. */
const inertial = (omega: number, v0: number, field = 0): Problem =>
  gas({
    id: 'rot-inertial', tEnd: Math.PI / (4 * omega),
    rotation: () => ({ omega: [0, 0, omega], centrifugal: false }),
    init: () => () => ({ rho: 1, vx: v0, vy: 0, vz: 0, p: 1, s: 0 }),
    ...(field ? { mhd: { face: () => (axis: number) => (axis === 2 ? field : 0) } } : {}),
  });

/** Checks a run of `inertial` at its end time: the velocity's phase and the kinetic energy. */
async function inertialCheck(sim: Sim | AmrSim, omega: number, tag: string, m: Record<string, number | string>) {
  const d0 = await sim.diagnostics();
  await sim.runToEnd();
  const d1 = await sim.diagnostics();
  const want = -2 * omega * d1.t;
  const got = Math.atan2(d1.momentum[1], d1.momentum[0]);
  m[`${tag}_phase_error`] = Math.abs(turn(want, got));
  m[`${tag}_kinetic_drift`] = Math.abs(d1.eKin / d0.eKin - 1);
  m[`${tag}_t`] = d1.t;
  if (d1.magnetic && d0.magnetic) m[`${tag}_field_drift`] = Math.abs(d1.magnetic.eMag / d0.magnetic.eMag - 1);
}

/** Isothermal gas at rest in the frame turning at omega about z, held by the centrifugal force. */
const balanced = (omega: number): Problem =>
  gas({
    id: 'rot-balance', box: 2, tEnd: 2, bcLo: all(REFLECT), bcHi: all(REFLECT),
    rotation: () => ({ omega: [0, 0, omega] }),
    init: () => (x, y) => {
      const rho = Math.exp((omega * omega * (x * x + y * y)) / 2);
      return { rho, vx: 0, vy: 0, vz: 0, p: rho, s: 0 };
    },
  });

/** Center of the gas the passive scalar marks, from a uniform run's state. */
async function scalarCenter(sim: Sim): Promise<[number, number]> {
  const { u2 } = await sim.readState();
  const n = sim.cfg.n;
  const dx = sim.cfg.problem.box / n;
  const lo = domainLo(sim.cfg);
  let s = 0;
  let sx = 0;
  let sy = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const w = u2[4 * (j * n + i) + 2];
      if (!(w > 1e-3)) continue;
      s += w;
      sx += w * ((i + 0.5) * dx + lo[0]);
      sy += w * ((j + 0.5) * dx + lo[1]);
    }
  return [sx / s, sy / s];
}

export const FRAME_TESTS: TestDef[] = [
  {
    id: 'rot-off',
    name: 'Rotating frame: a frame that does not turn changes nothing',
    criterion: 'A Kelvin-Helmholtz layer (2D, 64^2) with a rotation of zero ends in the same bits as without one',
    async run(gpu) {
      const kh = gas({
        id: 'rot-off', tEnd: 0.2,
        init: () => (_x, y) => {
          const inside = Math.abs(y) < 0.25;
          return { rho: inside ? 2 : 1, vx: inside ? 0.5 : -0.5, vy: 0.01 * Math.sin(4 * Math.PI * _x), vz: 0, p: 2.5, s: inside ? 1 : 0 };
        },
      });
      const run = (p: Problem) =>
        withSim(gpu, p, 64, { ndim: 2 }, async (sim) => {
          await sim.runToEnd();
          return sim.readState();
        });
      const a = await run(kh);
      const b = await run({ ...kh, rotation: () => ({ omega: [0, 0, 0] }) });
      let differ = 0;
      for (let i = 0; i < a.u1.length; i++) if (a.u1[i] !== b.u1[i] || a.u2[i] !== b.u2[i]) differ++;
      return { pass: differ === 0, metrics: { values_differing: differ } };
    },
  },
  {
    id: 'rot-inertial',
    name: 'Rotating frame: uniform flow turns at twice the frame rate (Coriolis only)',
    criterion:
      'Uniform gas moving at 0.1 in a periodic 2D box, in an f-plane turning at Omega = 1: after an eighth of a period of the inertial oscillation the velocity has turned by -2 Omega t to within 1e-4 rad and its kinetic energy is unchanged to 1e-5, on the uniform grid and on blocks',
    async run(gpu) {
      const omega = 1;
      const p = inertial(omega, 0.1);
      const m: Record<string, number | string> = {};
      await withSim(gpu, p, 32, { ndim: 2 }, (sim) => inertialCheck(sim, omega, 'uniform', m));
      await withAmr(gpu, p, 32, { ndim: 2, amrLevels: 0 }, (sim) => inertialCheck(sim, omega, 'blocks', m));
      const pass = ['uniform', 'blocks'].every((t) => (m[`${t}_phase_error`] as number) < 1e-4 && (m[`${t}_kinetic_drift`] as number) < 1e-5);
      return { pass, metrics: m };
    },
  },
  {
    id: 'rot-balance',
    name: 'Rotating frame: isothermal gas held by the centrifugal force stays at rest',
    criterion:
      'Isothermal gas (2D, 64^2, Omega = 1, density rising as exp(Omega^2 R^2 / 2 c^2) to the corners of a reflecting box) after more than a sound crossing: the largest Mach number stays below 1e-4 with the well-balanced reconstruction, at least a hundred times smaller than without it, on the uniform grid and on blocks',
    async run(gpu) {
      const p = balanced(1);
      const mach = async (wellBalanced: boolean, blocks: boolean) => {
        const go = async (sim: Sim | AmrSim) => {
          await sim.runToEnd();
          return (await sim.diagnostics()).maxMach;
        };
        const over = { ndim: 2 as const, wellBalanced };
        return blocks ? withAmr(gpu, p, 64, { ...over, amrLevels: 0 }, go) : withSim(gpu, p, 64, over, go);
      };
      const m = {
        max_mach: await mach(true, false),
        max_mach_plain: await mach(false, false),
        max_mach_blocks: await mach(true, true),
        max_mach_plain_blocks: await mach(false, true),
      };
      const pass = m.max_mach < 1e-4 && m.max_mach_blocks < 1e-4 && m.max_mach < 0.01 * m.max_mach_plain && m.max_mach_blocks < 0.01 * m.max_mach_plain_blocks;
      return { pass, metrics: m };
    },
  },
  {
    id: 'rot-blob',
    name: 'Rotating frame: gas moving straight in the inertial frame',
    criterion:
      'A dense blob carried by gas that moves uniformly in the inertial frame (2D, 192^2, box 12, Omega = 1, Coriolis and centrifugal forces): after a quarter turn the blob is centered on R(-Omega t)(x0 + v0 t) to within half a cell',
    async run(gpu) {
      const omega = 1;
      const x0 = [1, 0, 0];
      const v0 = [0, 0.5, 0];
      const t = Math.PI / (2 * omega);
      const p = gas({
        id: 'rot-blob', box: 12, tEnd: t, bcLo: all(OUTFLOW), bcHi: all(OUTFLOW),
        rotation: () => ({ omega: [0, 0, omega] }),
        init: () => (x, y) => {
          const v = toRotating(v0, [x, y, 0], [0, 0, omega]);
          const inside = Math.hypot(x - x0[0], y - x0[1]) < 0.3;
          return { rho: inside ? 10 : 1, vx: v[0], vy: v[1], vz: 0, p: 1, s: inside ? 1 : 0 };
        },
      });
      return withSim(gpu, p, 192, { ndim: 2 }, async (sim) => {
        await sim.runToEnd();
        const d = await sim.diagnostics();
        const [cx, cy] = await scalarCenter(sim);
        // Where the blob is in the inertial frame, then turned back by the frame's angle.
        const xi = x0[0] + v0[0] * d.t;
        const yi = x0[1] + v0[1] * d.t;
        const a = -omega * d.t;
        const want = [Math.cos(a) * xi - Math.sin(a) * yi, Math.sin(a) * xi + Math.cos(a) * yi];
        const off = Math.hypot(cx - want[0], cy - want[1]);
        const half = (0.5 * p.box) / 192;
        return { pass: off < half, metrics: { offset: off, half_cell: half, x: cx, y: cy, x_expected: want[0], y_expected: want[1], t: d.t } };
      });
    },
  },
  {
    id: 'rot-binary',
    name: 'Rotating frame: a binary at rest in its corotating frame',
    criterion:
      'Two equal point masses on a circular orbit of separation 1, in a 2D frame turning at the softened orbital rate (Coriolis and centrifugal): over two orbits the separation stays within 1e-3 of 1 and each mass stays within 0.01 of where it started, on the uniform grid and on blocks',
    async run(gpu) {
      const soft = 0.25;
      const omega = Math.pow(1 + soft * soft, -0.75);
      const p = gas({
        id: 'rot-binary', box: 4, tEnd: (4 * Math.PI) / omega, bcLo: all(OUTFLOW), bcHi: all(OUTFLOW),
        rotation: () => ({ omega: [0, 0, omega] }),
        init: () => () => ({ rho: 1e-6, vx: 0, vy: 0, vz: 0, p: 1e-6, s: 0 }),
        masses: () => [
          { gm: 0.5, pos: [-0.5, 0, 0], vel: [0, 0, 0], soft, sinkRadius: 0 },
          { gm: 0.5, pos: [0.5, 0, 0], vel: [0, 0, 0], soft, sinkRadius: 0 },
        ],
      });
      const orbit = async (sim: Sim | AmrSim) => {
        const d0 = await sim.diagnostics();
        const softUsed = d0.masses[0].soft;
        let sep = 0;
        let drift = 0;
        const tEnd = sim.cfg.tEnd;
        for (let i = 1; i <= 16; i++) {
          sim.cfg.tEnd = (tEnd * i) / 16;
          sim.writeParams();
          await sim.runToEnd();
          const d = await sim.diagnostics();
          const [a, b] = d.masses;
          sep = Math.max(sep, Math.abs(Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) - 1));
          drift = Math.max(drift, ...[0, 1].map((k) => Math.hypot(d.masses[k].pos[0] - d0.masses[k].pos[0], d.masses[k].pos[1] - d0.masses[k].pos[1])));
        }
        return { ok: softUsed === soft && sep < 1e-3 && drift < 0.01, sep, drift, softUsed };
      };
      const u = await withSim(gpu, p, 64, { ndim: 2 }, orbit);
      const b = await withAmr(gpu, p, 64, { ndim: 2, amrLevels: 0 }, orbit);
      return {
        pass: u.ok && b.ok,
        metrics: { separation_max_deviation: u.sep, position_drift: u.drift, blocks_separation_max_deviation: b.sep, blocks_position_drift: b.drift, softening: u.softUsed, omega },
      };
    },
  },
  {
    id: 'rot-mhd',
    name: 'Rotating frame: a uniform field along Omega with uniform flow (MHD)',
    criterion:
      'As rot-inertial with a uniform field of 0.5 along the rotation axis: the velocity turns by -2 Omega t to within 1e-4 rad, the kinetic energy keeps to 1e-5 and the magnetic energy to 1e-6, on the uniform grid and on blocks',
    async run(gpu) {
      const omega = 1;
      const p = inertial(omega, 0.1, 0.5);
      const m: Record<string, number | string> = {};
      await withSim(gpu, p, 32, { ndim: 2 }, (sim) => inertialCheck(sim, omega, 'uniform', m));
      await withAmr(gpu, p, 32, { ndim: 2, amrLevels: 0 }, (sim) => inertialCheck(sim, omega, 'blocks', m));
      const pass = ['uniform', 'blocks'].every(
        (t) => (m[`${t}_phase_error`] as number) < 1e-4 && (m[`${t}_kinetic_drift`] as number) < 1e-5 && (m[`${t}_field_drift`] as number) < 1e-6,
      );
      return { pass, metrics: m };
    },
  },
  ...SYM_TESTS,
  ...COSMO_TESTS,
  ...SHEAR_TESTS,
  ...CYL_TESTS,
  ...SPH_TESTS,
];
