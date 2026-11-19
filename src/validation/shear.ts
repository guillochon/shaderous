// Validation tests for the shearing box (docs/numerics.md section 22). Registered
// through validation/frames.ts.
import { all, PERIODIC, type Prim, type Problem } from '../problems/types';
import type { Sim } from '../sim';
import type { MhdPasses } from '../mhd/mhd';
import { withSim, type TestDef } from './tests';

const OMEGA = 1;
const Q = 1.5;

/** A 2D shearing sheet (Keplerian, Omega = 1, box 1) whose state is the background flow v_y = -q Omega x plus `extra`. */
function sheet(tEnd: number, extra: (x: number, y: number, z: number) => Partial<Prim> = () => ({}), over: Partial<Problem> = {}): Problem {
  return {
    id: 'shearing-sheet', label: 'Shearing sheet', blurb: '', eos: 'ideal', eosParams: { gamma: 5 / 3 },
    bcLo: all(PERIODIC), bcHi: all(PERIODIC), gravity: 'off', G: 0, box: 1, tEnd, supports2d: true,
    shear: { omega: OMEGA, q: Q },
    init: () => (x, y, z) => ({ rho: 1, vx: 0, vy: -Q * OMEGA * x, vz: 0, p: 1, s: 0, ...extra(x, y, z) }),
    ...over,
  };
}

/** The face field of a run (mhd/mhd.ts initialFaces' layout: (n + 1)^2 (nz + 1) texels of (b_x, b_y, b_z, 0)). */
async function faces(sim: Sim): Promise<Float32Array> {
  return (sim as unknown as { mhd: MhdPasses }).mhd.readFaces(sim.cur);
}

/**
 * The linear compressive shearing wave (Johnson & Gammie 2005) in float64: the amplitudes of
 * delta = drho / rho, u_x and u_y of a wave exp(i (k_x(t) x + k_y y)), k_x(t) = k_x0 + q Omega k_y t,
 * at times `ts`, from delta = A and u = 0 at t = 0. Adiabatic, sound speed c.
 */
function shearingWave(kx0: number, ky: number, c: number, A: number, ts: number[]): { re: number; im: number }[] {
  type C = [number, number];
  // State: [delta, ux, uy] complex.
  const f = (t: number, y: C[]): C[] => {
    const kx = kx0 + Q * OMEGA * ky * t;
    const [d, ux, uy] = y;
    // -i z = (z.im, -z.re)
    const mi = (z: C, s: number): C => [s * z[1], -s * z[0]];
    const div = mi([kx * ux[0] + ky * uy[0], kx * ux[1] + ky * uy[1]], 1);
    const px = mi(d, kx * c * c);
    const py = mi(d, ky * c * c);
    return [div, [2 * OMEGA * uy[0] + px[0], 2 * OMEGA * uy[1] + px[1]], [-(2 - Q) * OMEGA * ux[0] + py[0], -(2 - Q) * OMEGA * ux[1] + py[1]]];
  };
  const add = (y: C[], k: C[], h: number): C[] => y.map((z, i) => [z[0] + h * k[i][0], z[1] + h * k[i][1]] as C);
  let y: C[] = [[A, 0], [0, 0], [0, 0]];
  let t = 0;
  const out: { re: number; im: number }[] = [];
  for (const te of ts) {
    const n = Math.max(1, Math.ceil((te - t) / 1e-4));
    const h = (te - t) / n;
    for (let i = 0; i < n; i++) {
      const k1 = f(t, y);
      const k2 = f(t + h / 2, add(y, k1, h / 2));
      const k3 = f(t + h / 2, add(y, k2, h / 2));
      const k4 = f(t + h, add(y, k3, h));
      y = y.map((z, j) => [z[0] + (h / 6) * (k1[j][0] + 2 * k2[j][0] + 2 * k3[j][0] + k4[j][0]), z[1] + (h / 6) * (k1[j][1] + 2 * k2[j][1] + 2 * k3[j][1] + k4[j][1])] as C);
      t += h;
    }
    out.push({ re: y[0][0], im: y[0][1] });
  }
  return out;
}

/** Largest departure of the velocity from the background flow, and of the density from rho0, over the grid. */
async function departure(sim: Sim, rho0: number): Promise<{ dv: number; drho: number }> {
  const { u1 } = await sim.readState();
  const n = sim.cfg.n;
  let dv = 0;
  let drho = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const o = 4 * (j * n + i);
      const x = (i + 0.5) / n - 0.5;
      dv = Math.max(dv, Math.abs(u1[o + 1] / u1[o]), Math.abs(u1[o + 2] / u1[o] + Q * OMEGA * x));
      drho = Math.max(drho, Math.abs(u1[o] / rho0 - 1));
    }
  return { dv, drho };
}

export const SHEAR_TESTS: TestDef[] = [
  {
    id: 'shear-steady',
    name: 'Shearing box: the background flow is steady',
    criterion: 'Uniform gas of density and pressure 1.4 (the sound speed of density 1 and pressure 1) in a Keplerian shearing sheet (2D, 64^2), moving with the background flow v_y = -q Omega x: over 10 orbits (16,700 steps) the velocity stays within 2e-3 of the shear speed at the box edge and the density within 1e-3 of 1.4, and the mass is kept to 1e-6. (What departs is float32 rounding of the Coriolis and tidal increments to a velocity of order the shear speed, the same every step in a steady state, so it adds up: about 1e-4 of the shear speed an orbit. The density is 1.4, not 1: a float32 density that hovers about 1.0, a power of two, keeps an increment just below it that it loses just above it, where the spacing of floats doubles, and so loses mass: 2e-6 over the run on Metal, against 1e-7 or less at 1.3, 1.4 or 1.75.)',
    async run(gpu) {
      const rho0 = 1.4;
      return withSim(gpu, sheet(20 * Math.PI, () => ({ rho: rho0, p: rho0 })), 64, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const d = await departure(sim, rho0);
        const d1 = await sim.diagnostics();
        const edge = Q * OMEGA * 0.5;
        return {
          pass: d.dv / edge < 2e-3 && d.drho < 1e-3 && Math.abs(d1.mass / d0.mass - 1) < 1e-6,
          metrics: { velocity_departure: d.dv / edge, density_departure: d.drho, mass_change: d1.mass / d0.mass - 1, steps: d1.steps },
        };
      });
    },
  },
  {
    id: 'shear-mass',
    name: 'Shearing box: a dense blob across the shearing boundary',
    criterion: 'A blob of twice the density straddling the x faces of a Keplerian sheet (2D, 64^2), carried along y by the shear for two orbits: its mass (the passive scalar) and the total kept to 1e-6, x momentum to 1e-6 of the gas’s mass times the shear speed, and the density within [0.9, 2.1] (its edges smeared, nothing made at the boundary)',
    async run(gpu) {
      const blob = (x: number, y: number) => {
        const dx = Math.abs(x) - 0.5;
        return Math.hypot(dx, y) < 0.15 ? { rho: 2, s: 1 } : {};
      };
      return withSim(gpu, sheet(4 * Math.PI, blob), 64, { ndim: 2 }, async (sim) => {
        const d0 = await sim.diagnostics();
        const { u2: s0 } = await sim.readState();
        await sim.runToEnd();
        const d1 = await sim.diagnostics();
        const { u1, u2 } = await sim.readState();
        let m0 = 0;
        let m1 = 0;
        for (let q = 0; q < s0.length; q += 4) {
          m0 += s0[q + 2];
          m1 += u2[q + 2];
        }
        let lo = Infinity;
        let hi = -Infinity;
        for (let q = 0; q < u1.length; q += 4) {
          lo = Math.min(lo, u1[q]);
          hi = Math.max(hi, u1[q]);
        }
        const scale = d0.mass * Q * OMEGA * 0.5;
        const m = {
          mass_change: Math.abs(d1.mass / d0.mass - 1),
          blob_change: Math.abs(m1 / m0 - 1),
          x_momentum: Math.abs(d1.momentum[0] - d0.momentum[0]) / scale,
          density_min: lo, density_max: hi,
        };
        return { pass: m.mass_change < 1e-6 && m.blob_change < 1e-6 && m.x_momentum < 1e-6 && lo > 0.9 && hi < 2.1, metrics: m };
      });
    },
  },
  {
    id: 'shear-wave',
    name: 'Shearing box: a compressive shearing wave',
    criterion: 'A leading density wave of 1% (k_x = -2, k_y = 1 wavelengths across the box, c_s = 0.13, Keplerian sheet, 2D, 128^2) swung to trailing by the shear (Johnson & Gammie 2005): its amplitude, projected on the sheared wave exp(i (k_x(t) x + k_y y)), within 3% of the float64 linear solution at eight times to t = 4 (swing at t = 1.33, k_x = 4 at the end), and its phase within 0.05 rad',
    async run(gpu) {
      const A = 0.01;
      const kx0 = -2 * 2 * Math.PI;
      const ky = 2 * Math.PI;
      const gamma = 5 / 3;
      const p0 = 0.01;
      const c = Math.sqrt(gamma * p0);
      const ts = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4];
      const want = shearingWave(kx0, ky, c, A, ts);
      // Adiabatic: the pressure follows the density, p = p0 (1 + gamma delta). (Uniform pressure
      // would be an entropy wave, which the shear only carries.)
      const problem = sheet(0, (x, y) => ({ rho: 1 + A * Math.cos(kx0 * x + ky * y), p: p0 * (1 + gamma * A * Math.cos(kx0 * x + ky * y)) }));
      return withSim(gpu, problem, 128, { ndim: 2 }, async (sim) => {
        const n = sim.cfg.n;
        let amp = 0;
        let phase = 0;
        const m: Record<string, number> = {};
        for (let s = 0; s < ts.length; s++) {
          sim.cfg.tEnd = ts[s];
          sim.writeParams();
          await sim.runToEnd();
          const t = (await sim.diagnostics()).t;
          const kx = kx0 + Q * OMEGA * ky * t;
          const { u1 } = await sim.readState();
          // delta-hat = (2 / N) sum delta exp(-i (kx x + ky y)): the cos wave is half this mode.
          let re = 0;
          let im = 0;
          for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) {
              const x = (i + 0.5) / n - 0.5;
              const y = (j + 0.5) / n - 0.5;
              const d = u1[4 * (j * n + i)] - 1;
              const ph = kx * x + ky * y;
              re += d * Math.cos(ph);
              im -= d * Math.sin(ph);
            }
          re *= 2 / (n * n);
          im *= 2 / (n * n);
          const w = want[s];
          const ampErr = Math.abs(Math.hypot(re, im) / Math.hypot(w.re, w.im) - 1);
          let dph = Math.atan2(im, re) - Math.atan2(w.im, w.re);
          while (dph > Math.PI) dph -= 2 * Math.PI;
          while (dph < -Math.PI) dph += 2 * Math.PI;
          amp = Math.max(amp, ampErr);
          phase = Math.max(phase, Math.abs(dph));
          m[`amplitude_t${ts[s]}`] = Math.hypot(re, im) / A;
          m[`expected_t${ts[s]}`] = Math.hypot(w.re, w.im) / A;
        }
        m.amplitude_error = amp;
        m.phase_error = phase;
        return { pass: amp < 0.03 && phase < 0.05, metrics: m };
      });
    },
  },
  {
    id: 'shear-flux',
    name: 'Magnetized shearing box: net flux and divergence',
    criterion: 'A Keplerian box (3D, 16^3) with a net vertical field of 0.1, a periodic field from A_y = 0.02 cos(2 pi x) sin(2 pi z) / 2 pi crossing the x faces, and a velocity kick: over two orbits the net vertical flux (the mean b_z) and the net radial flux (the mean b_x) keep to 1e-6 of the vertical field, div B stays at round-off (below 1e-5), and the mass keeps to 1e-6',
    async run(gpu) {
      const k = 2 * Math.PI;
      const problem = sheet(4 * Math.PI, (x, y, z) => ({ vx: 0.05 * Math.sin(k * y) * Math.sin(k * z), vz: 0.05 * Math.sin(k * x) }), {
        mhd: {
          potential: () => (x, _y, z) => [0, (0.02 * Math.cos(k * x) * Math.sin(k * z)) / k, 0],
          face: () => (axis) => (axis === 2 ? 0.1 : 0),
        },
      });
      return withSim(gpu, problem, 16, {}, async (sim) => {
        const n = sim.cfg.n;
        const net = (f: Float32Array, a: number) => {
          let s = 0;
          for (let kk = 0; kk < n; kk++)
            for (let j = 0; j < n; j++)
              for (let i = 0; i < n; i++) s += f[4 * ((kk * (n + 1) + j) * (n + 1) + i) + a];
          return s / n ** 3;
        };
        const f0 = await faces(sim);
        const d0 = await sim.diagnostics();
        await sim.runToEnd();
        const f1 = await faces(sim);
        const d1 = await sim.diagnostics();
        const m = {
          bz_drift: Math.abs(net(f1, 2) - net(f0, 2)) / 0.1,
          bx_drift: Math.abs(net(f1, 0) - net(f0, 0)) / 0.1,
          net_bz: net(f1, 2),
          net_by: net(f1, 1),
          div_b: d1.magnetic!.divB,
          mass_change: Math.abs(d1.mass / d0.mass - 1),
          steps: d1.steps,
        };
        return { pass: m.bz_drift < 1e-6 && m.bx_drift < 1e-6 && m.div_b < 1e-5 && m.mass_change < 1e-6, metrics: m };
      });
    },
  },
  {
    id: 'shear-mri',
    name: 'Magnetized shearing box: the magnetorotational instability',
    criterion: 'A Keplerian box (3D, 32^3, beta = 50) threaded by the vertical field whose channel mode of one wavelength across the box is the fastest (k v_A = sqrt(15/16) Omega), seeded with v_x = 1e-4 c_s sin(kz): the mode B_x(k) grows at 0.75 Omega within 5%, fitted from t = 3 to t = 6; and the disk diagnostics in the local frame give the vertical quality factor 2 pi v_A / (Omega dx) within 2% and a positive Maxwell stress',
    async run(gpu) {
      const k = 2 * Math.PI;
      const vA = Math.sqrt(15 / 16) / k;
      const p0 = 0.6;
      const cs = Math.sqrt((5 / 3) * p0);
      const problem = sheet(0, (_x, _y, z) => ({ p: p0, vx: 1e-4 * cs * Math.sin(k * z) }), {
        mhd: { face: () => (axis) => (axis === 2 ? vA : 0) },
      });
      return withSim(gpu, problem, 32, {}, async (sim) => {
        const n = sim.cfg.n;
        const mode = async () => {
          const f = await faces(sim);
          let sn = 0;
          let cn = 0;
          for (let kk = 0; kk < n; kk++) {
            let bx = 0;
            for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) bx += f[4 * ((kk * (n + 1) + j) * (n + 1) + i)];
            bx /= n * n;
            const z = (kk + 0.5) / n - 0.5;
            sn += bx * Math.sin(k * z);
            cn += bx * Math.cos(k * z);
          }
          return Math.hypot(sn, cn) * (2 / n);
        };
        const ts: number[] = [];
        const ls: number[] = [];
        for (let i = 0; i <= 6; i++) {
          sim.cfg.tEnd = 3 + 0.5 * i;
          sim.writeParams();
          await sim.runToEnd();
          ts.push((await sim.diagnostics()).t);
          ls.push(Math.log(await mode()));
        }
        // Least-squares slope of ln |B_x(k)| against t.
        const tm = ts.reduce((a, b) => a + b) / ts.length;
        const lm = ls.reduce((a, b) => a + b) / ls.length;
        let num = 0;
        let den = 0;
        ts.forEach((t, i) => {
          num += (t - tm) * (ls[i] - lm);
          den += (t - tm) ** 2;
        });
        const rate = num / den / OMEGA;
        const d = await sim.diagnostics();
        // The disk diagnostics in the local frame (mhd/mri.wgsl): Q_z = 2 pi v_A / (Omega dx),
        // the vertical field being unchanged by the channel mode, and a positive Maxwell stress.
        const qz = (2 * Math.PI * vA) / (OMEGA / n);
        const m = {
          growth_rate: rate, expected: 0.75, error: Math.abs(rate / 0.75 - 1), amplitude_end: Math.exp(ls[ls.length - 1]), steps: d.steps,
          q_z: d.magnetic?.disk?.qz ?? 0, q_z_expected: qz, alpha_maxwell: d.magnetic?.disk?.alphaMaxwell ?? 0,
        };
        return { pass: m.error < 0.05 && Math.abs(m.q_z / qz - 1) < 0.02 && m.alpha_maxwell > 0, metrics: m };
      });
    },
  },
  {
    id: 'shear-epicycle',
    name: 'Shearing box: epicycles',
    criterion: 'A uniform kick of 0.01 along x on the background flow of a Keplerian sheet (2D, 32^2) oscillates at the epicyclic frequency kappa = sqrt(2 (2 - q)) Omega: after one epicycle the kick is back to within 1e-3 of itself in phase and amplitude, and the epicyclic energy (dv_x^2 + 2 dv_y^2 / (2 - q)) / 2 keeps to 1e-4 throughout',
    async run(gpu) {
      const kappa = Math.sqrt(2 * (2 - Q)) * OMEGA;
      const period = (2 * Math.PI) / kappa;
      const kick = 0.01;
      return withSim(gpu, sheet(0, (x) => ({ vx: kick, vy: -Q * OMEGA * x })), 32, { ndim: 2 }, async (sim) => {
        const energy = (d: { momentum: number[]; mass: number }) => {
          const vx = d.momentum[0] / d.mass;
          const vy = d.momentum[1] / d.mass;
          return 0.5 * (vx * vx + (2 / (2 - Q)) * vy * vy);
        };
        const d0 = await sim.diagnostics();
        const e0 = energy(d0);
        let worst = 0;
        let last = d0;
        for (let i = 1; i <= 8; i++) {
          sim.cfg.tEnd = (period * i) / 8;
          sim.writeParams();
          await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(energy(last) / e0 - 1));
          // Where it should be: dv_x = kick cos(kappa t), dv_y = -kick (2 - q) Omega / kappa sin(kappa t).
        }
        const vx = last.momentum[0] / last.mass;
        const vy = last.momentum[1] / last.mass;
        const off = Math.hypot(vx - kick, vy) / kick;
        return { pass: off < 1e-3 && worst < 1e-4, metrics: { return_offset: off, energy_drift: worst, vx, vy, kappa } };
      });
    },
  },
];
