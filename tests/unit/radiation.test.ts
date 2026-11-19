import { describe, expect, it } from 'vitest';
import { eddington, eddingtonTensor, exchange, reduction, solveTemperature, TAU_0 } from '../../src/radiation/radiation';
import { marshakDiffusion, RefRad1D, type RefRadOptions } from '../../src/reference/radiation';
import { codeLightSpeed, opacity, physicalOpacity } from '../../src/opacity/opacity';
import { COMPOSITIONS } from '../../src/units/composition';
import { gravityUnits } from '../../src/units/units';
import { ASTRO } from '../../src/units/constants';

const ideal = (g: number) => ({
  eFromT: (rho: number, T: number) => (rho * T) / (g - 1),
  tOf: (rho: number, e: number) => ((g - 1) * e) / rho,
});

describe('M1 closure', () => {
  it('has the right limits and grows with f', () => {
    expect(eddington(0)).toBeCloseTo(1 / 3, 14);
    expect(eddington(1)).toBeCloseTo(1, 14);
    let prev = 0;
    for (let f = 0; f <= 1; f += 0.01) {
      const x = eddington(f);
      expect(x).toBeGreaterThanOrEqual(prev);
      prev = x;
    }
  });
  it('gives a tensor of trace 1 whose wave speeds are bounded by c', () => {
    for (const f of [0, 0.3, 0.7, 1]) {
      const D = eddingtonTensor(1, [f * 0.6, f * 0.8, 0], 1);
      expect(D[0] + D[4] + D[8]).toBeCloseTo(1, 12);
      for (const d of [D[0], D[4], D[8]]) expect(Math.sqrt(d)).toBeLessThanOrEqual(1 + 1e-12);
    }
  });
});

describe('reduced speed of light per cell', () => {
  it('is chat / c in thin cells and c in opaque ones', () => {
    expect(reduction(0.01, 0.01, 1, true)).toBeCloseTo(0.01, 14);
    expect(reduction(TAU_0 * 100, 0.01, 1, true)).toBe(1);
    expect(reduction(TAU_0 * 10, 0.01, 1, true)).toBeCloseTo(0.1, 12);
    expect(reduction(1e6, 0.01, 1, false)).toBeCloseTo(0.01, 14);
  });
});

describe('local implicit exchange', () => {
  const { eFromT, tOf } = ideal(5 / 3);
  it('solves the temperature equation as bisection does', () => {
    for (const [K, C] of [[1, 3], [1e-6, 2], [1e4, 0.5], [0, 1]]) {
      const T = solveTemperature(1, K, C, eFromT, tOf);
      let lo = 0;
      let hi = 10;
      for (let i = 0; i < 200; i++) {
        const m = 0.5 * (lo + hi);
        if (eFromT(1, m) + K * m ** 4 > C) hi = m;
        else lo = m;
      }
      expect(T).toBeCloseTo(lo, 10);
    }
  });
  it('reaches the equilibrium of the pair at enormous optical depth', () => {
    const a = 1;
    const x = exchange(1, 1, 0.2, [0.1, 0, 0], 1e20, 1e20, 1, 1, a, 1, eFromT, tOf);
    // e(T) + a T^4 = 1.2, E = a T^4
    expect(x.eint + x.E).toBeCloseTo(1.2, 12);
    expect(x.E).toBeCloseTo(a * x.T ** 4, 12);
    expect(Math.abs(x.F[0])).toBeLessThan(1e-18);
    expect(x.dmom[0]).toBeCloseTo(0.1, 12);
  });
});

describe('opacities and constants in code units', () => {
  it('has the speed of light and electron scattering of a solar-mass, solar-radius unit system', () => {
    const u = gravityUnits('sun', ASTRO.Rsun, ASTRO.Msun);
    const c = codeLightSpeed(u);
    // c / sqrt(G M / R) = 2.998e8 / 4.37e5
    expect(c).toBeCloseTo(299792458 / Math.sqrt((6.6743e-11 * ASTRO.Msun) / ASTRO.Rsun), 0);
    const k = physicalOpacity(u, COMPOSITIONS.solar);
    // 0.2 (1 + X) cm^2/g, X = 0.7381, in R_sun^2 / M_sun
    const unit = (ASTRO.Rsun ** 2 / ASTRO.Msun) * 10;
    expect(k.scattering! * unit).toBeCloseTo(0.2 * (1 + 0.7381), 2);
    // Kramers falls as T^-3.5
    const p = { absorption: 0, scattering: 0, kramers: 1, tFloor: 0 };
    expect(opacity('kramers', p, 1, 2).planck / opacity('kramers', p, 1, 1).planck).toBeCloseTo(2 ** -3.5, 12);
  });
});

const gaussianVariance = (s: RefRad1D, floor: number) => {
  let m0 = 0;
  let m1 = 0;
  let m2 = 0;
  for (let i = 0; i < s.o.n; i++) {
    const x = (i + 0.5) * s.o.dx - 0.5;
    const w = s.E[i] - floor;
    m0 += w;
    m1 += w * x;
    m2 += w * x * x;
  }
  return m2 / m0 - (m1 / m0) ** 2;
};

function diffusion(tauCell: number, localC: boolean) {
  const n = 128;
  const dx = 1 / n;
  const kappa = tauCell / dx;
  const D = 1 / (3 * kappa);
  const s0 = 8 * dx;
  const t = (1.5 * s0 * s0) / D;
  // A reduced speed that makes the run about 1500 sweeps long, never above c.
  const chat = Math.min(1, (0.5 * dx * 1500) / t);
  const floor = 1e-6;
  const sim = new RefRad1D({ n, dx, c: 1, chat, a: 1, gamma: 5 / 3, bc: 'periodic', localC, opacity: () => ({ planck: 0, rosseland: kappa }) });
  for (let i = 0; i < n; i++) sim.E[i] = floor + Math.exp(-(((i + 0.5) * dx - 0.5) ** 2) / (2 * s0 * s0));
  const v0 = gaussianVariance(sim, floor);
  const sweeps = Math.ceil(t / sim.sweepDt);
  for (let k = 0; k < sweeps; k++) sim.sweep(t / sweeps);
  const v1 = gaussianVariance(sim, floor);
  return { ratio: (v1 - v0) / (2 * D * t), chat, sweeps, finite: sim.E.every((e) => Number.isFinite(e) && e >= 0) };
}

describe('1D reference scheme', () => {
  it('relaxes gas and radiation to the equilibrium temperature, exactly with r = 1', () => {
    const run = (chat: number, localC: boolean, kappa: number) => {
      const o: RefRadOptions = { n: 8, dx: 1 / 8, c: 1, chat, a: 1, gamma: 5 / 3, bc: 'reflect', localC, opacity: () => ({ planck: kappa, rosseland: kappa }) };
      const sim = new RefRad1D(o);
      sim.eint.fill(1);
      sim.E.fill(0);
      const before = sim.totals();
      for (let k = 0; k < 400; k++) sim.sweep(sim.sweepDt);
      return { T: sim.temperature(0), E: sim.E[0], drift: Math.abs(sim.totals().conserved - before.conserved) / before.conserved };
    };
    // e(T) + a T^4 = 1 with e = 1.5 T: T = 0.5652...
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 100; i++) {
      const m = 0.5 * (lo + hi);
      if (1.5 * m + m ** 4 > 1) hi = m;
      else lo = m;
    }
    const exact = run(1, true, 100);
    expect(exact.T).toBeCloseTo(lo, 8);
    expect(exact.E).toBeCloseTo(lo ** 4, 8);
    expect(exact.drift).toBeLessThan(1e-12);
    // A uniform reduced speed: the radiation weighs c / chat times too much.
    const uniform = run(0.1, false, 100);
    expect(Math.abs(uniform.T - lo) / lo).toBeGreaterThan(0.05);
    // The per-cell rule in an opaque box: the true speed, and the exact temperature.
    const local = run(0.1, true, 100);
    expect(local.T).toBeCloseTo(lo, 8);
  });

  it('diffuses at c / (3 kappa rho) from tau_c = 1 to 1e4, with the per-cell rule', () => {
    for (const tau of [1, 10, 100, 1e4]) {
      const d = diffusion(tau, true);
      expect(d.finite).toBe(true);
      // Within 10% of the true diffusion (the scheme's own dissipation adds some at tau ~ 1).
      expect(Math.abs(d.ratio - 1), `tau ${tau}: ratio ${d.ratio} at chat ${d.chat}`).toBeLessThan(0.1);
    }
  });

  it('streams a pulse through vacuum at chat, with f near 1', () => {
    const n = 256;
    const dx = 1 / n;
    const chat = 0.1;
    const sim = new RefRad1D({ n, dx, c: 1, chat, a: 1, gamma: 5 / 3, bc: 'outflow', localC: true, opacity: () => ({ planck: 0, rosseland: 0 }) });
    const floor = 1e-10;
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) * dx;
      const e = Math.exp(-(((x - 0.2) / 0.03) ** 2));
      sim.E[i] = floor + e;
      sim.F[i] = e;   // c E: free streaming to +x
    }
    const t = 4;   // chat t = 0.4
    const k = Math.ceil(t / sim.sweepDt);
    for (let s = 0; s < k; s++) sim.sweep(t / k);
    let best = 0;
    for (let i = 1; i < n; i++) if (sim.E[i] > sim.E[best]) best = i;
    const x = (best + 0.5) * dx;
    expect(x).toBeCloseTo(0.6, 1);
    expect(sim.F[best] / sim.E[best]).toBeGreaterThan(0.9);
  });

  it('carries trapped radiation with the gas (beta tau = 1e4 at chat = c / 100); the lab frame lags', () => {
    const n = 128;
    const dx = 1 / n;
    const beta = 1e-2;
    const kappa = 1e6;   // tau of the box 1e6: beta tau = 1e4
    const make = (labFrame: boolean) => {
      const sim = new RefRad1D({ n, dx, c: 1, chat: 0.01, a: 1, gamma: 5 / 3, bc: 'periodic', localC: !labFrame, v: beta, labFrame, opacity: () => ({ planck: kappa, rosseland: kappa }) });
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) * dx;
        const T = 1 + 0.5 * Math.exp(-(((x - 0.3) / 0.05) ** 2));
        sim.eint[i] = 1.5 * T;
        sim.E[i] = T ** 4;
      }
      return sim;
    };
    const peak = (s: RefRad1D) => {
      let m1 = 0;
      let m0 = 0;
      for (let i = 0; i < n; i++) {
        const w = s.E[i] - 1;
        m0 += w;
        m1 += w * (i + 0.5) * dx;
      }
      return m1 / m0;
    };
    const t = 20;   // the gas moves 0.2
    const gas = make(false);
    const lab = make(true);
    const dtH = (0.8 * dx) / beta / 4;
    for (let k = 0; k < Math.round(t / dtH); k++) {
      gas.step(dtH, 4);
      lab.step(dtH, 4);
    }
    expect(Math.abs(peak(gas) - 0.5)).toBeLessThan(0.01);
    // The lab frame's pulse has moved about (chat / c) (4/3) of the gas's way.
    expect(Math.abs(peak(lab) - 0.3)).toBeLessThan(0.02);
  });
});

/** Linear interpolation of ys(xs) at x. */
const interp = (xs: Float64Array, ys: Float64Array, x: number) => {
  for (let i = 1; i < xs.length; i++)
    if (xs[i] >= x) {
      const w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] * (1 - w) + ys[i] * w;
    }
  return ys[ys.length - 1];
};

describe('Su & Olson Marshak wave', () => {
  it('has a converged float64 diffusion solution', () => {
    const a = marshakDiffusion({ c: 1, k: 1, u0: 1, length: 30, cells: 1500, t: 10, steps: 2000, marshak: true });
    const b = marshakDiffusion({ c: 1, k: 1, u0: 1, length: 30, cells: 3000, t: 10, steps: 4000, marshak: true });
    for (const x of [0.5, 1, 2, 4]) expect(Math.abs(interp(a.x, a.u, x) - interp(b.x, b.u, x))).toBeLessThan(2e-4);
  });

  it('M1 approaches the diffusion solution as the wave becomes diffusive (Marshak boundary on both)', () => {
    const n = 128;
    const L = 16;
    const dx = L / n;
    const R = 1e-12;
    const rho = 1e6;
    // Material energy a T^4 (heat capacity 4 a T^3, epsilon = 1), with a trace of ideal gas.
    const eint = (r: number, T: number) => 1.5 * R * r * T + T ** 4;
    const temperature = (r: number, e: number) => {
      let lo = 0;
      let hi = Math.max(1, e) * 10;
      for (let i = 0; i < 100; i++) {
        const m = 0.5 * (lo + hi);
        if (eint(r, m) > e) hi = m;
        else lo = m;
      }
      return lo;
    };
    const s = new RefRad1D({ n, dx, c: 1, chat: 1, a: 1, gamma: 5 / 3, bc: 'outflow', localC: true, dims: 2, holdFirst: true, eos: { eintFromT: eint, temperature }, opacity: () => ({ planck: 1 / rho, rosseland: 1 / rho }) });
    s.rho.fill(rho);
    for (let i = 0; i < n; i++) {
      s.eint[i] = eint(rho, i === 0 ? 1 : 1e-3);
      s.E[i] = i === 0 ? 0.5 : 1e-12;
    }
    // Half-range isotropic radiation falling in: E = E_inc / 2, F = c E_inc / 4.
    s.F[0] = 0.25;
    let t = 0;
    for (const [T, xs, tol] of [[3.1623, [0.5, 1, 2], 0.03], [10, [0.5, 1, 2, 3, 5], 0.03]] as const) {
      const k = Math.ceil((T - t) / s.sweepDt);
      for (let q = 0; q < k; q++) s.sweep((T - t) / k);
      t = T;
      const ref = marshakDiffusion({ c: 1, k: 1, u0: 1, length: 30, cells: 1500, t: T, steps: 2000, marshak: true });
      for (const x of xs) {
        // Cell i's center is (i - 1/2) dx from the face where the radiation falls in.
        const i = Math.round(x / dx + 0.5);
        const xc = (i - 0.5) * dx;
        expect(Math.abs(s.E[i] - interp(ref.x, ref.u, xc)), `u at x ${xc}, tau ${T}`).toBeLessThan(tol);
        expect(Math.abs(s.temperature(i) ** 4 - interp(ref.x, ref.v, xc)), `v at x ${xc}, tau ${T}`).toBeLessThan(tol);
      }
    }
  });
});

describe('an opaque slab lit from behind', () => {
  it('shines steadily: its surface does not send the light out in pulses', () => {
    // A held cell at E = 1, seven cells of optical depth 5 each, then transparent gas (the
    // cooling sphere of the GPU tests along a line). With the monotonized-central slope on E the
    // light at the far end swung by a quarter around its mean with a period of some 40 sweeps.
    const n = 40;
    const dx = 1 / 64;
    const s = new RefRad1D({ n, dx, c: 100, chat: 5, a: 1, gamma: 5 / 3, bc: 'outflow', localC: true, dims: 2, holdFirst: true, opacity: () => ({ planck: 0, rosseland: (5 / dx) * 1e-4 }) });
    for (let i = 0; i < n; i++) {
      s.rho[i] = i <= 7 ? 1e4 : 1e-4;
      s.eint[i] = s.rho[i] * 1.5e-6;
      s.E[i] = i === 0 ? 1 : 1e-8;
    }
    while (s.t < 3) s.sweep(s.sweepDt);
    let lo = Infinity;
    let hi = 0;
    for (let k = 0; k < 320; k++) {
      s.sweep(s.sweepDt);
      lo = Math.min(lo, s.F[n - 1]);
      hi = Math.max(hi, s.F[n - 1]);
    }
    expect((hi - lo) / (hi + lo)).toBeLessThan(2e-3);
    // The converged luminosity (four times the cells) is 0.88.
    expect(Math.abs(0.5 * (hi + lo) / 0.88 - 1)).toBeLessThan(0.05);
  });
});
