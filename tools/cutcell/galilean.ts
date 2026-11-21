// Galilean drag on the cut-cell prototype: a cylinder at rest in a stream against one moving
// through still liquid, in a periodic box (plans/wind-tunnel.md 8.1.1).
// node --experimental-strip-types tools/cutcell/galilean.ts [box] [n] [tEnd] [small] [muscl]
import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
const box = Number(process.argv[2] ?? 4), N = Number(process.argv[3] ?? 256), tEnd = Number(process.argv[4] ?? 2.2);
const small = Number(process.argv[5] ?? 0.5), muscl = (process.argv[6] ?? '1') === '1';
const dx = box / N, D = 0.25;
const staircase = (process.argv[7] ?? '0') === '1';
const cfg = { ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0.0025, c: 10, small, muscl, tol: 1e-8, subSteps: 4, staircase };
const x0 = box / 2 - 0.8;
function history(moving: boolean) {
  const s = new CutCell(cfg, moving ? rigid(circle(D / 2), [x0, 0.01], [-1, 0]) : rigid(circle(D / 2), [0, 0.01]), () => ({ rho: 1, vx: moving ? 0 : 1, vy: 0 }));
  const out: [number, number, number][] = [];
  const t0 = performance.now();
  while (s.t < tEnd) {
    const dt = Math.min(s.dtMax(), tEnd - s.t + 1e-12);
    s.step(dt);
    out.push([s.t, s.force.px + s.force.vx, s.force.py + s.force.vy]);
  }
  console.error(`${moving ? 'moving' : 'rest'}: ${s.steps} steps, ${((performance.now() - t0) / s.steps).toFixed(0)} ms/step, lost ${s.lost.toExponential(1)}`);
  return out;
}
const a = history(false), b = history(true);
// Resample on a common grid of times from 1 to tEnd, and a running mean over one cell crossed.
const at = (h: [number, number, number][], t: number) => { let i = h.findIndex((r) => r[0] >= t); if (i <= 0) i = 1; const [t0, f0] = h[i - 1], [t1, f1] = h[i]; return f0 + ((t - t0) / (t1 - t0)) * (f1 - f0); };
const ts: number[] = []; for (let t = 1; t <= tEnd - 0.01; t += 0.002) ts.push(t);
const run = (h: [number, number, number][], t: number) => { let s = 0; const k = 8; for (let i = 0; i < k; i++) s += at(h, t - (dx * i) / k); return s / k; };
const fa = ts.map((t) => run(a, t)), fb = ts.map((t) => run(b, t));
const raw = ts.map((t) => at(b, t));
const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;
const ma = mean(fa), mb = mean(fb);
const rms = Math.sqrt(mean(fa.map((v, i) => (fb[i] - v) ** 2))) / Math.abs(ma);
const noise = Math.sqrt(mean(raw.map((v, i) => (v - fb[i]) ** 2))) / Math.abs(mb);
console.log(JSON.stringify({ box, N, cells_across: D / dx, cd_rest: ma / (0.5 * D), cd_moving: mb / (0.5 * D), mean_difference: Math.abs(mb / ma - 1), rms_difference_over_mean: rms, moving_raw_noise_over_mean: noise, raw_min: Math.min(...raw), raw_max: Math.max(...raw) }));
