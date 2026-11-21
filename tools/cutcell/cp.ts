import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
const across = Number(process.argv[2] ?? 32);
const box = 4, D = 0.5, dx = D / across, N = Math.round(box / dx), cx = 0.003, cy = 0.007, R = D / 2;
const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0, c: 10, tol: 1e-9, subSteps: 1 }, rigid(circle(R), [cx, cy]), (x, y) => {
  const X = x - cx, Y = y - cy, r2 = X * X + Y * Y, R2 = R * R;
  if (r2 < R2) return { rho: 1, vx: 0, vy: 0 };
  return { rho: 1, vx: 1 - (R2 * (X * X - Y * Y)) / (r2 * r2), vy: (-2 * R2 * X * Y) / (r2 * r2) };
});
while (s.t < 0.3) s.step(Math.min(s.dtMax(), 0.3 - s.t + 1e-12));
// Far pressure: mean over cells at |x| > 1.5
let pf = 0, nf = 0; for (let k = 0; k < s.n; k++) { const i = k % N; const x = -box / 2 + (i + 0.5) * dx; if (Math.abs(x) > 1.5) { pf += s.p[k]; nf++; } } pf /= nf;
const bins = new Array(12).fill(0).map(() => [0, 0, 0, 0]);
for (const k of s.g.cut) {
  if (s.g.phi[k] <= 0.2) continue;
  const i = k % N, j = Math.floor(k / N);
  const x = -box / 2 + (i + 0.5) * dx - cx, y = -box / 2 + (j + 0.5) * dx - cy;
  const th = Math.atan2(y, x);
  const b = Math.min(11, Math.floor(((th + Math.PI) / (2 * Math.PI)) * 12));
  const cp = (s.p[k] - pf) / 0.5;
  const ex = 1 - 4 * Math.sin(th) ** 2;
  const v = Math.hypot(s.Mx[k], s.My[k]) / s.Q[k];
  bins[b][0] += cp; bins[b][1] += ex; bins[b][2] += v; bins[b][3]++;
}
console.log('angle(deg)  Cp  exact  |v|  exact|v| (2 sin)');
bins.forEach((b, q) => { const th = ((q + 0.5) / 12) * 360 - 180; console.log(th.toFixed(0).padStart(5), (b[0] / b[3]).toFixed(3), (b[1] / b[3]).toFixed(3), (b[2] / b[3]).toFixed(3), (2 * Math.abs(Math.sin((th * Math.PI) / 180))).toFixed(3)); });
console.log('drag Cd', (s.force.px / (0.5 * D)).toFixed(4));
