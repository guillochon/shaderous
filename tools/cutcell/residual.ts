// The scheme's error in one step from the exact potential flow past a cylinder (steady: every
// cell's acceleration should be nil): where it is, and how it scales.
// node --experimental-strip-types tools/cutcell/residual.ts [option=value ...]
import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
const extra: Record<string, number | boolean> = {};
for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); extra[k] = v === 'true' ? true : v === 'false' ? false : Number(v); }
const R = 0.25, cx = 0.003, cy = 0.007, box = 4;
const pot = (x: number, y: number) => {
  const X = x - cx, Y = y - cy, r2 = X * X + Y * Y, R2 = R * R;
  if (r2 < R2) return { vx: 0, vy: 0 };
  return { vx: 1 - (R2 * (X * X - Y * Y)) / (r2 * r2), vy: (-2 * R2 * X * Y) / (r2 * r2) };
};
for (const across of [16, 32, 64]) {
  const dx = (2 * R) / across, N = Math.round(box / dx);
  const c = 10;
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0, c, tol: 1e-12, subSteps: 1, ...extra }, rigid(circle(R), [cx, cy]), (x, y) => {
    const v = pot(x, y);
    return { rho: 1 + (0.5 * (1 - v.vx * v.vx - v.vy * v.vy)) / (c * c), ...v };
  });
  const dt = 0.05 * dx;
  // Two steps to let the solve bring the field to its discrete balance, then the one measured.
  for (let w = 0; w < Number(extra.warm ?? 3); w++) s.step(dt);
  const v0x = new Float64Array(s.n), v0y = new Float64Array(s.n);
  for (let k = 0; k < s.n; k++) if (s.Q[k] > 0) { v0x[k] = s.Mx[k] / s.Q[k]; v0y[k] = s.My[k] / s.Q[k]; }
  s.step(dt);
  // Distance of each cell from the wall, in cells; the residual by band.
  const bands = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // cut cells, within 3 cells, beyond
  let worst = 0, wk = 0;
  for (let k = 0; k < s.n; k++) {
    if (!(s.Q[k] > 0) || s.g.phi[k] <= 0) continue;
    const i = k % N, j = Math.floor(k / N);
    const x = -box / 2 + (i + 0.5) * dx - cx, y = -box / 2 + (j + 0.5) * dx - cy;
    const d = (Math.hypot(x, y) - R) / dx;
    const r = Math.hypot(s.Mx[k] / s.Q[k] - v0x[k], s.My[k] / s.Q[k] - v0y[k]) / dt;
    const b = s.g.phi[k] < 1 ? 0 : d < 3 ? 1 : 2;
    bands[b][0] += r * s.g.phi[k]; bands[b][1] += s.g.phi[k]; bands[b][2] = Math.max(bands[b][2], r);
    if (r > worst) { worst = r; wk = k; }
  }
  const f = (b: number[]) => `mean ${(b[0] / b[1]).toExponential(2)} max ${b[2].toExponential(2)}`;
  console.log(`${across} across ${JSON.stringify(extra)}: cut cells ${f(bands[0])} | within 3 cells ${f(bands[1])} | beyond ${f(bands[2])} | worst at phi ${s.g.phi[wk].toFixed(3)}`);
}
