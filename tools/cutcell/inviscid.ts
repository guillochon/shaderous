import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
// The inviscid cylinder in a stream: its drag should vanish. Mean drag coefficient over windows.
const muscl = (process.argv[2] ?? '1') === '1';
const extra: Record<string, unknown> = {};
for (const a of process.argv.slice(3)) { const [k, v] = a.split('='); extra[k] = v === 'true' ? true : v === 'false' ? false : isNaN(Number(v)) ? v : Number(v); }
const sizes = (extra.sizes as string | undefined)?.toString().split(':').map(Number) ?? [8, 16, 32, 64];
for (const across of sizes) {
  const box = 4, D = 0.5, dx = D / across, N = Math.round(box / dx);
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0, c: 10, tol: 1e-8, muscl, subSteps: 1, ...extra }, rigid(circle(D / 2), [0.003, 0.007]), (x, y) => {
    // The potential flow past the cylinder.
    const X = x - 0.003, Y = y - 0.007, r2 = X * X + Y * Y, R2 = (D / 2) ** 2;
    if (r2 < R2) return { rho: 1, vx: 0, vy: 0 };
    return { rho: 1, vx: 1 - (R2 * (X * X - Y * Y)) / (r2 * r2), vy: (-2 * R2 * X * Y) / (r2 * r2) };
  });
  const w = [[0.1, 0.3], [0.3, 0.6]].map(() => [0, 0]);
  while (s.t < 0.6) {
    s.step(Math.min(s.dtMax(), 0.6 - s.t + 1e-12));
    if (s.t > 0.1 && s.t <= 0.3) { w[0][0] += s.force.px; w[0][1]++; }
    if (s.t > 0.3) { w[1][0] += s.force.px; w[1][1]++; }
  }
  console.log(`${across} across ${JSON.stringify(extra)}: Cd over t=0.1-0.3 ${(w[0][0] / w[0][1] / (0.5 * D)).toExponential(3)}, over 0.3-0.6 ${(w[1][0] / w[1][1] / (0.5 * D)).toExponential(3)}`);
}
