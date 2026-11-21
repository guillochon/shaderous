import { CutCell, cutDefaults, circle, rectangle, rigid } from '../../src/reference/cutcell.ts';
// The open-ended piston: a plate across a periodic box, sliding at 0.5; the liquid must move with it.
for (const N of [32, 64]) {
  const L = 2, dx = L / N;
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -L / 2, y0: -L / 2, nu: 0.01 }, rigid(rectangle(0.2, 4), [-0.53, 0], [0.5, 0]), () => ({ rho: 1, vx: 0, vy: 0 }));
  let fmax = 0;
  while (s.t < 1) { s.step(Math.min(s.dtMax(), 1 - s.t + 1e-12)); if (s.t > 0.2) fmax = Math.max(fmax, Math.abs(s.force.px + s.force.vx)); }
  let m = 0, q = 0, lo = Infinity, hi = -Infinity;
  for (let k = 0; k < s.n; k++) if (s.Q[k] > 0) { m += s.Mx[k]; q += s.Q[k]; const v = s.Mx[k] / s.Q[k]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  console.log(`piston n=${N}: liquid momentum / mass ${(m / q).toFixed(5)} (0.5 wanted), cell speeds ${lo.toFixed(4)}..${hi.toFixed(4)}, |force| after t=0.2 at most ${fmax.toExponential(2)}, lost ${s.lost.toExponential(1)}`);
}
// The inviscid cylinder in a stream: its drag should vanish; measured over t = 0.3 to 0.6.
for (const across of [8, 16, 32]) {
  const box = 4, D = 0.5, dx = D / across, N = Math.round(box / dx);
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0, c: 10, tol: 1e-8 }, rigid(circle(D / 2), [0.003, 0.007]), () => ({ rho: 1, vx: 1, vy: 0 }));
  let f = 0, k = 0;
  while (s.t < 0.6) { s.step(Math.min(s.dtMax(), 0.6 - s.t + 1e-12)); if (s.t >= 0.3) { f += s.force.px; k++; } }
  console.log(`inviscid cylinder ${across} across: mean drag coefficient ${(f / k / (0.5 * D)).toExponential(3)}`);
}
