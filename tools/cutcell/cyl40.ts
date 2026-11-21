// The steady cylinder at Re = 40 (Cd 1.50 to 1.57: Fornberg 1980, Dennis & Chang 1970, Tritton
// 1959; blockage of a box 16 diameters across adds a few percent), on the cut-cell prototype.
// node --experimental-strip-types tools/cutcell/cyl40.ts [cellsAcross,...] [tEnd] [staircase 0|1] [option=value ...]
import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
const list = (process.argv[2] ?? '8,16').split(',').map(Number);
const tEnd = Number(process.argv[3] ?? 40);
const staircase = (process.argv[4] ?? '0') === '1';
const extra: Record<string, number | boolean> = {};
for (const a of process.argv.slice(5)) { const [k, v] = a.split('='); extra[k] = v === 'true' ? true : v === 'false' ? false : Number(v); }
const D = 1, box = 16;
for (const across of list) {
  const dx = D / across, N = Math.round(box / dx);
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: D / 40, c: 10, tol: 1e-8, subSteps: 1, staircase, sponge: { from: 4, rate: 4, U: 1 }, ...extra }, rigid(circle(D / 2), [-3.0037, 0.0051]), () => ({ rho: 1, vx: 1, vy: 0 }));
  let f = 0, fp = 0, k = 0;
  const t0 = performance.now();
  while (s.t < tEnd) {
    s.step(Math.min(s.dtMax(), tEnd - s.t + 1e-9));
    if (s.t > tEnd - 5 && s.t < tEnd - 1e-6) { f += s.force.px + s.force.vx; fp += s.force.px; k++; }
  }
  console.log(`Re 40, ${across} across${staircase ? ' (staircase)' : ''} ${JSON.stringify(extra)}: Cd ${(f / k / (0.5 * D)).toFixed(4)} (pressure ${(fp / k / (0.5 * D)).toFixed(4)}) over t = ${tEnd - 5} to ${tEnd}, ${s.steps} steps, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
