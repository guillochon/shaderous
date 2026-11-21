// The prototype's Galilean problem with the body at rest: its drag's pressure and viscous parts.
import { CutCell, cutDefaults, circle, rigid } from '../../src/reference/cutcell.ts';
const box = 4, N = 256, dx = box / N, D = 0.25;
const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0.0025, c: 10, tol: 1e-8, subSteps: 4 }, rigid(circle(D / 2), [0, 0.01]), () => ({ rho: 1, vx: 1, vy: 0 }));
let fp = 0, fv = 0, k = 0;
while (s.t < 2.2) {
  s.step(Math.min(s.dtMax(), 2.2 - s.t + 1e-12));
  if (s.t > 1) { fp += s.force.px; fv += s.force.vx; k++; }
}
console.log(`prototype: Cd ${((fp + fv) / k / (0.5 * D)).toFixed(4)}, pressure ${(fp / k / (0.5 * D)).toFixed(4)}, viscous ${(fv / k / (0.5 * D)).toFixed(4)}`);
