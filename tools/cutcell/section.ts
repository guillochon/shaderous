// A blade section of builtin:turbine-fast (src/geometry/builtin.ts turbineRotor) in a viscous
// stream, on the cut-cell prototype: its lift and drag (means over the second half of the run) at the angle of attack and Reynolds number
// the blade meets (plans/wind-tunnel.md 8.1.5). The section is the NACA four-digit symmetric
// thickness `t`, thickened to `minT` (of the chord) where thinner: a blunt trailing edge.
// node --experimental-strip-types tools/cutcell/section.ts cellsPerChord,... t minT alphaDeg Re [tEnd] [staircase 0|1] [roundNose 0|1]
import { CutCell, cutDefaults, rigid } from '../../src/reference/cutcell.ts';
const list = (process.argv[2] ?? '10,20').split(',').map(Number);
const t = Number(process.argv[3] ?? 0.2);
const minT = Number(process.argv[4] ?? 0);
const alpha = (Number(process.argv[5] ?? 9) * Math.PI) / 180;
const re = Number(process.argv[6] ?? 5000);
const tEnd = Number(process.argv[7] ?? 6);
const staircase = (process.argv[8] ?? '0') === '1';
const roundNose = (process.argv[9] ?? '0') === '1'; // the floor only aft of the thickest point (x >= 0.3): a round nose, a blunt base
function section(n = 160): [number, number][] {
  const naca = (x: number) => 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  const yt = (x: number) => (roundNose && x < 0.3 ? naca(x) : Math.max(naca(x), minT / 2));
  const xs = Array.from({ length: n + 1 }, (_, k) => 0.5 * (1 - Math.cos((Math.PI * k) / n)));
  // Lower side leading edge to trailing edge, then the upper back (counterclockwise); with minT > 0
  // the trailing edge's two corners are both kept (the blunt base).
  const lower = xs.slice(1).map((x) => [x - 0.25, -yt(x)] as [number, number]);
  const upper = xs.slice(1).reverse().map((x) => [x - 0.25, yt(x)] as [number, number]);
  return [[-0.25, 0], ...lower, ...upper.slice(minT > 0 ? 0 : 1)];
}
// Box 8 chords, the section 1.5 upstream of the middle, a sponge from 2.5 downstream (as cyl40.ts).
const box = 8;
for (const cells of list) {
  const dx = 1 / cells, N = Math.round(box / dx);
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 1 / re, c: 10, tol: 1e-8, subSteps: 1, staircase, sponge: { from: 2.5, rate: 4, U: 1 } }, rigid(section(), [-1.4963, 0.0051], [0, 0], -alpha), () => ({ rho: 1, vx: 1, vy: 0 }));
  const t0 = performance.now();
  let lift = 0, drag = 0, dragV = 0, k = 0;
  while (s.t < tEnd) {
    s.step(Math.min(s.dtMax(), tEnd - s.t + 1e-12));
    if (s.t > tEnd / 2) { lift += s.force.py + s.force.vy; drag += s.force.px + s.force.vx; dragV += s.force.vx; k++; }
  }
  const cl = lift / k / 0.5, cd = drag / k / 0.5;
  console.log(JSON.stringify({ cells, t, minT, alpha: (alpha * 180) / Math.PI, re, staircase, cl: +cl.toFixed(4), cd: +cd.toFixed(4), cdShear: +(dragV / k / 0.5).toFixed(4), lOverD: +(cl / cd).toFixed(2), steps: s.steps, seconds: Math.round((performance.now() - t0) / 1000) }));
}
