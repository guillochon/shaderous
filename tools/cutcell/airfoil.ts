// A NACA 0012 airfoil at 5 degrees in an inviscid stream, on the cut-cell prototype: its lift
// against the potential-flow value (Cl about 0.60: thin-airfoil 2 pi alpha times 1 + 0.77 t/c),
// at chord resolutions given on the command line.
// node --experimental-strip-types tools/cutcell/airfoil.ts [cellsPerChord,...] [tEnd] [staircase 0|1] [option=value ...]
import { CutCell, cutDefaults, rigid } from '../../src/reference/cutcell.ts';
const list = (process.argv[2] ?? '14,28').split(',').map(Number);
const tEnd = Number(process.argv[3] ?? 6);
const staircase = (process.argv[4] ?? '0') === '1';
const extra: Record<string, number | boolean> = {};
for (const a of process.argv.slice(5)) { const [k, v] = a.split('='); extra[k] = v === 'true' ? true : v === 'false' ? false : Number(v); }
const alpha = (5 * Math.PI) / 180;
function naca(t = 0.12, n = 160): [number, number][] {
  const yt = (x: number) => 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  const xs = Array.from({ length: n + 1 }, (_, k) => 0.5 * (1 - Math.cos((Math.PI * k) / n)));
  const lower = xs.map((x) => [x - 0.25, -yt(x)] as [number, number]); // leading edge to trailing edge, about the quarter chord
  const upper = xs.slice(1, -1).reverse().map((x) => [x - 0.25, yt(x)] as [number, number]);
  return [...lower, ...upper]; // counterclockwise
}
const box = 8;
for (const cells of list) {
  const dx = 1 / cells, N = Math.round(box / dx);
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -box / 2, y0: -box / 2, nu: 0, c: 10, tol: 1e-8, subSteps: 1, staircase, ...extra }, rigid(naca(), [0.0037, 0.0051], [0, 0], -alpha), () => ({ rho: 1, vx: 1, vy: 0 }));
  const t0 = performance.now();
  const hist: string[] = [];
  let next = 1;
  let lift = 0, drag = 0, k = 0;
  while (s.t < tEnd) {
    s.step(Math.min(s.dtMax(), tEnd - s.t + 1e-12));
    const L = s.force.py + s.force.vy, D = s.force.px + s.force.vx;
    if (s.t >= next) { hist.push(`t=${next}: Cl ${(L / 0.5).toFixed(4)} Cd ${(D / 0.5).toFixed(4)}`); next++; }
    if (s.t > tEnd - 1) { lift += L; drag += D; k++; }
  }
  console.log(`${cells} cells along the chord${staircase ? ' (staircase)' : ''} ${JSON.stringify(extra)}: ${hist.join(', ')}; over the last unit Cl ${(lift / k / 0.5).toFixed(4)}, Cd ${(drag / k / 0.5).toFixed(4)} (${s.steps} steps, ${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}
