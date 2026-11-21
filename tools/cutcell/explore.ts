// Quick checks of the cut-cell prototype (src/reference/cutcell.ts).
import { CutCell, cutDefaults, circle, rectangle, rigid } from '../../src/reference/cutcell.ts';

const N = 64, L = 2, dx = L / N;
const base = { ...cutDefaults, nx: N, ny: N, dx, x0: -L / 2, y0: -L / 2, nu: 0.01 };

function run(name: string, body: any, init: any, steps: number) {
  const s = new CutCell(base, body, init);
  const t0 = s.totals();
  let imp = [0, 0];
  let maxIt = 0;
  for (let k = 0; k < steps; k++) {
    const dt = s.dtMax();
    s.step(dt);
    imp[0] += (s.force.px + s.force.vx) * dt;
    imp[1] += (s.force.py + s.force.vy) * dt;
    maxIt = Math.max(maxIt, s.iters);
  }
  const t1 = s.totals();
  let vmax = 0, rmin = Infinity, rmax = 0;
  for (let k = 0; k < s.n; k++) if (s.g.phi[k] > 0 && s.Q[k] > 0) {
    vmax = Math.max(vmax, Math.hypot(s.Mx[k], s.My[k]) / s.Q[k]);
    rmin = Math.min(rmin, s.Q[k] / s.g.phi[k]); rmax = Math.max(rmax, s.Q[k] / s.g.phi[k]);
  }
  console.log(name, `t=${s.t.toFixed(3)} dmass=${(t1.mass / t0.mass - 1).toExponential(2)} lost=${s.lost.toExponential(2)}`,
    `balance x=${(t1.px - t0.px + imp[0]).toExponential(2)} y=${(t1.py - t0.py + imp[1]).toExponential(2)} (impulse ${imp[0].toFixed(4)})`,
    `F=${(s.force.px + s.force.vx).toFixed(4)},${(s.force.py + s.force.vy).toFixed(4)} vmax=${vmax.toFixed(4)} rho ${rmin.toFixed(5)}..${rmax.toFixed(5)} iters<=${maxIt}`);
  return s;
}
run('rest', rigid(circle(0.25), [0.01, 0.02]), () => ({ rho: 1, vx: 0, vy: 0 }), 20);
run('stream', rigid(circle(0.25), [0.01, 0.02]), () => ({ rho: 1, vx: 1, vy: 0 }), 200);
run('moving', rigid(circle(0.25), [0.3, 0.02], [-1, 0]), () => ({ rho: 1, vx: 0, vy: 0 }), 200);
run('turning box', rigid(rectangle(0.6, 0.1), [0.01, 0.02], [0, 0], 0.3, 2), () => ({ rho: 1, vx: 0, vy: 0 }), 200);
const pis = run('piston', rigid(rectangle(0.2, 4), [-0.5, 0], [0.5, 0]), () => ({ rho: 1, vx: 0, vy: 0 }), 40);
let mean = 0, cnt = 0; for (let k = 0; k < pis.n; k++) if (pis.g.phi[k] > 0.99) { mean += pis.Mx[k] / pis.Q[k]; cnt++; }
console.log('piston liquid mean vx', (mean / cnt).toFixed(5), 'wanted 0.5');
{
  const s = new CutCell(base, rigid(circle(0.25), [0.3, 0.02], [-1, 0]), () => ({ rho: 1, vx: 0, vy: 0 }));
  for (let k = 0; k < 120; k++) {
    s.step();
    let vm = 0, at = -1;
    for (let q = 0; q < s.n; q++) if (s.g.phi[q] > 0 && s.Q[q] > 0) { const v = Math.hypot(s.Mx[q], s.My[q]) / s.Q[q]; if (v > vm) { vm = v; at = q; } }
    if (k % 10 === 0 || vm > 2) console.log(k, 'vmax', vm.toFixed(3), 'phi', s.g.phi[at].toFixed(4), 'rho', (s.Q[at] / s.g.phi[at]).toFixed(4), 'i,j', at % N, Math.floor(at / N), 'w', s.g.wx[at].toFixed(3), s.g.wy[at].toFixed(3));
    if (vm > 2) break;
  }
}
