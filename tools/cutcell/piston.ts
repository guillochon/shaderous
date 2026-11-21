import { CutCell, cutDefaults, rectangle, type Body } from '../../src/reference/cutcell.ts';
// The open-ended piston, started smoothly over tau (longer than sound's crossing) to speed U.
const U = 0.5, tau = 0.5;
const x = (t: number) => (t < tau ? U * (t / 2 - (tau / (2 * Math.PI)) * Math.sin((Math.PI * t) / tau)) : U * (t - tau / 2));
const v = (t: number) => (t < tau ? (U / 2) * (1 - Math.cos((Math.PI * t) / tau)) : U);
for (const N of [32, 64, 128]) {
  const L = 2, dx = L / N;
  const body: Body = { shape: rectangle(0.2, 4), pos: (t) => [-0.53 + x(t), 0], angle: () => 0, vel: (t) => [v(t), 0], spin: () => 0 };
  const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -L / 2, y0: -L / 2, nu: 0.01 }, body, () => ({ rho: 1, vx: 0, vy: 0 }));
  const f: number[] = [], sp: number[] = [];
  while (s.t < 1.5) {
    s.step(Math.min(s.dtMax(), 1.5 - s.t + 1e-12));
    if (s.t > 0.8) {
      f.push(s.force.px + s.force.vx);
      let m = 0, q = 0; for (let k = 0; k < s.n; k++) { m += s.Mx[k]; q += s.Q[k]; } sp.push(m / q);
    }
  }
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  const rms = (a: number[]) => Math.sqrt(mean(a.map((y) => y * y)));
  console.log(`n=${N}: liquid speed ${mean(sp).toFixed(5)} (${Math.min(...sp).toFixed(4)}..${Math.max(...sp).toFixed(4)}; 0.5 wanted), force mean ${mean(f).toExponential(2)} rms ${rms(f).toExponential(2)} (dynamic pressure x height ${(0.5 * U * U * L).toFixed(3)})`);
}
