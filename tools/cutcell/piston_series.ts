import { CutCell, cutDefaults, rectangle, type Body } from '../../src/reference/cutcell.ts';
const U = 0.5, tau = 0.5, N = Number(process.argv[2] ?? 64), small = Number(process.argv[3] ?? 0.5);
const x = (t: number) => (t < tau ? U * (t / 2 - (tau / (2 * Math.PI)) * Math.sin((Math.PI * t) / tau)) : U * (t - tau / 2));
const v = (t: number) => (t < tau ? (U / 2) * (1 - Math.cos((Math.PI * t) / tau)) : U);
const L = 2, dx = L / N;
const body: Body = { shape: rectangle(0.2, 4), pos: (t) => [-0.53 + x(t), 0], angle: () => 0, vel: (t) => [v(t), 0], spin: () => 0 };
const s = new CutCell({ ...cutDefaults, nx: N, ny: N, dx, x0: -L / 2, y0: -L / 2, nu: 0.01, small }, body, () => ({ rho: 1, vx: 0, vy: 0 }));
let line = '';
while (s.t < 1.2) {
  s.step(Math.min(s.dtMax(), 1.2 - s.t + 1e-12));
  if (s.t > 0.8) {
    const front = -0.43 + x(s.t);
    line += `${s.t.toFixed(4)} ${(((front + 1) / dx) % 1).toFixed(2)} ${(s.force.px).toFixed(3)} ${s.force.vx.toFixed(3)}\n`;
  }
}
console.log(`dx/U=${(dx / U).toFixed(4)}  columns: t, wall's place within its cell, pressure force, viscous force`);
console.log(line.split('\n').filter((_, i) => i % 2 === 0).slice(0, 40).join('\n'));
