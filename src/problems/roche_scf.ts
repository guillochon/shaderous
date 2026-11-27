// A star in a binary's Roche potential, as the run will settle it (problems/settle.ts), solved
// ahead of the run on a small grid of its own: the field inside it (mhd/star_fields.ts) is
// tapered by this star's pressure, not by the round profile's, so that it fills the star the run
// settles, out to the tip at L1.
//
// The balance is the settling's: h(rho) + phi = C, with phi the star's own potential (solved on
// the grid), the companion's as a point, the frame's centrifugal term, and a uniform pull that
// holds the star where the orbit put it (the settling's holdPull: the rest of the binary's pull at
// the star's center of mass cancelled, with feedback on how far that center has strayed; left
// alone, a star that fills its lobe slides toward L1). C is fixed by the star's mass. No gas past the Roche lobe (where the two-point
// Roche potential is above L1's) or past the star's reach, as the settling has it. Solved by
// self-consistent-field rounds (Hachisu 1986, without his boundary points: the mass fixes C),
// from the round star, the potential by FFT convolution with 1/r on a grid padded to twice its
// size (Hockney & Eastwood): the isolated potential, exact for the grid's density.
import { lobePotential, type RocheBody } from './roche';
import type { Barotrope } from './settle';

export interface RocheStarInput {
  /** the star's center, its mass and the reach of its settling */
  center: number[];
  M: number;
  /** the round star's radius: the hold's stiffness is G M / R^3 */
  R: number;
  reach: number;
  /** the round star's density at radius r, to start from */
  rho0(r: number): number;
  barotrope: Barotrope;
  /** its pressure at a density */
  pressure(rho: number): number;
  G: number;
  omega: number;
  /** the binary's two bodies as points and the potential at L1 (BinaryFrame.lobe) */
  lobe: { bodies: RocheBody[]; phi: number };
  /** cells across the grid (a power of 2); default 32: the run's settled 0.6 M_sun donor at fill 0.97 to 10% in pressure everywhere, 1.5 s (64: 3-8%, 35 s) */
  n?: number;
}

export interface RocheStar {
  /** the pressure at (x, y, z) from the center (0 outside the star) */
  pressure(x: number, y: number, z: number): number;
  /** the density there */
  density(x: number, y: number, z: number): number;
  /** the rounds taken and the last round's largest change of density, over the central one */
  rounds: number;
  change: number;
}

/** In-place radix-2 FFT of n complex values at stride `step` from `off` (sign -1 forward, +1 inverse, unnormalized). */
function fft1(re: Float64Array, im: Float64Array, off: number, step: number, n: number, sign: number, br: Uint32Array, cs: Float64Array, sn: Float64Array) {
  for (let i = 0; i < n; i++) {
    const j = br[i];
    if (j > i) {
      const a = off + i * step;
      const b = off + j * step;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const tstep = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = cs[k * tstep];
        const wi = sign * sn[k * tstep];
        const a = off + (i + k) * step;
        const b = off + (i + k + half) * step;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** 3D FFT of an m^3 complex array (x fastest), in place. */
function fft3(re: Float64Array, im: Float64Array, m: number, sign: number) {
  const bits = Math.log2(m);
  const br = new Uint32Array(m);
  for (let i = 0; i < m; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); br[i] = r; }
  const cs = new Float64Array(m / 2);
  const sn = new Float64Array(m / 2);
  for (let k = 0; k < m / 2; k++) { cs[k] = Math.cos((2 * Math.PI * k) / m); sn[k] = Math.sin((2 * Math.PI * k) / m); }
  const m2 = m * m;
  for (let k = 0; k < m; k++) for (let j = 0; j < m; j++) fft1(re, im, k * m2 + j * m, 1, m, sign, br, cs, sn);
  for (let k = 0; k < m; k++) for (let i = 0; i < m; i++) fft1(re, im, k * m2 + i, m, m, sign, br, cs, sn);
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) fft1(re, im, j * m + i, m2, m, sign, br, cs, sn);
}

export function rocheStar(s: RocheStarInput): RocheStar {
  const n = s.n ?? 32;
  const m = 2 * n;
  const c = [s.center[0], s.center[1], s.center[2] ?? 0];
  const L = s.reach;
  const dx = (2 * L) / n;
  const pos = (i: number) => -L + (i + 0.5) * dx;
  const N3 = n * n * n;
  // The tidal potential: the companion as a point and the centrifugal term.
  const self = s.lobe.bodies.reduce((b, x, i) => (Math.hypot(x.at[0] - c[0], x.at[1] - c[1], (x.at[2] ?? 0) - c[2]) < Math.hypot(s.lobe.bodies[b].at[0] - c[0], s.lobe.bodies[b].at[1] - c[1], (s.lobe.bodies[b].at[2] ?? 0) - c[2]) ? i : b), 0);
  const others = s.lobe.bodies.filter((_, i) => i !== self);
  const om2 = s.omega * s.omega;
  const tidalAbs = (x: number, y: number, z: number) => {
    let p = -0.5 * om2 * (x * x + y * y);
    for (const b of others) p -= b.gm / Math.hypot(x - b.at[0], y - b.at[1], z - (b.at[2] ?? 0), b.soft);
    return p;
  };
  const t0 = tidalAbs(c[0], c[1], c[2]);
  const tidal = (x: number, y: number, z: number) => tidalAbs(c[0] + x, c[1] + y, c[2] + z) - t0;
  // The hold (problems/settle.ts holdPull), its gains as there.
  const e = 1e-4 * L;
  const gradAt = (p: number[]) =>
    [0, 1, 2].map((a) => {
      const q1 = [c[0] + p[0], c[1] + p[1], c[2] + p[2]];
      const q0 = [...q1];
      q1[a] += e;
      q0[a] -= e;
      return (tidalAbs(q1[0], q1[1], q1[2]) - tidalAbs(q0[0], q0[1], q0[2])) / (2 * e);
    });
  const stiff = (s.G * s.M) / s.R ** 3;
  const held = [0, 0, 0];
  const outside = (x: number, y: number, z: number) => x * x + y * y + z * z > L * L || lobePotential(s.lobe.bodies, s.omega, c[0] + x, c[1] + y, c[2] + z) > s.lobe.phi;
  // Density and pressure against the enthalpy, on a table up to well past the center's.
  const hTop = (() => {
    let hi = 1;
    const rc = s.rho0(0);
    while (s.barotrope.rhoOfH(hi) < 4 * rc) hi *= 2;
    return hi;
  })();
  const NH = 8000;
  const rhoT = new Float64Array(NH + 1);
  const pT = new Float64Array(NH + 1);
  for (let i = 0; i <= NH; i++) {
    const r = s.barotrope.rhoOfH((i / NH) * hTop);
    rhoT[i] = r;
    pT[i] = r > 0 ? s.pressure(r) : 0;
  }
  const look = (t: Float64Array, h: number) => {
    if (!(h > 0)) return 0;
    const u = Math.min(h / hTop, 1) * NH;
    const i = Math.min(Math.floor(u), NH - 1);
    return t[i] + (u - i) * (t[i + 1] - t[i]);
  };
  // The fixed part of each cell's potential, and which cells may hold gas.
  const ext = new Float64Array(N3);
  const open = new Uint8Array(N3);
  let rho = new Float64Array(N3);
  for (let k = 0; k < n; k++)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const q = (k * n + j) * n + i;
        const x = pos(i);
        const y = pos(j);
        const z = pos(k);
        ext[q] = tidal(x, y, z);
        open[q] = outside(x, y, z) ? 0 : 1;
        rho[q] = open[q] ? s.rho0(Math.hypot(x, y, z)) : 0;
      }
  // The Green's function of the padded grid: -G dV / r, and a uniform cube's own at r = 0.
  const gre = new Float64Array(m * m * m);
  const gim = new Float64Array(m * m * m);
  const dV = dx * dx * dx;
  for (let k = 0; k < m; k++)
    for (let j = 0; j < m; j++)
      for (let i = 0; i < m; i++) {
        const di = i < n ? i : m - i;
        const dj = j < n ? j : m - j;
        const dk = k < n ? k : m - k;
        const r = Math.hypot(di, dj, dk) * dx;
        gre[(k * m + j) * m + i] = r > 0 ? (-s.G * dV) / r : (-s.G * dV * 2.380077) / dx;
      }
  fft3(gre, gim, m, -1);
  const re = new Float64Array(m * m * m);
  const im = new Float64Array(m * m * m);
  const phiSelf = () => {
    re.fill(0);
    im.fill(0);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) re[(k * m + j) * m + i] = rho[(k * n + j) * n + i];
    fft3(re, im, m, -1);
    for (let q = 0; q < re.length; q++) {
      const a = re[q] * gre[q] - im[q] * gim[q];
      const b = re[q] * gim[q] + im[q] * gre[q];
      re[q] = a;
      im[q] = b;
    }
    fft3(re, im, m, 1);
    const out = new Float64Array(N3);
    const norm = 1 / (m * m * m);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) out[(k * n + j) * n + i] = re[(k * m + j) * m + i] * norm;
    return out;
  };
  // Rounds: the density that balances the potential of the last, at the star's mass.
  const massOf = (base: Float64Array, C: number) => {
    let sum = 0;
    for (let q = 0; q < N3; q++) if (open[q]) sum += look(rhoT, C - base[q]);
    return sum * dV;
  };
  let rounds = 0;
  let change = Infinity;
  let h = new Float64Array(N3);
  const rc0 = s.rho0(0);
  for (; rounds < 60 && change > 1e-4; rounds++) {
    const phi = phiSelf();
    // The star's center of mass from its place, and the pull that holds it there.
    const com = [0, 0, 0];
    let mass = 0;
    for (let k = 0; k < n; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const r = rho[(k * n + j) * n + i];
          mass += r;
          com[0] += r * pos(i);
          com[1] += r * pos(j);
          com[2] += r * pos(k);
        }
    for (let a = 0; a < 3; a++) com[a] /= mass;
    const grad = gradAt(com);
    for (let a = 0; a < 3; a++) held[a] += 0.05 * stiff * com[a];
    const lam = com.map((d, a) => -grad[a] + 0.3 * stiff * d + held[a]);
    const base = new Float64Array(N3);
    let lo = Infinity;
    for (let k = 0; k < n; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const q = (k * n + j) * n + i;
          base[q] = phi[q] + ext[q] + lam[0] * pos(i) + lam[1] * pos(j) + lam[2] * pos(k);
          if (open[q]) lo = Math.min(lo, base[q]);
        }
    // C: between the deepest potential (no mass) and that plus the table's top enthalpy.
    let a = lo;
    let b = lo + hTop;
    for (let it = 0; it < 60; it++) {
      const mid = 0.5 * (a + b);
      if (massOf(base, mid) > s.M) b = mid;
      else a = mid;
    }
    const C = 0.5 * (a + b);
    const next = new Float64Array(N3);
    change = 0;
    for (let q = 0; q < N3; q++) {
      h[q] = C - base[q];
      next[q] = open[q] ? look(rhoT, h[q]) : 0;
      change = Math.max(change, Math.abs(next[q] - rho[q]) / rc0);
    }
    rho = next;
  }
  // The enthalpy between cell centers (trilinear); the lobe and the reach cut it exactly.
  const hAt = (x: number, y: number, z: number) => {
    const u = [x, y, z].map((v) => (v + L) / dx - 0.5);
    const i0 = u.map((v) => Math.min(Math.max(Math.floor(v), 0), n - 2));
    const t = u.map((v, a) => Math.min(Math.max(v - i0[a], 0), 1));
    let sum = 0;
    for (let dk = 0; dk < 2; dk++)
      for (let dj = 0; dj < 2; dj++)
        for (let di = 0; di < 2; di++) {
          const w = (di ? t[0] : 1 - t[0]) * (dj ? t[1] : 1 - t[1]) * (dk ? t[2] : 1 - t[2]);
          sum += w * h[((i0[2] + dk) * n + (i0[1] + dj)) * n + (i0[0] + di)];
        }
    return sum;
  };
  return {
    pressure: (x, y, z) => (outside(x, y, z) ? 0 : look(pT, hAt(x, y, z))),
    density: (x, y, z) => (outside(x, y, z) ? 0 : look(rhoT, hAt(x, y, z))),
    rounds,
    change,
  };
}
