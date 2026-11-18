// Float64 prototype of the implicit-pressure step (plans/incompressible.md, section 4): the flux
// split into advection, taken explicitly, and pressure, taken implicitly (Kwatra et al. 2009),
// so that the timestep follows the flow speed. It is the scheme the shaders of src/pressure are
// written from, kept here so that its choices can be measured without a GPU, and so that a
// shader bug can be told from a limit of the scheme. No gravity, viscosity or species.
//
// One step, for the conserved state (rho, rho v, E, e_int, rho s):
//   1. advect: MUSCL-Hancock with the pressure taken out of the predictor and of the fluxes;
//   2. the advected pressure p^a (equation of state with an energy), or the pressure of the
//      advected density (without one);
//   3. solve  alpha p - div(beta grad p) = source  for the new pressure (conjugate gradients);
//   4. correct the momentum, the energy and, for a gas without an energy, the mass flux, with
//      differences of face values.
import type { Eos } from '../eos/eos';
import type { Bc3 } from '../problems/types';
import { LIMITERS } from './hydro';

export interface SplitConfig {
  nx: number;
  ny: number;
  nz: number;
  dx: number;
  eos: Eos;
  /** per axis: 0 periodic, 1 outflow, 2 reflecting */
  bcLo: Bc3;
  bcHi: Bc3;
  /** Courant number of the flow speed */
  cfl: number;
  /** ceiling of the step, in acoustic steps (Courant number of the sound speed) */
  acousticCfl: number;
  rhoFloor: number;
  limiter: keyof typeof LIMITERS;
  /** the advective flux: local Lax-Friedrichs, or upwinded on the mean face velocity */
  flux: 'llf' | 'upwind';
  /**
   * p^a from the pressure equation (in flux form, with the reconstructed face pressures), from
   * the advected internal energy, or from the pressure equation by first-order upwind
   * differences of the cells' own pressures and velocities (what the shaders do: it needs
   * nothing from the flux pass).
   */
  pa: 'pressure' | 'energy' | 'upwind';
  /**
   * The pressure gradient in the predictor's velocity: none, that of the state at the start of
   * the step, or that of the last solve (lagged, as projection methods take it).
   */
  predGradP: 'none' | 'state' | 'solved';
  /**
   * A gas without an energy (its pressure a function of its density): 'consistent' redoes the
   * mass flux with the corrected face momentum (backward Euler for sound, and no divergence left
   * as the sound speed grows); 'kwatra' leaves the density as advection left it.
   */
  baro: 'consistent' | 'kwatra';
  /**
   * A gas with an energy: 'kwatra' leaves the mass, momentum and energy that advection carried
   * on the uncorrected face velocities; 'carried' moves them on to the corrected ones (the
   * difference of the two face velocities times the upwind state), so that the density and the
   * energy see the divergence the solve left, not the predictor's.
   */
  gas: 'kwatra' | 'carried';
  /** relative residual the solve stops at, and its cap on iterations */
  tol: number;
  maxIter: number;
  /** round the state to single precision after every step */
  single: boolean;
}

export const splitDefaults = {
  cfl: 0.4, acousticCfl: 20, rhoFloor: 1e-10, limiter: 'koren' as const, flux: 'llf' as const, pa: 'pressure' as const,
  predGradP: 'none' as const, baro: 'consistent' as const, gas: 'kwatra' as const, tol: 1e-10, maxIter: 2000, single: false,
};

const NQ = 6; // rho, vx, vy, vz, p, s
const NU = 8; // rho, mx, my, mz, E, e_int, rho s, and the pressure (for its advection)

export class RefSplit {
  readonly cfg: SplitConfig;
  readonly ncell: number;
  /** conserved: rho, mx, my, mz, E, e_int, rho s */
  U: Float64Array[];
  /** the pressure of the last solve */
  p: Float64Array;
  t = 0;
  dt = 0;
  steps = 0;
  /** iterations of the last solve, and the largest face divergence it left, |div v| dx */
  iters = 0;
  private W: Float64Array;
  private nbr: Int32Array[][] = [];
  private dims: number;

  constructor(cfg: SplitConfig) {
    this.cfg = cfg;
    const { nx, ny, nz } = cfg;
    this.ncell = nx * ny * nz;
    this.dims = (nx > 1 ? 1 : 0) + (ny > 1 ? 1 : 0) + (nz > 1 ? 1 : 0);
    this.U = Array.from({ length: 7 }, () => new Float64Array(this.ncell));
    this.p = new Float64Array(this.ncell);
    this.W = new Float64Array(this.ncell * 7);
    // Neighbors: an index, or -1 beyond an outflow face, -2 beyond a wall.
    const n = [nx, ny, nz];
    const stride = [1, nx, nx * ny];
    for (let a = 0; a < 3; a++) {
      const lo = new Int32Array(this.ncell);
      const hi = new Int32Array(this.ncell);
      for (let o = 0; o < this.ncell; o++) {
        const i = Math.floor(o / stride[a]) % n[a];
        lo[o] = i > 0 ? o - stride[a] : cfg.bcLo[a] === 0 ? o + (n[a] - 1) * stride[a] : -cfg.bcLo[a];
        hi[o] = i < n[a] - 1 ? o + stride[a] : cfg.bcHi[a] === 0 ? o - (n[a] - 1) * stride[a] : -cfg.bcHi[a];
      }
      this.nbr.push([lo, hi]);
    }
  }

  index(i: number, j: number, k: number): number {
    return (k * this.cfg.ny + j) * this.cfg.nx + i;
  }

  init(fn: (x: number, y: number, z: number) => { rho: number; vx: number; vy: number; vz: number; p: number; s: number }) {
    const { nx, ny, nz, dx, eos } = this.cfg;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const w = fn((i + 0.5 - nx / 2) * dx, (j + 0.5 - ny / 2) * dx, (k + 0.5 - nz / 2) * dx);
          const p = eos.hasEnergy ? w.p : eos.pressure(w.rho, 0);
          const eint = eos.eintFromP(w.rho, p);
          const o = this.index(i, j, k);
          this.U[0][o] = w.rho;
          this.U[1][o] = w.rho * w.vx;
          this.U[2][o] = w.rho * w.vy;
          this.U[3][o] = w.rho * w.vz;
          this.U[4][o] = eint + 0.5 * w.rho * (w.vx ** 2 + w.vy ** 2 + w.vz ** 2);
          this.U[5][o] = eint;
          this.U[6][o] = w.rho * w.s;
          this.p[o] = p;
        }
    this.t = 0;
    this.steps = 0;
  }

  /** Primitives of every cell; returns the largest flow rate, signal rate and pressure acceleration. */
  private primitives(): [number, number, number] {
    const { eos, rhoFloor, dx } = this.cfg;
    const [R, MX, MY, MZ, E, EI, SC] = this.U;
    const W = this.W;
    let flow = 0;
    let sound = 0;
    for (let o = 0; o < this.ncell; o++) {
      const rho = Math.max(R[o], rhoFloor);
      const v = [MX[o] / rho, MY[o] / rho, MZ[o] / rho];
      const eint = eos.hasEnergy ? E[o] - 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) : EI[o];
      const p = eos.pressure(rho, eint);
      const cs = eos.soundSpeed(rho, p);
      const w = o * 7;
      W[w] = rho;
      W[w + 1] = v[0];
      W[w + 2] = v[1];
      W[w + 3] = v[2];
      W[w + 4] = p;
      W[w + 5] = SC[o] / rho;
      W[w + 6] = cs;
      const f = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]);
      flow = Math.max(flow, f);
      sound = Math.max(sound, f + this.dims * cs);
    }
    let acc = 0;
    for (let o = 0; o < this.ncell; o++) {
      let g2 = 0;
      for (let a = 0; a < 3; a++) {
        const [l, h] = [this.nbr[a][0][o], this.nbr[a][1][o]];
        const pl = l >= 0 ? W[l * 7 + 4] : W[o * 7 + 4];
        const ph = h >= 0 ? W[h * 7 + 4] : W[o * 7 + 4];
        g2 += ((ph - pl) / (2 * dx)) ** 2;
      }
      acc = Math.max(acc, Math.sqrt(g2) / W[o * 7]);
    }
    return [flow, sound, acc];
  }

  /** A neighbor's primitives as cell `o` sees them along `axis`: its own beyond an open face, its image beyond a wall. */
  private q(o: number, nb: number, axis: number, out: Float64Array) {
    const src = (nb >= 0 ? nb : o) * 7;
    for (let v = 0; v < 7; v++) out[v] = this.W[src + v];
    if (nb === -2) out[1 + axis] = -out[1 + axis];
  }

  /** Advance one step; returns dt. `tEnd` clamps the step so the run lands on it exactly. */
  step(tEnd = Infinity): number {
    const c = this.cfg;
    const { eos, dx } = c;
    const N = this.ncell;
    const W = this.W;
    const [flow, sound, acc] = this.primitives();
    let dt = Math.min((c.cfl * dx) / Math.max(flow, 1e-300), (c.acousticCfl * c.cfl * dx) / sound);
    if (acc > 0) dt = Math.min(dt, c.cfl * Math.sqrt(dx / acc));
    dt = Math.min(dt, Math.max(tEnd - this.t, 0));
    this.dt = dt;
    const dtdx = dt / dx;
    const h = 0.5 * dtdx;
    const lim = LIMITERS[c.limiter];

    // ---- predictor: half-step primitives and edge increments, by the advective Jacobian ----
    const QH = new Float64Array(N * NQ);
    const SU = new Float64Array(N * 3 * NQ);
    const SD = new Float64Array(N * 3 * NQ);
    const m = new Float64Array(7);
    const p = new Float64Array(7);
    const s = new Float64Array(3 * NQ);
    for (let o = 0; o < N; o++) {
      const w = o * 7;
      const scale = [W[w], W[w + 6], W[w + 6], W[w + 6], eos.hasEnergy ? W[w + 4] : W[w] * W[w + 6] ** 2, 0];
      let rmin = 0;
      for (let a = 0; a < 3; a++) {
        this.q(o, this.nbr[a][0][o], a, m);
        this.q(o, this.nbr[a][1][o], a, p);
        for (let v = 0; v < NQ; v++) {
          const su = lim(W[w + v] - m[v], p[v] - W[w + v], scale[v]);
          const sd = lim(p[v] - W[w + v], W[w + v] - m[v], scale[v]);
          SU[(o * 3 + a) * NQ + v] = su;
          SD[(o * 3 + a) * NQ + v] = sd;
          s[a * NQ + v] = 0.5 * (su + sd);
          if (v === 0) rmin = Math.max(rmin, Math.abs(su), Math.abs(sd));
        }
      }
      const rho = W[w];
      const v = [W[w + 1], W[w + 2], W[w + 3]];
      const dv = s[1] + s[NQ + 2] + s[2 * NQ + 3];
      const adv = (n: number) => v[0] * s[n] + v[1] * s[NQ + n] + v[2] * s[2 * NQ + n];
      const qh = QH.subarray(o * NQ, o * NQ + NQ);
      qh[0] = rho - h * (adv(0) + rho * dv);
      for (let d = 0; d < 3; d++) {
        let gp = 0;
        if (c.predGradP === 'state') gp = s[d * NQ + 4];
        else if (c.predGradP === 'solved') {
          const [l, hh] = [this.nbr[d][0][o], this.nbr[d][1][o]];
          gp = 0.5 * (this.p[hh >= 0 ? hh : o] - this.p[l >= 0 ? l : o]);
        }
        qh[1 + d] = v[d] - h * (adv(1 + d) + gp / rho);
      }
      qh[4] = W[w + 4] - h * adv(4);
      qh[5] = W[w + 5] - h * adv(5);
      if (!(qh[0] - 0.5 * rmin > 0)) {
        for (let n = 0; n < NQ; n++) qh[n] = W[w + n];
        SU.fill(0, o * 3 * NQ, (o + 1) * 3 * NQ);
        SD.fill(0, o * 3 * NQ, (o + 1) * 3 * NQ);
      }
    }

    // ---- advective fluxes, gathered per cell ----
    const side = (o: number, axis: number, sgn: number, out: Float64Array): number => {
      const S = sgn > 0 ? SU : SD;
      const f = (n: number) => QH[o * NQ + n] + 0.5 * sgn * S[(o * 3 + axis) * NQ + n];
      const rho = Math.max(f(0), c.rhoFloor);
      const v = [f(1), f(2), f(3)];
      const pr = eos.hasEnergy ? f(4) : eos.pressure(rho, 0);
      const eint = eos.eintFromP(rho, pr);
      out[0] = rho;
      out[1] = rho * v[0];
      out[2] = rho * v[1];
      out[3] = rho * v[2];
      out[4] = eint + 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
      out[5] = eint;
      out[6] = rho * f(5);
      out[7] = pr;
      return v[axis];
    };
    const UL = new Float64Array(NU);
    const UR = new Float64Array(NU);
    const F = new Float64Array(NU);
    /** Flux through the face on side `hi` (0 lower, 1 upper) of cell o; returns the face velocity. */
    const faceFlux = (o: number, axis: number, hi: number): number => {
      const nb = this.nbr[axis][hi][o];
      if (nb === -2) {
        F.fill(0);
        return 0;
      }
      let ul: number, ur: number;
      if (nb === -1) {
        ul = ur = side(o, axis, hi ? 1 : -1, UL);
        UR.set(UL);
      } else if (hi) {
        ul = side(o, axis, 1, UL);
        ur = side(nb, axis, -1, UR);
      } else {
        ul = side(nb, axis, 1, UL);
        ur = side(o, axis, -1, UR);
      }
      const u = 0.5 * (ul + ur);
      if (c.flux === 'llf') {
        const a = Math.max(Math.abs(ul), Math.abs(ur));
        for (let n = 0; n < NU; n++) F[n] = 0.5 * (UL[n] * ul + UR[n] * ur) - 0.5 * a * (UR[n] - UL[n]);
      } else {
        const up = u >= 0 ? UL : UR;
        for (let n = 0; n < NU; n++) F[n] = u * up[n];
      }
      return u;
    };
    const next = this.U.map((a) => new Float64Array(a));
    const pa = new Float64Array(N);
    /** the mass flux through each cell's upper faces, for the mass correction */
    const FM = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    /** and the face velocity that carried it */
    const FU = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    for (let o = 0; o < N; o++) {
      const d = new Float64Array(NU);
      let dvn = 0;
      for (let a = 0; a < 3; a++)
        for (let hi = 0; hi < 2; hi++) {
          const u = faceFlux(o, a, hi);
          const sg = hi ? -1 : 1;
          for (let n = 0; n < NU; n++) d[n] += sg * F[n];
          dvn += sg * u;
          if (hi) {
            FM[a][o] = F[0];
            FU[a][o] = u;
          }
        }
      for (let n = 0; n < 7; n++) next[n][o] += dtdx * d[n];
      next[0][o] = Math.max(next[0][o], c.rhoFloor);
      pa[o] = W[o * 7 + 4] + dtdx * (d[7] - W[o * 7 + 4] * dvn);
      if (c.pa === 'upwind') {
        let adv = 0;
        for (let a = 0; a < 3; a++) {
          const [l, hh] = [this.nbr[a][0][o], this.nbr[a][1][o]];
          const u = W[o * 7 + 1 + a];
          const pc = W[o * 7 + 4];
          adv += Math.max(u, 0) * (pc - (l >= 0 ? W[l * 7 + 4] : pc)) + Math.min(u, 0) * ((hh >= 0 ? W[hh * 7 + 4] : pc) - pc);
        }
        pa[o] = W[o * 7 + 4] - dtdx * adv;
      }
    }

    // ---- the pressure equation: alpha p - div(beta grad p) = b ----
    const [R, MX, MY, MZ, E, EI] = next;
    const MOM = [MX, MY, MZ];
    const alpha = new Float64Array(N);
    const b = new Float64Array(N);
    const consistent = !eos.hasEnergy && c.baro === 'consistent';
    for (let o = 0; o < N; o++) {
      const w = o * 7;
      let rc2 = W[w] * W[w + 6] ** 2;
      if (!eos.hasEnergy) {
        // The pressure of the advected density, and the sound speed there.
        const rho = consistent ? R[o] : W[w];
        const pr = eos.pressure(rho, 0);
        if (consistent) pa[o] = pr;
        rc2 = (consistent ? 1 : rho) * eos.soundSpeed(rho, pr) ** 2;
      } else if (c.pa === 'energy') {
        // The advected internal energy already holds the part of the compression that scales the
        // state as a whole; what is left for the solve is the rest of rho c^2.
        const rho = R[o];
        const ke = (0.5 * (MX[o] ** 2 + MY[o] ** 2 + MZ[o] ** 2)) / rho;
        const e = E[o] - ke;
        pa[o] = eos.pressure(rho, e);
        const eps = 1e-6;
        const hom = (eos.pressure(rho * (1 + eps), e * (1 + eps)) - eos.pressure(rho * (1 - eps), e * (1 - eps))) / (2 * eps);
        rc2 = Math.max(rho * eos.soundSpeed(rho, pa[o]) ** 2 - hom, 1e-300);
      }
      alpha[o] = 1 / (rc2 * dt * dt);
    }
    // Face coefficients and starred face velocities, on each cell's upper faces. A consistent
    // barotropic solve is for the mass flux itself: beta = 1, and the source is what the face
    // momentum of the new state carries beyond the flux the density already took.
    const beta = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    const vs = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    /** an open face's own term: its ghost holds the cell's advected pressure at the face */
    const open = new Float64Array(N);
    for (let o = 0; o < N; o++)
      for (let a = 0; a < 3; a++) {
        const nb = this.nbr[a][1][o];
        if (nb >= 0) {
          const [rl, rr] = [R[o], R[nb]];
          if (consistent) {
            beta[a][o] = 1;
            vs[a][o] = 0.5 * (MOM[a][o] + MOM[a][nb]) - FM[a][o];
          } else {
            beta[a][o] = 2 / (rl + rr);
            vs[a][o] = (MOM[a][o] + MOM[a][nb]) / (rl + rr);
          }
        }
        for (let hi = 0; hi < 2; hi++) if (this.nbr[a][hi][o] === -1) open[o] += 2 * (consistent ? 1 : 1 / R[o]);
      }
    const idx2 = 1 / (dx * dx);
    const apply = (x: Float64Array, y: Float64Array) => {
      for (let o = 0; o < N; o++) {
        let acc2 = (alpha[o] + open[o] * idx2) * x[o];
        for (let a = 0; a < 3; a++) {
          const hi = this.nbr[a][1][o];
          const lo = this.nbr[a][0][o];
          if (hi >= 0) acc2 -= beta[a][o] * (x[hi] - x[o]) * idx2;
          if (lo >= 0) acc2 -= beta[a][lo] * (x[lo] - x[o]) * idx2;
        }
        y[o] = acc2;
      }
    };
    let bmax = 0;
    for (let o = 0; o < N; o++) {
      let div = 0;
      for (let a = 0; a < 3; a++) {
        const lo = this.nbr[a][0][o];
        const hi = this.nbr[a][1][o];
        // An open face moves at its cell's own velocity.
        const own = consistent ? 0 : MOM[a][o] / R[o];
        div += (hi >= 0 ? vs[a][o] : hi === -1 ? own : 0) - (lo >= 0 ? vs[a][lo] : lo === -1 ? own : 0);
      }
      b[o] = (alpha[o] + open[o] * idx2) * pa[o] - div / (dt * dx);
      bmax = Math.max(bmax, Math.abs(div / (dt * dx)), Math.abs(alpha[o] * (pa[o] - this.p[o])));
    }
    // Conjugate gradients with the diagonal as preconditioner, started from the last pressure.
    const x = this.p;
    const r = new Float64Array(N);
    const z = new Float64Array(N);
    const d = new Float64Array(N);
    const Ad = new Float64Array(N);
    const diag = new Float64Array(N);
    let mean = 0;
    let singular = true;
    for (let o = 0; o < N; o++) {
      let dg = alpha[o] + open[o] * idx2;
      if (dg > 1e-12 * idx2) singular = false;
      for (let a = 0; a < 3; a++) {
        if (this.nbr[a][1][o] >= 0) dg += beta[a][o] * idx2;
        const lo = this.nbr[a][0][o];
        if (lo >= 0) dg += beta[a][lo] * idx2;
      }
      diag[o] = dg;
      mean += b[o];
    }
    if (singular) for (let o = 0; o < N; o++) b[o] -= mean / N;
    apply(x, r);
    let rz = 0;
    for (let o = 0; o < N; o++) {
      r[o] = b[o] - r[o];
      z[o] = r[o] / diag[o];
      d[o] = z[o];
      rz += r[o] * z[o];
    }
    // The largest source term: the divergence, the change of the advected pressure, or (a gas
    // at rest with a pressure jump has neither) the residual the solve starts from.
    for (let o = 0; o < N; o++) bmax = Math.max(bmax, Math.abs(r[o]));
    const stop = c.tol * bmax;
    let it = 0;
    for (; it < c.maxIter; it++) {
      let rmax = 0;
      for (let o = 0; o < N; o++) rmax = Math.max(rmax, Math.abs(r[o]));
      if (rmax <= stop || rz === 0) break;
      apply(d, Ad);
      let dAd = 0;
      for (let o = 0; o < N; o++) dAd += d[o] * Ad[o];
      const al = rz / dAd;
      let rz2 = 0;
      for (let o = 0; o < N; o++) {
        x[o] += al * d[o];
        r[o] -= al * Ad[o];
        z[o] = r[o] / diag[o];
        rz2 += r[o] * z[o];
      }
      const be = rz2 / rz;
      rz = rz2;
      for (let o = 0; o < N; o++) d[o] = z[o] + be * d[o];
    }
    this.iters = it;

    // ---- correct, with differences of face values ----
    const P = this.p;
    const carried = eos.hasEnergy && c.gas === 'carried';
    const S = carried ? [R, MX, MY, MZ, E].map((a) => Float64Array.from(a)) : [];
    const DQ = carried ? S.map(() => new Float64Array(N)) : [];
    for (let o = 0; o < N; o++) {
      let de = 0;
      let dvh = 0;
      let dm = 0;
      for (let a = 0; a < 3; a++) {
        // (face pressure, corrected face velocity or mass flux change) on the lower and upper face
        const fp = [0, 0];
        const fv = [0, 0];
        for (let hi = 0; hi < 2; hi++) {
          const nb = this.nbr[a][hi][o];
          const l = hi ? o : nb;
          const rr = hi ? nb : o;
          if (nb >= 0) {
            const g = (P[rr] - P[l]) / dx;
            fp[hi] = consistent ? 0.5 * (P[l] + P[rr]) : (R[rr] * P[l] + R[l] * P[rr]) / (R[l] + R[rr]);
            fv[hi] = vs[a][l] - dt * beta[a][l] * g;
            if (carried) {
              const du = fv[hi] - FU[a][l];
              const donor = du > 0 ? l : rr;
              for (let n = 0; n < 5; n++) DQ[n][o] += (hi ? 1 : -1) * du * S[n][donor];
            }
          } else if (nb === -2) {
            fp[hi] = P[o];
          } else {
            // Open: the face holds the cell's advected pressure, and the gas follows its gradient.
            fp[hi] = pa[o];
            const g = ((hi ? 1 : -1) * 2 * (pa[o] - P[o])) / dx;
            fv[hi] = consistent ? -dt * g : MOM[a][o] / R[o] - (dt / R[o]) * g;
          }
        }
        MOM[a][o] -= dtdx * (fp[1] - fp[0]);
        de += fp[1] * fv[1] - fp[0] * fv[0];
        dvh += fv[1] - fv[0];
        dm += fv[1] - fv[0];
      }
      if (consistent) R[o] = Math.max(R[o] - dtdx * dm, c.rhoFloor);
      if (eos.hasEnergy) {
        E[o] -= dtdx * de;
        EI[o] -= dtdx * P[o] * dvh;
      }
    }
    if (carried) {
      const T = [R, MX, MY, MZ, E];
      for (let o = 0; o < N; o++) for (let n = 0; n < 5; n++) T[n][o] -= dtdx * DQ[n][o];
    }
    for (let o = 0; o < N; o++) {
      const ke = (0.5 * (MX[o] ** 2 + MY[o] ** 2 + MZ[o] ** 2)) / R[o];
      if (eos.hasEnergy) {
        EI[o] = E[o] - ke;
      } else {
        EI[o] = eos.eintFromP(R[o], eos.pressure(R[o], 0));
        E[o] = ke + EI[o];
      }
    }
    if (c.single) for (const a of next) for (let o = 0; o < N; o++) a[o] = Math.fround(a[o]);
    this.U = next;
    this.t += dt;
    this.steps++;
    return dt;
  }

  run(tEnd: number, maxSteps = 100000) {
    while (this.t < tEnd * (1 - 1e-12) && this.steps < maxSteps) this.step(tEnd);
  }

  /** Total kinetic energy (per cell volume). */
  kinetic(): number {
    const [R, MX, MY, MZ] = this.U;
    let e = 0;
    for (let o = 0; o < this.ncell; o++) e += (0.5 * (MX[o] ** 2 + MY[o] ** 2 + MZ[o] ** 2)) / R[o];
    return e;
  }
}
