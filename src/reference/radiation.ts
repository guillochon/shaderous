// Float64 reference of the radiation sweep in 1D (radiation/sweep.wgsl), for the unit tests and
// as a check on the GPU: the same face flux, reconstruction, reduced speed of light per cell and
// local implicit exchange, on a line of cells of ideal gas (R = 1, so T = (gamma - 1) e / rho).
//
// The gas may move at one uniform velocity: the radiation is then carried with it as the hydro
// step carries it (radiation/advect.wgsl), upwind, before the sweeps. In `labFrame` mode the moments
// are instead lab-frame ones with the reduced speed on every term (as mixed-frame codes with a
// reduced speed of light have them): nothing carries them with the gas, and the trapped
// radiation is moved only by the v/c terms of the source, at (chat / c) v.
import { eddington, exchange, RAD_CFL, reduction, type RadSolver } from '../radiation/radiation';

export type RadBc = 'periodic' | 'outflow' | 'reflect';

export interface RefRadOptions {
  n: number;
  dx: number;
  c: number;
  chat: number;
  a: number;
  gamma: number;
  /** opacities per unit mass at (rho, T) */
  opacity: (rho: number, T: number) => { planck: number; rosseland: number };
  bc: RadBc;
  localC: boolean;
  solver?: RadSolver;
  /**
   * Courant number of the sweep: dt_r = cfl dx / (d chat). Default 0.5: linear reconstruction with
   * a forward Euler step is stable to there (at 0.8 a streaming pulse gained energy sevenfold).
   */
  cfl?: number;
  /** dimensions the rate counts (the GPU's d): 1 here unless a test mimics 2D or 3D */
  dims?: number;
  /** a uniform gas velocity (along the line) */
  v?: number;
  labFrame?: boolean;
  /** second-order reconstruction (default) or first order */
  order?: 1 | 2;
}

/** Monotonized-central slope, as mc() in radiation/sweep.wgsl. */
export function mcSlope(a: number, b: number): number {
  if (a * b <= 0) return 0;
  const c = 0.5 * (a + b);
  return Math.sign(c) * Math.min(2 * Math.abs(a), 2 * Math.abs(b), Math.abs(c));
}

export class RefRad1D {
  readonly o: Required<Omit<RefRadOptions, 'v' | 'labFrame'>> & { v: number; labFrame: boolean };
  rho: Float64Array;
  eint: Float64Array;
  E: Float64Array;
  F: Float64Array;
  /** gas momentum gained from the radiation, per cell */
  mom: Float64Array;
  /** energy that has left through the two ends (outflow) */
  escaped = 0;
  t = 0;

  constructor(opts: RefRadOptions) {
    this.o = { solver: 'hll', cfl: RAD_CFL, dims: 1, order: 2, v: 0, labFrame: false, ...opts };
    const n = opts.n;
    this.rho = new Float64Array(n).fill(1);
    this.eint = new Float64Array(n).fill(1);
    this.E = new Float64Array(n);
    this.F = new Float64Array(n);
    this.mom = new Float64Array(n);
  }

  temperature(i: number, eint = this.eint[i]): number {
    return ((this.o.gamma - 1) * eint) / this.rho[i];
  }

  /** The longest sweep the scheme takes: cfl dx / (d chat). */
  get sweepDt(): number {
    return (this.o.cfl * this.o.dx) / (this.o.dims * this.o.chat);
  }

  /** The cell a ghost index maps to, and the sign of F there. */
  private map(i: number): [number, number] {
    const n = this.o.n;
    if (i >= 0 && i < n) return [i, 1];
    if (this.o.bc === 'periodic') return [((i % n) + n) % n, 1];
    if (this.o.bc === 'outflow') return [i < 0 ? 0 : n - 1, 1];
    // reflecting: the mirror image, F reversed
    return [i < 0 ? -1 - i : 2 * n - 1 - i, -1];
  }

  private cell(i: number): { E: number; F: number; chi: number } {
    const [m, s] = this.map(i);
    const T = this.temperature(m);
    return { E: this.E[m], F: s * this.F[m], chi: this.o.opacity(this.rho[m], T).rosseland * this.rho[m] };
  }

  /** r of cell i. */
  r(i: number): number {
    const k = this.o.opacity(this.rho[i], this.temperature(i)).rosseland;
    return this.o.labFrame ? Math.min(1, this.o.chat / this.o.c) : reduction(k * this.rho[i] * this.o.dx, this.o.chat, this.o.c, this.o.localC);
  }

  /**
   * The flux of (E, F) through the face between cells i and i + 1, from the four cells i - 1 to
   * i + 2: as face_flux in radiation/sweep.wgsl.
   */
  faceFlux(i: number): [number, number] {
    const { c } = this.o;
    const q0 = this.cell(i - 1);
    const q1 = this.cell(i);
    const q2 = this.cell(i + 1);
    const q3 = this.cell(i + 2);
    const red = (q: { E: number; F: number }) => (q.E > 0 ? q.F / (c * q.E) : 0);
    let EL = q1.E;
    let ER = q2.E;
    let fL = red(q1);
    let fR = red(q2);
    if (this.o.order === 2) {
      const eL = q1.E + 0.5 * mcSlope(q1.E - q0.E, q2.E - q1.E);
      const eR = q2.E - 0.5 * mcSlope(q2.E - q1.E, q3.E - q2.E);
      const gL = fL + 0.5 * mcSlope(fL - red(q0), fR - fL);
      const gR = fR - 0.5 * mcSlope(fR - fL, red(q3) - fR);
      // Positivity: a face whose reconstructed state is not physical takes the cells' own.
      if (eL > 0 && eR > 0 && Math.abs(gL) <= 1 && Math.abs(gR) <= 1) {
        EL = eL;
        ER = eR;
        fL = gL;
        fR = gR;
      }
    }
    const FL = fL * c * EL;
    const FR = fR * c * ER;
    const chiL = eddington(Math.abs(fL));
    const chiR = eddington(Math.abs(fR));
    // Along the line, D_xx = chi.
    const S = this.o.solver === 'glf' ? c : c * Math.min(1, Math.max(Math.sqrt(chiL), Math.sqrt(chiR), Math.abs(fL), Math.abs(fR)));
    const tau = 0.5 * (q1.chi + q2.chi) * this.o.dx;
    const eps = Math.min(1, 1 / Math.max(tau, 1e-30));
    const fluxE = 0.5 * (FL + FR) - 0.5 * eps * S * (ER - EL);
    const fluxF = 0.5 * c * c * (chiL * EL + chiR * ER) - 0.5 * eps * S * (FR - FL);
    return [fluxE, fluxF];
  }

  /** One sweep of length dt: explicit transport, then the local implicit exchange. */
  sweep(dt: number) {
    const { n, dx, c, a, gamma } = this.o;
    const flux: [number, number][] = [];
    for (let i = -1; i < n; i++) flux.push(this.faceFlux(i));
    const E1 = new Float64Array(n);
    const F1 = new Float64Array(n);
    const eint1 = new Float64Array(n);
    const eFromT = (rho: number, T: number) => (rho * T) / (gamma - 1);
    const tOf = (rho: number, e: number) => ((gamma - 1) * e) / rho;
    for (let i = 0; i < n; i++) {
      const r = this.r(i);
      const [lo, hi] = [flux[i], flux[i + 1]];
      const Es = this.E[i] - (r * dt * (hi[0] - lo[0])) / dx;
      const Fs = this.F[i] - (r * dt * (hi[1] - lo[1])) / dx;
      const T = this.temperature(i);
      const k = this.o.opacity(this.rho[i], T);
      const x = exchange(this.rho[i], this.eint[i], Math.max(Es, 0), [Fs], k.planck, k.rosseland, r, c, a, dt, eFromT, tOf);
      let E = x.E;
      // The work the radiation's force does on the gas (none at rest), taken from the radiation.
      // (The gas here is rigid, at one velocity: the GPU takes it half way through the sweep's kick.)
      if (!this.o.labFrame) E -= r * this.o.v * x.dmom[0];
      E1[i] = Math.max(E, 0);
      F1[i] = x.F[0];
      eint1[i] = x.eint;
      this.mom[i] += x.dmom[0];
    }
    if (this.o.bc === 'outflow') this.escaped += dt * (flux[n][0] - flux[0][0]);
    this.E = E1;
    this.F = F1;
    this.eint = eint1;
    this.t += dt;
  }

  /**
   * Carry the radiation with the gas over dt (uniform velocity, density and so mass flux):
   * second-order upwind in E and F per unit mass, as radiation/advect.wgsl, plus the work term
   * div(v . P).
   */
  advect(dt: number) {
    const { n, dx } = this.o;
    const v = this.o.v;
    if (v === 0 || this.o.labFrame) return;
    const up = (q: Float64Array, i: number) => {
      // face between i and i + 1
      const [l] = this.map(i);
      const [r] = this.map(i + 1);
      const [ll] = this.map(i - 1);
      const [rr] = this.map(i + 2);
      if (v >= 0) return q[l] + 0.5 * (1 - (v * dt) / dx) * mcSlope(q[l] - q[ll], q[r] - q[l]);
      return q[r] - 0.5 * (1 + (v * dt) / dx) * mcSlope(q[r] - q[l], q[rr] - q[r]);
    };
    const carry = (q: Float64Array, extra?: (i: number) => number) => {
      const f: number[] = [];
      for (let i = -1; i < n; i++) f.push(v * up(q, i) + (extra?.(i) ?? 0));
      return q.map((x, i) => x - (dt / dx) * (f[i + 1] - f[i]));
    };
    // The work term's flux v . P through the face, from the arithmetic means of the two cells.
    // (It is P : grad v that vanishes for a uniform velocity; div(v . P) does not, and the
    // sweep's - v . g cancels it in trapped radiation.)
    const vP = (i: number) => {
      const [l] = this.map(i);
      const [r] = this.map(i + 1);
      const p = (k: number) => eddington(this.E[k] > 0 ? Math.abs(this.F[k]) / (this.o.c * this.E[k]) : 0) * this.E[k];
      return v * 0.5 * (p(l) + p(r));
    };
    this.E = carry(this.E, vP);
    this.F = carry(this.F);
    // The gas's own heat moves with it too (the hydro step's job on the GPU).
    this.eint = carry(this.eint);
  }

  /**
   * Lab-frame mode's v/c terms over dt, cell by cell: the source of a mixed-frame scheme to first
   * order in v/c with the reduced speed on it, r c kappa_R rho (v E (1 + chi) / c) in the flux
   * equation (what keeps trapped radiation moving with the gas, at r times its speed).
   */
  private labSource(dt: number) {
    const { n, c } = this.o;
    for (let i = 0; i < n; i++) {
      const r = this.r(i);
      const k = this.o.opacity(this.rho[i], this.temperature(i)).rosseland;
      const chi = eddington(this.E[i] > 0 ? Math.abs(this.F[i]) / (c * this.E[i]) : 0);
      const target = this.o.v * this.E[i] * (1 + chi);
      // F relaxes to v E (1 + chi) at the rate r c kappa rho, implicitly.
      const th = r * c * k * this.rho[i] * dt;
      this.F[i] = (this.F[i] + th * target) / (1 + th);
    }
  }

  /**
   * One hydro step of length dt: the radiation carried with the gas, then `sweeps` sweeps (or as
   * many as the sweep's own step needs, if not given).
   */
  step(dt: number, sweeps?: number) {
    const k = sweeps ?? Math.max(1, Math.ceil(dt / this.sweepDt - 1e-9));
    this.advect(dt);
    for (let s = 0; s < k; s++) {
      if (this.o.labFrame) this.labSource(dt / k);
      this.sweep(dt / k);
    }
  }

  /** Total radiation energy, and total energy E_gas + E / r (per unit area). */
  totals(): { erad: number; egas: number; conserved: number } {
    let erad = 0;
    let egas = 0;
    let conserved = 0;
    for (let i = 0; i < this.o.n; i++) {
      erad += this.E[i] * this.o.dx;
      egas += this.eint[i] * this.o.dx;
      conserved += (this.eint[i] + this.E[i] / this.r(i)) * this.o.dx;
    }
    return { erad, egas, conserved };
  }
}
