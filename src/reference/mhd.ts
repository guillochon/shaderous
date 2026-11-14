// Float64 CPU implementation of the MHD scheme in src/mhd/*.wgsl: the same van Leer
// predictor-corrector with constrained transport, first-order stage 1, reconstructed stage 2,
// HLLD or HLLE fluxes, the Gardiner & Stone (2005) edge fields with Athena++'s upwind weights,
// the stage-1 shock flag and HLLE fallback, and the dual-energy update. It exists so that shader
// bugs can be told apart from properties of the scheme. Faces are plain float64 (no
// compensation needed), and there are no fluff limits, gravity or composition.
import type { Eos } from '../eos/eos';
import type { Bc3 } from '../problems/types';
import { LIMITERS } from './hydro';

export interface RefMhdConfig {
  n: number;
  /** cells along z: n, or 1 in 2D */
  nz: number;
  dx: number;
  eos: Eos;
  bcLo: Bc3;
  bcHi: Bc3;
  cfl: number;
  rhoFloor: number;
  pFloor: number;
  etaDual: number;
  shockThresh: number;
  riemann: 'hlld' | 'hlle';
  limiter: 'mc' | 'vanleer' | 'minmod' | 'koren';
}

/** A face state in the frame of the face: v = (normal, t1, t2), bt = (B_t1, B_t2). */
interface Side {
  rho: number;
  v: number[];
  p: number;
  s: number;
  cs: number;
  eint: number;
  bt: number[];
}
/** Flux: [m, mom (3, rotated), E, F(B_t1), F(B_t2)], with ei, s, vn. */
interface Flux {
  f: number[];
  ei: number;
  s: number;
  vn: number;
}

const pt = (s: Side, bn: number) => s.p + 0.5 * (bn * bn + s.bt[0] ** 2 + s.bt[1] ** 2);
function cf(s: Side, bn: number): number {
  const a2 = s.cs * s.cs;
  const sum = a2 + (bn * bn + s.bt[0] ** 2 + s.bt[1] ** 2) / s.rho;
  return Math.sqrt(0.5 * (sum + Math.sqrt(Math.max(sum * sum - (4 * a2 * bn * bn) / s.rho, 0))));
}
const energy = (s: Side, bn: number) => s.eint + 0.5 * s.rho * (s.v[0] ** 2 + s.v[1] ** 2 + s.v[2] ** 2) + 0.5 * (bn * bn + s.bt[0] ** 2 + s.bt[1] ** 2);
function cons(s: Side, bn: number): number[] {
  return [s.rho, s.rho * s.v[0], s.rho * s.v[1], s.rho * s.v[2], energy(s, bn), s.bt[0], s.bt[1]];
}
function phys(s: Side, bn: number): number[] {
  const P = pt(s, bn);
  const vb = s.v[0] * bn + s.v[1] * s.bt[0] + s.v[2] * s.bt[1];
  const m = s.rho * s.v[0];
  return [m, m * s.v[0] + P - bn * bn, m * s.v[1] - bn * s.bt[0], m * s.v[2] - bn * s.bt[1], (energy(s, bn) + P) * s.v[0] - bn * vb, s.bt[0] * s.v[0] - bn * s.v[1], s.bt[1] * s.v[0] - bn * s.v[2]];
}
const lin = (a: number[], b: number[], x: number, y: number) => a.map((v, i) => x * v + y * b[i]);

function fluxOf(f: number[], vn: number, l: Side, r: Side): Flux {
  const up = f[0] >= 0;
  return { f, vn, ei: f[0] * (up ? l.eint / l.rho : r.eint / r.rho), s: f[0] * (up ? l.s : r.s) };
}

function hlle(l: Side, r: Side, bn: number): Flux {
  const sl = Math.min(l.v[0] - cf(l, bn), r.v[0] - cf(r, bn));
  const sr = Math.max(l.v[0] + cf(l, bn), r.v[0] + cf(r, bn));
  const fl = phys(l, bn);
  const fr = phys(r, bn);
  if (sl >= 0) return fluxOf(fl, l.v[0], l, r);
  if (sr <= 0) return fluxOf(fr, r.v[0], l, r);
  const ul = cons(l, bn);
  const ur = cons(r, bn);
  const f = fl.map((_, i) => (sr * fl[i] - sl * fr[i] + sl * sr * (ur[i] - ul[i])) / (sr - sl));
  const ql = l.rho * (sl - l.v[0]);
  const qr = r.rho * (sr - r.v[0]);
  return fluxOf(f, (pt(r, bn) - pt(l, bn) + l.v[0] * ql - r.v[0] * qr) / (ql - qr), l, r);
}

function hlld(l: Side, r: Side, bn: number, w: number): Flux {
  if (w >= 1) return hlle(l, r, bn);
  const sl = Math.min(l.v[0] - cf(l, bn), r.v[0] - cf(r, bn));
  const sr = Math.max(l.v[0] + cf(l, bn), r.v[0] + cf(r, bn));
  const fl = phys(l, bn);
  const fr = phys(r, bn);
  if (sl >= 0) return fluxOf(fl, l.v[0], l, r);
  if (sr <= 0) return fluxOf(fr, r.v[0], l, r);
  const ql = l.rho * (sl - l.v[0]);
  const qr = r.rho * (sr - r.v[0]);
  const sm = (qr * r.v[0] - ql * l.v[0] - pt(r, bn) + pt(l, bn)) / (qr - ql);
  const pts = (qr * pt(l, bn) - ql * pt(r, bn) + ql * qr * (r.v[0] - l.v[0])) / (qr - ql);
  const star = (s: Side, sa: number) => {
    const rho = (s.rho * (sa - s.v[0])) / (sa - sm);
    const den = s.rho * (sa - s.v[0]) * (sa - sm) - bn * bn;
    let vt = [s.v[1], s.v[2]];
    let bt = [...s.bt];
    if (Math.abs(den) > 1e-6 * (s.rho * (sa - s.v[0]) ** 2 + bn * bn)) {
      vt = [0, 1].map((i) => s.v[i + 1] - (bn * s.bt[i] * (sm - s.v[0])) / den);
      bt = [0, 1].map((i) => (s.bt[i] * (s.rho * (sa - s.v[0]) ** 2 - bn * bn)) / den);
    }
    const vb = s.v[0] * bn + s.v[1] * s.bt[0] + s.v[2] * s.bt[1];
    const vbs = sm * bn + vt[0] * bt[0] + vt[1] * bt[1];
    const e = ((sa - s.v[0]) * energy(s, bn) - pt(s, bn) * s.v[0] + pts * sm + bn * (vb - vbs)) / (sa - sm);
    return { rho, vt, bt, e, u: [rho, rho * sm, rho * vt[0], rho * vt[1], e, bt[0], bt[1]] };
  };
  const ls = star(l, sl);
  const rs = star(r, sr);
  const sql = Math.sqrt(ls.rho);
  const sqr = Math.sqrt(rs.rho);
  const sal = sm - Math.abs(bn) / sql;
  const sar = sm + Math.abs(bn) / sqr;
  const ul = cons(l, bn);
  const ur = cons(r, bn);
  let f: number[];
  if (sal >= 0) {
    f = lin(fl, lin(ls.u, ul, 1, -1), 1, sl);
  } else if (sar <= 0) {
    f = lin(fr, lin(rs.u, ur, 1, -1), 1, sr);
  } else {
    const sg = bn >= 0 ? 1 : -1;
    const isum = 1 / (sql + sqr);
    const vt = [0, 1].map((i) => (sql * ls.vt[i] + sqr * rs.vt[i] + (rs.bt[i] - ls.bt[i]) * sg) * isum);
    const bt = [0, 1].map((i) => (sql * rs.bt[i] + sqr * ls.bt[i] + sql * sqr * (rs.vt[i] - ls.vt[i]) * sg) * isum);
    const vbss = sm * bn + vt[0] * bt[0] + vt[1] * bt[1];
    if (sm >= 0) {
      const e = ls.e - sql * (sm * bn + ls.vt[0] * ls.bt[0] + ls.vt[1] * ls.bt[1] - vbss) * sg;
      const uss = [ls.rho, ls.rho * sm, ls.rho * vt[0], ls.rho * vt[1], e, bt[0], bt[1]];
      const fs = lin(fl, lin(ls.u, ul, 1, -1), 1, sl);
      f = lin(fs, lin(uss, ls.u, 1, -1), 1, sal);
    } else {
      const e = rs.e + sqr * (sm * bn + rs.vt[0] * rs.bt[0] + rs.vt[1] * rs.bt[1] - vbss) * sg;
      const uss = [rs.rho, rs.rho * sm, rs.rho * vt[0], rs.rho * vt[1], e, bt[0], bt[1]];
      const fs = lin(fr, lin(rs.u, ur, 1, -1), 1, sr);
      f = lin(fs, lin(uss, rs.u, 1, -1), 1, sar);
    }
  }
  const out = fluxOf(f, sm, l, r);
  if (w > 0) {
    const h = hlle(l, r, bn);
    return { f: out.f.map((v, i) => v + (h.f[i] - v) * w), ei: out.ei + (h.ei - out.ei) * w, s: out.s + (h.s - out.s) * w, vn: out.vn + (h.vn - out.vn) * w };
  }
  return out;
}

/** Cyclic rotation putting axis a first, and its inverse. */
const rot = (v: number[], a: number) => [v[a], v[(a + 1) % 3], v[(a + 2) % 3]];
const unrot = (v: number[], a: number) => {
  const o = [0, 0, 0];
  for (let d = 0; d < 3; d++) o[(a + d) % 3] = v[d];
  return o;
};

/** A cell's primitives: rho, v (3), p, s, cs, B (3). */
interface Q {
  rho: number;
  v: number[];
  p: number;
  s: number;
  cs: number;
  B: number[];
}

export class RefMhd {
  readonly cfg: RefMhdConfig;
  /** per cell: rho, mx, my, mz, E, e_int, rho s */
  U: Float64Array[];
  /** face fields b[a], texel (i, j, k) of a grid with one more texel per axis (one along z in 2D) */
  B: Float64Array[];
  t = 0;
  steps = 0;
  private fz: number;
  private flat: boolean;

  constructor(cfg: RefMhdConfig) {
    this.cfg = cfg;
    const cells = cfg.n * cfg.n * cfg.nz;
    this.flat = cfg.nz === 1;
    this.fz = this.flat ? 1 : cfg.nz + 1;
    this.U = Array.from({ length: 7 }, () => new Float64Array(cells));
    this.B = Array.from({ length: 3 }, () => new Float64Array((cfg.n + 1) ** 2 * this.fz));
  }

  private cell(i: number, j: number, k: number) {
    return (k * this.cfg.n + j) * this.cfg.n + i;
  }
  private tex(t: number[]) {
    const f = this.cfg.n + 1;
    return ((this.flat ? 0 : t[2]) * f + t[1]) * f + t[0];
  }
  private dims(): number[] {
    return [this.cfg.n, this.cfg.n, this.cfg.nz];
  }

  /**
   * Fill from primitive initial conditions (positions relative to the box center) and face
   * fields in the layout of mhd/mhd.ts initialFaces.
   */
  init(fn: (x: number, y: number, z: number) => { rho: number; vx: number; vy: number; vz: number; p: number; s: number }, faces: ArrayLike<number>) {
    const { n, nz, dx, eos } = this.cfg;
    for (let q = 0; q < this.B[0].length; q++) for (let a = 0; a < 3; a++) this.B[a][q] = faces[4 * q + a];
    const half = (n * dx) / 2;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const w = fn((i + 0.5) * dx - half, (j + 0.5) * dx - half, this.flat ? 0 : (k + 0.5) * dx - half);
          const eint = eos.eintFromP(w.rho, w.p);
          const o = this.cell(i, j, k);
          const B = this.bcell([i, j, k]);
          const U = this.U;
          U[0][o] = w.rho;
          U[1][o] = w.rho * w.vx;
          U[2][o] = w.rho * w.vy;
          U[3][o] = w.rho * w.vz;
          U[4][o] = eint + 0.5 * w.rho * (w.vx ** 2 + w.vy ** 2 + w.vz ** 2) + 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2);
          U[5][o] = eint;
          U[6][o] = w.rho * w.s;
        }
    this.t = 0;
    this.steps = 0;
  }

  /** (index, flip, side) along an axis, as hydro/common.wgsl bc_axis maps it. */
  private bcAxis(i: number, a: number): [number, number, number] {
    const n = this.dims()[a];
    const lo = this.cfg.bcLo[a];
    const hi = this.cfg.bcHi[a];
    if (i < 0) {
      if (lo === 0) return [Math.max(i + n, 0), 1, 0];
      if (lo === 1) return [0, 1, 0];
      if (lo === 3) return [0, 1, -1];
      return [-1 - i, -1, 0];
    }
    if (i >= n) {
      if (hi === 0) return [Math.min(i - n, n - 1), 1, 0];
      if (hi === 1) return [n - 1, 1, 0];
      if (hi === 3) return [n - 1, 1, 1];
      return [2 * n - 1 - i, -1, 0];
    }
    return [i, 1, 0];
  }

  private faceTexel(a: number, t: number[]): number[] {
    const n = this.dims();
    const s = [...t];
    for (let b = 0; b < 3; b++) {
      if (b === a) {
        if (s[a] >= n[a] && this.cfg.bcLo[a] === 0) s[a] -= n[a];
        s[a] = Math.min(Math.max(s[a], 0), n[a]);
      } else {
        s[b] = this.bcAxis(s[b], b)[0];
      }
    }
    return s;
  }
  private bface(B: Float64Array[], a: number, t: number[]): number {
    return B[a][this.tex(this.faceTexel(a, t))];
  }
  private bcell(c: number[], B = this.B): number[] {
    return [0, 1, 2].map((a) => {
      const u = [...c];
      u[a]++;
      return 0.5 * (this.bface(B, a, c) + this.bface(B, a, u));
    });
  }

  private prims(U: Float64Array[], o: number) {
    const { eos, rhoFloor, pFloor } = this.cfg;
    const rho = Math.max(U[0][o], rhoFloor);
    let p = eos.pressure(rho, U[5][o]);
    if (!(p >= pFloor)) p = pFloor;
    return { rho, v: [U[1][o] / rho, U[2][o] / rho, U[3][o] / rho], p, s: U[6][o] / rho, cs: eos.soundSpeed(rho, p) };
  }

  private load(U: Float64Array[], B: Float64Array[], c: number[]): Q {
    const m = [0, 1, 2].map((a) => this.bcAxis(c[a], a));
    const mc = m.map((x) => x[0]);
    const w = this.prims(U, this.cell(mc[0], mc[1], mc[2]));
    const v = w.v.map((x, a) => {
      const y = x * m[a][1];
      const side = m[a][2];
      return side !== 0 ? Math.max(y * side, 0) * side : y;
    });
    const Bc = this.bcell(mc, B).map((x, a) => x * m[a][1]);
    return { rho: w.rho, v, p: w.p, s: w.s, cs: w.cs, B: Bc };
  }

  private side(r: number[], a: number, lo: number[], s: number): Side {
    const { eos, rhoFloor, pFloor } = this.cfg;
    const rho = Math.max(r[0], lo[0], rhoFloor);
    const p = Math.max(r[4], lo[1], pFloor);
    void s;
    return { rho, v: rot([r[1], r[2], r[3]], a), p, s: r[5], cs: eos.soundSpeed(rho, p), eint: eos.eintFromP(rho, p), bt: [r[6], r[7]] };
  }

  /** Flux through the lower face along a of cell t, and the upwind weight. */
  private face(U: Float64Array[], B: Float64Array[], flags: Float64Array | null, t: number[], a: number, stage: number, dt: number) {
    const c = this.cfg;
    const e = [0, 0, 0];
    e[a] = 1;
    const sub = (x: number[], k: number) => x.map((v, i) => v - k * e[i]);
    const bn = this.bface(B, a, t);
    const rq = (q: Q) => {
      const b = rot(q.B, a);
      return [q.rho, q.v[0], q.v[1], q.v[2], q.p, q.s, b[1], b[2]];
    };
    const qL = this.load(U, B, sub(t, 1));
    const qR = this.load(U, B, t);
    let l = rq(qL);
    let r = rq(qR);
    if (stage === 2) {
      const ll = rq(this.load(U, B, sub(t, 2)));
      const rr = rq(this.load(U, B, sub(t, -1)));
      const lim = LIMITERS[c.limiter];
      const scales = (q: Q) => {
        const bs = Math.hypot(...q.B);
        return [q.rho, q.cs, q.cs, q.cs, q.p, 0, bs, bs];
      };
      const sL = scales(qL);
      const sR = scales(qR);
      const fl = l.map((v, i) => v + 0.5 * lim(v - ll[i], r[i] - v, sL[i]));
      const fr = r.map((v, i) => v - 0.5 * lim(rr[i] - v, v - l[i], sR[i]));
      if (fl[0] > 0 && fl[4] > 0) l = fl;
      if (fr[0] > 0 && fr[4] > 0) r = fr;
    }
    const lo = [0.5 * Math.min(qL.rho, qR.rho), 0.5 * Math.min(qL.p, qR.p)];
    const sl = this.side(l, a, lo, 0);
    const sr = this.side(r, a, lo, 0);
    let w = 0;
    if (stage === 2 && flags) {
      const fi = (x: number[]) => {
        const m = [0, 1, 2].map((d) => this.bcAxis(x[d], d)[0]);
        return flags[this.cell(m[0], m[1], m[2])];
      };
      w = Math.max(fi(sub(t, 1)), fi(t));
    }
    const F = c.riemann === 'hlld' ? hlld(sl, sr, bn, w) : hlle(sl, sr, bn);
    const mom = unrot([F.f[1], F.f[2], F.f[3]], a);
    const vc = (1024 * dt * F.f[0]) / (c.dx * (sl.rho + sr.rho));
    return { m: F.f[0], mom, e: F.f[4], bt: [F.f[5], F.f[6]], ei: F.ei, s: F.s, vn: F.vn, wgt: 0.5 + Math.min(Math.max(vc, -0.5), 0.5) };
  }

  private shockFlag(U: Float64Array[], B: Float64Array[], c: number[]): number {
    let conv = 0;
    let jump = 0;
    const axes = this.flat ? 2 : 3;
    for (let a = 0; a < axes; a++) {
      const lo = [...c];
      const hi = [...c];
      lo[a]--;
      hi[a]++;
      const m = this.load(U, B, lo);
      const p = this.load(U, B, hi);
      conv += p.v[a] - m.v[a];
      const pm = m.p + 0.5 * (m.B[0] ** 2 + m.B[1] ** 2 + m.B[2] ** 2);
      const pp = p.p + 0.5 * (p.B[0] ** 2 + p.B[1] ** 2 + p.B[2] ** 2);
      jump = Math.max(jump, Math.abs(pp - pm) / Math.min(pp, pm));
    }
    return conv < 0 && jump > this.cfg.shockThresh ? 1 : 0;
  }

  /** One stage: from (U0, B0) by dt with fluxes of (Us, Bs). Returns the new state and the stage-1 flags. */
  private stage(U0: Float64Array[], B0: Float64Array[], Us: Float64Array[], Bs: Float64Array[], flags: Float64Array | null, stage: number, dt: number) {
    const c = this.cfg;
    const n = c.n;
    const nz = c.nz;
    const axes = this.flat ? 2 : 3;
    // Fluxes of every face of cells t in [-1, n] (z: the layer in 2D).
    const fd = [n + 2, n + 2, this.flat ? 1 : nz + 2];
    const fo = [1, 1, this.flat ? 0 : 1];
    const fidx = (t: number[]) => ((this.flat ? 0 : t[2] + fo[2]) * fd[1] + (t[1] + 1)) * fd[0] + (t[0] + 1);
    const F: ReturnType<RefMhd['face']>[][] = [[], [], []];
    const dims = this.dims();
    for (let z = 0; z < fd[2]; z++)
      for (let y = 0; y < fd[1]; y++)
        for (let x = 0; x < fd[0]; x++) {
          const t = [x - 1, y - 1, z - fo[2]];
          for (let a = 0; a < axes; a++) if (t[a] >= 0 && t[a] <= dims[a]) F[a][fidx(t)] = this.face(Us, Bs, flags, t, a, stage, dt);
        }
    const fc = (a: number, t: number[]) => {
      const f = F[a][fidx(t)];
      return [f.bt[0], f.bt[1], f.wgt];
    };
    const eref = (t: number[]) => {
      const q = this.load(Us, Bs, t);
      // -v x B = B x v
      return [q.B[1] * q.v[2] - q.B[2] * q.v[1], q.B[2] * q.v[0] - q.B[0] * q.v[2], q.B[0] * q.v[1] - q.B[1] * q.v[0]];
    };
    const corner = (cc: number, t: number[]) => {
      const a = (cc + 1) % 3;
      const b = (cc + 2) % 3;
      const ea = [0, 0, 0];
      ea[a] = 1;
      const eb = [0, 0, 0];
      eb[b] = 1;
      const tma = t.map((v, i) => v - ea[i]);
      const tmb = t.map((v, i) => v - eb[i]);
      const e00 = eref(t.map((v, i) => v - ea[i] - eb[i]))[cc];
      const e10 = eref(tmb)[cc];
      const e01 = eref(tma)[cc];
      const e11 = eref(t)[cc];
      const fa1 = fc(a, t);
      const fa0 = fc(a, tmb);
      const fb1 = fc(b, t);
      const fb0 = fc(b, tma);
      const ea1 = -fa1[0];
      const ea0 = -fa0[0];
      const eb1 = fb1[1];
      const eb0 = fb0[1];
      const dl2 = (1 - fa0[2]) * (eb1 - e10) + fa0[2] * (eb0 - e00);
      const dr2 = (1 - fa1[2]) * (eb1 - e11) + fa1[2] * (eb0 - e01);
      const dl1 = (1 - fb0[2]) * (ea1 - e01) + fb0[2] * (ea0 - e00);
      const dr1 = (1 - fb1[2]) * (ea1 - e11) + fb1[2] * (ea0 - e10);
      return 0.25 * (dl1 + dr1 + dl2 + dr2 + eb0 + eb1 + ea0 + ea1);
    };
    // Edge impulses on texels [0, n]^3.
    const ed = [n + 1, n + 1, this.flat ? 1 : nz + 1];
    const eidx = (t: number[]) => ((this.flat ? 0 : t[2]) * ed[1] + t[1]) * ed[0] + t[0];
    const I = Array.from({ length: 3 }, () => new Float64Array(ed[0] * ed[1] * ed[2]));
    const dtdx = dt / c.dx;
    for (let z = 0; z < ed[2]; z++)
      for (let y = 0; y < ed[1]; y++)
        for (let x = 0; x < ed[0]; x++) {
          const t = [x, y, z];
          const q = eidx(t);
          I[2][q] = dtdx * corner(2, t);
          if (this.flat) {
            I[0][q] = -dtdx * fc(1, t)[0];
            I[1][q] = dtdx * fc(0, t)[1];
          } else {
            I[0][q] = dtdx * corner(0, t);
            I[1][q] = dtdx * corner(1, t);
          }
        }
    const newFace = (a: number, t: number[]) => {
      const b = (a + 1) % 3;
      const cc = (a + 2) % 3;
      const tb = [...t];
      tb[b]++;
      const tc = [...t];
      tc[cc]++;
      let v = this.bface(B0, a, t);
      if (!(this.flat && b === 2)) v += -I[cc][eidx(tb)] + I[cc][eidx(t)];
      if (!(this.flat && cc === 2)) v += I[b][eidx(tc)] - I[b][eidx(t)];
      return v;
    };
    const Bn = B0.map((x) => new Float64Array(x));
    for (let z = 0; z < ed[2]; z++)
      for (let y = 0; y < ed[1]; y++)
        for (let x = 0; x < ed[0]; x++) {
          const t = [x, y, z];
          for (let a = 0; a < 3; a++) {
            let ok = t[a] < dims[a] || (t[a] === dims[a] && c.bcLo[a] !== 0);
            for (let b = 0; b < 3; b++) if (b !== a && t[b] >= dims[b]) ok = false;
            // Texels that hold no face (never read) are zero, as the GPU leaves them.
            Bn[a][this.tex(t)] = ok ? newFace(a, t) : 0;
          }
        }
    const Un = U0.map((x) => new Float64Array(x));
    const flagsOut = new Float64Array(n * n * nz);
    const { eos } = c;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          const t = [i, j, k];
          const o = this.cell(i, j, k);
          const net = new Array(9).fill(0); // m, mom x3, e, ei, s, vn
          for (let a = 0; a < axes; a++) {
            const up = [...t];
            up[a]++;
            const lo = F[a][fidx(t)];
            const hi = F[a][fidx(up)];
            const vl = [lo.m, ...lo.mom, lo.e, lo.ei, lo.s, lo.vn];
            const vh = [hi.m, ...hi.mom, hi.e, hi.ei, hi.s, hi.vn];
            for (let q = 0; q < 8; q++) net[q] += vl[q] - vh[q];
          }
          const bc = this.bcell(t, Bn);
          const ph = stage === 1 ? this.prims(U0, o).p : this.prims(Us, o).p;
          if (stage === 1) flagsOut[o] = this.shockFlag(U0, B0, t);
          const rho = Math.max(U0[0][o] + dtdx * net[0], c.rhoFloor);
          const mom = [1, 2, 3].map((d) => U0[d][o] + dtdx * net[d]);
          let E = U0[4][o] + dtdx * net[4];
          let eint = U0[5][o] + dtdx * (net[5] + Math.max(ph, c.pFloor) * net[7]);
          const ke = (0.5 * (mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2)) / rho + 0.5 * (bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2);
          const eth = E - ke;
          const emin = eos.eintFromP(rho, c.pFloor);
          if (eth > c.etaDual * E && eth >= emin) {
            eint = eth;
          } else {
            if (!(eint >= emin)) eint = emin;
            E = ke + eint;
          }
          Un[0][o] = rho;
          Un[1][o] = mom[0];
          Un[2][o] = mom[1];
          Un[3][o] = mom[2];
          Un[4][o] = E;
          Un[5][o] = eint;
          Un[6][o] = U0[6][o] + dtdx * net[6];
        }
    return { U: Un, B: Bn, flags: flagsOut };
  }

  /** The timestep the GPU takes: CFL dx over the largest of |vx| + |vy| + |vz| + d c_f. */
  timestep(): number {
    const c = this.cfg;
    const d = this.flat ? 2 : 3;
    let rate = 0;
    for (let k = 0; k < c.nz; k++)
      for (let j = 0; j < c.n; j++)
        for (let i = 0; i < c.n; i++) {
          const o = this.cell(i, j, k);
          const w = this.prims(this.U, o);
          const B = this.bcell([i, j, k]);
          const fast = Math.sqrt(w.cs ** 2 + (B[0] ** 2 + B[1] ** 2 + B[2] ** 2) / w.rho);
          rate = Math.max(rate, Math.abs(w.v[0]) + Math.abs(w.v[1]) + Math.abs(w.v[2]) + d * fast);
        }
    return (c.cfl * c.dx) / rate;
  }

  step(): number {
    const dt = this.timestep();
    const h = this.stage(this.U, this.B, this.U, this.B, null, 1, 0.5 * dt);
    const f = this.stage(this.U, this.B, h.U, h.B, h.flags, 2, dt);
    this.U = f.U;
    this.B = f.B;
    this.t += dt;
    this.steps++;
    return dt;
  }
}
