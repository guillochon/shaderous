// Float64 CPU implementation of the hydro scheme in src/hydro/*.wgsl: same primitives pass,
// timestep, MUSCL-Hancock predictor, positivity fallback, shock-flagged HLLC/HLL blend,
// dual-energy update and floors, in the same gather form. It exists so that shader bugs can
// be told apart from limitations of the scheme. Gravity is not included.
import type { Eos } from '../eos/eos';
import type { Bc3 } from '../problems/types';

export interface RefConfig {
  nx: number;
  ny: number;
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
  riemann: 'hllc' | 'hll';
  limiter: 'mc' | 'vanleer' | 'minmod' | 'koren';
}

/** Edge increment from the differences (far, near) and the quantity's natural scale. */
type Limiter = (a: number, b: number, scale: number) => number;
export const KOREN_SMOOTH = 0.01;
export const LIMITERS: Record<RefConfig['limiter'], Limiter> = {
  mc: (a, b) => (a * b > 0 ? Math.sign(a) * Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b)) : 0),
  vanleer: (a, b) => (a * b > 0 ? (2 * a * b) / (a + b) : 0),
  minmod: (a, b) => (a * b > 0 ? Math.sign(a) * Math.min(Math.abs(a), Math.abs(b)) : 0),
  // edge increment (far, near): limited third order (Koren)
  koren: (far, near, scale) => {
    const free = (far + 2 * near) / 3;
    const lim = far * near > 0 ? Math.sign(near) * Math.min(2 * Math.abs(far), 2 * Math.abs(near), Math.abs(free)) : 0;
    const s = KOREN_SMOOTH * scale;
    const w = Math.min(Math.max(1.5 - (far * far + near * near) / Math.max(s * s, 1e-30), 0), 1);
    return lim + (free - lim) * w;
  },
};

interface Side {
  rho: number;
  v: [number, number, number]; // rotated so v[0] is normal to the face
  p: number;
  s: number;
  cs: number;
  eint: number;
}
interface Flux {
  m: number;
  mom: [number, number, number];
  e: number;
  ei: number;
  s: number;
  vn: number;
}

export function riemannFlux(l: Side, r: Side, wHll: number, kind: 'hllc' | 'hll'): Flux {
  const sl = Math.min(l.v[0] - l.cs, r.v[0] - r.cs);
  const sr = Math.max(l.v[0] + l.cs, r.v[0] + r.cs);
  const ke = (s: Side) => 0.5 * s.rho * (s.v[0] ** 2 + s.v[1] ** 2 + s.v[2] ** 2);
  const el = l.eint + ke(l);
  const er = r.eint + ke(r);
  const ml = l.v.map((c) => l.rho * c);
  const mr = r.v.map((c) => r.rho * c);
  const fpl = [ml[0] * l.v[0] + l.p, ml[1] * l.v[0], ml[2] * l.v[0]];
  const fpr = [mr[0] * r.v[0] + r.p, mr[1] * r.v[0], mr[2] * r.v[0]];
  const fel = (el + l.p) * l.v[0];
  const fer = (er + r.p) * r.v[0];

  let m: number, mom: number[], e: number, vn: number;
  if (sl >= 0) {
    [m, mom, e, vn] = [ml[0], fpl, fel, l.v[0]];
  } else if (sr <= 0) {
    [m, mom, e, vn] = [mr[0], fpr, fer, r.v[0]];
  } else {
    const ql = l.rho * (sl - l.v[0]);
    const qr = r.rho * (sr - r.v[0]);
    const sm = (r.p - l.p + l.v[0] * ql - r.v[0] * qr) / (ql - qr);
    const inv = 1 / (sr - sl);
    const hm = (sr * ml[0] - sl * mr[0] + sl * sr * (r.rho - l.rho)) * inv;
    const hp = [0, 1, 2].map((i) => (sr * fpl[i] - sl * fpr[i] + sl * sr * (mr[i] - ml[i])) * inv);
    const he = (sr * fel - sl * fer + sl * sr * (er - el)) * inv;
    let cm: number, cp: number[], ce: number;
    if (sm >= 0) {
      const f = ql / (sl - sm);
      const ms = [f * sm, f * l.v[1], f * l.v[2]];
      const es = f * (el / l.rho + (sm - l.v[0]) * (sm + l.p / ql));
      cm = ml[0] + sl * (f - l.rho);
      cp = [0, 1, 2].map((i) => fpl[i] + sl * (ms[i] - ml[i]));
      ce = fel + sl * (es - el);
    } else {
      const f = qr / (sr - sm);
      const ms = [f * sm, f * r.v[1], f * r.v[2]];
      const es = f * (er / r.rho + (sm - r.v[0]) * (sm + r.p / qr));
      cm = mr[0] + sr * (f - r.rho);
      cp = [0, 1, 2].map((i) => fpr[i] + sr * (ms[i] - mr[i]));
      ce = fer + sr * (es - er);
    }
    const w = kind === 'hll' ? 1 : wHll;
    m = cm + (hm - cm) * w;
    mom = cp.map((c, i) => c + (hp[i] - c) * w);
    e = ce + (he - ce) * w;
    vn = sm;
  }
  const up = m >= 0;
  return {
    m,
    mom: mom as [number, number, number],
    e,
    ei: m * (up ? l.eint / l.rho : r.eint / r.rho),
    s: m * (up ? l.s : r.s),
    vn,
  };
}

// Primitive vector layout used below: [rho, vx, vy, vz, p, s].
const NQ = 6;

interface Pred {
  q: Float64Array;
  su: Float64Array[]; // edge increments toward the upper face along x, y, z
  sd: Float64Array[]; // and from the lower face
  flag: number; // shock detector
  firstOrder: boolean; // positivity fallback: unpredicted state, no slopes
}

export class RefHydro {
  readonly cfg: RefConfig;
  readonly ncell: number;
  /** conserved: rho, mx, my, mz, E, e_int, rho*s */
  U: Float64Array[];
  t = 0;
  dt = 0;
  steps = 0;
  private W: Float64Array; // per cell: rho, vx, vy, vz, p, s, cs
  private preds: (Pred | undefined)[] = [];

  constructor(cfg: RefConfig) {
    this.cfg = cfg;
    this.ncell = cfg.nx * cfg.ny * cfg.nz;
    this.U = Array.from({ length: 7 }, () => new Float64Array(this.ncell));
    this.W = new Float64Array(this.ncell * 7);
  }

  index(i: number, j: number, k: number): number {
    return (k * this.cfg.ny + j) * this.cfg.nx + i;
  }

  /** Fill from primitive initial conditions; positions are cell centers with the box centered on 0. */
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
        }
    this.t = 0;
    this.steps = 0;
  }

  private bcAxis(i: number, n: number, lo: number, hi: number): [number, number] {
    if (i < 0) return lo === 0 ? [((i % n) + n) % n, 1] : lo === 1 ? [0, 1] : [-1 - i, -1];
    if (i >= n) return hi === 0 ? [i % n, 1] : hi === 1 ? [n - 1, 1] : [2 * n - 1 - i, -1];
    return [i, 1];
  }

  /** [rho, vx, vy, vz, p, s, cs] at a possibly out-of-range cell. */
  private loadQ(i: number, j: number, k: number): number[] {
    const c = this.cfg;
    const [x, fx] = this.bcAxis(i, c.nx, c.bcLo[0], c.bcHi[0]);
    const [y, fy] = this.bcAxis(j, c.ny, c.bcLo[1], c.bcHi[1]);
    const [z, fz] = this.bcAxis(k, c.nz, c.bcLo[2], c.bcHi[2]);
    const o = this.index(x, y, z) * 7;
    const W = this.W;
    return [W[o], W[o + 1] * fx, W[o + 2] * fy, W[o + 3] * fz, W[o + 4], W[o + 5], W[o + 6]];
  }

  private primitives() {
    const { eos, rhoFloor, pFloor } = this.cfg;
    const [R, MX, MY, MZ, , EI, SC] = this.U;
    let rate = 0;
    for (let o = 0; o < this.ncell; o++) {
      const rho = Math.max(R[o], rhoFloor);
      let eint = EI[o];
      let p = eos.pressure(rho, eint);
      if (!(p >= pFloor)) {
        p = pFloor;
        eint = eos.eintFromP(rho, p);
      }
      const cs = eos.soundSpeed(rho, p);
      const w = o * 7;
      this.W[w] = rho;
      this.W[w + 1] = MX[o] / rho;
      this.W[w + 2] = MY[o] / rho;
      this.W[w + 3] = MZ[o] / rho;
      this.W[w + 4] = p;
      this.W[w + 5] = SC[o] / rho;
      this.W[w + 6] = cs;
      rate = Math.max(rate, Math.abs(this.W[w + 1]) + Math.abs(this.W[w + 2]) + Math.abs(this.W[w + 3]) + 3 * cs);
    }
    return rate;
  }

  /** Natural scale of each primitive at a cell, by which the limiter judges smoothness. */
  private static scales(q: ArrayLike<number>): number[] {
    return [q[0], q[6], q[6], q[6], q[4], 0];
  }

  private predictRaw(i: number, j: number, k: number): Pred {
    const c = this.cfg;
    const lim = LIMITERS[c.limiter];
    const q = this.loadQ(i, j, k);
    const nb = [
      [this.loadQ(i - 1, j, k), this.loadQ(i + 1, j, k)],
      [this.loadQ(i, j - 1, k), this.loadQ(i, j + 1, k)],
      [this.loadQ(i, j, k - 1), this.loadQ(i, j, k + 1)],
    ];
    const scale = RefHydro.scales(q);
    const su = nb.map(([m, p]) => {
      const out = new Float64Array(NQ);
      for (let v = 0; v < NQ; v++) out[v] = lim(q[v] - m[v], p[v] - q[v], scale[v]);
      return out;
    });
    const sd = nb.map(([m, p]) => {
      const out = new Float64Array(NQ);
      for (let v = 0; v < NQ; v++) out[v] = lim(p[v] - q[v], q[v] - m[v], scale[v]);
      return out;
    });
    // Face-to-face difference across the cell, which drives the half-step evolution.
    const s = su.map((u, d) => u.map((x, v) => 0.5 * (x + sd[d][v])));
    const big = (v: number) => Math.max(...su.map((u) => Math.abs(u[v])), ...sd.map((u) => Math.abs(u[v])));
    const conv = nb[0][1][1] - nb[0][0][1] + (nb[1][1][2] - nb[1][0][2]) + (nb[2][1][3] - nb[2][0][3]);
    const jump = Math.max(...nb.map(([m, p]) => Math.abs(p[4] - m[4]) / Math.min(p[4], m[4])));
    const flag = conv < 0 && jump > c.shockThresh ? 1 : 0;

    const h = (0.5 * this.dt) / c.dx;
    const rho = q[0];
    const v = [q[1], q[2], q[3]];
    const cs = q[6];
    const dv = s[0][1] + s[1][2] + s[2][3];
    const adv = (n: number) => v[0] * s[0][n] + v[1] * s[1][n] + v[2] * s[2][n];
    const qh = new Float64Array(NQ);
    qh[0] = rho - h * (adv(0) + rho * dv);
    for (let d = 0; d < 3; d++) qh[1 + d] = v[d] - h * (adv(1 + d) + s[d][4] / rho);
    qh[4] = q[4] - h * (adv(4) + rho * cs * cs * dv);
    qh[5] = q[5] - h * adv(5);

    const rmin = qh[0] - 0.5 * big(0);
    const pmin = qh[4] - 0.5 * big(4);
    if (!(rmin > 0 && pmin > 0)) {
      const zero = () => [0, 1, 2].map(() => new Float64Array(NQ));
      return { q: Float64Array.from(q.slice(0, NQ)), su: zero(), sd: zero(), flag, firstOrder: true };
    }
    return { q: qh, su, sd, flag, firstOrder: false };
  }

  /**
   * Half-step state, slopes and flags of a cell. As on the GPU, a cell beyond a boundary takes
   * the half-step state and flags of the cell it maps to (mirrored if reflecting), with slopes
   * recomputed from the primitives through the boundary mapping.
   */
  private predict(i: number, j: number, k: number): Pred {
    const c = this.cfg;
    if (i >= 0 && j >= 0 && k >= 0 && i < c.nx && j < c.ny && k < c.nz) {
      const o = this.index(i, j, k);
      return (this.preds[o] ??= this.predictRaw(i, j, k));
    }
    const [x, fx] = this.bcAxis(i, c.nx, c.bcLo[0], c.bcHi[0]);
    const [y, fy] = this.bcAxis(j, c.ny, c.bcLo[1], c.bcHi[1]);
    const [z, fz] = this.bcAxis(k, c.nz, c.bcLo[2], c.bcHi[2]);
    const src = this.predict(x, y, z);
    const q = Float64Array.from(src.q);
    q[1] *= fx;
    q[2] *= fy;
    q[3] *= fz;
    const lim = LIMITERS[c.limiter];
    const w = this.loadQ(i, j, k);
    const e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const scale = RefHydro.scales(w);
    const edges = (up: boolean) =>
      e.map(([a, b, d]) => {
        const out = new Float64Array(NQ);
        if (src.firstOrder) return out;
        const m = this.loadQ(i - a, j - b, k - d);
        const pl = this.loadQ(i + a, j + b, k + d);
        for (let v = 0; v < NQ; v++) out[v] = up ? lim(w[v] - m[v], pl[v] - w[v], scale[v]) : lim(pl[v] - w[v], w[v] - m[v], scale[v]);
        return out;
      });
    return { q, su: edges(true), sd: edges(false), flag: src.flag, firstOrder: src.firstOrder };
  }

  private side(pr: Pred, axis: number, sgn: number): Side {
    const { eos, rhoFloor, pFloor } = this.cfg;
    const f = (n: number) => pr.q[n] + 0.5 * sgn * (sgn > 0 ? pr.su : pr.sd)[axis][n];
    const rho = Math.max(f(0), rhoFloor);
    // Barotropic: the face pressure is a function of the face density, not reconstructed.
    const p = eos.hasEnergy ? Math.max(f(4), pFloor) : Math.max(eos.pressure(rho, 0), pFloor);
    const v = [f(1), f(2), f(3)];
    return {
      rho,
      v: [v[axis], v[(axis + 1) % 3], v[(axis + 2) % 3]],
      p,
      s: f(5),
      cs: eos.soundSpeed(rho, p),
      eint: eos.eintFromP(rho, p),
    };
  }

  /** Advance one step; returns dt. `tEnd` clamps the step so the run lands on it exactly. */
  step(tEnd = Infinity): number {
    const c = this.cfg;
    const { eos } = c;
    const rate = this.primitives();
    this.dt = Math.min((c.cfl * c.dx) / rate, Math.max(tEnd - this.t, 0));
    this.preds = new Array(this.ncell);
    const dtdx = this.dt / c.dx;
    const next = this.U.map((a) => new Float64Array(a));
    const OFF = [
      [-1, 0, 0],
      [1, 0, 0],
      [0, -1, 0],
      [0, 1, 0],
      [0, 0, -1],
      [0, 0, 1],
    ];

    for (let k = 0; k < c.nz; k++)
      for (let j = 0; j < c.ny; j++)
        for (let i = 0; i < c.nx; i++) {
          const o = this.index(i, j, k);
          const self = this.predict(i, j, k);
          let dm = 0;
          const dmom = [0, 0, 0];
          let de = 0;
          let dei = 0;
          let ds = 0;
          let dvn = 0;
          for (let f = 0; f < 6; f++) {
            const axis = f >> 1;
            const upper = (f & 1) === 1;
            const nb = this.predict(i + OFF[f][0], j + OFF[f][1], k + OFF[f][2]);
            const L = upper ? this.side(self, axis, 1) : this.side(nb, axis, 1);
            const R = upper ? this.side(nb, axis, -1) : this.side(self, axis, -1);
            const w = eos.hasEnergy ? Math.max(self.flag, nb.flag) : 1;
            const F = riemannFlux(L, R, w, c.riemann);
            const sg = upper ? -1 : 1;
            dm += sg * F.m;
            for (let d = 0; d < 3; d++) dmom[(axis + d) % 3] += sg * F.mom[d];
            de += sg * F.e;
            dei += sg * F.ei;
            ds += sg * F.s;
            dvn += sg * F.vn;
          }
          const rho = Math.max(this.U[0][o] + dtdx * dm, c.rhoFloor);
          const mom = [1, 2, 3].map((d) => this.U[d][o] + dtdx * dmom[d - 1]);
          let E = this.U[4][o] + dtdx * de;
          let eint = this.U[5][o] + dtdx * (dei + Math.max(self.q[4], c.pFloor) * dvn);
          const ke = (0.5 * (mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2)) / rho;
          if (eos.hasEnergy) {
            const eth = E - ke;
            const emin = eos.eintFromP(rho, c.pFloor);
            if (eth > c.etaDual * E && eth >= emin) {
              eint = eth;
            } else {
              if (!(eint >= emin)) eint = emin;
              E = ke + eint;
            }
          } else {
            eint = eos.eintFromP(rho, eos.pressure(rho, 0));
            E = ke + eint;
          }
          next[0][o] = rho;
          next[1][o] = mom[0];
          next[2][o] = mom[1];
          next[3][o] = mom[2];
          next[4][o] = E;
          next[5][o] = eint;
          next[6][o] = this.U[6][o] + dtdx * ds;
        }
    this.U = next;
    this.t += this.dt;
    this.steps++;
    return this.dt;
  }

  /** Run to `tEnd`. */
  run(tEnd: number, maxSteps = 100000) {
    while (this.t < tEnd * (1 - 1e-12) && this.steps < maxSteps) this.step(tEnd);
  }
}
