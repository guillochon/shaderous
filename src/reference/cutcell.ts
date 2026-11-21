// Float64 prototype of partly filled cells (plans/wind-tunnel.md 8.1, 8.1.1): a liquid with
// implicit pressure around a rigid body that may move, on a 2D periodic grid, with the body's
// surface cutting cells where it truly lies. Kept here, as reference/psplit.ts was for the
// pressure split, so the scheme can be measured before any shader is written. Self-contained
// (no imports), so a script can run it under `node --experimental-strip-types`.
//
// The state is stored per whole cell: Q = phi rho and M = phi rho v, phi the cell's share of gas.
// The geometry is exact for a convex polygon (a circle is a polygon of many sides): each cell's
// phi from clipping its square against the body, each face's open share a from clipping the
// face, and the wall's vector (its length times its normal, out of the gas) from closure,
// W = -sum_f a_f n_f dx, so a uniform pressure pushes no cell. One step, from t to t + dt:
//   1. geometry at t + dt/2 (the faces and walls the step's fluxes use) and t + dt (the volumes);
//   2. advection through the open faces: MUSCL (minmod), upwinded on the faces' mean velocity;
//      viscosity through the open faces; nothing through the wall;
//   3. the pressure, implicit: with Q = phi' (rho0 + p / c^2) at the step's end and the mass the
//      pressure moves through a face (dt^2 / dx^2) a (p_L - p_R) per unit cell area,
//        (phi' / c^2) p + (dt^2 / dx^2) sum_f a_f (p - p_nb) = Q* - phi' rho0,
//      solved by conjugate gradients: a cell the body is swallowing (phi' -> 0) is emptied by
//      the pressure, one it uncovers filled;
//   4. the correction: that mass moved, carrying its upwind velocity; the pressure force through
//      the open faces (their mean pressure) and on the wall (the cell's own); the viscous wall
//      stress at the true distance to the wall, point-implicit;
//   5. small cells: state redistribution (Berger & Giuliani 2021), each cell under half full
//      averaged with its neighbors away from the wall, conservatively.
// The force on the body is what the wall terms took from the gas, so the box's momentum balance
// holds to round-off.

export interface Body {
  /** the convex polygon in the body's own frame, counterclockwise */
  shape: [number, number][];
  /** where its frame's origin is, its turn (radians), at time t; with their rates */
  pos(t: number): [number, number];
  angle(t: number): number;
  vel(t: number): [number, number];
  spin(t: number): number;
}

export interface CutConfig {
  nx: number;
  ny: number;
  dx: number;
  /** lower corner of the box */
  x0: number;
  y0: number;
  rho0: number;
  /** sound speed */
  c: number;
  /** kinematic viscosity */
  nu: number;
  cfl: number;
  tol: number;
  maxIter: number;
  /** cells under this share of gas are merged with neighbors (0: no redistribution) */
  small: number;
  /** MUSCL slopes (false: first-order upwind) */
  muscl: boolean;
  /** samples of the body's place over a step that the faces' open shares are averaged over */
  subSteps: number;
  /** the wall's pressure carried from the cell's center to the wall along the pressure's gradient */
  wallGradient: boolean;
  /** beside the wall, the upwind cell's slope from the face's other cell (false: none) */
  oneSided: boolean;
  /** a cut face's pressure taken at its open part's centroid */
  faceCentroid: boolean;
  /** the mass the pressure moves carries the mean of its two cells' velocities, not the upwind one */
  centralCarry: boolean;
  /** the old way, for comparison: each cell whole gas or whole body by its center, faces open where both cells are gas */
  staircase?: boolean;
  /** the slopes' limiter: minmod, monotonized central, or none (central slopes) */
  limiter: 'minmod' | 'mc' | 'none';
  /** advect with each face's velocity from the last step (Rhie and Chow: its cells' mean with its own pressure correction; false: the mean of its cells') */
  faceVelocity: boolean;
  /** the wall's shear from the velocity this many cells out along the wall's normal (0: the cell's own, point-implicit) */
  wallProbe: number;
  /** a sponge from x = `from` to the box's downstream edge, its strength rising to `rate` per unit time, toward the stream U along x */
  sponge?: { from: number; rate: number; U: number };
}

export const cutDefaults: Omit<CutConfig, 'nx' | 'ny' | 'dx' | 'x0' | 'y0'> = {
  rho0: 1, c: 10, nu: 0, cfl: 0.4, tol: 1e-10, maxIter: 2000, small: 0.5, muscl: true, subSteps: 8, wallGradient: true, oneSided: true, faceCentroid: true, centralCarry: true, limiter: 'minmod', faceVelocity: false, wallProbe: 1.5,
};

/** A circle of `n` sides (vertices on the circle), or a rectangle, about the body's origin. */
export const circle = (r: number, n = 256): [number, number][] => Array.from({ length: n }, (_, k) => [r * Math.cos((2 * Math.PI * k) / n), r * Math.sin((2 * Math.PI * k) / n)]);
export const rectangle = (w: number, h: number): [number, number][] => [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]];

/** A body moving at constant velocity and spin from where it starts. */
export function rigid(shape: [number, number][], p0: [number, number], v: [number, number] = [0, 0], a0 = 0, w = 0): Body {
  return { shape, pos: (t) => [p0[0] + v[0] * t, p0[1] + v[1] * t], angle: (t) => a0 + w * t, vel: () => v, spin: () => w };
}

/** Each cell's gas share, faces' open shares (x faces: the cell's lower x face; y likewise), wall vector over dx, the gas centroid's distance to the wall and the nearest wall point. */
export interface Geometry {
  phi: Float64Array;
  ax: Float64Array;
  ay: Float64Array;
  /** the open parts' first moments about the faces' middles (in faces): x faces along y, y faces along x */
  mx: Float64Array;
  my: Float64Array;
  wx: Float64Array;
  wy: Float64Array;
  dist: Float64Array;
  px: Float64Array;
  py: Float64Array;
  /** cells the body touches */
  cut: number[];
}

type Pt = [number, number];
const area = (p: Pt[]) => {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i], b = p[(i + 1) % p.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2;
};
const centroid = (p: Pt[]): Pt => {
  let cx = 0, cy = 0, s = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i], b = p[(i + 1) % p.length];
    const cr = a[0] * b[1] - b[0] * a[1];
    s += cr;
    cx += (a[0] + b[0]) * cr;
    cy += (a[1] + b[1]) * cr;
  }
  return s === 0 ? [0, 0] : [cx / (3 * s), cy / (3 * s)];
};

/** The square clipped to the convex polygon (Sutherland-Hodgman, the polygon's edges as half-planes). */
function clip(sq: Pt[], poly: Pt[]): Pt[] {
  let out = sq;
  for (let e = 0; e < poly.length && out.length; e++) {
    const a = poly[e], b = poly[(e + 1) % poly.length];
    const side = (p: Pt) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const inp = out;
    out = [];
    for (let i = 0; i < inp.length; i++) {
      const p = inp[i], q = inp[(i + 1) % inp.length];
      const sp = side(p), sq2 = side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq2 >= 0)) {
        const t = sp / (sp - sq2);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

/** The share of the segment p..q inside the convex polygon (Cyrus-Beck). */
function covered(p: Pt, q: Pt, poly: Pt[]): number {
  const c = coveredPart(p, q, poly);
  return c ? c[1] - c[0] : 0;
}

/** The part [t0, t1] of the segment p..q (t from 0 at p to 1 at q) inside the convex polygon, or null. */
function coveredPart(p: Pt, q: Pt, poly: Pt[]): [number, number] | null {
  let t0 = 0, t1 = 1;
  for (let e = 0; e < poly.length; e++) {
    const a = poly[e], b = poly[(e + 1) % poly.length];
    // inside: (b - a) x (x - a) >= 0
    const f0 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const df = (b[0] - a[0]) * (q[1] - p[1]) - (b[1] - a[1]) * (q[0] - p[0]);
    if (df === 0) {
      if (f0 < 0) return null;
      continue;
    }
    const t = -f0 / df;
    if (df > 0) t0 = Math.max(t0, t);
    else t1 = Math.min(t1, t);
    if (t0 >= t1) return null;
  }
  return [t0, t1];
}

/** Where the open part of a face lies along it: its centroid's offset from the face's middle (in faces), times its open share. */
function openMoment(c: [number, number] | null): number {
  if (!c) return 0;
  // The open part is [0, t0] and [t1, 1]: its first moment about the middle, 1/2.
  const [t0, t1] = c;
  return (t0 * t0 - t0 - t1 * t1 + t1) / 2;
}

/** Signed distance from a point to the convex polygon (negative inside), and its nearest boundary point. */
function polyDistance(x: Pt, poly: Pt[]): { d: number; p: Pt } {
  let best = Infinity, bp: Pt = poly[0], inside = true;
  for (let e = 0; e < poly.length; e++) {
    const a = poly[e], b = poly[(e + 1) % poly.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    if (ex * (x[1] - a[1]) - ey * (x[0] - a[0]) < 0) inside = false;
    const t = Math.max(0, Math.min(1, ((x[0] - a[0]) * ex + (x[1] - a[1]) * ey) / (ex * ex + ey * ey)));
    const px = a[0] + t * ex, py = a[1] + t * ey;
    const d = Math.hypot(x[0] - px, x[1] - py);
    if (d < best) {
      best = d;
      bp = [px, py];
    }
  }
  return { d: inside ? -best : best, p: bp };
}

export class CutCell {
  readonly n: number;
  Q: Float64Array;
  Mx: Float64Array;
  My: Float64Array;
  p: Float64Array;
  g: Geometry;
  t = 0;
  steps = 0;
  /** the force on the body over the last step (pressure and viscous parts), per unit depth */
  force = { px: 0, py: 0, vx: 0, vy: 0 };
  /** mass left in cells the body covered (should be round-off) */
  lost = 0;
  iters = 0;
  /** each face's velocity from the last step's mass flux (NaN: none) */
  ufx: Float64Array;
  ufy: Float64Array;
  readonly cfg: CutConfig;
  readonly body: Body | null;

  constructor(cfg: CutConfig, body: Body | null, init: (x: number, y: number) => { rho: number; vx: number; vy: number }) {
    this.cfg = cfg;
    this.body = body;
    this.n = cfg.nx * cfg.ny;
    this.Q = new Float64Array(this.n);
    this.Mx = new Float64Array(this.n);
    this.My = new Float64Array(this.n);
    this.p = new Float64Array(this.n);
    this.ufx = new Float64Array(this.n).fill(NaN);
    this.ufy = new Float64Array(this.n).fill(NaN);
    this.g = this.geometry(0);
    for (let j = 0; j < cfg.ny; j++)
      for (let i = 0; i < cfg.nx; i++) {
        const k = j * cfg.nx + i;
        const w = init(cfg.x0 + (i + 0.5) * cfg.dx, cfg.y0 + (j + 0.5) * cfg.dx);
        const phi = this.g.phi[k];
        this.Q[k] = phi * w.rho;
        this.Mx[k] = phi * w.rho * w.vx;
        this.My[k] = phi * w.rho * w.vy;
      }
  }

  /** The body's polygon in the box at time t. */
  polygon(t: number): Pt[] {
    const b = this.body!;
    const [cx, cy] = b.pos(t);
    const a = b.angle(t), c = Math.cos(a), s = Math.sin(a);
    return b.shape.map(([x, y]) => [cx + c * x - s * y, cy + s * x + c * y] as Pt);
  }

  geometry(t: number): Geometry {
    const { nx, ny, dx, x0, y0 } = this.cfg;
    const n = this.n;
    const g: Geometry = {
      phi: new Float64Array(n).fill(1), ax: new Float64Array(n).fill(1), ay: new Float64Array(n).fill(1), mx: new Float64Array(n), my: new Float64Array(n),
      wx: new Float64Array(n), wy: new Float64Array(n), dist: new Float64Array(n).fill(Infinity), px: new Float64Array(n), py: new Float64Array(n), cut: [],
    };
    if (!this.body) return g;
    const poly = this.polygon(t);
    let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
    for (const [x, y] of poly) {
      lx = Math.min(lx, x); hx = Math.max(hx, x); ly = Math.min(ly, y); hy = Math.max(hy, y);
    }
    const i0 = Math.max(0, Math.floor((lx - x0) / dx) - 1), i1 = Math.min(nx - 1, Math.floor((hx - x0) / dx) + 1);
    const j0 = Math.max(0, Math.floor((ly - y0) / dx) - 1), j1 = Math.min(ny - 1, Math.floor((hy - y0) / dx) + 1);
    // Faces: the lower x face of (i, j) and the lower y face, for every cell of the bounds and one beyond.
    for (let j = j0; j <= j1 + 1 && j < ny; j++)
      for (let i = i0; i <= i1 + 1 && i < nx; i++) {
        const k = j * nx + i;
        const xa = x0 + i * dx, ya = y0 + j * dx;
        const cxp = coveredPart([xa, ya], [xa, ya + dx], poly), cyp = coveredPart([xa, ya], [xa + dx, ya], poly);
        g.ax[k] = 1 - (cxp ? cxp[1] - cxp[0] : 0);
        g.ay[k] = 1 - (cyp ? cyp[1] - cyp[0] : 0);
        g.mx[k] = openMoment(cxp);
        g.my[k] = openMoment(cyp);
      }
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const k = j * nx + i;
        const xa = x0 + i * dx, ya = y0 + j * dx;
        const sq: Pt[] = [[xa, ya], [xa + dx, ya], [xa + dx, ya + dx], [xa, ya + dx]];
        const part = clip(sq, poly);
        let ab = part.length >= 3 ? area(part) : 0;
        if (this.cfg.staircase) ab = polyDistance([xa + dx / 2, ya + dx / 2], poly).d < 0 ? dx * dx : 0;
        if (ab <= 0) continue;
        let phi = 1 - ab / (dx * dx);
        if (phi < 1e-12) phi = 0;
        g.phi[k] = phi;
        g.cut.push(k);
        if (phi === 0) continue;
        const kr = j * nx + ((i + 1) % nx), kt = ((j + 1) % ny) * nx + i;
        g.wx[k] = g.ax[k] - g.ax[kr];
        g.wy[k] = g.ay[k] - g.ay[kt];
        // The gas's centroid, its distance to the wall and the wall's nearest point.
        const cb = centroid(part);
        const sx = xa + dx / 2, sy = ya + dx / 2;
        const gx = (dx * dx * sx - ab * cb[0]) / (dx * dx - ab), gy = (dx * dx * sy - ab * cb[1]) / (dx * dx - ab);
        const pd = polyDistance([gx, gy], poly);
        g.dist[k] = Math.max(pd.d, 1e-3 * dx);
        g.px[k] = pd.p[0];
        g.py[k] = pd.p[1];
      }
    if (this.cfg.staircase) {
      // Faces open where both cells are gas; walls by closure (whole faces).
      for (let j = j0; j <= j1 + 1 && j < ny; j++)
        for (let i = i0; i <= i1 + 1 && i < nx; i++) {
          const k = j * nx + i;
          g.ax[k] = g.phi[k] > 0 && g.phi[j * nx + ((i - 1 + nx) % nx)] > 0 ? 1 : 0;
          g.ay[k] = g.phi[k] > 0 && g.phi[((j - 1 + ny) % ny) * nx + i] > 0 ? 1 : 0;
          g.mx[k] = g.my[k] = 0;
        }
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const k = j * nx + i;
          if (g.phi[k] === 0) continue;
          g.wx[k] = g.ax[k] - g.ax[j * nx + ((i + 1) % nx)];
          g.wy[k] = g.ay[k] - g.ay[((j + 1) % ny) * nx + i];
          if (g.wx[k] || g.wy[k]) {
            if (!g.cut.includes(k)) g.cut.push(k);
            const pd = polyDistance([x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dx], poly);
            g.dist[k] = Math.max(pd.d, 0.5 * dx);
            g.px[k] = pd.p[0];
            g.py[k] = pd.p[1];
          }
        }
    }
    return g;
  }

  /**
   * The faces the step's fluxes go through: each face's open share averaged over the step (the
   * trapezoidal rule on `subSteps` equal parts, so that both ends count: a face open at the step's
   * end is never shut through it), so that a face the body closes or opens within the step
   * counts for the part of it that it was open, and the wall vectors by closure with those
   * shares; the wall's place (distance, nearest point) at the middle.
   */
  private average(t: number, dt: number, g0: Geometry, g1: Geometry): Geometry {
    const { nx, ny } = this.cfg;
    const m = this.cfg.subSteps;
    const ax = new Float64Array(this.n), ay = new Float64Array(this.n), mx = new Float64Array(this.n), my = new Float64Array(this.n);
    const cuts = new Set<number>();
    let mid: Geometry | null = null;
    const others: Geometry[] = [];
    const half = m >= 2 ? m >> 1 : -1;
    for (let q = 0; q <= m; q++) {
      const g = q === 0 ? g0 : q === m ? g1 : this.geometry(t + (q * dt) / m);
      const w = (q === 0 || q === m ? 0.5 : 1) / m;
      for (let k = 0; k < this.n; k++) {
        ax[k] += g.ax[k] * w;
        ay[k] += g.ay[k] * w;
        mx[k] += g.mx[k] * w;
        my[k] += g.my[k] * w;
      }
      for (const k of g.cut) cuts.add(k);
      if (q === half) mid = g;
      else others.push(g);
    }
    // With one part, the middle is not sampled: take it.
    if (!mid) mid = this.geometry(t + dt / 2);
    const cut = [...cuts];
    const wx = new Float64Array(this.n), wy = new Float64Array(this.n);
    for (const k of cut) {
      const i = k % nx, j = Math.floor(k / nx);
      wx[k] = ax[k] - ax[j * nx + ((i + 1) % nx)];
      wy[k] = ay[k] - ay[((j + 1) % ny) * nx + i];
    }
    const dist = Float64Array.from(mid!.dist), px = Float64Array.from(mid!.px), py = Float64Array.from(mid!.py);
    // A cell open for only part of the step: the wall's place when it was.
    for (const k of cut)
      if (!Number.isFinite(dist[k]))
        for (const o of others)
          if (Number.isFinite(o.dist[k])) {
            dist[k] = o.dist[k];
            px[k] = o.px[k];
            py[k] = o.py[k];
            break;
          }
    return { phi: mid!.phi, ax, ay, mx, my, wx, wy, dist, px, py, cut };
  }

  /** The body's velocity at a point, at time t. */
  wallVelocity(x: number, y: number, t: number): Pt {
    const b = this.body!;
    const [cx, cy] = b.pos(t), [vx, vy] = b.vel(t), w = b.spin(t);
    return [vx - w * (y - cy), vy + w * (x - cx)];
  }

  maxSpeed(): number {
    let m = 0;
    for (let k = 0; k < this.n; k++) if (this.Q[k] > 0) m = Math.max(m, Math.hypot(this.Mx[k], this.My[k]) / this.Q[k]);
    if (this.body) {
      const [vx, vy] = this.body.vel(this.t);
      let r = 0;
      for (const [x, y] of this.body.shape) r = Math.max(r, Math.hypot(x, y));
      m = Math.max(m, Math.hypot(vx, vy) + Math.abs(this.body.spin(this.t)) * r);
    }
    return m;
  }

  dtMax(): number {
    const { dx, cfl, nu } = this.cfg;
    let dt = (cfl * dx) / Math.max(this.maxSpeed(), 1e-12);
    if (nu > 0) dt = Math.min(dt, (0.2 * dx * dx) / nu);
    return dt;
  }

  step(dt = this.dtMax()) {
    const { nx, ny, dx, rho0, c, nu } = this.cfg;
    const n = this.n;
    const g0 = this.g;
    const g1 = this.body ? this.geometry(this.t + dt) : g0;
    const gh = this.body ? this.average(this.t, dt, g0, g1) : g0;
    const { Q, Mx, My } = this;
    // Primitives at the step's start; a cell with no gas yet takes none.
    const rho = new Float64Array(n), vx = new Float64Array(n), vy = new Float64Array(n), has = new Uint8Array(n);
    for (let k = 0; k < n; k++)
      if (g0.phi[k] > 0 && Q[k] > 0) {
        has[k] = 1;
        rho[k] = Q[k] / g0.phi[k];
        vx[k] = Mx[k] / Q[k];
        vy[k] = My[k] / Q[k];
      }
    const dQ = new Float64Array(n), dMx = new Float64Array(n), dMy = new Float64Array(n);
    const at = (i: number, j: number) => ((j + ny) % ny) * nx + ((i + nx) % nx);
    const lim = this.cfg.limiter;
    const minmod = (a: number, b: number) => {
      if (lim === 'none') return 0.5 * (a + b);
      if (a * b <= 0) return 0;
      if (lim === 'mc') return Math.sign(a) * Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b));
      return Math.abs(a) < Math.abs(b) ? a : b;
    };
    const muscl = this.cfg.muscl;
    // One face, between cells L and R along axis `ax` (0: x, 1: y), open share a.
    const face = (L: number, R: number, LL: number, RR: number, a: number, axis: number, f: number) => {
      if (a <= 0 || !(has[L] || has[R])) return;
      let un = has[L] && has[R] ? 0.5 * ((axis ? vy[L] : vx[L]) + (axis ? vy[R] : vx[R])) : has[L] ? (axis ? vy[L] : vx[L]) : (axis ? vy[R] : vx[R]);
      // The face's own velocity from the last step's corrected flux, where it was open then too.
      if (this.cfg.faceVelocity) {
        const last = (axis ? this.ufy : this.ufx)[f];
        if (Number.isFinite(last)) un = last;
      }
      // The upwind cell's values at the face, with a limited slope where its far neighbor has gas.
      const up = un >= 0 ? L : R, far = un >= 0 ? LL : RR, near = un >= 0 ? R : L;
      let r = rho[up], u = vx[up], v = vy[up];
      if (!has[up]) {
        r = rho[near]; u = vx[near]; v = vy[near];
      } else if (muscl && has[far] && has[near]) {
        r += 0.5 * minmod(rho[up] - rho[far], rho[near] - rho[up]);
        u += 0.5 * minmod(vx[up] - vx[far], vx[near] - vx[up]);
        v += 0.5 * minmod(vy[up] - vy[far], vy[near] - vy[up]);
      } else if (muscl && this.cfg.oneSided && has[near]) {
        // Beside the wall (no gas beyond the upwind cell): the slope toward the face, from the
        // face's other cell, unlimited (it has no second side to limit it against).
        r += 0.5 * (rho[near] - rho[up]);
        u += 0.5 * (vx[near] - vx[up]);
        v += 0.5 * (vy[near] - vy[up]);
      }
      const F = (dt / dx) * a * un * r; // mass per unit cell area
      dQ[L] -= F; dQ[R] += F;
      dMx[L] -= F * u; dMx[R] += F * u;
      dMy[L] -= F * v; dMy[R] += F * v;
      if (nu > 0 && has[L] && has[R]) {
        const mu = nu * 0.5 * (rho[L] + rho[R]);
        const s = ((dt / (dx * dx)) * a * mu);
        dMx[L] += s * (vx[R] - vx[L]); dMx[R] -= s * (vx[R] - vx[L]);
        dMy[L] += s * (vy[R] - vy[L]); dMy[R] -= s * (vy[R] - vy[L]);
      }
    };
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        face(at(i - 1, j), k, at(i - 2, j), at(i + 1, j), gh.ax[k], 0, k);
        face(at(i, j - 1), k, at(i, j - 2), at(i, j + 1), gh.ay[k], 1, k);
      }
    for (let k = 0; k < n; k++) {
      Q[k] += dQ[k];
      Mx[k] += dMx[k];
      My[k] += dMy[k];
    }

    // The pressure: (phi'/c^2) p + s sum_f a_f (p - p_nb) = Q* - phi' rho0, s = dt^2 / dx^2.
    const s = (dt * dt) / (dx * dx);
    const active = new Uint8Array(n);
    const diag = new Float64Array(n);
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const aSum = gh.ax[k] + gh.ax[at(i + 1, j)] + gh.ay[k] + gh.ay[at(i, j + 1)];
        if (g1.phi[k] > 0 || aSum > 0) active[k] = 1;
        diag[k] = g1.phi[k] / (c * c) + s * aSum;
      }
    const apply = (x: Float64Array, out: Float64Array) => {
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const k = j * nx + i;
          if (!active[k]) {
            out[k] = 0;
            continue;
          }
          out[k] = diag[k] * x[k] - s * (gh.ax[k] * x[at(i - 1, j)] + gh.ax[at(i + 1, j)] * x[at(i + 1, j)] + gh.ay[k] * x[at(i, j - 1)] + gh.ay[at(i, j + 1)] * x[at(i, j + 1)]);
        }
    };
    const b = new Float64Array(n);
    for (let k = 0; k < n; k++) b[k] = active[k] ? Q[k] - g1.phi[k] * rho0 : 0;
    this.iters = this.cg(apply, b, this.p, diag, active);
    const p = this.p;

    // The correction: the mass the pressure moves, with its upwind velocity, and the pressure's force.
    const vxs = new Float64Array(n), vys = new Float64Array(n);
    for (let k = 0; k < n; k++)
      if (Q[k] > 0) {
        vxs[k] = Mx[k] / Q[k];
        vys[k] = My[k] / Q[k];
      }
    const cMx = new Float64Array(n), cMy = new Float64Array(n), cQ = new Float64Array(n);
    const pflux = [new Float64Array(n), new Float64Array(n)];
    const corr = (L: number, R: number, a: number, axis: number, mom: number, fk: number) => {
      if (a <= 0) return;
      const G = s * a * (p[L] - p[R]);
      pflux[axis][fk] = G;
      const up = G >= 0 ? L : R;
      const other = up === L ? R : L;
      let ux = Q[up] > 0 ? vxs[up] : vxs[other], uy = Q[up] > 0 ? vys[up] : vys[other];
      // Between two cells that hold gas well, the mean of their velocities (second order);
      // beside one nearly empty or emptying, the upwind one, so that a cell's momentum leaves
      // with its mass.
      if (this.cfg.centralCarry && Q[L] > 0 && Q[R] > 0 && g1.phi[L] >= this.cfg.small && g1.phi[R] >= this.cfg.small) {
        ux = 0.5 * (vxs[L] + vxs[R]);
        uy = 0.5 * (vys[L] + vys[R]);
      }
      cQ[L] -= G; cQ[R] += G;
      cMx[L] -= G * ux; cMx[R] += G * ux;
      cMy[L] -= G * uy; cMy[R] += G * uy;
      // The face's pressure at its open part's centroid: the mean, moved along the face by the
      // mean of the two cells' differences across it (a times the offset is the moment m).
      let f = a * 0.5 * (p[L] + p[R]);
      if (this.cfg.faceCentroid && mom !== 0) {
        const tl = axis === 0 ? this.gradAt(p, g1.phi, L, 1) : this.gradAt(p, g1.phi, L, 0);
        const tr = axis === 0 ? this.gradAt(p, g1.phi, R, 1) : this.gradAt(p, g1.phi, R, 0);
        f += mom * 0.5 * (tl + tr);
      }
      f *= dt / dx;
      if (axis === 0) {
        cMx[L] -= f; cMx[R] += f;
      } else {
        cMy[L] -= f; cMy[R] += f;
      }
    };
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        corr(at(i - 1, j), k, gh.ax[k], 0, gh.mx[k], k);
        corr(at(i, j - 1), k, gh.ay[k], 1, gh.my[k], k);
      }
    // The faces' velocities for the next step's advection (Rhie and Chow): the mean of the two
    // cells' velocities after advection and viscosity, with the face's own pressure correction
    // (the mass the solve moved through it over its open share and mean density), not the cells'
    // central one: what makes the faces' velocities balance the volumes, as the cells' do not
    // beside a wall.
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        for (const axis of [0, 1]) {
          const L = axis ? at(i, j - 1) : at(i - 1, j);
          const a = axis ? gh.ay[k] : gh.ax[k];
          const rl = Q[L] + cQ[L], rr = Q[k] + cQ[k];
          const out = axis ? this.ufy : this.ufx;
          if (a > 0 && g1.phi[L] > 0 && g1.phi[k] > 0 && rl > 0 && rr > 0 && Q[L] > 0 && Q[k] > 0) {
            const rf = 0.5 * (rl / g1.phi[L] + rr / g1.phi[k]);
            const vs = axis ? 0.5 * (vys[L] + vys[k]) : 0.5 * (vxs[L] + vxs[k]);
            out[k] = vs + pflux[axis][k] / ((dt / dx) * a * rf);
          } else out[k] = NaN;
        }
      }
    const F = { px: 0, py: 0, vx: 0, vy: 0 };
    for (let k = 0; k < n; k++) {
      Q[k] += cQ[k];
      Mx[k] += cMx[k];
      My[k] += cMy[k];
    }
    if (this.body) {
      // The wall: its pressure (the cell's), and its viscous stress at the true distance.
      const cells = new Set([...gh.cut, ...g1.cut]);
      const u0x = new Float64Array(n), u0y = new Float64Array(n);
      for (let k = 0; k < n; k++)
        if (Q[k] > 0 && g1.phi[k] > 0) {
          u0x[k] = Mx[k] / Q[k];
          u0y[k] = My[k] / Q[k];
        }
      for (const k of cells) {
        const wx = gh.wx[k], wy = gh.wy[k];
        if (wx === 0 && wy === 0) continue;
        // The pressure at the wall: the cell's, carried to the wall's nearest point along the
        // pressure's gradient (central differences, one-sided where a neighbor has no gas).
        let pw = p[k];
        if (this.cfg.wallGradient) {
          const i = k % nx, j = Math.floor(k / nx);
          const gx = this.grad(p, g1.phi, at(i - 1, j), k, at(i + 1, j)), gy = this.grad(p, g1.phi, at(i, j - 1), k, at(i, j + 1));
          const ox = gh.px[k] - (this.cfg.x0 + (i + 0.5) * dx), oy = gh.py[k] - (this.cfg.y0 + (j + 0.5) * dx);
          if (Number.isFinite(ox + oy)) pw += (gx * ox + gy * oy) / dx;
        }
        const fx = (dt / dx) * pw * wx, fy = (dt / dx) * pw * wy;
        Mx[k] -= fx;
        My[k] -= fy;
        F.px += (fx * dx * dx) / dt;
        F.py += (fy * dx * dx) / dt;
        if (nu > 0 && Q[k] > 0 && g1.phi[k] > 0 && this.cfg.wallProbe > 0) {
          // The wall's shear from the velocity at a point `wallProbe` cells out from the wall
          // along its normal (bilinear among the cells with gas around it), not from the cell's
          // own velocity over its centroid's distance, which in a small cell is a sliver's.
          const [sx, sy] = this.wallVelocity(gh.px[k], gh.py[k], this.t + dt / 2);
          const wl = Math.hypot(wx, wy);
          const h = this.cfg.wallProbe * dx;
          const [ux, uy] = this.probe(u0x, u0y, g1.phi, gh.px[k] - (h * wx) / wl, gh.py[k] - (h * wy) / wl);
          const rk = Q[k] / g1.phi[k];
          const ex = -(dt * nu * rk * wl * (ux - sx)) / (h * dx), ey = -(dt * nu * rk * wl * (uy - sy)) / (h * dx);
          Mx[k] += ex;
          My[k] += ey;
          F.vx -= (ex * dx * dx) / dt;
          F.vy -= (ey * dx * dx) / dt;
        } else if (nu > 0 && Q[k] > 0 && g1.phi[k] > 0) {
          const [sx, sy] = this.wallVelocity(gh.px[k], gh.py[k], this.t + dt / 2);
          const kk = (dt * nu * Math.hypot(wx, wy)) / (Math.min(gh.dist[k], 1e30) * g1.phi[k] * dx);
          const ux = Mx[k] / Q[k], uy = My[k] / Q[k];
          const nx2 = (ux + kk * sx) / (1 + kk), ny2 = (uy + kk * sy) / (1 + kk);
          const ex = Q[k] * (nx2 - ux), ey = Q[k] * (ny2 - uy);
          Mx[k] += ex;
          My[k] += ey;
          F.vx -= (ex * dx * dx) / dt;
          F.vy -= (ey * dx * dx) / dt;
        }
      }
    }
    this.force = F;
    // A cell the body covered within the step with no open face left to the pressure (its last
    // sliver closed before the half step): what little it held goes to its neighbors with gas,
    // by their volume. Counted in `lost`, which should stay round-off small.
    for (const k of g1.cut)
      if (g1.phi[k] === 0 && (Q[k] !== 0 || Mx[k] !== 0 || My[k] !== 0)) {
        this.lost += Math.abs(Q[k]) * dx * dx;
        const i = k % nx, j = Math.floor(k / nx);
        let nb = [at(i - 1, j), at(i + 1, j), at(i, j - 1), at(i, j + 1)].filter((q) => g1.phi[q] > 0);
        if (!nb.length) nb = [at(i - 1, j - 1), at(i + 1, j - 1), at(i - 1, j + 1), at(i + 1, j + 1)].filter((q) => g1.phi[q] > 0);
        const v = nb.reduce((a, q) => a + g1.phi[q], 0);
        for (const q of nb) {
          const w = g1.phi[q] / v;
          Q[q] += w * Q[k];
          Mx[q] += w * Mx[k];
          My[q] += w * My[k];
        }
        if (nb.length) Q[k] = Mx[k] = My[k] = 0;
      }
    this.g = g1;
    if (this.body && this.cfg.small > 0) this.redistribute();
    // A sponge downstream: the gas pulled back to the free stream (rho0, U along x), so that a
    // wake does not come round the periodic box to the body, nor the body slow the whole stream.
    const sp = this.cfg.sponge;
    if (sp) {
      for (let i = 0; i < nx; i++) {
        const x = this.cfg.x0 + (i + 0.5) * dx;
        const w = x > sp.from ? Math.min(1, sp.rate * dt * ((x - sp.from) / (this.cfg.x0 + nx * dx - sp.from))) : 0;
        if (w <= 0) continue;
        for (let j = 0; j < ny; j++) {
          const k = j * nx + i;
          if (g1.phi[k] <= 0) continue;
          const q = g1.phi[k] * rho0;
          Q[k] += w * (q - Q[k]);
          Mx[k] += w * (q * sp.U - Mx[k]);
          My[k] += w * (0 - My[k]);
        }
      }
    }
    this.t += dt;
    this.steps++;
  }

  /** The velocity at a point: bilinear among the four cell centers around it that hold gas. */
  private probe(ux: Float64Array, uy: Float64Array, phi: Float64Array, x: number, y: number): [number, number] {
    const { nx, ny, dx, x0, y0 } = this.cfg;
    const fx = (x - x0) / dx - 0.5, fy = (y - y0) / dx - 0.5;
    const i = Math.floor(fx), j = Math.floor(fy);
    const tx = fx - i, ty = fy - j;
    let w = 0, a = 0, b = 0;
    for (const [di, dj, wt] of [[0, 0, (1 - tx) * (1 - ty)], [1, 0, tx * (1 - ty)], [0, 1, (1 - tx) * ty], [1, 1, tx * ty]] as const) {
      const k = (((j + dj) % ny) + ny) % ny * nx + ((((i + di) % nx) + nx) % nx);
      if (phi[k] <= 0 || wt <= 0) continue;
      // A cell under a quarter full holds a velocity redistribution has just averaged: half weight.
      const ww = wt * Math.min(1, phi[k] * 4);
      w += ww;
      a += ww * ux[k];
      b += ww * uy[k];
    }
    return w > 0 ? [a / w, b / w] : [0, 0];
  }

  /** A difference of p across cell k (per cell), central where both neighbors have gas. */
  private grad(p: Float64Array, phi: Float64Array, m: number, k: number, q: number): number {
    const a = phi[m] > 0, b = phi[q] > 0;
    return a && b ? 0.5 * (p[q] - p[m]) : b ? p[q] - p[k] : a ? p[k] - p[m] : 0;
  }

  /** The difference of p across cell k along an axis, per cell. */
  private gradAt(p: Float64Array, phi: Float64Array, k: number, axis: number): number {
    const { nx, ny } = this.cfg;
    const i = k % nx, j = Math.floor(k / nx);
    const at = (a: number, b: number) => ((b + ny) % ny) * nx + ((a + nx) % nx);
    return axis === 0 ? this.grad(p, phi, at(i - 1, j), k, at(i + 1, j)) : this.grad(p, phi, at(i, j - 1), k, at(i, j + 1));
  }

  /** Preconditioned conjugate gradients on the active cells; returns the iterations taken. */
  private cg(apply: (x: Float64Array, out: Float64Array) => void, b: Float64Array, x: Float64Array, diag: Float64Array, active: Uint8Array): number {
    const n = this.n;
    const r = new Float64Array(n), z = new Float64Array(n), d = new Float64Array(n), q = new Float64Array(n);
    for (let k = 0; k < n; k++) if (!active[k]) x[k] = 0;
    apply(x, q);
    let bn = 0;
    for (let k = 0; k < n; k++) {
      r[k] = b[k] - q[k];
      bn += b[k] * b[k];
    }
    bn = Math.sqrt(bn) || 1;
    let rz = 0;
    for (let k = 0; k < n; k++) {
      z[k] = active[k] ? r[k] / diag[k] : 0;
      d[k] = z[k];
      rz += r[k] * z[k];
    }
    let it = 0;
    for (; it < this.cfg.maxIter; it++) {
      let rn = 0;
      for (let k = 0; k < n; k++) rn += r[k] * r[k];
      if (Math.sqrt(rn) <= this.cfg.tol * bn) break;
      apply(d, q);
      let dq = 0;
      for (let k = 0; k < n; k++) dq += d[k] * q[k];
      const alpha = rz / dq;
      for (let k = 0; k < n; k++) {
        x[k] += alpha * d[k];
        r[k] -= alpha * q[k];
      }
      let rz2 = 0;
      for (let k = 0; k < n; k++) {
        z[k] = active[k] ? r[k] / diag[k] : 0;
        rz2 += r[k] * z[k];
      }
      const beta = rz2 / rz;
      rz = rz2;
      for (let k = 0; k < n; k++) d[k] = z[k] + beta * d[k];
    }
    return it;
  }

  /**
   * State redistribution (Berger & Giuliani 2021): a cell under `small` full is merged with its
   * neighbor away from the wall (along the axis the wall faces, or both axes and their diagonal
   * where it faces neither), and every cell's density and momentum density become the mean of
   * the merged neighborhoods' that hold it, each neighborhood's taken by volume with its cells
   * counted once among all the neighborhoods they are in. Conservative.
   */
  private redistribute() {
    const { nx, ny, small } = this.cfg;
    const g = this.g;
    const n = this.n;
    const at = (i: number, j: number) => ((j + ny) % ny) * nx + ((i + nx) % nx);
    const hoods = new Map<number, number[]>();
    const count = new Float64Array(n);
    for (let k = 0; k < n; k++) if (g.phi[k] > 0) count[k] = 1;
    for (const k of g.cut) {
      const phi = g.phi[k];
      if (phi <= 0 || phi >= small) continue;
      const wx = g.wx[k], wy = g.wy[k];
      const i = k % nx, j = Math.floor(k / nx);
      // Away from the wall: against the wall vector (which points into the body).
      const sx = wx > 0 ? -1 : 1, sy = wy > 0 ? -1 : 1;
      const ax = Math.abs(wx), ay = Math.abs(wy);
      let cand: number[];
      if (ax > 2 * ay) cand = [at(i + sx, j)];
      else if (ay > 2 * ax) cand = [at(i, j + sy)];
      else cand = [at(i + sx, j), at(i, j + sy), at(i + sx, j + sy)];
      const hood = [k, ...cand.filter((q) => g.phi[q] > 0)];
      if (hood.length < 2) continue;
      hoods.set(k, hood);
      for (const q of cand) if (g.phi[q] > 0) count[q]++;
    }
    if (!hoods.size) return;
    const { Q, Mx, My } = this;
    const sum = (k: number) => hoods.get(k) ?? [k];
    // Each neighborhood's volume and mean densities.
    const touched = new Set<number>();
    for (const h of hoods.values()) for (const q of h) touched.add(q);
    const mean = new Map<number, [number, number, number]>();
    const owners = new Map<number, number[]>();
    const heads = new Set<number>(touched);
    for (const k of heads) {
      if (g.phi[k] <= 0) continue;
      const h = sum(k);
      let v = 0, q = 0, mx = 0, my = 0;
      for (const m of h) {
        const w = 1 / count[m];
        v += g.phi[m] * w;
        q += Q[m] * w;
        mx += Mx[m] * w;
        my += My[m] * w;
        if (!owners.has(m)) owners.set(m, []);
        owners.get(m)!.push(k);
      }
      mean.set(k, [q / v, mx / v, my / v]);
    }
    for (const [m, list] of owners) {
      let r = 0, ux = 0, uy = 0;
      for (const k of list) {
        const u = mean.get(k)!;
        r += u[0];
        ux += u[1];
        uy += u[2];
      }
      const c = list.length;
      Q[m] = (g.phi[m] * r) / c;
      Mx[m] = (g.phi[m] * ux) / c;
      My[m] = (g.phi[m] * uy) / c;
    }
  }

  /** Totals: mass and momentum, per unit depth. */
  totals() {
    const a = this.cfg.dx * this.cfg.dx;
    let m = 0, px = 0, py = 0;
    for (let k = 0; k < this.n; k++) {
      m += this.Q[k] * a;
      px += this.Mx[k] * a;
      py += this.My[k] * a;
    }
    return { mass: m, px, py };
  }
}
