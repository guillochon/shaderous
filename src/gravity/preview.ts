// The path a point mass would take if it were placed now: the orbit preview of the sink tool
// (src/ui/app.ts). Integrated here on the CPU, in double precision, with the forces and the
// scheme the run uses for its point masses (gravity/bh_kick.wgsl, gravity/bh_drift.wgsl), but
// with the gas held as it is: a frozen list of lumps read back from the GPU
// (Simulation.readGasLumps). So it is the orbit in the gas as it stands, without the gas's
// response to the new mass, without accretion, and without the gas moving on meanwhile.

type V3 = [number, number, number];

/** A point mass: G M, where it is and how it moves, and its Plummer softening length. */
export interface Body {
  gm: number;
  pos: number[];
  vel: number[];
  soft: number;
}

/** A piece of the gas: its mass, at its center of mass, and the size of the region it sums. */
export interface Lump {
  m: number;
  pos: V3;
  h: number;
}

export interface OrbitInput {
  /** The run's masses, then the one being placed, last: its path is the one returned. */
  bodies: Body[];
  lumps: Lump[];
  G: number;
  /** P.bh.y: 0 inertial, 1 accelerating with the gas, 2 with the masses. */
  frame: number;
  /** side of the box (positions are relative to its center) */
  box: number;
}

export interface OrbitLimits {
  /** longest path, in box sides */
  length: number;
  steps: number;
  /** time from now at which the path ends */
  time: number;
  /** how far outside the box the mass may go, in box sides */
  outside: number;
  /** points the path is resampled to */
  points: number;
}

export const ORBIT_LIMITS: OrbitLimits = { length: 2, steps: 1200, time: Infinity, outside: 0.1, points: 160 };

export interface Orbit {
  /** xyz per point, relative to the box center: evenly spaced along the path */
  points: Float32Array;
  /** time from now at each point */
  times: Float32Array;
  /** why it ends: the caps, leaving the box, the time limit, or a mass that does not move */
  end: 'length' | 'steps' | 'left' | 'time' | 'rest';
}

/**
 * The gas's pull on other masses than the new one is found again only every so many steps: this
 * many, and more in a run with many masses, so that their sums do not outweigh the new one's.
 */
const GAS_EVERY = 4;

export function previewOrbit(input: OrbitInput, limits: Partial<OrbitLimits> = {}): Orbit {
  const lim = { ...ORBIT_LIMITS, ...limits };
  const { bodies, lumps, G, frame, box } = input;
  const n = bodies.length;
  const me = n - 1;
  const x = new Float64Array(3 * n);
  const v = new Float64Array(3 * n);
  const a = new Float64Array(3 * n);
  const gas = new Float64Array(3 * n); // the gas's pull on each mass
  const gm = bodies.map((b) => b.gm);
  // A softening length of zero would let a close pass run away: never less than a thousandth of the box.
  const soft = bodies.map((b) => Math.max(b.soft, 1e-3 * box));
  bodies.forEach((b, i) => {
    for (let k = 0; k < 3; k++) {
      x[3 * i + k] = b.pos[k] ?? 0;
      v[3 * i + k] = b.vel[k] ?? 0;
    }
  });
  const nl = lumps.length;
  const lm = new Float64Array(nl);
  const lx = new Float64Array(3 * nl);
  const lh2 = new Float64Array(nl);
  const lfar = new Float64Array(nl);
  let gasMass = 0;
  lumps.forEach((l, j) => {
    lm[j] = l.m;
    lx.set(l.pos, 3 * j);
    lh2[j] = (l.h * l.h) / 9;
    lfar[j] = 1 / (4 * l.h * l.h);
    gasMass += l.m;
  });
  const gmAll = gm.reduce((s, g) => s + g, 0);

  // The gas's pull on mass i: each lump as a point at its center of mass, softened by the
  // mass's own length (as gravity/bh.wgsl softens each cell's) and by a third of the lump's
  // size where the mass is in or beside it, so that a lump the mass is inside does not pull on
  // it like a point. Farther off the lump is a point: softened there too, the lumps of a star
  // four tenths of the box away pulled a percent too weakly.
  const gasPull = (i: number) => {
    const [px, py, pz] = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    const e2 = soft[i] * soft[i];
    let ax = 0;
    let ay = 0;
    let az = 0;
    for (let j = 0; j < nl; j++) {
      const dx = lx[3 * j] - px;
      const dy = lx[3 * j + 1] - py;
      const dz = lx[3 * j + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      // The lump's own softening, fading to none two of its sizes away.
      const near = 1 - d2 * lfar[j];
      const r2 = d2 + e2 + (near > 0 ? lh2[j] * near * near : 0);
      const w = lm[j] / (r2 * Math.sqrt(r2));
      ax += w * dx;
      ay += w * dy;
      az += w * dz;
    }
    gas[3 * i] = G * ax;
    gas[3 * i + 1] = G * ay;
    gas[3 * i + 2] = G * az;
  };

  // Accelerations at the current positions (gravity/bh_kick.wgsl).
  let step = 0;
  const gasEvery = Math.max(GAS_EVERY, n - 1);
  const accel = () => {
    if (nl) for (let i = 0; i < n; i++) if (i === me || step % gasEvery === 0) gasPull(i);
    // The frame: with the gas, it accelerates at the mean of the masses' pull on the gas (the
    // gas pulls on mass i with gm_i / G times `gas`, so the mass pulls back with minus that);
    // with the masses, at the acceleration of their center, which only the gas gives.
    const abar = [0, 0, 0];
    if (frame === 1 && gasMass > 0) {
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) abar[k] -= (gm[i] * gas[3 * i + k]) / (G * gasMass);
    } else if (frame === 2 && gmAll > 0) {
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) abar[k] += (gm[i] * gas[3 * i + k]) / gmAll;
    }
    for (let i = 0; i < n; i++) {
      let ax = gas[3 * i] - abar[0];
      let ay = gas[3 * i + 1] - abar[1];
      let az = gas[3 * i + 2] - abar[2];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = x[3 * i] - x[3 * j];
        const dy = x[3 * i + 1] - x[3 * j + 1];
        const dz = x[3 * i + 2] - x[3 * j + 2];
        const eps = Math.max(soft[i], soft[j]);
        const r2 = dx * dx + dy * dy + dz * dz + eps * eps;
        const w = gm[j] / (r2 * Math.sqrt(r2));
        ax -= w * dx;
        ay -= w * dy;
        az -= w * dz;
      }
      a[3 * i] = ax;
      a[3 * i + 1] = ay;
      a[3 * i + 2] = az;
    }
  };
  // In the masses' frame their center of mass is held at rest after every kick.
  const recenter = () => {
    if (frame !== 2 || !(gmAll > 0)) return;
    for (let k = 0; k < 3; k++) {
      let p = 0;
      for (let i = 0; i < n; i++) p += gm[i] * v[3 * i + k];
      for (let i = 0; i < n; i++) v[3 * i + k] -= p / gmAll;
    }
  };

  accel();
  const path: number[] = [x[3 * me], x[3 * me + 1], x[3 * me + 2]];
  const times: number[] = [0];
  const arc: number[] = [0];
  let t = 0;
  let end: Orbit['end'] = 'steps';
  const edge = (0.5 + lim.outside) * box;
  const maxLen = lim.length * box;
  for (; step < lim.steps; ) {
    // The step: a fraction of the time each mass's acceleration takes to move it its softening
    // length, and no mass farther than a hundredth of the box.
    let dt = Infinity;
    for (let i = 0; i < n; i++) {
      const am = Math.hypot(a[3 * i], a[3 * i + 1], a[3 * i + 2]);
      const vm = Math.hypot(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
      if (am > 0) dt = Math.min(dt, 0.1 * Math.sqrt(soft[i] / am));
      if (vm > 0) dt = Math.min(dt, (0.01 * box) / vm);
    }
    if (!Number.isFinite(dt)) {
      end = 'rest';
      break;
    }
    let last = false;
    if (t + dt >= lim.time) {
      dt = lim.time - t;
      last = true;
    }
    if (!(dt > 0)) {
      end = 'time';
      break;
    }
    for (let i = 0; i < 3 * n; i++) {
      v[i] += 0.5 * dt * a[i];
      x[i] += dt * v[i];
    }
    step++;
    accel();
    for (let i = 0; i < 3 * n; i++) v[i] += 0.5 * dt * a[i];
    recenter();
    t += dt;
    const o = path.length - 3;
    const p = [x[3 * me], x[3 * me + 1], x[3 * me + 2]];
    const s = arc[arc.length - 1] + Math.hypot(p[0] - path[o], p[1] - path[o + 1], p[2] - path[o + 2]);
    path.push(...p);
    times.push(t);
    arc.push(s);
    if (last) {
      end = 'time';
      break;
    }
    if (s >= maxLen) {
      end = 'length';
      break;
    }
    if (Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])) > edge) {
      end = 'left';
      break;
    }
  }

  // Evenly spaced along the path, so that what is drawn does not depend on the steps taken.
  const total = arc[arc.length - 1];
  const count = total > 0 ? Math.max(2, Math.min(lim.points, arc.length)) : 1;
  const points = new Float32Array(3 * count);
  const out = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < count; i++) {
    const s = count > 1 ? (total * i) / (count - 1) : 0;
    while (j < arc.length - 2 && arc[j + 1] < s) j++;
    const span = arc.length > 1 ? arc[j + 1] - arc[j] : 0;
    const f = span > 0 ? Math.min(1, Math.max(0, (s - arc[j]) / span)) : 0;
    const k = arc.length > 1 ? j + 1 : j;
    for (let c = 0; c < 3; c++) points[3 * i + c] = path[3 * j + c] + f * (path[3 * k + c] - path[3 * j + c]);
    out[i] = times[j] + f * (times[k] - times[j]);
  }
  return { points, times: out, end };
}

/** Most lumps a preview sums over at each of its steps. */
export const MAX_LUMPS = 768;

/**
 * The lumps a preview uses, from those read back: all of the gas, in no more than `max` lumps.
 * The box is cut as a tree: starting from the whole of it, the piece holding the most mass is
 * cut in two along each axis, again and again, until there are as many pieces as allowed or
 * each holds a single lump of those read back. So the gas is summed finely where most of it is
 * (a star keeps the resolution it was read at) and coarsely where it is thin, and none is left
 * out: the thin gas around a star, dropped, shifted the pull on a mass beside it by a percent.
 */
export function fitLumps(lumps: Lump[], box: number, ndim: number, max = MAX_LUMPS): Lump[] {
  const src = lumps.filter((l) => l.m > 0);
  const n = src.length;
  if (!n) return [];
  // The lumps are sorted in place into the pieces of the tree (an index into them, cut by
  // counting), since a fine 2D mesh has tens of thousands of them and this runs several times
  // a second: each piece is a stretch [lo, hi) of the index.
  const m = new Float64Array(n);
  const u = [new Float64Array(n), new Float64Array(n), new Float64Array(n)]; // positions in [0, 1)
  for (let i = 0; i < n; i++) {
    m[i] = src[i].m;
    for (let c = 0; c < 3; c++) u[c][i] = src[i].pos[c] / box + 0.5;
  }
  const index = new Int32Array(n);
  const spare = new Int32Array(n);
  for (let i = 0; i < n; i++) index[i] = i;
  const code = new Uint8Array(n);
  interface Piece { lo: number; hi: number; k: number; m: number }
  const pieces: Piece[] = [{ lo: 0, hi: n, k: 1, m: m.reduce((t, v) => t + v, 0) }];
  const children = ndim === 3 ? 8 : 4;
  const count = new Int32Array(children + 1);
  const mass = new Float64Array(children);
  for (;;) {
    if (pieces.length + children - 1 > max) break;
    // The heaviest piece that can still be cut (4096 cells a side is finer than any grid's lumps).
    let at = -1;
    for (let i = 0; i < pieces.length; i++) {
      const q = pieces[i];
      if (q.hi - q.lo > 1 && q.k < 4096 && (at < 0 || q.m > pieces[at].m)) at = i;
    }
    if (at < 0) break;
    const { lo, hi, k } = pieces[at];
    const k2 = 2 * k;
    const half = (v: number) => Math.min(k2 - 1, Math.max(0, Math.floor(v * k2))) & 1;
    count.fill(0);
    mass.fill(0);
    for (let j = lo; j < hi; j++) {
      const i = index[j];
      const c = half(u[0][i]) + 2 * half(u[1][i]) + (ndim === 3 ? 4 * half(u[2][i]) : 0);
      code[j] = c;
      count[c + 1]++;
      mass[c] += m[i];
    }
    for (let c = 0; c < children; c++) count[c + 1] += count[c];
    const cut: Piece[] = [];
    for (let c = 0; c < children; c++) if (count[c + 1] > count[c]) cut.push({ lo: lo + count[c], hi: lo + count[c + 1], k: k2, m: mass[c] });
    for (let j = lo; j < hi; j++) spare[lo + count[code[j]]++] = index[j];
    index.set(spare.subarray(lo, hi), lo);
    pieces.splice(at, 1, ...cut);
  }
  return pieces.map((q) => {
    if (q.hi - q.lo === 1) return src[index[q.lo]];
    const pos: [number, number, number] = [0, 0, 0];
    for (let j = q.lo; j < q.hi; j++) {
      const l = src[index[j]];
      for (let c = 0; c < 3; c++) pos[c] += (l.m * l.pos[c]) / q.m;
    }
    return { m: q.m, pos, h: box / q.k };
  });
}
