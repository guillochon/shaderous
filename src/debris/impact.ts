// What a giant impact leaves: the planet, the disk around it, and what escapes.
//
// The analysis of the moon-forming impact calculations (Canup 2004, section 2.3), on the cells
// of a run. The planet is found by iteration: about the center of the densest gas, each cell is
// on the Kepler orbit its position and velocity give it around the planet's mass, and it is
//   escaping   if that orbit is unbound,
//   the planet if the orbit's pericenter is inside the planet's radius,
//   the disk   otherwise,
// with the planet's mass and radius (that of a body of its mass at the mean density given, the
// Earth's by default) taken again from the cells that fell to it, until they settle. Gas is
// treated as moving freely: pressure holds up some of what counts here as the planet, which is
// the convention.
//
// From the disk's mass and angular momentum comes the mass of the moon it would gather into
// (Ida, Canup & Stewart 1997; Kokubo, Ida & Makino 2000), for a disk that loses a twentieth of
// its mass outward:
//   M_moon = 1.9 L_D / sqrt(G M_p a_R) - 1.1 M_D - 1.9 M_esc,   M_esc = 0.05 M_D
// with a_R = 2.9 R_p the Roche radius.

/** Cells of a run: centers, sizes, and the conserved state (rho, momentum) four per cell. */
export interface ImpactCells {
  count: number;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  z: ArrayLike<number>;
  dx: ArrayLike<number>;
  u1: ArrayLike<number>;
}

export interface ImpactDebris {
  /** masses of the planet, the disk and the escaping gas */
  planet: number;
  disk: number;
  escaping: number;
  /** the planet's radius, as used */
  radius: number;
  /** the disk's angular momentum about the planet, along the whole system's */
  diskL: number;
  /** that over what the disk's mass would have on a circular orbit at the Roche radius */
  diskLRoche: number;
  /** the moon the disk would gather into (0 if the disk is too compact) */
  moon: number;
  /** of each part's mass, the share that is each tracer's (`tracers`), in order: planet, disk, escaping */
  shares: number[][];
  /** where the planet is and how it moves */
  center: [number, number, number];
  velocity: [number, number, number];
}

/** Roche radius over the planet's radius, for a moon of lunar density about an Earth. */
export const ROCHE = 2.9;

/**
 * `tracers` are per-cell mass fractions (iron's, the impactor's), each summed by mass into
 * `shares`. `meanDensity` is the planet's, mass over volume, in the cells' units. Cells thinner
 * than `least` are the ambient gas and are left out.
 */
export function impactDebris(c: ImpactCells, G: number, meanDensity: number, tracers: ArrayLike<number>[] = [], least = 0): ImpactDebris {
  const n = c.count;
  // The densest gas: where the planet is.
  let peak = 0;
  for (let q = 0; q < n; q++) peak = Math.max(peak, c.u1[4 * q]);
  const m = new Float64Array(n);
  let M = 0;
  const cen = [0, 0, 0];
  const vel = [0, 0, 0];
  for (let q = 0; q < n; q++) {
    const rho = c.u1[4 * q];
    m[q] = rho > least ? rho * c.dx[q] ** 3 : 0;
    if (rho > 0.5 * peak && m[q] > 0) {
      M += m[q];
      cen[0] += m[q] * c.x[q];
      cen[1] += m[q] * c.y[q];
      cen[2] += m[q] * c.z[q];
      vel[0] += c.u1[4 * q + 1] * c.dx[q] ** 3;
      vel[1] += c.u1[4 * q + 2] * c.dx[q] ** 3;
      vel[2] += c.u1[4 * q + 3] * c.dx[q] ** 3;
    }
  }
  if (!(M > 0)) {
    return { planet: 0, disk: 0, escaping: 0, radius: 0, diskL: 0, diskLRoche: 0, moon: 0, shares: [0, 1, 2].map(() => tracers.map(() => 0)), center: [0, 0, 0], velocity: [0, 0, 0] };
  }
  for (let a = 0; a < 3; a++) {
    cen[a] /= M;
    vel[a] /= M;
  }
  // 0 planet, 1 disk, 2 escaping.
  const kind = new Uint8Array(n);
  const radiusOf = (mass: number) => Math.cbrt((3 * mass) / (4 * Math.PI * meanDensity));
  // The whole system's angular momentum about the planet gives the axis.
  let R = radiusOf(M);
  for (let round = 0; round < 40; round++) {
    let Mp = 0;
    const nc = [0, 0, 0];
    const nv = [0, 0, 0];
    const mu = G * M;
    for (let q = 0; q < n; q++) {
      if (m[q] === 0) continue;
      const rho = c.u1[4 * q];
      const rx = c.x[q] - cen[0];
      const ry = c.y[q] - cen[1];
      const rz = c.z[q] - cen[2];
      const vx = c.u1[4 * q + 1] / rho - vel[0];
      const vy = c.u1[4 * q + 2] / rho - vel[1];
      const vz = c.u1[4 * q + 3] / rho - vel[2];
      const r = Math.hypot(rx, ry, rz);
      let k = 0;
      if (r > R) {
        const e = 0.5 * (vx * vx + vy * vy + vz * vz) - mu / r;
        if (e >= 0) k = 2;
        else {
          const h2 = (ry * vz - rz * vy) ** 2 + (rz * vx - rx * vz) ** 2 + (rx * vy - ry * vx) ** 2;
          // Pericenter of the orbit: a (1 - ecc), with a = -mu / 2e and ecc^2 = 1 + 2 e h^2 / mu^2.
          const ecc = Math.sqrt(Math.max(0, 1 + (2 * e * h2) / (mu * mu)));
          k = (-mu / (2 * e)) * (1 - ecc) > R ? 1 : 0;
        }
      }
      kind[q] = k;
      if (k === 0) {
        Mp += m[q];
        nc[0] += m[q] * c.x[q];
        nc[1] += m[q] * c.y[q];
        nc[2] += m[q] * c.z[q];
        nv[0] += m[q] * (c.u1[4 * q + 1] / rho);
        nv[1] += m[q] * (c.u1[4 * q + 2] / rho);
        nv[2] += m[q] * (c.u1[4 * q + 3] / rho);
      }
    }
    if (!(Mp > 0)) break;
    const change = Math.abs(Mp - M) / Mp;
    M = Mp;
    R = radiusOf(M);
    for (let a = 0; a < 3; a++) {
      cen[a] = nc[a] / Mp;
      vel[a] = nv[a] / Mp;
    }
    if (change < 1e-6) break;
  }
  const mass = [0, 0, 0];
  const shares = [0, 1, 2].map(() => tracers.map(() => 0));
  const Ltot = [0, 0, 0];
  const Ld = [0, 0, 0];
  for (let q = 0; q < n; q++) {
    if (m[q] === 0) continue;
    const rho = c.u1[4 * q];
    const k = kind[q];
    mass[k] += m[q];
    tracers.forEach((t, i) => (shares[k][i] += m[q] * t[q]));
    const r = [c.x[q] - cen[0], c.y[q] - cen[1], c.z[q] - cen[2]];
    const v = [c.u1[4 * q + 1] / rho - vel[0], c.u1[4 * q + 2] / rho - vel[1], c.u1[4 * q + 3] / rho - vel[2]];
    const l = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]];
    for (let a = 0; a < 3; a++) {
      Ltot[a] += m[q] * l[a];
      if (k === 1) Ld[a] += m[q] * l[a];
    }
  }
  for (let k = 0; k < 3; k++) for (let i = 0; i < tracers.length; i++) shares[k][i] = mass[k] > 0 ? shares[k][i] / mass[k] : 0;
  const ln = Math.hypot(Ltot[0], Ltot[1], Ltot[2]);
  const diskL = ln > 0 ? (Ld[0] * Ltot[0] + Ld[1] * Ltot[1] + Ld[2] * Ltot[2]) / ln : 0;
  const roche = Math.sqrt(G * M * ROCHE * R);
  const moon = Math.max(0, (1.9 * diskL) / roche - 1.1 * mass[1] - 1.9 * 0.05 * mass[1]);
  return {
    planet: mass[0], disk: mass[1], escaping: mass[2], radius: R, diskL, diskLRoche: mass[1] > 0 ? diskL / (mass[1] * roche) : 0, moon, shares,
    center: [cen[0], cen[1], cen[2]], velocity: [vel[0], vel[1], vel[2]],
  };
}

/** A snapshot's blocks as cells (io/snapshot.ts): what impactDebris takes, and a per-cell array of each variable asked for. */
export function snapshotCells(
  snap: { rootSize: number[]; blockSize: number[]; lo: number[]; hi: number[]; levels: Int32Array; locations: Int32Array; data: Record<string, Float32Array | undefined> },
  names: readonly string[],
): { cells: ImpactCells; fields: Float32Array[] } {
  const [bx, by, bz] = snap.blockSize;
  const per = bx * by * bz;
  const n = snap.levels.length * per;
  const cells = { count: n, x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n), dx: new Float32Array(n), u1: new Float32Array(4 * n) };
  const { rho, vel1, vel2, vel3 } = snap.data;
  if (!rho || !vel1 || !vel2 || !vel3) throw new Error('the snapshot has no density or velocity');
  const d0 = (snap.hi[0] - snap.lo[0]) / snap.rootSize[0];
  for (let b = 0; b < snap.levels.length; b++) {
    const dx = d0 / 2 ** snap.levels[b];
    const ox = snap.lo[0] + snap.locations[3 * b] * bx * dx;
    const oy = snap.lo[1] + snap.locations[3 * b + 1] * by * dx;
    const oz = snap.lo[2] + snap.locations[3 * b + 2] * bz * dx;
    let q = b * per;
    for (let k = 0; k < bz; k++)
      for (let j = 0; j < by; j++)
        for (let i = 0; i < bx; i++, q++) {
          cells.x[q] = ox + (i + 0.5) * dx;
          cells.y[q] = oy + (j + 0.5) * dx;
          cells.z[q] = oz + (k + 0.5) * dx;
          cells.dx[q] = dx;
          const r = rho[q];
          cells.u1[4 * q] = r;
          cells.u1[4 * q + 1] = r * vel1[q];
          cells.u1[4 * q + 2] = r * vel2[q];
          cells.u1[4 * q + 3] = r * vel3[q];
        }
  }
  return { cells, fields: names.map((v) => snap.data[v] ?? new Float32Array(n)) };
}
