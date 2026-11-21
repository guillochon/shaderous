// From triangles to a signed distance field (plans/wind-tunnel.md 3.2).
//
// 1. A bounding-volume tree over the triangles (median splits, a few triangles a leaf), each node
//    also holding its triangles' area-weighted normal and centroid, for the winding number's
//    far field.
// 2. Distance on a grid in the body's frame: exact (to the nearest triangle) within a band of
//    `band` cells of the surface, and clamped to the band's width beyond.
// 3. Sign by the generalized winding number (Jacobson, Kavan & Sorkine-Hornung 2013): the solid
//    angle the triangles subtend, over 4 pi; inside above 1/2. (The triangles' winding is first
//    made consistent and outward, mesh.ts orient: a flipped triangle subtracts its angle.) A
//    node far from the point (its
//    distance more than BETA times its radius) counts as its dipole (Barill et al. 2018). Holes
//    and pieces that pass through each other change the number a little, not
//    the answer, where ray parity fails outright. Within the band each point takes its own
//    number; beyond it, the number of a coarse grid of spacing band x h: a point beyond the band
//    is at least band x h from the surface, and its coarse sample within sqrt(3)/2 of that, so
//    the two are on the same side.
// 4. A report: volume, area, bounds, frontal area along each axis, the thinnest part, and how far
//    the winding number strays from 0 and 1 away from the surface (how broken the mesh is).
import { bounds, measure, orient, type Mesh } from './mesh';

/** A node is taken as its dipole beyond this many of its radii (Barill et al. use 2). */
const BETA = 2;
const LEAF = 4;

export interface DistanceField {
  /** center of grid point (0, 0, 0), in the mesh's frame */
  origin: [number, number, number];
  /** spacing */
  h: number;
  dims: [number, number, number];
  /** signed distance at each point, x fastest; negative inside; clamped to +-band beyond the band */
  data: Float32Array;
  /** the band's width, in lengths (band cells x h) */
  band: number;
}

export interface MeshReport {
  triangles: number;
  /** enclosed volume (divergence theorem) and the volume of the field's inside points */
  volume: number;
  insideVolume: number;
  area: number;
  min: [number, number, number];
  max: [number, number, number];
  /** area of the body's shadow along x, y and z: the reference areas of drag along each */
  frontal: [number, number, number];
  /** thinnest part: the shortest way through the body from any triangle, inward along its normal, in cells */
  thinnest: number;
  /** largest distance of the winding number from 0 or 1 away from the surface: 0 for a closed mesh, near 0.5 for a badly broken one */
  stray: number;
  /** triangles turned to make the winding consistent and outward */
  flipped: number;
  /** grid points of the field */
  points: number;
  /** seconds the field took */
  seconds: number;
}

/** The tree: per node its box, children or triangle range, and the far field's moments. */
class Bvh {
  readonly box: Float64Array; // 6 per node: min xyz, max xyz
  readonly kids: Int32Array; // 2 per node: left, right (-1 for a leaf)
  readonly range: Int32Array; // 2 per node: first triangle (in `order`), count
  readonly nrm: Float64Array; // 3 per node: area-weighted normal sum (half the cross products)
  readonly cen: Float64Array; // 3 per node: area-weighted centroid
  readonly rad: Float64Array; // per node: distance from the centroid to the box's farthest corner
  readonly order: Uint32Array;
  readonly tri: Float64Array; // 9 per triangle: its three corners
  nodes = 0;

  constructor(mesh: Mesh) {
    const n = mesh.indices.length / 3;
    const P = mesh.positions;
    this.tri = new Float64Array(9 * n);
    for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) for (let a = 0; a < 3; a++) this.tri[9 * t + 3 * k + a] = P[3 * mesh.indices[3 * t + k] + a];
    const cap = 2 * Math.ceil(n / LEAF) * 2 + 8;
    this.box = new Float64Array(6 * cap);
    this.kids = new Int32Array(2 * cap).fill(-1);
    this.range = new Int32Array(2 * cap);
    this.nrm = new Float64Array(3 * cap);
    this.cen = new Float64Array(3 * cap);
    this.rad = new Float64Array(cap);
    this.order = new Uint32Array(n);
    for (let t = 0; t < n; t++) this.order[t] = t;
    const centroid = new Float64Array(3 * n);
    for (let t = 0; t < n; t++) for (let a = 0; a < 3; a++) centroid[3 * t + a] = (this.tri[9 * t + a] + this.tri[9 * t + 3 + a] + this.tri[9 * t + 6 + a]) / 3;
    // Iterative build: a stack of (node, first, count).
    const stack: number[] = [this.alloc(), 0, n];
    while (stack.length) {
      const count = stack.pop()!;
      const first = stack.pop()!;
      const node = stack.pop()!;
      this.range[2 * node] = first;
      this.range[2 * node + 1] = count;
      const b = this.box.subarray(6 * node, 6 * node + 6);
      b.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
      let nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0, wsum = 0;
      for (let i = first; i < first + count; i++) {
        const t = this.order[i];
        const T = this.tri.subarray(9 * t, 9 * t + 9);
        for (let k = 0; k < 3; k++)
          for (let a = 0; a < 3; a++) {
            b[a] = Math.min(b[a], T[3 * k + a]);
            b[3 + a] = Math.max(b[3 + a], T[3 * k + a]);
          }
        const ux = T[3] - T[0], uy = T[4] - T[1], uz = T[5] - T[2];
        const vx = T[6] - T[0], vy = T[7] - T[1], vz = T[8] - T[2];
        const qx = 0.5 * (uy * vz - uz * vy), qy = 0.5 * (uz * vx - ux * vz), qz = 0.5 * (ux * vy - uy * vx);
        nx += qx;
        ny += qy;
        nz += qz;
        const w = Math.hypot(qx, qy, qz);
        wsum += w;
        cx += w * centroid[3 * t];
        cy += w * centroid[3 * t + 1];
        cz += w * centroid[3 * t + 2];
      }
      this.nrm.set([nx, ny, nz], 3 * node);
      const c = wsum > 0 ? [cx / wsum, cy / wsum, cz / wsum] : [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
      this.cen.set(c, 3 * node);
      let r = 0;
      for (let k = 0; k < 8; k++) r = Math.max(r, Math.hypot((k & 1 ? b[3] : b[0]) - c[0], (k & 2 ? b[4] : b[1]) - c[1], (k & 4 ? b[5] : b[2]) - c[2]));
      this.rad[node] = r;
      if (count <= LEAF) continue;
      // Split at the median along the box's longest side.
      const ext = [b[3] - b[0], b[4] - b[1], b[5] - b[2]];
      const axis = ext[0] >= ext[1] && ext[0] >= ext[2] ? 0 : ext[1] >= ext[2] ? 1 : 2;
      const seg = Array.from(this.order.subarray(first, first + count)).sort((p, q) => centroid[3 * p + axis] - centroid[3 * q + axis]);
      this.order.set(seg, first);
      const half = count >> 1;
      const l = this.alloc();
      const rr = this.alloc();
      this.kids[2 * node] = l;
      this.kids[2 * node + 1] = rr;
      stack.push(l, first, half, rr, first + half, count - half);
    }
  }

  private alloc(): number {
    return this.nodes++;
  }

  /** Squared distance from p to the node's box. */
  boxDist2(node: number, x: number, y: number, z: number): number {
    const b = this.box;
    const o = 6 * node;
    const dx = Math.max(b[o] - x, 0, x - b[o + 3]);
    const dy = Math.max(b[o + 1] - y, 0, y - b[o + 4]);
    const dz = Math.max(b[o + 2] - z, 0, z - b[o + 5]);
    return dx * dx + dy * dy + dz * dz;
  }

  /** After nearest2: the nearest triangle's faceSide (see triDist2). */
  side = 0;

  /** Squared distance to the nearest triangle, or `cap2` if none is nearer. */
  nearest2(x: number, y: number, z: number, cap2: number): number {
    let best = cap2;
    this.side = 0;
    const stack = [0];
    while (stack.length) {
      const node = stack.pop()!;
      if (this.boxDist2(node, x, y, z) >= best) continue;
      const l = this.kids[2 * node];
      if (l < 0) {
        const f = this.range[2 * node], c = this.range[2 * node + 1];
        for (let i = f; i < f + c; i++) {
          const d2 = triDist2(this.tri, this.order[i], x, y, z);
          if (d2 < best) {
            best = d2;
            this.side = faceSide;
          }
        }
        continue;
      }
      const r = this.kids[2 * node + 1];
      // Nearer child last, so it is searched first.
      if (this.boxDist2(l, x, y, z) < this.boxDist2(r, x, y, z)) stack.push(r, l);
      else stack.push(l, r);
    }
    return best;
  }

  /** Generalized winding number about p; `beta` Infinity for the exact sum, with no far field. */
  winding(x: number, y: number, z: number, beta = BETA): number {
    let w = 0;
    const stack = [0];
    while (stack.length) {
      const node = stack.pop()!;
      const dx = this.cen[3 * node] - x, dy = this.cen[3 * node + 1] - y, dz = this.cen[3 * node + 2] - z;
      const d = Math.hypot(dx, dy, dz);
      if (d > beta * this.rad[node]) {
        w += (dx * this.nrm[3 * node] + dy * this.nrm[3 * node + 1] + dz * this.nrm[3 * node + 2]) / (d * d * d);
        continue;
      }
      const l = this.kids[2 * node];
      if (l < 0) {
        const f = this.range[2 * node], c = this.range[2 * node + 1];
        for (let i = f; i < f + c; i++) w += solidAngle(this.tri, this.order[i], x, y, z);
        continue;
      }
      stack.push(l, this.kids[2 * node + 1]);
    }
    return w / (4 * Math.PI);
  }

  /** Distance along the ray from p in direction u to the nearest triangle it crosses, or Infinity. */
  ray(px: number, py: number, pz: number, ux: number, uy: number, uz: number, skip: number): number {
    let best = Infinity;
    const stack = [0];
    const inv = [1 / ux, 1 / uy, 1 / uz];
    while (stack.length) {
      const node = stack.pop()!;
      // Slab test.
      const o = 6 * node;
      let t0 = 0, t1 = best;
      const p = [px, py, pz];
      for (let a = 0; a < 3; a++) {
        let ta = (this.box[o + a] - p[a]) * inv[a];
        let tb = (this.box[o + 3 + a] - p[a]) * inv[a];
        if (ta > tb) [ta, tb] = [tb, ta];
        t0 = Math.max(t0, ta);
        t1 = Math.min(t1, tb);
      }
      if (!(t0 <= t1)) continue;
      const l = this.kids[2 * node];
      if (l < 0) {
        const f = this.range[2 * node], c = this.range[2 * node + 1];
        for (let i = f; i < f + c; i++) {
          const t = this.order[i];
          if (t === skip) continue;
          const h = rayTri(this.tri, t, px, py, pz, ux, uy, uz);
          if (h > 0 && h < best) best = h;
        }
        continue;
      }
      stack.push(l, this.kids[2 * node + 1]);
    }
    return best;
  }
}

/** Squared distance from p to triangle t (Ericson, Real-Time Collision Detection 5.1.5). */
// Set by triDist2: where the nearest point is inside the triangle's face, the side of it p is on
// (+1 the side its winding's normal points to, -1 the other); 0 where it is on an edge or corner.
let faceSide = 0;

function triDist2(T: Float64Array, t: number, px: number, py: number, pz: number): number {
  const o = 9 * t;
  faceSide = 0;
  const ax = T[o], ay = T[o + 1], az = T[o + 2];
  const abx = T[o + 3] - ax, aby = T[o + 4] - ay, abz = T[o + 5] - az;
  const acx = T[o + 6] - ax, acy = T[o + 7] - ay, acz = T[o + 8] - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  let qx: number, qy: number, qz: number;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;
  const bpx = px - T[o + 3], bpy = py - T[o + 4], bpz = pz - T[o + 5];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz;
  } else {
    const cpx = px - T[o + 6], cpy = py - T[o + 7], cpz = pz - T[o + 8];
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const w = d2 / (d2 - d6);
      qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz;
    } else {
      const va = d3 * d6 - d5 * d4;
      if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        qx = T[o + 3] + w * (T[o + 6] - T[o + 3]); qy = T[o + 4] + w * (T[o + 7] - T[o + 4]); qz = T[o + 5] + w * (T[o + 8] - T[o + 5]);
      } else {
        const den = 1 / (va + vb + vc);
        const v = vb * den, w = vc * den;
        qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
        const side = apx * (aby * acz - abz * acy) + apy * (abz * acx - abx * acz) + apz * (abx * acy - aby * acx);
        faceSide = side > 0 ? 1 : side < 0 ? -1 : 0;
      }
    }
  }
  return (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2;
}

/** Solid angle of triangle t seen from p (Van Oosterom & Strackee 1983), signed by its winding. */
function solidAngle(T: Float64Array, t: number, px: number, py: number, pz: number): number {
  const o = 9 * t;
  const ax = T[o] - px, ay = T[o + 1] - py, az = T[o + 2] - pz;
  const bx = T[o + 3] - px, by = T[o + 4] - py, bz = T[o + 5] - pz;
  const cx = T[o + 6] - px, cy = T[o + 7] - py, cz = T[o + 8] - pz;
  const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz), lc = Math.hypot(cx, cy, cz);
  const det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  const den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc + (bx * cx + by * cy + bz * cz) * la + (cx * ax + cy * ay + cz * az) * lb;
  return 2 * Math.atan2(det, den);
}

/** Möller-Trumbore: distance along the ray to triangle t, or -1. */
function rayTri(T: Float64Array, t: number, px: number, py: number, pz: number, ux: number, uy: number, uz: number): number {
  const o = 9 * t;
  const e1x = T[o + 3] - T[o], e1y = T[o + 4] - T[o + 1], e1z = T[o + 5] - T[o + 2];
  const e2x = T[o + 6] - T[o], e2y = T[o + 7] - T[o + 1], e2z = T[o + 8] - T[o + 2];
  const hx = uy * e2z - uz * e2y, hy = uz * e2x - ux * e2z, hz = ux * e2y - uy * e2x;
  const a = e1x * hx + e1y * hy + e1z * hz;
  if (Math.abs(a) < 1e-30) return -1;
  const f = 1 / a;
  const sx = px - T[o], sy = py - T[o + 1], sz = pz - T[o + 2];
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < 0 || u > 1) return -1;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = f * (ux * qx + uy * qy + uz * qz);
  if (v < 0 || u + v > 1) return -1;
  return f * (e2x * qx + e2y * qy + e2z * qz);
}

export interface SdfOptions {
  /** grid spacing, in the mesh's lengths */
  h: number;
  /** exact distances within this many cells of the surface (default 3) */
  band?: number;
  /** points along the longest side at most (default 256): a coarser spacing past it */
  maxPoints?: number;
}

/** The signed distance field of a mesh (one part, in its own frame) and its report. */
export function buildSdf(input: Mesh, opts: SdfOptions): { field: DistanceField; report: MeshReport } {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  // A consistent winding first: the winding number needs one (mesh.ts orient).
  const { mesh, flipped, pieces, open } = orient(input);
  // The side of the nearest face gives the sign only for one closed piece: where pieces touch
  // or pass through each other, the nearest face may be another piece's, inside the body.
  const closed = pieces === 1 && open === 0;
  const bandCells = opts.band ?? 3;
  const maxPoints = opts.maxPoints ?? 256;
  const { min, max } = bounds(mesh);
  const pad = bandCells + 1;
  let h = opts.h;
  const longest = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  if (longest / h + 2 * pad + 1 > maxPoints) h = longest / (maxPoints - 2 * pad - 1);
  const dims = [0, 1, 2].map((a) => Math.ceil((max[a] - min[a]) / h) + 2 * pad + 1) as [number, number, number];
  const origin = [0, 1, 2].map((a) => (min[a] + max[a]) / 2 - ((dims[a] - 1) * h) / 2) as [number, number, number];
  const [nx, ny, nz] = dims;
  const N = nx * ny * nz;
  const R = bandCells * h;
  const bvh = new Bvh(mesh);
  const at = (i: number, j: number, k: number): [number, number, number] => [origin[0] + i * h, origin[1] + j * h, origin[2] + k * h];

  // Candidates: points within R of some triangle's box.
  const near = new Uint8Array(N);
  const T = bvh.tri;
  const nt = T.length / 9;
  for (let t = 0; t < nt; t++) {
    const lo = [0, 1, 2].map((a) => Math.max(0, Math.floor((Math.min(T[9 * t + a], T[9 * t + 3 + a], T[9 * t + 6 + a]) - R - origin[a]) / h)));
    const hi = [0, 1, 2].map((a) => Math.min(dims[a] - 1, Math.ceil((Math.max(T[9 * t + a], T[9 * t + 3 + a], T[9 * t + 6 + a]) + R - origin[a]) / h)));
    for (let k = lo[2]; k <= hi[2]; k++) for (let j = lo[1]; j <= hi[1]; j++) for (let i = lo[0]; i <= hi[0]; i++) near[i + nx * (j + ny * k)] = 1;
  }

  // The coarse grid's winding numbers, spacing S = R (at least h): every S-th point.
  const step = Math.max(1, bandCells);
  const cd = dims.map((d) => Math.ceil((d - 1) / step) + 1);
  const coarse = new Float32Array(cd[0] * cd[1] * cd[2]);
  let stray = 0;
  // (The stray is measured on at most about 2000 of the coarse samples, with the far field taken
  // only beyond 8 node radii: at 2 its own error, a few percent, would hide a broken mesh's. The
  // exact sum, over every triangle, took most of the field's time on large meshes.)
  const every = Math.max(1, Math.floor(coarse.length / 2000));
  for (let k = 0; k < cd[2]; k++)
    for (let j = 0; j < cd[1]; j++)
      for (let i = 0; i < cd[0]; i++) {
        const [x, y, z] = at(Math.min(i * step, nx - 1), Math.min(j * step, ny - 1), Math.min(k * step, nz - 1));
        const w = bvh.winding(x, y, z);
        coarse[i + cd[0] * (j + cd[1] * k)] = w;
        // (How broken: measured where the surface is at least a cell away.)
        if ((i + cd[0] * (j + cd[1] * k)) % every === 0 && bvh.nearest2(x, y, z, h * h) >= h * h) {
          const we = bvh.winding(x, y, z, 8);
          stray = Math.max(stray, Math.min(Math.abs(we), Math.abs(1 - we)));
        }
      }

  const data = new Float32Array(N);
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const q = i + nx * (j + ny * k);
        const [x, y, z] = at(i, j, k);
        if (near[q]) {
          const d2 = bvh.nearest2(x, y, z, R * R);
          if (d2 < R * R) {
            const d = Math.sqrt(d2);
            // One closed piece, the nearest point inside a face: the side of it (the winding is
            // consistent and outward, orient). Otherwise the winding number.
            const inside = closed && bvh.side !== 0 ? bvh.side < 0 : bvh.winding(x, y, z) > 0.5;
            data[q] = inside ? -d : d;
            continue;
          }
        }
        const w = coarse[Math.round(i / step) + cd[0] * (Math.round(j / step) + cd[1] * Math.round(k / step))];
        data[q] = w > 0.5 ? -R : R;
      }
  const field: DistanceField = { origin, h, dims, data, band: R };

  // The report.
  const m = measure(mesh);
  let inside = 0;
  for (let q = 0; q < N; q++) if (data[q] < 0) inside++;
  const frontal: [number, number, number] = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const [u, v] = [(a + 1) % 3, (a + 2) % 3];
    let cols = 0;
    for (let jv = 0; jv < dims[v]; jv++)
      for (let ju = 0; ju < dims[u]; ju++) {
        for (let ja = 0; ja < dims[a]; ja++) {
          const idx = [0, 0, 0];
          idx[a] = ja;
          idx[u] = ju;
          idx[v] = jv;
          if (data[idx[0] + nx * (idx[1] + ny * idx[2])] < 0) {
            cols++;
            break;
          }
        }
      }
    frontal[a] = cols * h * h;
  }
  // Thickness: from each triangle's centroid, inward along its normal, to the next surface. A
  // winding the other way (a flipped triangle) looks outward and finds nothing, or the far side
  // of the body from outside: only hits are counted, and the thinnest of them is reported.
  let thinnest = Infinity;
  for (let t = 0; t < nt; t++) {
    const o = 9 * t;
    const ux = T[o + 3] - T[o], uy = T[o + 4] - T[o + 1], uz = T[o + 5] - T[o + 2];
    const vx = T[o + 6] - T[o], vy = T[o + 7] - T[o + 1], vz = T[o + 8] - T[o + 2];
    let qx = uy * vz - uz * vy, qy = uz * vx - ux * vz, qz = ux * vy - uy * vx;
    const len = Math.hypot(qx, qy, qz);
    if (!(len > 0)) continue;
    qx /= -len;
    qy /= -len;
    qz /= -len;
    const cx = (T[o] + T[o + 3] + T[o + 6]) / 3, cy = (T[o + 1] + T[o + 4] + T[o + 7]) / 3, cz = (T[o + 2] + T[o + 5] + T[o + 8]) / 3;
    const d = bvh.ray(cx, cy, cz, qx, qy, qz, t);
    if (d < thinnest) thinnest = d;
  }
  const seconds = ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) / 1000;
  const report: MeshReport = {
    triangles: nt, volume: m.volume, insideVolume: inside * h ** 3, area: m.area, min, max, frontal,
    thinnest: thinnest / h, stray, flipped, points: N, seconds,
  };
  return { field, report };
}

/** The field at a point of the mesh's frame, trilinear; beyond the grid, outside (+band). */
export function sampleSdf(f: DistanceField, x: number, y: number, z: number): number {
  const [nx, ny, nz] = f.dims;
  const gx = (x - f.origin[0]) / f.h, gy = (y - f.origin[1]) / f.h, gz = (z - f.origin[2]) / f.h;
  if (!(gx >= 0 && gy >= 0 && gz >= 0 && gx <= nx - 1 && gy <= ny - 1 && gz <= nz - 1)) return f.band;
  const i = Math.min(Math.floor(gx), nx - 2), j = Math.min(Math.floor(gy), ny - 2), k = Math.min(Math.floor(gz), nz - 2);
  const fx = gx - i, fy = gy - j, fz = gz - k;
  const d = f.data;
  const q = i + nx * (j + ny * k);
  const sx = 1, sy = nx, sz = nx * ny;
  const c00 = d[q] * (1 - fx) + d[q + sx] * fx;
  const c10 = d[q + sy] * (1 - fx) + d[q + sy + sx] * fx;
  const c01 = d[q + sz] * (1 - fx) + d[q + sz + sx] * fx;
  const c11 = d[q + sz + sy] * (1 - fx) + d[q + sz + sy + sx] * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}
