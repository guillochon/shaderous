// The reference state a gas is stored as its deviation from (SimConfig.reference,
// hydro/ref_on.wgsl; docs/numerics.md section 7). Built once from the initial state, in double
// precision, and handed to the GPU as one row of values along an axis (Params.reft).
//
// The passes never compute the reference's own forces: they take it to be in hydrostatic balance
// with the trapezoid rule of the balanced reconstruction (hydro/q.wgsl),
//   p[j+1] - p[j] = (rho[j] + rho[j+1]) / 2 * g dx,
// g the uniform field along the axis. So a column is built from the initial state's mean density
// in each layer, and its pressure is integrated from the top layer's mean down, which makes it
// balanced exactly in double precision. A uniform reference (no gravity) is balanced as it is.

/** Whether and how the gas is stored as its deviation from a reference state. */
export type RefKind = 'off' | 'uniform' | 'column';
export const REF_KINDS: readonly RefKind[] = ['off', 'uniform', 'column'];

/** Most cells a column's reference may have along its axis (Params.reft in hydro/common.wgsl). */
export const REF_CELLS = 2048;

/** A reference: along `axis` (0 to 2), or uniform (3), with one value per cell along it. */
export interface RefState {
  axis: number;
  rho: Float64Array;
  p: Float64Array;
  /** internal energy per volume, p / (gamma - 1) */
  e: Float64Array;
}

/**
 * The reference for an initial state of nx * ny * nz cells (x fastest), given each cell's
 * density and pressure in double precision. `g` is the uniform field (zero for 'uniform'),
 * `gamma` the ideal gas's. Returns why not, if the column cannot be built.
 */
export function buildReference(kind: Exclude<RefKind, 'off'>, dims: readonly number[], rho: Float64Array, p: Float64Array, gamma: number, g: readonly number[], dx: number): RefState | string {
  const [nx, ny, nz] = dims;
  const cells = nx * ny * nz;
  if (kind === 'uniform') {
    let sr = 0;
    let sp = 0;
    for (let q = 0; q < cells; q++) {
      sr += rho[q];
      sp += p[q];
    }
    const r = sr / cells;
    const pr = sp / cells;
    return { axis: 3, rho: Float64Array.of(r), p: Float64Array.of(pr), e: Float64Array.of(pr / (gamma - 1)) };
  }
  const axis = g.findIndex((v) => v !== 0);
  const n = dims[axis];
  const mr = new Float64Array(n);
  const mp = new Float64Array(n);
  const stride = [1, nx, nx * ny];
  for (let q = 0; q < cells; q++) {
    const j = Math.floor(q / stride[axis]) % n;
    mr[j] += rho[q];
    mp[j] += p[q];
  }
  const per = cells / n;
  for (let j = 0; j < n; j++) {
    mr[j] /= per;
    mp[j] /= per;
  }
  // The top is where the potential is highest: the far end if the field points down the axis.
  const ga = g[axis];
  const top = ga < 0 ? n - 1 : 0;
  const step = ga < 0 ? -1 : 1;
  const pr = new Float64Array(n);
  pr[top] = mp[top];
  // Going down, the pressure grows by the weight of the gas between two cells' centers.
  for (let j = top + step; j >= 0 && j < n; j += step) pr[j] = pr[j - step] + 0.5 * (mr[j] + mr[j - step]) * Math.abs(ga) * dx;
  for (let j = 0; j < n; j++) {
    if (!(mr[j] > 0) || !(pr[j] > 0)) return 'the column’s reference has a density or pressure that is not positive';
  }
  const e = pr.map((v) => v / (gamma - 1));
  return { axis, rho: mr, p: pr, e };
}

/** The reference's values at the cell q of an nx * ny * nz grid (x fastest): (rho, p, e). */
export function refAt(ref: RefState, dims: readonly number[], q: number): [number, number, number] {
  if (ref.axis > 2) return [ref.rho[0], ref.p[0], ref.e[0]];
  const stride = [1, dims[0], dims[0] * dims[1]];
  const j = Math.floor(q / stride[ref.axis]) % dims[ref.axis];
  return [ref.rho[j], ref.p[j], ref.e[j]];
}

/** The Params.refx and Params.reft words (hydro/common.wgsl), at float offset 140 of the block. */
export function packReference(f: Float32Array, at: number, ref: RefState | null) {
  if (!ref) return;
  f.set([1, ref.axis, 0, 0], at);
  for (let j = 0; j < ref.rho.length; j++) f.set([ref.rho[j], ref.p[j], ref.e[j], 0], at + 4 + 4 * j);
}
