// Partly filled cells on the uniform grid (plans/wind-tunnel.md 8.1; the float64 prototype is
// src/reference/cutcell.ts): obstacles at rest cut the cells their surfaces cross.
//
// The geometry is made once, on the CPU, from each obstacle's signed distance (exact for spheres,
// boxes and half-spaces; a mesh's distance field): each cell's gas share phi, and the open share
// of each of its upper faces (one texel holds a face's share, so both of its cells read the same
// bits), into CUT (phi, a_x, a_y, a_z); and into CUTW the distance from the cell's center to the
// surface (positive in the gas) and the body's mark, for the wall's shear. A cell is solid (its
// mark the body's) only where it holds no gas at all.
//
// The passes (hydro/cut_*.wgsl) run after the pressure's correction: the wall's shear from the
// velocity 1.5 cells out along its normal, then state redistribution (Berger & Giuliani 2021) as
// a gather, which keeps a cell holding a sliver of gas stable. The update, the pressure's passes
// and the diagnostics read CUT through the `cut` slot.
import { setSharedBinding, tex3d, writeTexture } from '../gpu/device';

export const CUT_BINDING = 61;
export const CUT_WALL_BINDING = 62;
/** Cells under this share of gas are merged with neighbors after each step. */
export const CUT_SMALL = 0.5;
/**
 * Below this share a cell is taken as wholly solid (and a face below it as closed): a pocket of
 * slivers at a thin body's corner (a turbine blade's, 1% of a cell beside cells of 1%) leaves a
 * sliver no neighborhood with enough gas to merge with, and it runs away.
 */
export const CUT_MIN = 0.05;
/** The wall's shear is from the velocity this many cells out from the wall, along its normal. */
export const CUT_PROBE = 1.5;
/** Samples per side of a cell (and of a face) for its share. */
const SUB = 4;

export interface CutGeometry {
  /** per cell: gas share, upper faces' open shares along x, y, z */
  cut: Float32Array;
  /** per cell: distance from its center to the nearest surface (positive in the gas), the body's mark, 0, 0 */
  wall: Float32Array;
  /** per cell: the body's mark where the cell holds no gas, else 0 */
  solid: Uint8Array;
  /** cells cut (0 < phi < 1), for reports */
  cutCells: number;
}

/**
 * The shares of gas and of open face area on a grid, from the obstacles' signed distances (each
 * sample's share is 1/2 + d/h, clamped, h the sample's size: second order on a smooth surface).
 */
export function cutGeometry(bodies: { mark: number; distance(x: number, y: number, z: number): number }[], dims: [number, number, number], dx: number, lo: readonly number[]): CutGeometry {
  const [nx, ny, nz] = dims;
  const three = nz > 1;
  const cells = nx * ny * nz;
  const cut = new Float32Array(cells * 4).fill(1);
  const wall = new Float32Array(cells * 4);
  const solid = new Uint8Array(cells);
  const dist = (x: number, y: number, z: number): [number, number] => {
    let d = Infinity, m = 0;
    for (const b of bodies) {
      const e = b.distance(x, y, z);
      if (e < d) {
        d = e;
        m = b.mark;
      }
    }
    return [d, m];
  };
  // A sample's gas share: a patch of side h, the surface's normal n across it, at distance d from
  // its center: linear over the patch's extent along n, h times the sum of n's components along
  // the patch's own directions (a patch parallel to the surface is all gas or all solid).
  const share = (d: number, w: number) => (w <= 1e-9 * dx ? (d > 0 ? 1 : 0) : Math.min(1, Math.max(0, 0.5 + d / w)));
  // The surface's normal near a point: the distance's gradient, by differences.
  const normal = (x: number, y: number, z: number): [number, number, number] => {
    const e = dx / 8;
    const g: [number, number, number] = [
      dist(x + e, y, z)[0] - dist(x - e, y, z)[0],
      dist(x, y + e, z)[0] - dist(x, y - e, z)[0],
      three ? dist(x, y, z + e)[0] - dist(x, y, z - e)[0] : 0,
    ];
    const l = Math.hypot(...g) || 1;
    return [g[0] / l, g[1] / l, g[2] / l];
  };
  const h = dx / SUB;
  const reach = dx * (0.5 * Math.sqrt(three ? 3 : 2) + 0.05);
  const at = (i: number, j: number, k: number) => (k * ny + j) * nx + i;
  const center = (i: number, j: number, k: number): [number, number, number] => [lo[0] + (i + 0.5) * dx, lo[1] + (j + 0.5) * dx, three ? lo[2] + (k + 0.5) * dx : 0];
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const q = at(i, j, k);
        const [x, y, z] = center(i, j, k);
        const [d, m] = dist(x, y, z);
        wall[4 * q] = d;
        wall[4 * q + 1] = m;
        // The cell's gas share.
        let phi = d > 0 ? 1 : 0;
        if (Math.abs(d) < reach) {
          const nn = normal(x, y, z);
          const w = h * (Math.abs(nn[0]) + Math.abs(nn[1]) + (three ? Math.abs(nn[2]) : 0));
          let s = 0;
          const nk = three ? SUB : 1;
          for (let c = 0; c < nk; c++)
            for (let b = 0; b < SUB; b++)
              for (let a = 0; a < SUB; a++) {
                const zz = three ? z - dx / 2 + (c + 0.5) * h : 0;
                s += share(dist(x - dx / 2 + (a + 0.5) * h, y - dx / 2 + (b + 0.5) * h, zz)[0], w);
              }
          phi = s / (SUB * SUB * nk);
        }
        cut[4 * q] = phi < CUT_MIN ? 0 : phi;
        // Its upper faces' open shares.
        for (let ax = 0; ax < (three ? 3 : 2); ax++) {
          const f: [number, number, number] = [x, y, z];
          f[ax] += dx / 2;
          const [df] = dist(f[0], f[1], f[2]);
          let a = df > 0 ? 1 : 0;
          if (Math.abs(df) < reach) {
            // Samples over the face: the two axes across it (one in 2D).
            const t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
            const nn = normal(f[0], f[1], f[2]);
            const w = h * ((three || t1 < 2 ? Math.abs(nn[t1]) : 0) + (three || t2 < 2 ? Math.abs(nn[t2]) : 0));
            let s = 0, n = 0;
            for (let v = 0; v < (three ? SUB : 1); v++)
              for (let u = 0; u < SUB; u++) {
                const p: [number, number, number] = [f[0], f[1], f[2]];
                if (three || t1 < 2) p[t1] += -dx / 2 + (u + 0.5) * h;
                if (three) p[t2] += -dx / 2 + (v + 0.5) * h;
                else if (t2 < 2) p[t2] += -dx / 2 + (u + 0.5) * h;
                s += share(dist(p[0], p[1], p[2])[0], w);
                n++;
              }
            a = s / n;
          }
          cut[4 * q + 1 + ax] = a < CUT_MIN ? 0 : a;
        }
      }
  // A cell with no gas closes its faces; a cell with gas whose faces are all closed holds none.
  const axes = three ? 3 : 2;
  const below = (i: number, j: number, k: number, ax: number) => {
    const c = [i, j, k];
    c[ax] = (c[ax] - 1 + dims[ax]) % dims[ax];
    return at(c[0], c[1], c[2]);
  };
  for (let pass = 0; pass < 2; pass++)
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const q = at(i, j, k);
          if (cut[4 * q] === 0) {
            for (let ax = 0; ax < axes; ax++) {
              cut[4 * q + 1 + ax] = 0;
              cut[4 * below(i, j, k, ax) + 1 + ax] = 0;
            }
            continue;
          }
          let open = 0;
          for (let ax = 0; ax < axes; ax++) open += cut[4 * q + 1 + ax] + cut[4 * below(i, j, k, ax) + 1 + ax];
          if (open === 0) cut[4 * q] = 0;
        }
  let cutCells = 0;
  for (let q = 0; q < cells; q++) {
    const phi = cut[4 * q];
    if (phi === 0) solid[q] = Math.max(1, Math.round(wall[4 * q + 1]));
    else if (phi < 1) cutCells++;
  }
  return { cut, wall, solid, cutCells };
}

/** The `cut` slot: CUT and the cell's shares, or 1 everywhere without partly filled cells. */
export function cutSlot(on: boolean): string {
  if (!on) {
    return 'const CUT = false;\nfn cut_at(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(1.0); }\nfn cut_wall(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(0.0); }\n';
  }
  return `const CUT = true;
const CUT_SMALL = ${CUT_SMALL};
const CUT_PROBE = ${CUT_PROBE};
@group(0) @binding(${CUT_BINDING}) var CUTT: texture_3d<f32>;
@group(0) @binding(${CUT_WALL_BINDING}) var CUTW: texture_3d<f32>;
// A cell of the grid, any cell beyond its edges taken around (the obstacles keep away from them).
fn cut_cell(c: vec3<i32>) -> vec3<i32> {
  let n = vec3<i32>(textureDimensions(CUTT));
  return (c % n + n) % n;
}
fn cut_at(c: vec3<i32>) -> vec4<f32> { return textureLoad(CUTT, cut_cell(c), 0); }
fn cut_wall(c: vec3<i32>) -> vec4<f32> { return textureLoad(CUTW, cut_cell(c), 0); }
`;
}

/** The textures of a run's cut geometry, shared with every pass that reads the slot. */
export class CutTextures {
  readonly cut: GPUTexture;
  readonly wall: GPUTexture;
  constructor(private device: GPUDevice, readonly geometry: CutGeometry, dims: [number, number, number]) {
    const [nx, ny, nz] = dims;
    this.cut = tex3d(device, nx, 'rgba32float', 'cut cells: shares', nz, ny);
    this.wall = tex3d(device, nx, 'rgba32float', 'cut cells: walls', nz, ny);
    writeTexture(device, this.cut, nx, 4, geometry.cut, nz, ny);
    writeTexture(device, this.wall, nx, 4, geometry.wall, nz, ny);
    this.share();
  }
  /** Make these the textures the passes built from now on bind. */
  share() {
    setSharedBinding(this.device, CUT_BINDING, this.cut.createView());
    setSharedBinding(this.device, CUT_WALL_BINDING, this.wall.createView());
  }
  destroy() {
    this.cut.destroy();
    this.wall.destroy();
  }
}
