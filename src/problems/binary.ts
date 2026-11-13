import { all, OUTFLOW, type PointMass, type Problem } from './types';
import { ASTRO } from '../units/constants';
import { gravityUnits } from '../units/units';

// Binary accretion: two point masses on a Kepler orbit inside a gas disk that orbits them both.
//
// Units: G = 1, total mass 1, semi-major axis 1, so the binary's period is 2 pi. The gas is
// globally isothermal with sound speed `cs` (Mach 10 at r = 1) and starts on circular orbits
// outside a cavity of radius 2.5. The binary clears the cavity, pulls streams across it, and
// each mass takes in what reaches its sink. The masses feel one another and the gas, in an
// inertial frame; the disk is light, so the orbit barely changes.
//
// The disk has a constant kinematic viscosity `nu`, in units of a^2 Omega_b: what carries gas
// inward to the cavity's edge. 1e-3 is alpha = 0.1 at r = 1 for this sound speed
// (nu = alpha cs^2 / Omega).
//
// In 2D the density is the disk's surface density. In 3D the disk has a Gaussian vertical
// profile of scale height cs / Omega.
//
// Beyond r = `rDamp` the gas is relaxed toward the disk as it started (hydro/damping.wgsl), at
// a rate that rises from 0 there to `damp` times per local orbital period at the box's edge
// and stays so into the corners. Both can be changed in the Boundaries section during a run. The outer disk is then a reservoir that viscosity draws on, and
// the outflow boundaries never meet gas that has spread out to them: without it, gas flowed in
// at the corners, the disk grew to a fifth of the binary's mass, and its pull threw the binary
// out of the box.
//
// The run follows the binary's center of mass (gravity/bh_kick.wgsl), so the damping zone
// and the box move with it, as an outer disk would. On an adaptive mesh the gas around both
// masses is kept at the finest level, and their softening and sinks are sized by its cells.
export const BINARY = { cs: 0.1, rCavity: 2.5, sigma0: 1e-4, floor: 1e-4, box: 12, nu: 1e-3, rDamp: 4.5, damp: 3 };

// The setup is scale-free; physical units put it at one scale: two massive black holes of
// 1e5 solar masses each (a total of 2e5, the unit of mass, whatever the mass ratio) 0.01 pc
// apart. The orbital period is then about 210 years.
export const BINARY_UNITS = gravityUnits('massive black holes', 0.01 * ASTRO.pc, 2e5 * ASTRO.Msun);

/** A number as a WGSL f32 literal. */
const f32 = (v: number) => (Number.isInteger(v) ? `${v}.0` : String(v));

/** The disk's initial surface density at radius r (also its density in 2D). */
const surface = (r: number) => {
  const { rCavity, sigma0, floor, box } = BINARY;
  return sigma0 * (Math.exp(-((rCavity / r) ** 4)) * Math.exp(-((r / (0.42 * box)) ** 8)) + floor);
};

/**
 * The same disk, in WGSL, as the damping zone's target. Must match init() below. The zone
 * starts at radius P.diff.z; P.diff.w is its rate at the box's edge, per local orbital period.
 */
const DAMPING = `const DAMPING = true;
fn damping_target(x: vec3<f32>) -> DampTarget {
  let half = ${f32(BINARY.box / 2)};
  let r0 = P.diff.z;
  let r = max(length(x.xy), 1e-6);
  if (P.diff.w <= 0.0 || r <= r0 || r0 >= half) { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }
  let ramp = min((r - r0) / (half - r0), 1.0);
  let edge = min(r, half);
  let rate = P.diff.w * ramp * ramp * pow(edge, -1.5) / 6.28318531;
  let sigma = ${f32(BINARY.sigma0)} * (exp(-pow(${f32(BINARY.rCavity)} / r, 4.0)) * exp(-pow(r / ${f32(0.42 * BINARY.box)}, 8.0)) + ${f32(BINARY.floor)});
  let v = r / pow(r * r + 0.25, 0.75);
  var rho = sigma;
  if (P.dims.z > 1u) {
    let h = max(${f32(BINARY.cs)} * r / v, 2.0 * g_dx);
    rho = sigma / (2.50662827 * h) * exp(-x.z * x.z / (2.0 * h * h)) + ${f32(BINARY.sigma0 * BINARY.floor / BINARY.box)};
  }
  return DampTarget(rate, rho, vec3<f32>(-v * x.y / r, v * x.x / r, 0.0), ${f32(BINARY.cs * BINARY.cs)} * rho);
}
`;

/**
 * The two masses at pericenter, on the x axis. With `incl` 0 they orbit counter-clockwise
 * about the origin, as the disk does. Otherwise the orbit is tilted about the x axis (the line
 * of nodes, through pericenter) by `incl` degrees: 180 is a retrograde orbit in the disk's plane.
 */
export function binaryMasses(q: number, ecc: number, soft: number, incl = 0): PointMass[] {
  const m1 = 1 / (1 + q);
  const m2 = q / (1 + q);
  const r = 1 - ecc;
  // Pericenter speed of the Kepler orbit, lowered to match the softened pull between the two
  // (so that a circular orbit stays circular).
  const v = Math.sqrt((1 + ecc) / (1 - ecc)) * Math.pow((r * r) / (r * r + soft * soft), 0.75);
  // Exact at the ends, so that a flat orbit has no vertical velocity at all.
  const a = (incl * Math.PI) / 180;
  const cy = incl === 0 ? 1 : incl === 180 ? -1 : Math.cos(a);
  const cz = incl === 0 || incl === 180 ? 0 : Math.sin(a);
  return [
    { gm: m1, pos: [-m2 * r, 0, 0], vel: [0, -m2 * v * cy, -m2 * v * cz], soft, sinkRadius: soft },
    { gm: m2, pos: [m1 * r, 0, 0], vel: [0, m1 * v * cy, m1 * v * cz], soft, sinkRadius: soft },
  ];
}

export const binary: Problem = {
  id: 'binary',
  label: 'Binary accretion',
  blurb: 'Two point masses orbiting inside a gas disk: they clear a cavity, draw streams across it and accrete.',
  eos: 'barotropic',
  eosParams: { gamma: 1, K: BINARY.cs ** 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'off',
  G: 1,
  units: BINARY_UNITS,
  box: BINARY.box,
  tEnd: Infinity,
  checkpointDt: 1,
  rhoFloor: 1e-9,
  viscosity: BINARY.nu,
  refineMasses: true,
  frame: 'masses',
  damping: { wgsl: DAMPING, radius: BINARY.rDamp, rate: BINARY.damp },
  supports2d: true,
  view: { field: 0, log: true, mode: 'slice', axis: 2, pitch: 1.15 },
  params: [
    { id: 'q', label: 'Mass ratio', title: 'Mass of the lighter body over the heavier one. Sets the orbit: changing it starts the run again', value: 1, min: 0.01, max: 1, factor: 1.25 },
    { id: 'ecc', label: 'Eccentricity', title: 'Eccentricity of the binary’s orbit. Changing it starts the run again', value: 0, min: 0, max: 0.8, step: 0.05 },
    { id: 'soft', label: 'Softening and sink radius', title: 'Radius within which each mass’s pull is softened and it takes in gas (at least two cells). Changes take effect without a restart', value: 0.06, min: 0.01, max: 0.5, step: 0.01, live: true },
    { id: 'incl', label: 'Binary inclination (°)', title: 'Angle between the binary’s orbit and the disk, in degrees, tilted about the line through pericenter: 0 orbits with the disk, 90 is polar, 180 against it. Changing it starts the run again', value: 0, min: 0, max: 180, step: 1, kind: 'slider', dims: 3 },
    { id: 'retro', label: 'Retrograde', title: 'The binary orbits against the disk’s rotation. Changing it starts the run again', value: 0, min: 0, max: 1, kind: 'checkbox', dims: 2 },
  ],
  // Softening and sink are at least two cells across, or the grid cannot see them.
  masses: (p, _G, dx, ndim) => binaryMasses(p.q, p.ecc, Math.max(p.soft, 2 * dx), ndim === 3 ? p.incl : p.retro ? 180 : 0),
  init: (n, _eos, ndim) => {
    const { cs, sigma0, floor, box } = BINARY;
    const dx = box / n;
    return (x, y, z) => {
      const r = Math.max(Math.hypot(x, y), 1e-6);
      const sigma = surface(r);
      // Circular speed about the total mass, flattened inside the binary's own orbit.
      const v = r / Math.pow(r * r + 0.25, 0.75);
      let rho = sigma;
      if (ndim === 3) {
        const h = Math.max((cs * r) / v, 2 * dx);
        rho = (sigma / (Math.sqrt(2 * Math.PI) * h)) * Math.exp((-z * z) / (2 * h * h)) + (sigma0 * floor) / box;
      }
      return { rho, vx: (-v * y) / r, vy: (v * x) / r, vz: 0, p: cs * cs * rho, s: sigma > 0.5 * sigma0 ? 1 : 0 };
    };
  },
};
