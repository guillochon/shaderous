import { all, OUTFLOW, type PointMass, type Problem } from './types';

// Binary accretion: two point masses on a Kepler orbit inside a gas disc that orbits them both.
//
// Units: G = 1, total mass 1, semi-major axis 1, so the binary's period is 2 pi. The gas is
// globally isothermal with sound speed `cs` (Mach 10 at r = 1) and starts on circular orbits
// outside a cavity of radius 2.5. The binary clears the cavity, pulls streams across it, and
// each mass takes in what reaches its sink. The masses feel one another and the gas, in an
// inertial frame; the disc is light, so the orbit barely changes.
//
// The disc has a constant kinematic viscosity `nu`, in units of a^2 Omega_b: what carries gas
// inward to the cavity's edge. 1e-3 is alpha = 0.1 at r = 1 for this sound speed
// (nu = alpha cs^2 / Omega).
//
// In 2D the density is the disc's surface density. In 3D the disc has a Gaussian vertical
// profile of scale height cs / Omega.
export const BINARY = { cs: 0.1, rCavity: 2.5, sigma0: 1e-4, floor: 1e-4, box: 12, nu: 1e-3 };

/** The two masses at pericentre, on the x axis, orbiting counter-clockwise about the origin. */
export function binaryMasses(q: number, ecc: number, soft: number): PointMass[] {
  const m1 = 1 / (1 + q);
  const m2 = q / (1 + q);
  const r = 1 - ecc;
  // Pericentre speed of the Kepler orbit, lowered to match the softened pull between the two
  // (so that a circular orbit stays circular).
  const v = Math.sqrt((1 + ecc) / (1 - ecc)) * Math.pow((r * r) / (r * r + soft * soft), 0.75);
  return [
    { gm: m1, pos: [-m2 * r, 0, 0], vel: [0, -m2 * v, 0], soft, sinkRadius: soft },
    { gm: m2, pos: [m1 * r, 0, 0], vel: [0, m1 * v, 0], soft, sinkRadius: soft },
  ];
}

export const binary: Problem = {
  id: 'binary',
  label: 'Binary accretion',
  blurb: 'Two point masses orbiting inside a gas disc: they clear a cavity, draw streams across it and accrete.',
  eos: 'barotropic',
  eosParams: { gamma: 1, K: BINARY.cs ** 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'off',
  G: 1,
  box: BINARY.box,
  tEnd: Infinity,
  rhoFloor: 1e-9,
  viscosity: BINARY.nu,
  supports2d: true,
  view: { field: 0, log: true, mode: 'slice', axis: 2, pitch: 1.15 },
  params: [
    { id: 'q', label: 'Mass ratio', title: 'Mass of the lighter body over the heavier one. Sets the orbit: changing it starts the run again', value: 1, min: 0.01, max: 1, factor: 1.25 },
    { id: 'ecc', label: 'Eccentricity', title: 'Eccentricity of the binary’s orbit. Changing it starts the run again', value: 0, min: 0, max: 0.8, step: 0.05 },
    { id: 'soft', label: 'Softening and sink radius', title: 'Radius within which each mass’s pull is softened and it takes in gas (at least two cells). Changes take effect without a restart', value: 0.06, min: 0.01, max: 0.5, step: 0.01, live: true },
  ],
  // Softening and sink are at least two cells across, or the grid cannot see them.
  masses: (p, _G, dx) => binaryMasses(p.q, p.ecc, Math.max(p.soft, 2 * dx)),
  init: (n, _eos, ndim) => {
    const { cs, rCavity, sigma0, floor, box } = BINARY;
    const rOut = 0.42 * box;
    const dx = box / n;
    return (x, y, z) => {
      const r = Math.max(Math.hypot(x, y), 1e-6);
      const sigma = sigma0 * (Math.exp(-((rCavity / r) ** 4)) * Math.exp(-((r / rOut) ** 8)) + floor);
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
