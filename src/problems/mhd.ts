// Standard tests of ideal MHD (mhd/mhd.ts). Units: magnetic pressure B^2 / 2, so the Gaussian
// fields of the papers are divided by sqrt(4 pi). Coordinates are relative to the box center.
import { all, OUTFLOW, PERIODIC, type Problem } from './types';

const SQ4PI = Math.sqrt(4 * Math.PI);

const base = {
  eos: 'ideal' as const,
  gravity: 'off' as const,
  G: 0,
  box: 1,
  supports2d: true,
  pFloor: 1e-10,
  rhoFloor: 1e-10,
};

/**
 * Brio & Wu (1988) shock tube along x, gamma = 2: B_x = 0.75 throughout and B_y reversing across
 * the membrane. Its solution holds a compound wave (a slow shock with a rotational
 * discontinuity attached) as well as fast rarefactions, a slow shock and a contact.
 */
export const brioWu: Problem = {
  ...base,
  id: 'brio-wu',
  label: 'Brio-Wu shock tube',
  blurb: 'The MHD shock tube of Brio & Wu (1988) along x: fast rarefactions, a compound wave, a contact and a slow shock.',
  eosParams: { gamma: 2 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  tEnd: 0.1,
  checkpointDt: 0.01,
  grid: { ndim: 2, n: 256 },
  view: { mode: 'slice', field: 0, log: false },
  mhd: { face: () => (axis, x) => (axis === 0 ? 0.75 : axis === 1 ? (x < 0 ? 1 : -1) : 0) },
  init: () => (x) => (x < 0 ? { rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 1 } : { rho: 0.125, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }),
};

/**
 * Orszag-Tang vortex (as in Athena): rho = 25 / (36 pi), p = 5 / (12 pi), gamma = 5/3,
 * v = (-sin 2 pi y, sin 2 pi x), B = (-sin 2 pi y, sin 4 pi x) / sqrt(4 pi), on the unit box,
 * periodic. The vortices steepen into shocks that interact; a standard check of the whole scheme.
 */
export const orszagTang: Problem = {
  ...base,
  id: 'orszag-tang',
  label: 'Orszag-Tang vortex',
  blurb: 'Orszag-Tang vortex: a periodic magnetized vortex whose waves steepen into interacting shocks.',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  tEnd: 0.5,
  checkpointDt: 0.05,
  grid: { ndim: 2, n: 256 },
  view: { mode: 'slice', field: 3, log: false },
  mhd: {
    potential: () => (x, y) => {
      const X = x + 0.5;
      const Y = y + 0.5;
      return [0, 0, (Math.cos(4 * Math.PI * X) / (4 * Math.PI) + Math.cos(2 * Math.PI * Y) / (2 * Math.PI)) / SQ4PI];
    },
  },
  init: () => (x, y) => {
    const X = x + 0.5;
    const Y = y + 0.5;
    return { rho: 25 / (36 * Math.PI), vx: -Math.sin(2 * Math.PI * Y), vy: Math.sin(2 * Math.PI * X), vz: 0, p: 5 / (12 * Math.PI), s: X < 0.5 ? 1 : 0 };
  },
};

/**
 * Field loop advection (Gardiner & Stone 2005): a weak loop of field, A_z = A0 (R - r) inside
 * radius R, carried across a periodic box by a uniform flow at beta of about 10^6. After t = 1
 * the flow (2, 1) has brought it back to where it started. A scheme whose edge fields are not
 * consistent with the flow distorts or diffuses the loop.
 */
export const fieldLoop: Problem = {
  ...base,
  id: 'field-loop',
  label: 'Field loop advection',
  blurb: 'A weak magnetic field loop carried across a periodic box by a uniform flow, back to its start at t = 1 (Gardiner & Stone 2005).',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  tEnd: 1,
  checkpointDt: 0.1,
  grid: { ndim: 2, n: 128 },
  view: { mode: 'slice', field: 25, log: false },
  mhd: {
    potential: () => (x, y) => {
      const r = Math.hypot(x, y);
      return [0, 0, r < 0.3 ? 1e-3 * (0.3 - r) : 0];
    },
  },
  init: () => (x, y) => ({ rho: 1, vx: 2, vy: 1, vz: 0, p: 1, s: Math.hypot(x, y) < 0.3 ? 1 : 0 }),
};

/**
 * MHD rotor (Balsara & Spicer 1999; the first rotor of Toth 2000): a dense disk spinning in a
 * uniform field B_x = 5 / sqrt(4 pi), gamma = 1.4. Torsional Alfven waves carry its angular
 * momentum away and squeeze it into an oblong.
 */
export const mhdRotor: Problem = {
  ...base,
  id: 'mhd-rotor',
  label: 'MHD rotor',
  blurb: 'A dense disk spinning in a uniform magnetic field: torsional Alfven waves brake it (Balsara & Spicer 1999).',
  eosParams: { gamma: 1.4 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  tEnd: 0.15,
  checkpointDt: 0.015,
  grid: { ndim: 2, n: 256 },
  view: { mode: 'slice', field: 0, log: false },
  mhd: { face: () => (axis) => (axis === 0 ? 5 / SQ4PI : 0) },
  init: () => (x, y) => {
    const r = Math.hypot(x, y);
    const r0 = 0.1;
    const r1 = 0.115;
    const f = (r1 - r) / (r1 - r0);
    const rho = r < r0 ? 10 : r < r1 ? 1 + 9 * f : 1;
    // Angular speed 2 / r0 inside; tapered to rest across the ring.
    const w = r < r0 ? 2 / r0 : r < r1 ? (f * 2) / r : 0;
    return { rho, vx: -w * y, vy: w * x, vz: 0, p: 1, s: r < r1 ? 1 : 0 };
  },
};

/**
 * MHD blast wave (Gardiner & Stone 2008, square box): p = 10 inside r < 0.1 and 0.1 outside, in a
 * field B = (1, 1, 0) / sqrt(2) at 45 degrees, gamma = 5/3. The field makes the blast oblong.
 */
export const mhdBlast: Problem = {
  ...base,
  id: 'mhd-blast',
  label: 'MHD blast wave',
  blurb: 'A blast wave in a magnetized gas: the field at 45 degrees stretches it along itself (Gardiner & Stone 2008).',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  tEnd: 0.2,
  checkpointDt: 0.02,
  grid: { ndim: 2, n: 256 },
  view: { mode: 'slice', field: 3, log: true },
  mhd: { face: () => (axis) => (axis < 2 ? Math.SQRT1_2 : 0) },
  init: (_n, _e, ndim) => (x, y, z) => {
    const r = ndim === 2 ? Math.hypot(x, y) : Math.hypot(x, y, z);
    return { rho: 1, vx: 0, vy: 0, vz: 0, p: r < 0.1 ? 10 : 0.1, s: r < 0.1 ? 1 : 0 };
  },
};

/**
 * Circularly polarized Alfven wave (Toth 2000), along the diagonal of the x-y plane: rho = 1,
 * p = 0.1, B_parallel = 1, perpendicular field and velocity of amplitude 0.1, wavelength 1/sqrt(2)
 * so the box holds one along each axis. It is an exact nonlinear solution: after one period,
 * t = 1/sqrt(2), the state should be what it was, and the difference measures the scheme's error.
 */
const AW = { amp: 0.1, lambda: Math.SQRT1_2 };
const awPhase = (x: number, y: number) => (2 * Math.PI * (x + y)) / (Math.SQRT2 * AW.lambda);
export const alfvenWave: Problem = {
  ...base,
  id: 'alfven-wave',
  label: 'Circularly polarized Alfven wave',
  blurb: 'A circularly polarized Alfven wave along the diagonal: an exact nonlinear solution that returns to its start after each period.',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  tEnd: AW.lambda,
  checkpointDt: AW.lambda / 10,
  grid: { ndim: 2, n: 128 },
  view: { mode: 'slice', field: 7, log: false },
  mhd: {
    // In the frame (x1 along the wave, x2 across it in the plane, z): A_z = x2 B_par + amp
    // lambda / (2 pi) cos(phase) and A_2 = amp lambda / (2 pi) sin(phase), so B_1 = B_par,
    // B_2 = amp sin(phase), B_3 = amp cos(phase).
    potential: () => (x, y) => {
      const x2 = (y - x) / Math.SQRT2;
      const ph = awPhase(x, y);
      const k = (AW.amp * AW.lambda) / (2 * Math.PI);
      const a2 = k * Math.sin(ph);
      // A_2 along (-1, 1) / sqrt(2).
      return [-a2 / Math.SQRT2, a2 / Math.SQRT2, x2 + k * Math.cos(ph)];
    },
  },
  init: () => (x, y) => {
    const ph = awPhase(x, y);
    const v2 = AW.amp * Math.sin(ph);
    return { rho: 1, vx: -v2 / Math.SQRT2, vy: v2 / Math.SQRT2, vz: AW.amp * Math.cos(ph), p: 0.1, s: 0 };
  },
};
