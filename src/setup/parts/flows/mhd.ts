// Standard tests of ideal MHD (mhd/mhd.ts). Units: magnetic pressure B^2 / 2, so the Gaussian
// fields of the papers are divided by sqrt(4 pi). Coordinates are relative to the box center.
import type { FlowDef } from '../flow';

const SQ4PI = Math.sqrt(4 * Math.PI);

type None = Record<string, never>;

/**
 * Brio & Wu (1988) shock tube along x, gamma = 2: B_x = 0.75 throughout and B_y reversing across
 * the membrane. Its solution holds a compound wave (a slow shock with a rotational
 * discontinuity attached) as well as fast rarefactions, a slow shock and a contact.
 */
const brioWu: FlowDef<None> = {
  name: 'brio-wu',
  label: 'Brio-Wu shock tube',
  tip: 'The MHD shock tube of Brio & Wu (1988) along x (for gamma = 2): density 1 and pressure 1 on the left, 0.125 and 0.1 on the right, B_x = 0.75 throughout and B_y = 1 reversing across the membrane.',
  wholeBox: true,
  fields: {},
  face: () => (axis, x) => (axis === 0 ? 0.75 : axis === 1 ? (x < 0 ? 1 : -1) : 0),
  init: () => (x) => (x < 0 ? { rho: 1, vx: 0, vy: 0, vz: 0, p: 1, s: 1 } : { rho: 0.125, vx: 0, vy: 0, vz: 0, p: 0.1, s: 0 }),
};

/**
 * Orszag-Tang vortex (as in Athena): rho = 25 / (36 pi), p = 5 / (12 pi), gamma = 5/3,
 * v = (-sin 2 pi y, sin 2 pi x), B = (-sin 2 pi y, sin 4 pi x) / sqrt(4 pi), on the unit box,
 * periodic. The vortices steepen into shocks that interact; a standard check of the whole scheme.
 */
const orszagTang: FlowDef<None> = {
  name: 'orszag-tang',
  label: 'Orszag-Tang vortex',
  tip: 'The Orszag-Tang vortex on a periodic unit box (for gamma = 5/3): uniform gas with a velocity vortex and a field of twice its wavenumber in x.',
  wholeBox: true,
  fields: {},
  potential: () => (x, y) => {
    const X = x + 0.5;
    const Y = y + 0.5;
    return [0, 0, (Math.cos(4 * Math.PI * X) / (4 * Math.PI) + Math.cos(2 * Math.PI * Y) / (2 * Math.PI)) / SQ4PI];
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
 *
 * Dense: the loop's gas twice as dense as its surroundings, at the same pressure: a contact that
 * the adaptive mesh refines around and must follow across the box, refining ahead of the loop
 * and merging behind it, with the field re-meshed divergence-free each time
 * (mhd/amr_regrid.wgsl). On a uniform grid it is the field loop with a passive density contrast.
 */
const fieldLoop: FlowDef<{ dense: boolean }> = {
  name: 'field-loop',
  label: 'Field loop advection',
  tip: 'A weak loop of magnetic field of radius 0.3, A_z = 0.001 (0.3 - r), in gas of density 1 and pressure 1 moving at (2, 1): on a periodic unit box it is back where it started at t = 1 (Gardiner & Stone 2005).',
  wholeBox: true,
  fields: {
    dense: { label: 'Dense loop', default: false, tip: 'The gas of the loop is twice as dense as the gas around it, at the same pressure' },
  },
  potential: () => (x, y) => {
    const r = Math.hypot(x, y);
    return [0, 0, r < 0.3 ? 1e-3 * (0.3 - r) : 0];
  },
  init: (_ctx, p) => {
    if (p.dense)
      return (x, y) => {
        const inside = Math.hypot(x, y) < 0.3;
        return { rho: inside ? 2 : 1, vx: 2, vy: 1, vz: 0, p: 1, s: inside ? 1 : 0 };
      };
    return (x, y) => ({ rho: 1, vx: 2, vy: 1, vz: 0, p: 1, s: Math.hypot(x, y) < 0.3 ? 1 : 0 });
  },
};

/**
 * The gas of the MHD rotor (Balsara & Spicer 1999; the first rotor of Toth 2000): a dense disk
 * spinning in gas at rest, gamma = 1.4. With a uniform field B_x = 5 / sqrt(4 pi) (a field
 * part), torsional Alfven waves carry its angular momentum away and squeeze it into an oblong.
 */
const rotor: FlowDef<None> = {
  name: 'mhd-rotor',
  label: 'Rotor',
  tip: 'A disk of density 10 and radius 0.1 spinning at angular speed 20 in gas of density 1 at rest, pressure 1 throughout, with a ring to radius 0.115 where density and speed taper (Balsara & Spicer 1999). The field is a part of its own.',
  wholeBox: true,
  fields: {},
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
 * Circularly polarized Alfven wave (Toth 2000), along the diagonal of the x-y plane: rho = 1,
 * p = 0.1, B_parallel = 1, perpendicular field and velocity of amplitude 0.1, wavelength 1/sqrt(2)
 * so the box holds one along each axis. It is an exact nonlinear solution: after one period,
 * t = 1/sqrt(2), the state should be what it was, and the difference measures the scheme's error.
 */
const AW = { amp: 0.1, lambda: Math.SQRT1_2 };
const awPhase = (x: number, y: number) => (2 * Math.PI * (x + y)) / (Math.SQRT2 * AW.lambda);
const alfvenWave: FlowDef<None> = {
  name: 'alfven-wave',
  label: 'Circularly polarized Alfven wave',
  tip: 'A circularly polarized Alfven wave along the diagonal of the x-y plane of a periodic unit box: density 1, pressure 0.1, a field of 1 along the wave, and a perpendicular field and velocity of amplitude 0.1 (Toth 2000). It returns to its start after each period, 1 / sqrt(2).',
  wholeBox: true,
  fields: {},
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
  init: () => (x, y) => {
    const ph = awPhase(x, y);
    const v2 = AW.amp * Math.sin(ph);
    return { rho: 1, vx: -v2 / Math.SQRT2, vy: v2 / Math.SQRT2, vz: AW.amp * Math.cos(ph), p: 0.1, s: 0 };
  },
};

export const flows = [orszagTang, rotor, brioWu, fieldLoop, alfvenWave];
