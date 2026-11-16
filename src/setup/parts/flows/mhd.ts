// Standard tests of ideal MHD (mhd/mhd.ts). Units: magnetic pressure B^2 / 2, so the Gaussian
// fields of the papers are divided by sqrt(4 pi). Coordinates are relative to the box center.
import type { FlowDef } from '../flow';

const SQ4PI = Math.sqrt(4 * Math.PI);


/**
 * Brio & Wu (1988) shock tube along x, gamma = 2: B_x = 0.75 throughout and B_y reversing across
 * the membrane. Its solution holds a compound wave (a slow shock with a rotational
 * discontinuity attached) as well as fast rarefactions, a slow shock and a contact.
 */
const brioWu: FlowDef<{ rhoL: number; pL: number; rhoR: number; pR: number; bx: number; by: number }> = {
  name: 'brio-wu',
  label: 'Brio-Wu shock tube',
  tip: 'The MHD shock tube of Brio & Wu (1988) along x (for gamma = 2): gas at rest in two states either side of a membrane at the center, a field B_x throughout and B_y reversing across the membrane.',
  wholeBox: true,
  fields: {
    rhoL: { label: 'Density, left', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas left of the membrane' },
    pL: { label: 'Pressure, left', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure of the gas left of the membrane' },
    rhoR: { label: 'Density, right', default: 0.125, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas right of the membrane' },
    pR: { label: 'Pressure, right', default: 0.1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure of the gas right of the membrane' },
    bx: { label: 'Field along the tube', default: 0.75, min: -100, max: 100, step: 0.05, tip: 'B_x, the same on both sides (magnetic pressure is B\u00b2 / 2)' },
    by: { label: 'Field across the tube', default: 1, min: 0, max: 100, step: 0.05, tip: 'B_y on the left; on the right it is reversed' },
  },
  face: (_ctx, q) => (axis, x) => (axis === 0 ? q.bx : axis === 1 ? (x < 0 ? q.by : -q.by) : 0),
  init: (_ctx, q) => (x) => (x < 0 ? { rho: q.rhoL, vx: 0, vy: 0, vz: 0, p: q.pL, s: 1 } : { rho: q.rhoR, vx: 0, vy: 0, vz: 0, p: q.pR, s: 0 }),
};

/**
 * Orszag-Tang vortex (as in Athena): rho = 25 / (36 pi), p = 5 / (12 pi), gamma = 5/3,
 * v = (-sin 2 pi y, sin 2 pi x), B = (-sin 2 pi y, sin 4 pi x) / sqrt(4 pi), on the unit box,
 * periodic. The vortices steepen into shocks that interact; a standard check of the whole scheme.
 */
const orszagTang: FlowDef<{ v0: number; b0: number }> = {
  name: 'orszag-tang',
  label: 'Orszag-Tang vortex',
  tip: 'The Orszag-Tang vortex on a periodic unit box (for gamma = 5/3): uniform gas of density 25 / 36 pi and pressure 5 / 12 pi (sound speed 1) with a velocity vortex and a field of twice its wavenumber in x.',
  wholeBox: true,
  fields: {
    v0: { label: 'Velocity amplitude', default: 1, min: 0, max: 100, step: 0.1, tip: 'Peak speed of the vortex, in units of the sound speed: 1 in the standard test' },
    b0: { label: 'Field amplitude', default: 1, min: 0, max: 100, step: 0.1, tip: 'Strength of the field over the standard test\u2019s, which is 1 / sqrt(4 pi)' },
  },
  potential: (_ctx, q) => (x, y) => {
    const X = x + 0.5;
    const Y = y + 0.5;
    return [0, 0, (q.b0 * (Math.cos(4 * Math.PI * X) / (4 * Math.PI) + Math.cos(2 * Math.PI * Y) / (2 * Math.PI))) / SQ4PI];
  },
  init: (_ctx, q) => (x, y) => {
    const X = x + 0.5;
    const Y = y + 0.5;
    return { rho: 25 / (36 * Math.PI), vx: -q.v0 * Math.sin(2 * Math.PI * Y), vy: q.v0 * Math.sin(2 * Math.PI * X), vz: 0, p: 5 / (12 * Math.PI), s: X < 0.5 ? 1 : 0 };
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
const fieldLoop: FlowDef<{ dense: boolean; radius: number; a0: number; vx: number; vy: number; rhoIn: number }> = {
  name: 'field-loop',
  label: 'Field loop advection',
  tip: 'A weak loop of magnetic field, A_z = A0 (R - r) inside radius R, in gas of density 1 and pressure 1 carried by a uniform flow: at (2, 1) on a periodic unit box it is back where it started at t = 1 (Gardiner & Stone 2005).',
  wholeBox: true,
  fields: {
    dense: { label: 'Dense loop', default: false, tip: 'The gas of the loop is denser than the gas around it, at the same pressure' },
    rhoIn: { label: 'Density of a dense loop', default: 2, min: 1, max: 1e3, factor: 1.25, tip: 'Density of the gas inside a dense loop (the gas around it has density 1)' },
    radius: { label: 'Loop radius', default: 0.3, min: 0.02, max: 0.5, step: 0.02, tip: 'Radius of the loop, in code lengths' },
    a0: { label: 'Field strength', default: 1e-3, min: 0, max: 10, factor: 2, tip: 'A0, the strength of the field in the loop (magnetic pressure is B\u00b2 / 2): 1e-3 is a plasma beta of about 1e6' },
    vx: { label: 'Velocity x', default: 2, min: -100, max: 100, step: 0.25, tip: 'Velocity of the flow along x' },
    vy: { label: 'Velocity y', default: 1, min: -100, max: 100, step: 0.25, tip: 'Velocity of the flow along y' },
  },
  potential: (_ctx, q) => (x, y) => {
    const r = Math.hypot(x, y);
    return [0, 0, r < q.radius ? q.a0 * (q.radius - r) : 0];
  },
  init: (_ctx, p) => {
    if (p.dense)
      return (x, y) => {
        const inside = Math.hypot(x, y) < p.radius;
        return { rho: inside ? p.rhoIn : 1, vx: p.vx, vy: p.vy, vz: 0, p: 1, s: inside ? 1 : 0 };
      };
    return (x, y) => ({ rho: 1, vx: p.vx, vy: p.vy, vz: 0, p: 1, s: Math.hypot(x, y) < p.radius ? 1 : 0 });
  },
};

/**
 * The gas of the MHD rotor (Balsara & Spicer 1999; the first rotor of Toth 2000): a dense disk
 * spinning in gas at rest, gamma = 1.4. With a uniform field B_x = 5 / sqrt(4 pi) (a field
 * part), torsional Alfven waves carry its angular momentum away and squeeze it into an oblong.
 */
const rotor: FlowDef<{ rho: number; v0: number; r0: number; r1: number; p: number }> = {
  name: 'mhd-rotor',
  label: 'Rotor',
  tip: 'A dense disk spinning rigidly in gas of density 1 at rest, at one pressure, with a ring around it where density and speed taper (Balsara & Spicer 1999). The field is a part of its own.',
  wholeBox: true,
  fields: {
    rho: { label: 'Disk density', default: 10, min: 1, max: 1e4, factor: 1.25, tip: 'Density of the spinning disk (the gas around it has density 1)' },
    v0: { label: 'Rim speed', default: 2, min: 0, max: 100, step: 0.25, tip: 'Speed of the disk at its rim: its angular speed is this over its radius' },
    r0: { label: 'Disk radius', default: 0.1, min: 0.01, max: 0.45, step: 0.01, tip: 'Radius of the disk, in code lengths' },
    r1: { label: 'Outer radius of the ring', default: 0.115, min: 0.011, max: 0.5, step: 0.005, tip: 'Where the taper ends and the gas is at rest: larger than the disk\u2019s radius' },
    p: { label: 'Pressure', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure, the same everywhere' },
  },
  init: (_ctx, q) => (x, y) => {
    const r = Math.hypot(x, y);
    const r0 = q.r0;
    const r1 = Math.max(q.r1, r0 * (1 + 1e-6));
    const f = (r1 - r) / (r1 - r0);
    const rho = r < r0 ? q.rho : r < r1 ? 1 + (q.rho - 1) * f : 1;
    // Angular speed v0 / r0 inside; tapered to rest across the ring.
    const w = r < r0 ? q.v0 / r0 : r < r1 ? (f * q.v0) / r : 0;
    return { rho, vx: -w * y, vy: w * x, vz: 0, p: q.p, s: r < r1 ? 1 : 0 };
  },
};

/**
 * Circularly polarized Alfven wave (Toth 2000), along the diagonal of the x-y plane: rho = 1,
 * p = 0.1, B_parallel = 1, perpendicular field and velocity of amplitude 0.1, wavelength 1/sqrt(2)
 * so the box holds one along each axis. It is an exact nonlinear solution: after one period,
 * t = 1/sqrt(2), the state should be what it was, and the difference measures the scheme's error.
 */
const AW = { lambda: Math.SQRT1_2 };
const awPhase = (x: number, y: number) => (2 * Math.PI * (x + y)) / (Math.SQRT2 * AW.lambda);
const alfvenWave: FlowDef<{ amp: number; p: number }> = {
  name: 'alfven-wave',
  label: 'Circularly polarized Alfven wave',
  tip: 'A circularly polarized Alfven wave along the diagonal of the x-y plane of a periodic unit box: density 1, a field of 1 along the wave, and a perpendicular field and velocity that turn about it (Toth 2000). It returns to its start after each period, 1 / sqrt(2).',
  wholeBox: true,
  fields: {
    amp: { label: 'Amplitude', default: 0.1, min: 0, max: 2, step: 0.05, tip: 'Amplitude of the perpendicular field and velocity, over the field along the wave: the wave is an exact solution at any amplitude' },
    p: { label: 'Pressure', default: 0.1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Gas pressure, the same everywhere: 0.1 is a plasma beta of 0.2' },
  },
  // In the frame (x1 along the wave, x2 across it in the plane, z): A_z = x2 B_par + amp
  // lambda / (2 pi) cos(phase) and A_2 = amp lambda / (2 pi) sin(phase), so B_1 = B_par,
  // B_2 = amp sin(phase), B_3 = amp cos(phase).
  potential: (_ctx, q) => (x, y) => {
    const x2 = (y - x) / Math.SQRT2;
    const ph = awPhase(x, y);
    const k = (q.amp * AW.lambda) / (2 * Math.PI);
    const a2 = k * Math.sin(ph);
    // A_2 along (-1, 1) / sqrt(2).
    return [-a2 / Math.SQRT2, a2 / Math.SQRT2, x2 + k * Math.cos(ph)];
  },
  init: (_ctx, q) => (x, y) => {
    const ph = awPhase(x, y);
    const v2 = q.amp * Math.sin(ph);
    return { rho: 1, vx: -v2 / Math.SQRT2, vy: v2 / Math.SQRT2, vz: q.amp * Math.cos(ph), p: q.p, s: 0 };
  },
};

export const flows = [orszagTang, rotor, brioWu, fieldLoop, alfvenWave];
