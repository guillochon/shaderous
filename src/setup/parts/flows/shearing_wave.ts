// A shearing sheet (physics.shear): the background flow v_y = -q Omega x of the local model of a
// disk, with a compressive wave in density on it, cos(kx x + ky y), of kx and ky wavelengths
// across the box. The shear swings a leading wave (kx < 0 for ky > 0) to trailing, amplifying
// it on the way (Johnson & Gammie 2005). Gas at rest otherwise: uniform density and pressure.
import type { FlowDef } from '../flow';

const shearingWave: FlowDef<{ amplitude: number; kx: number; ky: number; pressure: number }> = {
  name: 'shearing-wave',
  label: 'Shearing wave',
  tip: 'The background flow of a shearing sheet, v_y = -q Omega x, with a small density wave cos(kx x + ky y) on it (kx and ky in wavelengths across the box). A leading wave (kx < 0) is swung to trailing by the shear and amplified on the way. Needs physics.shear.',
  wholeBox: true,
  fields: {
    amplitude: { label: 'Amplitude', default: 0.01, min: 0, max: 0.5, factor: 2, tip: 'Relative amplitude of the density wave' },
    kx: { label: 'Wavenumber along x', default: -2, min: -32, max: 32, step: 1, tip: 'Wavelengths across the box along x (negative: a leading wave)' },
    ky: { label: 'Wavenumber along y', default: 1, min: -32, max: 32, step: 1, tip: 'Wavelengths across the box along y' },
    pressure: { label: 'Pressure', default: 0.01, min: 1e-8, max: 100, factor: 2, tip: 'Uniform pressure of the gas (its density averages 1)' },
  },
  init: (ctx, p) => {
    const sh = ctx.spec.physics?.shear;
    if (!sh) throw new Error('the shearing wave needs physics.shear');
    const q = sh.q ?? 1.5;
    const omega = sh.omega ?? 1;
    const L = ctx.box;
    const k = [(2 * Math.PI * p.kx) / L, (2 * Math.PI * p.ky) / L];
    // A sound wave: the pressure follows the density adiabatically (at uniform pressure it would
    // be an entropy wave, which the shear only carries round).
    const gamma = ctx.spec.physics?.eosParams?.gamma ?? 5 / 3;
    return (x, y) => {
      const w = Math.cos(k[0] * x + k[1] * y);
      return { rho: 1 + p.amplitude * w, vx: 0, vy: -q * omega * x, vz: 0, p: p.pressure * (1 + gamma * p.amplitude * w), s: 0.5 + 0.5 * w };
    };
  },
};

/** A deterministic hash of a position to [-1, 1), for seeding noise that every grid samples alike. */
function hash3(x: number, y: number, z: number, seed: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7 + seed * 19.19) * 43758.5453;
  return 2 * (s - Math.floor(s)) - 1;
}

const shearingNoise: FlowDef<{ noise: number; pressure: number }> = {
  name: 'shearing-noise',
  label: 'Shearing box',
  tip: 'The background flow of a shearing box, v_y = -q Omega x, in gas of uniform density 1 and pressure, with small random velocities on it (a fraction of the sound speed) to seed instabilities such as the magnetorotational. Needs physics.shear.',
  wholeBox: true,
  fields: {
    noise: { label: 'Velocity noise', default: 1e-3, min: 0, max: 0.5, factor: 2, tip: 'Random velocities, as a fraction of the sound speed' },
    pressure: { label: 'Pressure', default: 0.6, min: 1e-8, max: 100, factor: 2, tip: 'Uniform pressure of the gas (its density is 1)' },
  },
  init: (ctx, p) => {
    const sh = ctx.spec.physics?.shear;
    if (!sh) throw new Error('the shearing box flow needs physics.shear');
    const q = sh.q ?? 1.5;
    const omega = sh.omega ?? 1;
    const gamma = ctx.spec.physics?.eosParams?.gamma ?? 5 / 3;
    const a = p.noise * Math.sqrt(gamma * p.pressure);
    return (x, y, z) => ({ rho: 1, vx: a * hash3(x, y, z, 1), vy: -q * omega * x + a * hash3(x, y, z, 2), vz: a * hash3(x, y, z, 3), p: p.pressure, s: 0 });
  },
};

export const flows = [shearingWave, shearingNoise];
