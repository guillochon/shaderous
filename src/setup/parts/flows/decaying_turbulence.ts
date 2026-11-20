// Decaying turbulence: a periodic box of gas given a random, divergence-free velocity field at
// large scales and left alone. The eddies break up, and the flow decays. A band of the box is
// marked by the passive scalar (and can be denser), so that how the turbulence mixes it can be
// measured. Every seed is a different flow of the same statistics: the training and test sets
// of the turbulence closure (tools/closure) are made of these.
import type { FlowDef } from '../flow';
import { pressureForSoundSpeed } from './low_mach';

interface Decay {
  mach: number;
  kpeak: number;
  seed: number;
  ratio: number;
}

/** Uniform numbers in [0, 1) from a seed (mulberry32): the same in every browser and in Node. */
function random(seed: number): () => number {
  let a = (Math.floor(seed) | 0) + 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Mode {
  k: [number, number, number]; // wavenumber, in turns per box side
  a: [number, number, number]; // the vector potential's amplitude
  phase: number;
}

/**
 * The modes of the vector potential A = sum a cos(2 pi k.x + phase), whose curl is the velocity:
 * every wavevector of whole turns per box out to twice the peak, with random phases (and, in 3D,
 * random directions across k), the velocity's spectrum E(k) ~ k^4 exp(-2 (k / kpeak)^2) and its
 * rms speed 1. In 2D the potential is along z alone, so the flow stays in the plane.
 */
export function decayModes(q: Decay, ndim: 2 | 3): Mode[] {
  const rnd = random(q.seed);
  const kmax = Math.ceil(2 * q.kpeak);
  const modes: Mode[] = [];
  let power = 0;
  for (let kz = ndim === 3 ? -kmax : 0; kz <= (ndim === 3 ? kmax : 0); kz++)
    for (let ky = -kmax; ky <= kmax; ky++)
      for (let kx = 0; kx <= kmax; kx++) {
        // Half of the wavevectors: the other half are their conjugates.
        if (kx === 0 && (ky < 0 || (ky === 0 && kz <= 0))) continue;
        const k = Math.hypot(kx, ky, kz);
        // (The random numbers are drawn for every wavevector, so that a mode keeps its phase when
        // the peak moves.)
        const phase = 2 * Math.PI * rnd();
        const dir: [number, number, number] = ndim === 3 ? [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5] : [0, 0, 1];
        if (k > kmax) continue;
        // The velocity of a mode is k times its potential: its energy per mode is E(k) over the
        // modes of its shell, k^(ndim - 1) of them.
        const v = Math.sqrt(k ** 4 * Math.exp(-2 * (k / q.kpeak) ** 2) / k ** (ndim - 1));
        // Across k: only that part of the potential makes a velocity.
        const dot = (dir[0] * kx + dir[1] * ky + dir[2] * kz) / (k * k);
        const perp: [number, number, number] = [dir[0] - dot * kx, dir[1] - dot * ky, dir[2] - dot * kz];
        const len = Math.hypot(...perp) || 1;
        const a = v / k / len;
        modes.push({ k: [kx, ky, kz], a: [perp[0] * a, perp[1] * a, perp[2] * a], phase });
        power += 0.5 * v * v;
      }
  const norm = 1 / Math.sqrt(power);
  for (const m of modes) m.a = [m.a[0] * norm, m.a[1] * norm, m.a[2] * norm];
  return modes;
}

const decay: FlowDef<Decay & Record<string, number>> = {
  name: 'decaying-turbulence',
  label: 'Decaying turbulence',
  tip: 'A periodic box given a random, divergence-free velocity field at large scales (rms speed 1) and left to decay. The passive scalar marks a band across the middle of the box, which can be denser than the rest.',
  wholeBox: true,
  fields: {
    mach: { label: 'Mach number', default: 0.3, min: 0.01, max: 3, factor: 1.25, tip: 'The rms Mach number of the starting flow, in the gas outside the band: it sets the pressure' },
    kpeak: { label: 'Peak wavenumber', default: 4, min: 1, max: 16, step: 1, tip: 'The wavenumber, in turns per box side, at which the starting velocity has the most energy' },
    seed: { label: 'Seed', default: 1, min: 0, max: 1e6, step: 1, tip: 'Seed of the random phases: each is a different flow with the same statistics' },
    ratio: { label: 'Density of the band', default: 1, min: 0.01, max: 100, factor: 1.25, tip: 'Density of the marked band over that of the rest, at one pressure' },
  },
  init: (ctx, q) => {
    const modes = decayModes(q, ctx.ndim);
    const p0 = ctx.eos.hasEnergy ? pressureForSoundSpeed(ctx.eos, 1 / q.mach) : 0;
    const w = (2 * Math.PI) / ctx.box;
    return (x, y, z) => {
      // v = curl A: each mode gives -(k x a) sin(k.x + phase), times 2 pi / box.
      let vx = 0;
      let vy = 0;
      let vz = 0;
      for (const m of modes) {
        const s = -w * Math.sin(w * (m.k[0] * x + m.k[1] * y + m.k[2] * z) + m.phase);
        vx += s * (m.k[1] * m.a[2] - m.k[2] * m.a[1]);
        vy += s * (m.k[2] * m.a[0] - m.k[0] * m.a[2]);
        vz += s * (m.k[0] * m.a[1] - m.k[1] * m.a[0]);
      }
      // The band: the middle half of the box in y, its edges a few hundredths of the box wide.
      const s = 0.5 * (1 + Math.tanh(Math.cos(w * y) / 0.15));
      return { rho: 1 + (q.ratio - 1) * s, vx: vx / w, vy: vy / w, vz: ctx.ndim === 3 ? vz / w : 0, p: p0, s };
    };
  },
};

export const flows = [decay];
