// Kelvin-Helmholtz: a dense slab sliding through lighter gas, with a seeded sideways kick.
import { composition, meanMolecularWeight } from '../../../units/composition';
import type { FlowDef } from '../flow';

interface Shear {
  helium: boolean;
  rhoIn: number;
  rhoOut: number;
  shear: number;
  p: number;
  width: number;
  amp: number;
  modes: number;
  sigma: number;
  zmod: number;
  smooth: number;
  seed: number;
}

/**
 * The slab's sideways kick: one mode in x (`modes` wavelengths across a unit box), confined to
 * the two edges of the slab, weakly modulated in z so the flow is not exactly two-dimensional.
 */
function kicker(q: Shear): (x: number, y: number, z: number) => number {
  const sig = q.sigma;
  const edge = 0.5 * q.width;
  const k = 2 * q.modes;
  // With a seed: every wavelength from the seeded mode's to half of it, with random
  // phases and amplitudes (mulberry32: the same in every browser and in Node), at each edge its
  // own, and the same rms as the one mode. Each seed is then another run of the same flow.
  const waves: { k: number; a: number; up: number; down: number }[] = [];
  if (q.seed > 0) {
    let a = (Math.floor(q.seed) | 0) + 0x9e3779b9;
    const rnd = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let m = q.modes; m <= 2 * q.modes; m++) waves.push({ k: 2 * Math.PI * m, a: 0.5 + rnd(), up: 2 * Math.PI * rnd(), down: 2 * Math.PI * rnd() });
    const norm = 1 / Math.sqrt(waves.reduce((t, w) => t + w.a * w.a, 0));
    for (const w of waves) w.a *= norm;
  }
  return (x, y, z) => {
    const up = Math.exp(-((y - edge) ** 2) / (2 * sig * sig));
    const down = Math.exp(-((y + edge) ** 2) / (2 * sig * sig));
    const along = 1 + q.zmod * Math.cos(2 * Math.PI * z);
    if (!waves.length) return q.amp * Math.sin(k * Math.PI * x) * along * (up + down);
    let v = 0;
    for (const w of waves) v += w.a * (up * Math.sin(w.k * x + w.up) + down * Math.sin(w.k * x + w.down));
    return q.amp * v * along;
  };
}

/** How much of the slab a point at height y is in: 1 inside, 0 outside, across edges `smooth` wide (0: sharp). */
function inSlab(y: number, edge: number, smooth: number): number {
  if (!(smooth > 0)) return Math.abs(y) < edge ? 1 : 0;
  return 0.5 * (Math.tanh((y + edge) / smooth) - Math.tanh((y - edge) / smooth));
}

/**
 * Helium over hydrogen: mean molecular weights of the two gases, fully ionized (1/2 and 4/3).
 * At one pressure and one temperature the helium is denser by their ratio, 8/3.
 */
const MU_H = meanMolecularWeight(composition('hydrogen', { h1: 1 }));
const MU_HE = meanMolecularWeight(composition('helium', { he4: 1 }));

const shear: FlowDef<Shear & Record<string, number | boolean>> = {
  name: 'kelvin-helmholtz',
  label: 'Kelvin-Helmholtz shear layer',
  tip: 'A dense slab across the middle of the box (in y), sliding through lighter gas at one pressure, with one seeded mode at its two edges. With helium, the slab is helium in hydrogen (species he4 and h1, which then set the equation of state), denser by the ratio of their mean molecular weights.',
  wholeBox: true,
  fields: {
    helium: { label: 'Helium slab in hydrogen', default: false, tip: 'The slab is helium and the gas around it hydrogen, at one pressure and temperature: needs the species he4 and h1. The slab\u2019s density is then 8/3 of the gas around it' },
    rhoIn: { label: 'Slab density', default: 2, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the slab (without helium)' },
    rhoOut: { label: 'Density outside', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the gas on either side of the slab' },
    shear: { label: 'Shear speed', default: 1, min: 0, max: 100, step: 0.1, tip: 'Speed of the slab relative to the gas around it: the slab moves at half of it along +x, the rest at half of it along -x' },
    p: { label: 'Pressure', default: 2.5, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Pressure, the same everywhere' },
    width: { label: 'Slab thickness', default: 0.5, min: 0.05, max: 0.9, step: 0.05, tip: 'Thickness of the slab, in code lengths (the preset\u2019s box has side 1)' },
    amp: { label: 'Kick amplitude', default: 0.1, min: 0, max: 1, step: 0.01, tip: 'Peak sideways velocity of the seeded mode' },
    modes: { label: 'Wavelengths across', default: 2, min: 1, max: 32, step: 1, tip: 'Wavelengths of the seeded mode along x in a length of 1' },
    sigma: { label: 'Kick width', default: 0.035, min: 0.005, max: 0.2, step: 0.005, tip: 'Width in y of the layer at each edge of the slab where the kick acts' },
    smooth: { label: 'Edge width', default: 0, min: 0, max: 0.2, step: 0.005, tip: 'Width of the slab\u2019s edges: density, speed and dye go from the slab\u2019s to the surroundings\u2019 as a tanh of this width. 0 is a sharp edge, which no resolution converges on; with a width and a viscosity the flow has an answer' },
    seed: { label: 'Seed', default: 0, min: 0, max: 1e6, step: 1, tip: '0: the one seeded mode. Otherwise the kick is a random mix of wavelengths (from the seeded mode\u2019s to half of it) with the same rms, another for each seed' },
    zmod: { label: 'Variation along z', default: 0.2, min: 0, max: 1, step: 0.05, tip: 'How much the kick varies along z (one wavelength a unit length), so that a 3D run is not exactly two-dimensional. 0 is none' },
  },
  init: (_ctx, p) => {
    const kick = kicker(p);
    const edge = 0.5 * p.width;
    const [vIn, vOut] = [0.5 * p.shear, -0.5 * p.shear];
    if (p.helium)
      return (x, y, z) => {
        const inner = Math.abs(y) < edge;
        return { rho: inner ? (p.rhoOut * MU_HE) / MU_H : p.rhoOut, vx: inner ? vIn : vOut, vy: kick(x, y, z), vz: 0, p: p.p, s: inner ? 1 : 0, X: inner ? [1, 0] : [0, 1] };
      };
    // (A sharp edge, as it always was, to the bit.)
    if (!(p.smooth > 0))
      return (x, y, z) => {
        const inner = Math.abs(y) < edge;
        return { rho: inner ? p.rhoIn : p.rhoOut, vx: inner ? vIn : vOut, vy: kick(x, y, z), vz: 0, p: p.p, s: inner ? 1 : 0 };
      };
    return (x, y, z) => {
      const f = inSlab(y, edge, p.smooth);
      return { rho: p.rhoOut + (p.rhoIn - p.rhoOut) * f, vx: vOut + (vIn - vOut) * f, vy: kick(x, y, z), vz: 0, p: p.p, s: f };
    };
  },
};

export const flows = [shear];
