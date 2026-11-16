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
}

/**
 * The slab's sideways kick: one mode in x (`modes` wavelengths across a unit box), confined to
 * the two edges of the slab, weakly modulated in z so the flow is not exactly two-dimensional.
 */
function kicker(q: Shear): (x: number, y: number, z: number) => number {
  const sig = q.sigma;
  const edge = 0.5 * q.width;
  const k = 2 * q.modes;
  return (x, y, z) => {
    const env = Math.exp(-((y - edge) ** 2) / (2 * sig * sig)) + Math.exp(-((y + edge) ** 2) / (2 * sig * sig));
    return q.amp * Math.sin(k * Math.PI * x) * (1 + q.zmod * Math.cos(2 * Math.PI * z)) * env;
  };
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
    return (x, y, z) => {
      const inner = Math.abs(y) < edge;
      return { rho: inner ? p.rhoIn : p.rhoOut, vx: inner ? vIn : vOut, vy: kick(x, y, z), vz: 0, p: p.p, s: inner ? 1 : 0 };
    };
  },
};

export const flows = [shear];
