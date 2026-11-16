// Kelvin-Helmholtz: a dense slab sliding through lighter gas, with a seeded sideways kick.
import { composition, meanMolecularWeight } from '../../../units/composition';
import type { FlowDef } from '../flow';

/** The slab's sideways kick: a single mode in x, weakly modulated in z so the flow is not exactly two-dimensional. */
function kick(x: number, y: number, z: number): number {
  const sig = 0.035;
  const env = Math.exp(-((y - 0.25) ** 2) / (2 * sig * sig)) + Math.exp(-((y + 0.25) ** 2) / (2 * sig * sig));
  return 0.1 * Math.sin(4 * Math.PI * x) * (1 + 0.2 * Math.cos(2 * Math.PI * z)) * env;
}

/**
 * Helium over hydrogen: mean molecular weights of the two gases, fully ionized (1/2 and 4/3).
 * At one pressure and one temperature the helium is denser by their ratio, 8/3.
 */
const MU_H = meanMolecularWeight(composition('hydrogen', { h1: 1 }));
const MU_HE = meanMolecularWeight(composition('helium', { he4: 1 }));

const shear: FlowDef<{ helium: boolean }> = {
  name: 'kelvin-helmholtz',
  label: 'Kelvin-Helmholtz shear layer',
  tip: 'A dense slab (|y| < 1/4 of a unit box) sliding at speed 1 through lighter gas, with one seeded mode. With helium, the slab is helium in hydrogen (species he4 and h1, which then set the equation of state), denser by the ratio of their mean molecular weights.',
  wholeBox: true,
  fields: {
    helium: { label: 'Helium slab in hydrogen', default: false, tip: 'The slab is helium and the gas around it hydrogen, at one pressure and temperature: needs the species he4 and h1' },
  },
  init: (_ctx, p) => {
    if (p.helium)
      return (x, y, z) => {
        const inner = Math.abs(y) < 0.25;
        return { rho: inner ? MU_HE / MU_H : 1, vx: inner ? 0.5 : -0.5, vy: kick(x, y, z), vz: 0, p: 2.5, s: inner ? 1 : 0, X: inner ? [1, 0] : [0, 1] };
      };
    return (x, y, z) => {
      const inner = Math.abs(y) < 0.25;
      return { rho: inner ? 2 : 1, vx: inner ? 0.5 : -0.5, vy: kick(x, y, z), vz: 0, p: 2.5, s: inner ? 1 : 0 };
    };
  },
};

export const flows = [shear];
