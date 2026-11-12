import { all, PERIODIC, MASS_FRACTION_FIELD, type Problem } from './types';
import { COMPOSITIONS, composition, meanMolecularWeight } from '../units/composition';

/** The slab's sideways kick: a single mode in x, weakly modulated in z so the flow is not exactly two-dimensional. */
function kick(x: number, y: number, z: number): number {
  const sig = 0.035;
  const env = Math.exp(-((y - 0.25) ** 2) / (2 * sig * sig)) + Math.exp(-((y + 0.25) ** 2) / (2 * sig * sig));
  return 0.1 * Math.sin(4 * Math.PI * x) * (1 + 0.2 * Math.cos(2 * Math.PI * z)) * env;
}

export const kelvinHelmholtz: Problem = {
  id: 'kelvin-helmholtz',
  label: 'Kelvin-Helmholtz',
  blurb: 'Dense slab sliding through lighter gas; the passive scalar tags the slab.',
  eos: 'ideal',
  eosParams: { gamma: 1.4 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: Infinity,
  checkpointDt: 0.1,
  supports2d: true,
  view: { field: 0, log: false, mode: 'slice' },
  init: () => (x, y, z) => {
    const inner = Math.abs(y) < 0.25;
    return { rho: inner ? 2 : 1, vx: inner ? 0.5 : -0.5, vy: kick(x, y, z), vz: 0, p: 2.5, s: inner ? 1 : 0 };
  },
};

/**
 * Helium over hydrogen: mean molecular weights of the two gases, fully ionized (1/2 and 4/3).
 * At one pressure and one temperature the helium is denser by their ratio, 8/3.
 */
const MU_H = meanMolecularWeight(composition('hydrogen', { h1: 1 }));
const MU_HE = meanMolecularWeight(composition('helium', { he4: 1 }));

/**
 * The same shear flow with the slab helium and the gas around it hydrogen, both fully ionized
 * and at one temperature: the density contrast is that of the two gases' mean molecular
 * weights, and the species set the equation of state (species/mixture.ts), so the temperature
 * is flat across the unmixed layers and takes intermediate values only where they mix.
 */
export const kelvinHelmholtzHeH: Problem = {
  ...kelvinHelmholtz,
  id: 'kelvin-helmholtz-he-h',
  label: 'Kelvin-Helmholtz: helium in hydrogen',
  blurb:
    'A helium slab sliding through hydrogen at the same pressure and temperature, denser by the ratio of their mean molecular weights (8/3). The species set the equation of state: the temperature is uniform where the gases are unmixed, and the mass fractions show how they mix.',
  eosParams: { gamma: 5 / 3 },
  composition: COMPOSITIONS.solar,
  species: ['he4', 'h1'],
  view: { field: MASS_FRACTION_FIELD, log: false, mode: 'slice' },
  init: () => (x, y, z) => {
    const inner = Math.abs(y) < 0.25;
    return {
      rho: inner ? MU_HE / MU_H : 1, vx: inner ? 0.5 : -0.5, vy: kick(x, y, z), vz: 0, p: 2.5, s: inner ? 1 : 0,
      X: inner ? [1, 0] : [0, 1],
    };
  },
};
