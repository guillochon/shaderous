// What the gas is made of: a list of species and their mass fractions.
//
// Today a problem has one composition everywhere, and only its mean molecular weight mu
// reaches the physics (temperature in kelvin, the radiation EOS). The shape is that of a
// nuclear network's state, so that a burning network can take it over unchanged:
//
//   - `species` fixes the network's isotopes and their order. Once composition varies in
//     space, each species' mass fraction becomes an advected scalar in that order (rho X_i
//     conserved), and these functions run per cell on those scalars.
//   - Everything derived here (mu, abar, zbar, ye) is what EOS and rate routines take, so a
//     per-cell version is the same arithmetic over the cell's X_i.
//
// A is the nucleon number, used as the mass in atomic mass units; that is the convention of
// network codes, and it is good to under 1% for everything up to iron.

export interface Species {
  name: string;
  /** nucleon number (mass in atomic mass units) */
  A: number;
  /** nuclear charge */
  Z: number;
}

/**
 * How many free electrons each species gives: every one ('full'), none ('neutral'), or a
 * mean number per ion for each species, in the order of `species`.
 */
export type Ionization = 'full' | 'neutral' | number[];

export interface Composition {
  name: string;
  species: Species[];
  /** mass fractions, in the order of `species`; they sum to 1 */
  X: number[];
  ionization: Ionization;
}

export const SPECIES = {
  h1: { name: 'h1', A: 1, Z: 1 },
  he3: { name: 'he3', A: 3, Z: 2 },
  he4: { name: 'he4', A: 4, Z: 2 },
  c12: { name: 'c12', A: 12, Z: 6 },
  n14: { name: 'n14', A: 14, Z: 7 },
  o16: { name: 'o16', A: 16, Z: 8 },
  ne20: { name: 'ne20', A: 20, Z: 10 },
  mg24: { name: 'mg24', A: 24, Z: 12 },
  si28: { name: 'si28', A: 28, Z: 14 },
  s32: { name: 's32', A: 32, Z: 16 },
  fe56: { name: 'fe56', A: 56, Z: 26 },
  ni56: { name: 'ni56', A: 56, Z: 28 },
} as const satisfies Record<string, Species>;

/** A composition from mass fractions keyed by species name; the fractions are normalized. */
export function composition(name: string, fractions: Partial<Record<keyof typeof SPECIES, number>>, ionization: Ionization = 'full'): Composition {
  const entries = Object.entries(fractions).filter(([, x]) => x! > 0) as [keyof typeof SPECIES, number][];
  const total = entries.reduce((s, [, x]) => s + x, 0);
  if (!(total > 0)) throw new Error(`composition ${name}: no species`);
  return { name, species: entries.map(([k]) => SPECIES[k]), X: entries.map(([, x]) => x / total), ionization };
}

const electronsPerIon = (c: Composition, i: number): number =>
  c.ionization === 'full' ? c.species[i].Z : c.ionization === 'neutral' ? 0 : c.ionization[i];

/** Mean molecular weight of all particles, ions and electrons: 1/mu = sum X_i (1 + z_i) / A_i. */
export function meanMolecularWeight(c: Composition, X = c.X): number {
  let inv = 0;
  for (let i = 0; i < c.species.length; i++) inv += (X[i] * (1 + electronsPerIon(c, i))) / c.species[i].A;
  return 1 / inv;
}

/** Network-style averages: abar = 1 / sum(X_i / A_i), zbar = abar sum(X_i Z_i / A_i), ye = zbar / abar. */
export function nuclearAverages(c: Composition, X = c.X): { abar: number; zbar: number; ye: number } {
  let ymol = 0;
  let zy = 0;
  for (let i = 0; i < c.species.length; i++) {
    ymol += X[i] / c.species[i].A;
    zy += (X[i] * c.species[i].Z) / c.species[i].A;
  }
  const abar = 1 / ymol;
  return { abar, zbar: abar * zy, ye: zy };
}

// Metals of solar gas are mostly C, N, O, Ne, Fe; lumped as o16 they give mu within 0.2%.
// Fractions from Asplund et al. (2009), present-day photosphere.
export const COMPOSITIONS = {
  /** solar gas, fully ionized: a stellar interior (mu = 0.60) */
  solar: composition('solar', { h1: 0.7381, he4: 0.2485, o16: 0.0134 }),
  /** solar gas, neutral and atomic: the warm and cold interstellar medium (mu = 1.25) */
  solarNeutral: composition('solar, neutral', { h1: 0.7381, he4: 0.2485, o16: 0.0134 }, 'neutral'),
  /** pure ionized hydrogen (mu = 1/2) */
  hydrogen: composition('hydrogen', { h1: 1 }),
  /** equal parts carbon and oxygen by mass: a white dwarf (mu = 1.75) */
  carbonOxygen: composition('C/O', { c12: 0.5, o16: 0.5 }),
} as const;
