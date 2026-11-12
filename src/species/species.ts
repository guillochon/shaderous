// Species: mass fractions the gas carries, any number of them (Problem.species).
//
// They are stored apart from the hydrodynamic state, as rho X_i in groups of four: an
// rgba32float texture whose groups are stacked along z (group g of cell (x, y, z) is the texel
// (x, y, z + g nz)). The hydro passes never touch them. The update records the mass flux
// through every face (hydro/species_on.wgsl), and a pass of its own then moves every species
// with it (hydro/species.wgsl), so a problem without species pays nothing and one with many
// pays only for that pass.
import type { Problem } from '../problems/types';

/** Mass fractions per texel. */
export const SPECIES_PER_GROUP = 4;

/** A species name as the menus show it: 'c12' as ¹²C, 'he4' as ⁴He; any other name as it is. */
export function speciesLabel(name: string): string {
  const m = /^([a-z]{1,2})(\d+)$/i.exec(name);
  if (!m) return name;
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  return [...m[2]].map((d) => sup[+d]).join('') + m[1][0].toUpperCase() + m[1].slice(1).toLowerCase();
}

/** How many groups of four a problem's species fill (0 without species). */
export const speciesGroups = (p: Pick<Problem, 'species'>): number => Math.ceil((p.species?.length ?? 0) / SPECIES_PER_GROUP);

/**
 * The `species` slot of the update shaders: whether they record face mass fluxes, and where.
 * The flux buffer only exists (and is only bound) with species.
 */
export const speciesSlot = (groups: number): string => (groups > 0 ? 'hydro/species_on.wgsl' : 'hydro/species_off.wgsl');

/** WGSL constants of the species pass: the number of groups. */
export const speciesConst = (groups: number): string => `const NG: u32 = ${groups}u;\n`;

/**
 * Mass fractions of one cell as stored: rho X_i, padded with zeros to whole groups, written
 * into `out` (groups apart by `stride` floats) at `o`. Missing or unnormalized fractions are
 * normalized; a cell with none gets all of its mass in the first species.
 */
export function packSpecies(out: Float32Array, o: number, stride: number, groups: number, rho: number, X: readonly number[] | undefined) {
  let sum = 0;
  if (X) for (const x of X) sum += Math.max(x, 0);
  for (let g = 0; g < groups; g++) {
    for (let k = 0; k < SPECIES_PER_GROUP; k++) {
      const i = g * SPECIES_PER_GROUP + k;
      const x = sum > 0 ? Math.max(X![i] ?? 0, 0) / sum : i === 0 ? 1 : 0;
      out[o + g * stride + k] = rho * x;
    }
  }
}
