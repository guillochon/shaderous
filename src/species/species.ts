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

/**
 * The <flag_species> slot of amr/flag.wgsl: the refinement indicator of the mass fractions, read
 * from the fractions with ghosts (SX, binding 5; groups `depth` texels apart along z), or none
 * without species. Not Lohner's normalized second derivative, as for density and pressure: a
 * front that has spread over a few coarse cells is a smooth profile to it, and is never refined
 * again (the shock-bubble interface left the fine blocks behind that way). A mass fraction is
 * near piecewise constant, so its jump between neighbors says where an interface is: the
 * indicator is SPECIES_JUMP_SCALE times the largest jump of any species to a neighbor, so a jump
 * of 0.1 is the default refinement threshold, 0.8, and below 0.025 a block may merge. A species
 * at a trace level asks for nothing.
 */
/** What a jump in a mass fraction between neighbors counts for in the refinement indicator. */
export const SPECIES_JUMP_SCALE = 8;

export function flagSpeciesSlot(groups: number, depth: number): string {
  if (groups === 0) return 'fn species_indicator(c: vec3<i32>) -> f32 { return 0.0; }\n';
  return `@group(0) @binding(5) var SXF: texture_3d<f32>;
const FLAG_NG: u32 = ${groups}u;
const FLAG_SP_DEPTH: i32 = ${depth};

fn species_indicator(c: vec3<i32>) -> f32 {
  var jump = vec4<f32>(0.0);
  let axes = select(3, 2, is2d());
  for (var g = 0u; g < FLAG_NG; g++) {
    let o = vec3<i32>(0, 0, i32(g) * FLAG_SP_DEPTH);
    let x0 = textureLoad(SXF, c + o, 0);
    for (var a = 0; a < axes; a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      // Not across a wall: the gas kept inside an obstacle may hold anything.
      if (textureLoad(W1, c + e, 0).x >= 0.0) { jump = max(jump, abs(textureLoad(SXF, c + e + o, 0) - x0)); }
      if (textureLoad(W1, c - e, 0).x >= 0.0) { jump = max(jump, abs(textureLoad(SXF, c - e + o, 0) - x0)); }
    }
  }
  return ${SPECIES_JUMP_SCALE.toFixed(1)} * max(max(jump.x, jump.y), max(jump.z, jump.w));
}
`;
}

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
