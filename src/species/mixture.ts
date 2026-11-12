// Species that set the equation of state.
//
// A run whose species are all nuclides (SPECIES in units/composition.ts) is a mixture whose
// make-up varies from cell to cell, and its equation of state follows. Every module takes what
// it needs from four numbers per cell, each a mass-weighted sum over the species and so linear
// in the mass fractions X_i, which is how they mix:
//
//   x  particles (ions and electrons) per atomic mass unit, times mu_ref: mu_ref / mu, the gas
//      constant relative to the reference composition's (code T is T_K / (mu_ref theta), so
//      p_gas = x rho T)
//   y  electrons per nucleon relative to the reference's, Ye / Ye_ref (the degenerate
//      electrons' density scale goes as 1 / Ye)
//   z  hydrogen nuclei per atomic mass unit, times mu_ref (the h of eos/ionizing.wgsl)
//   w  every other particle per atomic mass unit, times mu_ref (its o)
//
// The reference composition is the problem's own (default solar); its kelvin scale stays the
// run's, and in it the four are (1, 1, h, o) as the uniform modules have them. How many free
// electrons each species gives is the reference's ionization.
//
// A run without species, or with species that are only tracers (any name that is not a
// nuclide), keeps one composition everywhere, and its shaders are built exactly as before
// (eos/comp_off.wgsl). Otherwise the moments live in a texture of their own, one texel per
// cell (eos/comp_on.wgsl): the species passes write them whenever they write rho X, and the
// hydro passes read them before every call to the equation of state.
import { COMPOSITIONS, SPECIES, meanMolecularWeight, nuclearAverages, type Composition, type Species } from '../units/composition';
import { SPECIES_PER_GROUP } from './species';

export type Moments = [number, number, number, number];

/** The run's species as nuclides, or null when there are none or any of them is not one. */
export function speciesNuclides(names: readonly string[] | undefined): Species[] | null {
  if (!names || names.length === 0) return null;
  const out: Species[] = [];
  for (const n of names) {
    const s = (SPECIES as Record<string, Species>)[n];
    if (!s) return null;
    out.push(s);
  }
  return out;
}

/** Free electrons per ion of a species, as the reference composition has it ionized. */
function electrons(s: Species, ref: Composition): number {
  if (ref.ionization === 'full') return s.Z;
  if (ref.ionization === 'neutral') return 0;
  const i = ref.species.findIndex((r) => r.name === s.name);
  return i >= 0 ? ref.ionization[i] : s.Z;
}

/** Each species' contribution per unit mass fraction to the four moments. */
export function momentCoefficients(nuclides: readonly Species[], ref: Composition = COMPOSITIONS.solar): Moments[] {
  const mu = meanMolecularWeight(ref);
  const ye = nuclearAverages(ref).ye;
  return nuclides.map((s) => {
    const n = (mu * (1 + electrons(s, ref))) / s.A;
    const h = s.name === 'h1';
    return [n, s.Z / s.A / ye, h ? mu / s.A : 0, h ? 0 : n];
  });
}

/**
 * The moments of a cell with mass fractions X (normalized as packSpecies stores them: missing
 * or unnormalized fractions are scaled to sum to one, and a cell with none is all the first
 * species).
 */
export function mixMoments(coeffs: readonly Moments[], X: readonly number[] | undefined): Moments {
  let sum = 0;
  if (X) for (const x of X) sum += Math.max(x, 0);
  if (!(sum > 0)) return [...coeffs[0]] as Moments;
  const m: Moments = [0, 0, 0, 0];
  coeffs.forEach((k, i) => {
    const x = Math.max(X![i] ?? 0, 0) / sum;
    for (let j = 0; j < 4; j++) m[j] += x * k[j];
  });
  return m;
}

/** The problem's coefficients, or null when its composition is the same everywhere. */
export function problemCoefficients(p: { species?: readonly string[]; composition?: Composition }): Moments[] | null {
  const nuc = speciesNuclides(p.species);
  return nuc ? momentCoefficients(nuc, p.composition ?? COMPOSITIONS.solar) : null;
}

const f = (x: number): string => {
  const s = Math.fround(x).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/**
 * The <comp> slot of every shader (included by hydro/common.wgsl): for a uniform composition
 * eos/comp_off.wgsl, sized for the species passes when there are tracers; otherwise the
 * varying one with the coefficients written in.
 */
export function compSlot(coeffs: readonly Moments[] | null, groups: number): string {
  if (!coeffs) {
    if (groups === 0) return 'eos/comp_off.wgsl';
    return `const COMP = false;
const COMP_NG: u32 = ${groups}u;
var<private> g_comp: vec4<f32>;
fn comp_use(k: vec4<f32>) {}
fn comp_tex(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
fn comp_store(c: vec3<i32>, k: vec4<f32>) {}
fn comp_moments(u: array<vec4<f32>, COMP_NG>) -> vec4<f32> { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
`;
  }
  // One matrix per group of four, its columns the species' coefficients (zero for padding).
  const mats: string[] = [];
  for (let g = 0; g < groups; g++) {
    const cols: string[] = [];
    for (let k = 0; k < SPECIES_PER_GROUP; k++) {
      const c = coeffs[g * SPECIES_PER_GROUP + k] ?? [0, 0, 0, 0];
      cols.push(c.map(f).join(', '));
    }
    mats.push(`mat4x4<f32>(${cols.join(', ')})`);
  }
  const terms = mats.map((m, g) => `${m} * u[${g}]`).join(' + ');
  const sum = mats.map((_m, g) => `dot(u[${g}], vec4<f32>(1.0))`).join(' + ');
  return `const COMP_NG: u32 = ${groups}u;
${COMP_ON_HEADER}
// Moments of a cell from its rho X (or X), groups of four, normalized by their sum as
// fractions() normalizes them; a cell with none is all the first species.
fn comp_moments(u: array<vec4<f32>, COMP_NG>) -> vec4<f32> {
  let sum = ${sum};
  if (!(sum > 0.0)) { return vec4<f32>(${coeffs[0].map(f).join(', ')}); }
  return (${terms}) / sum;
}
`;
}

const COMP_ON_HEADER = `// Composition that varies from cell to cell (species/mixture.ts): the equation of state reads
// the moments of the cell it is evaluated for from g_comp, which every caller sets first with
// comp_use(). KC holds them per cell for the passes that read them; KO is where the passes that
// move the species write them.
const COMP = true;
var<private> g_comp: vec4<f32>;
@group(0) @binding(27) var KC: texture_3d<f32>;
@group(0) @binding(26) var KO: texture_storage_3d<rgba32float, write>;
fn comp_use(k: vec4<f32>) { g_comp = k; }
fn comp_tex(c: vec3<i32>) -> vec4<f32> { return textureLoad(KC, c, 0); }
fn comp_store(c: vec3<i32>, k: vec4<f32>) { textureStore(KO, c, k); }`;

/**
 * The <comp_src> slot of amr/assemble.wgsl: where the composition varies, the moments of every
 * texel it builds, from rho X wherever it lives (`depth` texels along z between groups of four).
 */
export function compSrcSlot(coeffs: readonly Moments[] | null, depth: number): string {
  if (!coeffs) return 'amr/comp_src_off.wgsl';
  return `// The composition moments of every texel amr/assemble.wgsl builds (species/mixture.ts), from
// rho X where the state comes from: a leaf's own cell, the sum over the child cells covering a
// parent's, or the coarser cell that covers a ghost beyond a face to a coarser leaf (taken as
// it is: piecewise constant).
@group(0) @binding(25) var SC: texture_3d<f32>;   // rho X, groups of four stacked along z
const COMP_SP_DEPTH: i32 = ${depth};

fn comp_rhox_leaf(slot: i32, l: vec3<i32>, u: ptr<function, array<vec4<f32>, COMP_NG>>) {
  let c = atl(blk_origin(slot), l);
  for (var g = 0u; g < COMP_NG; g++) { (*u)[g] += textureLoad(SC, c + vec3<i32>(0, 0, i32(g) * COMP_SP_DEPTH), 0); }
}

fn comp_rhox(slot: i32, l: vec3<i32>) -> array<vec4<f32>, COMP_NG> {
  var u: array<vec4<f32>, COMP_NG>;
  if (blk_leaf(slot)) {
    comp_rhox_leaf(slot, l, &u);
    return u;
  }
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let ch = bt(slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  // Refined further: 2:1 balance keeps every leaf's stencil from reaching here.
  if (ch < 0 || !blk_leaf(ch)) { return u; }
  let f = 2 * (l - (BS / 2) * h);
  let nz = select(2, 1, is2d());
  for (var k = 0; k < nz; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) { comp_rhox_leaf(ch, f + vec3<i32>(i, j, k), &u); }
    }
  }
  return u;
}

fn comp_assembled(slot: i32, l: vec3<i32>, s: Src) -> vec4<f32> {
  if (s.kind == 0) { return comp_moments(comp_rhox(s.slot, s.l)); }
  let gf = blk_coord(slot) * BS + l;
  let nc = i32(P.amr.y) * BS * (1 << u32(bt(s.slot, 0u)));
  var gc = gf >> vec3<u32>(1u);
  gc = ((gc % nc) + nc) % nc;
  var lc = gc - blk_coord(s.slot) * BS;
  if (is2d()) { lc.z = 0; }
  return comp_moments(comp_rhox(s.slot, lc));
}
`;
}
