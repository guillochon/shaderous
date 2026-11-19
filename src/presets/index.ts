// The presets: one setup document per file in this folder (setup/spec.ts). Each is compiled to
// the Problem the solvers take; the menu, its families and variants come from the files' own
// `group`, `family` and `variant`.
import { compiledFrom, type Compiled } from '../setup/compile';
import { resolveExtends } from '../setup/extends';
import type { SetupSpec } from '../setup/spec';
import type { Problem } from '../problems/types';

const files = import.meta.glob('./*.json', { eager: true, import: 'default' }) as Record<string, SetupSpec>;

/** The files as they are in the repository, by preset id. */
export const PRESET_FILES: Record<string, SetupSpec> = Object.fromEntries(Object.values(files).map((f) => [f.id!, f]));

/**
 * The order of the preset list: groups, problems and variants appear as they first do here. A
 * file not listed goes last, in its own group.
 */
export const PRESET_ORDER: readonly string[] = [
  'kelvin-helmholtz', 'kelvin-helmholtz-he-h', 'kelvin-helmholtz-mhd', 'rayleigh-taylor', 'thermal-instability', 'thermal-instability-hhe', 'driven-turbulence', 'jeans',
  'taylor-green', 'shear-layers', 'cylinder-wake', 'lid-driven-cavity', 'lid-driven-cavity-liquid', 'gresho',
  'sedov', 'sedov-rad', 'blast-axisymmetric', 'blast-spherical', 'sod-x', 'sod-y', 'sod-z', 'sod-diag', 'strong-shock', 'forward-step', 'shock-bubble',
  'radiating-sphere', 'sedov-rad-transport', 'radiation-shadow', 'radiative-shock', 'radiative-shock-supercritical', 'radiation-star',
  'detonation-stable', 'detonation-pulsating', 'detonation-cellular', 'detonation-co',
  'polytrope', 'polytrope-dipole', 'polytrope-random', 'white-dwarf', 'sub-chandra', 'evrard', 'sphere', 'isothermal-collapse', 'second-collapse', 'zeldovich-pancake',
  'wd-collision', 'wd-collision-burning',
  'planet', 'moon-forming-impact',
  'tidal-disruption', 'tidal-disruption-hhe', 'tidal-disruption-full', 'tidal-disruption-n3', 'tidal-disruption-n3-full', 'tidal-disruption-dipole', 'tidal-disruption-random', 'tidal-disruption-full-random',
  'common-envelope', 'common-envelope-ideal',
  'binary', 'mri-disk', 'shearing-wave', 'mri-box',
  'orszag-tang', 'mhd-blast', 'mhd-rotor', 'brio-wu', 'field-loop', 'field-loop-dense', 'alfven-wave', 'alfven-collision', 'alfven-packets', 'tearing', 'island-coalescence', 'x-point', 'driven-turbulence-mhd',
  'advection', 'empty', 'empty-mhd',
];

const whole = new Map<string, SetupSpec>();
/** A preset's whole document (its `extends` resolved), or undefined if this build has no such preset. */
export function presetSpec(id: string): SetupSpec | undefined {
  let s = whole.get(id);
  if (!s && PRESET_FILES[id]) whole.set(id, (s = resolveExtends(PRESET_FILES[id], (parent) => PRESET_FILES[parent])));
  return s;
}

const rank = (id: string) => {
  const i = PRESET_ORDER.indexOf(id);
  return i < 0 ? PRESET_ORDER.length : i;
};
export const PRESET_IDS: string[] = Object.keys(PRESET_FILES).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));

const compiled = new Map<string, Compiled>();
/** A preset, compiled (once). */
export function preset(id: string): Compiled {
  let c = compiled.get(id);
  if (!c) {
    const spec = presetSpec(id);
    if (!spec) throw new Error(`unknown problem: ${id}`);
    compiled.set(id, (c = compiledFrom(spec)));
  }
  return c;
}

/** A problem in the menu: one or more variants of the same setup, each a preset of its own. */
export interface ProblemFamily {
  id: string;
  label: string;
  variants: { label: string; problem: Problem }[];
}

/** The problem menu, in loose groups. The first variant of a problem is the one it opens with. */
export const MENU: { group: string; families: ProblemFamily[] }[] = [];
for (const id of PRESET_IDS) {
  const { spec, problem } = preset(id);
  const groupName = spec.group ?? 'Other';
  let group = MENU.find((g) => g.group === groupName);
  if (!group) MENU.push((group = { group: groupName, families: [] }));
  const famId = spec.family ?? id;
  let fam = group.families.find((f) => f.id === famId);
  if (!fam) group.families.push((fam = { id: famId, label: spec.familyName ?? spec.name ?? id, variants: [] }));
  fam.variants.push({ label: spec.variant ?? 'Standard', problem });
}

export const FAMILIES: ProblemFamily[] = MENU.flatMap((g) => g.families);
export const PROBLEMS: Problem[] = FAMILIES.flatMap((f) => f.variants.map((v) => v.problem));

export const problemById = (id: string): Problem => preset(id).problem;

/** The menu entry a preset belongs to. */
export const familyOf = (id: string): ProblemFamily => {
  const f = FAMILIES.find((q) => q.variants.some((v) => v.problem.id === id));
  if (!f) throw new Error(`unknown problem: ${id}`);
  return f;
};

/** A menu entry by its id. */
export const familyById = (id: string): ProblemFamily => {
  const f = FAMILIES.find((q) => q.id === id);
  if (!f) throw new Error(`unknown problem: ${id}`);
  return f;
};
