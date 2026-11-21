// The option table: every setting of a setup, listed once. The dialog's forms, the conflict
// checks, the defaults, the codec's ranges, the link keys and the scripting interface's list of
// controls are all read from it.
//
// Which options cannot be combined is data too: each option (or one choice of a select, or a
// kind of part) may name what it `requires` and what it `excludes`, as rules over the setup's
// facts (FACTS below). The dialog greys out an option whose rules do not hold and makes its
// tooltip from the rule that disables it.
import { BOUNDARIES } from '../problems/boundaries';
import { COOLING_LABELS } from '../cooling/cooling';
import { OPACITY_LABELS } from '../opacity/opacity';
import { MAX_RAD_SWEEPS, RADIATION_LABELS } from '../radiation/radiation';
import { EOS_LABELS, eosHasEnergy } from '../eos/eos';
import { FUNCTIONS } from './expr';
import { FLOWS } from './parts/flow';
import { FIELDS, GRAVITY_MODES, LIMITER_KINDS, RIEMANN_KINDS } from '../problems/types';
import { VIEW_MODES } from '../render/view_options';
import { COMPOSITIONS } from '../units/composition';
import { ANEOS_LABELS, ANEOS_MATERIALS } from '../eos/aneos';
import { speciesNuclides } from '../species/mixture';
import { AXES, MESH_ANCHORS, REFINE_MODES, TUNNEL_FLOORS, TUNNEL_SIDES } from './spec';
import { DISK_HEIGHTS, DISK_PROFILES, FRAMES, getPath, MESH_UPS, ORBIT_TYPES, SHAPE_TYPES, STAR_FIELD_TYPES, STAR_TYPES, type Part, type PartKind, type SetupSpec } from './spec';
import { DENSITY_UNITS, LENGTH_UNITS, MASS_UNITS, VELOCITY_UNITS } from './units';


export type Value = string | number | boolean;

/** A condition on a fact of the setup: it has this value (one of these), or simply holds. */
export interface Rule {
  fact: string;
  is?: Value | Value[];
  /** why, where the bare names would not explain it */
  reason?: string;
}

export type Apply = 'live' | 'shaders' | 'restart';
export type Page = 'domain' | 'gas' | 'gravity' | 'boundaries' | 'mesh' | 'solver' | 'parts' | 'run';

export interface Choice {
  value: Value;
  label: string;
  tip?: string;
  requires?: Rule[];
  excludes?: Rule[];
}

export type Control =
  | { type: 'number'; min?: number; max?: number; step?: number; factor?: number; integer?: boolean; /** empty means none (an end time) */ nullable?: boolean; /** a length, mass, ... shown in physical units beside the code value */ quantity?: 'length' | 'velocity' | 'mass' | 'density' | 'time' }
  | { type: 'select'; choices: Choice[] }
  | { type: 'check' }
  | { type: 'text'; long?: boolean }
  | { type: 'vec3' }
  /** the six faces of the box */
  | { type: 'faces' }
  /** one or several other parts, by id, of these kinds */
  | { type: 'ref'; kinds: PartKind[]; count: 1 | 2 | 'any' }
  /** a number for each species, by name: mass fractions, unless `range` says otherwise */
  | { type: 'fractions'; range?: { min: number; max?: number; step?: number }; /** what each number is ('mass fraction') */ of?: string }
  /** names, comma separated */
  | { type: 'names' }
  /** fields of their own, present or not as a whole; `forms` where the object comes in more than one shape */
  | { type: 'group'; optional: boolean; fields: Option[]; forms?: GroupForms };

/**
 * The shapes a group's object may take, told apart by which of its keys it has: units by a
 * length and a mass, or by a length, a density and a velocity. The dialog offers them as a
 * menu at the head of the group and shows only the fields of the shape it has.
 */
export interface GroupForms {
  label: string;
  tip: string;
  choices: GroupForm[];
}

export interface GroupForm {
  value: string;
  label: string;
  tip?: string;
  /** keys of the object only this shape has (the first field of each path) */
  keys: string[];
  /** values written, over the fields' defaults, when the object takes this shape */
  set?: Record<string, unknown>;
  requires?: Rule[];
  excludes?: Rule[];
}

/** The shape a group's object has: the first whose keys it holds, else the first. */
export function formOf(forms: GroupForms, value: unknown): GroupForm {
  const obj = value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  return forms.choices.find((c) => c.keys.some((k) => obj[k] !== undefined)) ?? forms.choices[0];
}

/** Whether a field of a group (its path within the group) belongs to a shape other than this one. */
export function otherForm(forms: GroupForms, form: GroupForm, fieldPath: string): boolean {
  const key = fieldPath.split('.')[0];
  return !form.keys.includes(key) && forms.choices.some((c) => c.keys.includes(key));
}

export interface Option {
  /** where it sits in the document ('physics.gravity'); for a part's field, relative to the part ('soft', 'damp.rate') */
  path: string;
  label: string;
  tip: string;
  page: Page;
  control: Control;
  /** what changing it costs a running simulation */
  apply: Apply;
  /** its value when the document does not give one (absent: none) */
  default?: Value | number[] | null;
  requires?: Rule[];
  excludes?: Rule[];
  /** its key in a link ('gamma', 'bc', ...) */
  linkKey?: string;
  /** the SimConfig field it sets ('eosParams.gamma'), for an option the run holds directly */
  cfg?: string;
  /** shown only while this other field of the same part (or option of the document) has one of these values */
  show?: { path: string; is: Value[] };
}

export interface PartOption {
  kind: PartKind;
  label: string;
  tip: string;
  requires?: Rule[];
  excludes?: Rule[];
  /** most of them a setup may hold (absent: any number) */
  max?: number;
  fields: Option[];
  /** a new part of this kind */
  make(): Part;
}

// ---- facts ----

const parts = (s: SetupSpec) => s.parts ?? [];
const count = (s: SetupSpec, f: (p: Part) => boolean) => parts(s).filter(f).length;
// A 2D run has no z faces: what its document says of them does not count.
const faces = (s: SetupSpec): number[] => {
  const k = (s.domain?.ndim ?? 3) === 2 ? 2 : 3;
  return [...(s.domain?.bc?.lo ?? [1, 1, 1]).slice(0, k), ...(s.domain?.bc?.hi ?? [1, 1, 1]).slice(0, k)];
};
const flowDef = (p: Part) => (p.kind === 'flow' ? FLOWS[p.name] : undefined);
/** The gas the run has: a wind tunnel's own (the liquid below Mach 0.3, else an ideal gas: parts/tunnel.ts), or the document's. */
const eosKind = (s: SetupSpec): string => {
  const t = parts(s).find((p) => p.kind === 'tunnel');
  if (t && t.kind === 'tunnel') return (t.mach ?? 0.1) < 0.3 ? 'liquid' : 'ideal';
  return s.physics?.eos ?? 'ideal';
};

/** What rules are about: each a named property of a setup, with how to say it. */
export const FACTS: Record<string, { label: string; of: (s: SetupSpec) => Value; say?: (v: Value) => string }> = {
  'domain.ndim': { label: 'Dimensions', of: (s) => s.domain?.ndim ?? 3, say: (v) => `${v}D` },
  'domain.supports2d': { label: 'A setup that makes sense in 2D', of: (s) => s.domain?.supports2d ?? false },
  // Space in three dimensions: a 3D grid, or a 2D one in (R, z), whose cells are rings.
  'domain.space3': { label: '3D, or an axisymmetric (R, z) or spherical (r, theta) run', of: (s) => (s.domain?.ndim ?? 3) === 3 || ['cylindrical', 'spherical'].includes(s.domain?.geometry ?? 'cartesian') },
  'gravity.selfFlat': { label: 'Self-gravity', of: (s) => ['periodic', 'isolated'].includes(s.physics?.gravity ?? 'off') && !['cylindrical', 'spherical'].includes(s.domain?.geometry ?? 'cartesian') },
  'parts.starFlat': { label: 'A star', of: (s) => count(s, (p) => p.kind === 'star') > 0 && s.domain?.geometry !== 'cylindrical' },
  'domain.geometry': { label: 'Geometry', of: (s) => s.domain?.geometry ?? 'cartesian', say: (v) => (v === 'cylindrical' ? 'axisymmetric (R, z)' : String(v)) },
  'symmetric.x': { label: 'A setup symmetric under the x mirror', of: (s) => !!s.domain?.symmetric?.includes('x') },
  'symmetric.y': { label: 'A setup symmetric under the y mirror', of: (s) => !!s.domain?.symmetric?.includes('y') },
  'symmetric.z': { label: 'A setup symmetric under the z mirror', of: (s) => !!s.domain?.symmetric?.includes('z') },
  'symmetric.field': { label: 'A magnetic field symmetric under the mirrors, or none', of: (s) => !!s.domain?.symmetricField || !FACTS.magnetic.of(s) },
  'mesh.adaptive': { label: 'An adaptive mesh', of: (s) => (s.domain?.amr ?? 0) > 0 },
  'physics.eos': { label: 'Equation of state', of: (s) => eosKind(s), say: (v) => EOS_LABELS[v as keyof typeof EOS_LABELS] ?? String(v) },
  'eos.energy': { label: 'An equation of state with an energy', of: (s) => eosHasEnergy(eosKind(s) as never) },
  'eos.tabulated': { label: 'A tabulated equation of state (white dwarf matter, or hydrogen and helium)', of: (s) => eosKind(s) === 'helmholtz' || eosKind(s) === 'hhe' },
  'domain.curvilinear': { label: 'An axisymmetric (R, z) or spherical (r, theta) run', of: (s) => ['cylindrical', 'spherical'].includes(s.domain?.geometry ?? 'cartesian') },
  'eos.degenerate': { label: 'The degenerate or tabulated (Helmholtz) equation of state', of: (s) => ['degenerate', 'helmholtz'].includes(eosKind(s)) },
  'physics.gravity': { label: 'Gravity', of: (s) => s.physics?.gravity ?? 'off', say: (v) => ({ off: 'Off', constant: 'Constant', periodic: 'Self, periodic', isolated: 'Self, isolated' })[String(v)] ?? String(v) },
  'gravity.self': { label: 'Self-gravity', of: (s) => ['periodic', 'isolated'].includes(s.physics?.gravity ?? 'off') },
  'bc.periodic': { label: 'A periodic face', of: (s) => faces(s).some((b) => b === 0) },
  'bc.allPeriodic': { label: 'Every face periodic', of: (s) => faces(s).every((b) => b === 0) },
  // (Comoving coordinates take their units from the box's size in Mpc/h, and drop the document's: compile.ts scales.)
  'units': { label: 'Physical units', of: (s) => (s.physics?.cosmology ? !!s.physics.cosmology.boxMpc : !!s.domain?.units) },
  // What a flow is given as its units (compile.ts scales): the document's, which comoving coordinates set aside.
  'units.own': { label: 'Physical units of its own (not comoving coordinates)', of: (s) => !!s.domain?.units && !s.physics?.cosmology },
  'species.nuclides': { label: 'Species that are all nuclides (they set the equation of state)', of: (s) => { const sp = s.physics?.species; return speciesNuclides(sp === 'aprox13' ? ['he4'] : sp) !== null; } },
  'species.fuelAsh': { label: 'The species fuel and ash', of: (s) => Array.isArray(s.physics?.species) && ['fuel', 'ash'].every((n) => (s.physics!.species as string[]).includes(n)) },
  'species.aprox13': { label: 'The 13 species of the burning network', of: (s) => s.physics?.species === 'aprox13' },
  'burn.network': { label: 'Network burning', of: (s) => !!s.physics?.burn && 'network' in s.physics.burn },
  'radiation': { label: 'Radiation transport', of: (s) => (s.physics?.radiation ?? 'off') !== 'off' },
  'magnetic': { label: 'A magnetic field', of: (s) => parts(s).some((p) => p.kind === 'field' || (p.kind === 'star' && !!p.field) || (p.kind === 'disk' && !!p.loops) || !!flowDef(p)?.potential || !!flowDef(p)?.face) },
  'parts.mass': { label: 'A point mass', of: (s) => count(s, (p) => p.kind === 'mass' || (p.kind === 'star' && !!p.core)) > 0 },
  'parts.star': { label: 'A star', of: (s) => count(s, (p) => p.kind === 'star') > 0 },
  'parts.star.polytrope': { label: 'A polytropic star', of: (s) => count(s, (p) => p.kind === 'star' && p.type === 'polytrope') > 0 },
  'parts.star.white-dwarf': { label: 'A white dwarf', of: (s) => count(s, (p) => p.kind === 'star' && p.type === 'white-dwarf') > 0 },
  'parts.star.planet': { label: 'A planet', of: (s) => count(s, (p) => p.kind === 'star' && p.type === 'planet') > 0 },
  'eos.aneos': { label: 'The ANEOS equation of state (rock, iron and water)', of: (s) => eosKind(s) === 'aneos' },
  'parts.disk': { label: 'A disk', of: (s) => count(s, (p) => p.kind === 'disk') > 0 },
  'parts.body': { label: 'A point mass or a star', of: (s) => count(s, (p) => p.kind === 'mass' || p.kind === 'star') > 0 },
  'parts.moving': { label: 'An obstacle that moves', of: (s) => parts(s).some((p) => p.kind === 'obstacle' && !!p.motion) },
  'species.any': { label: 'Species', of: (s) => !!s.physics?.species && (s.physics.species === 'aprox13' || (Array.isArray(s.physics.species) && s.physics.species.length > 0)) },
  'parts.damping': { label: 'A damping zone', of: (s) => count(s, (p) => (p.kind === 'disk' && !!p.damp) || (p.kind === 'tunnel' && (p.sponge ?? 0.1) > 0)) > 0 },
  // (A wind tunnel's liquid is solved with implicit pressure: parts/tunnel.ts.)
  'pressure.implicit': { label: 'Implicit pressure', of: (s) => (s.physics?.pressure ?? 'explicit') === 'implicit' || (eosKind(s) === 'liquid' && parts(s).some((p) => p.kind === 'tunnel')) },
  'physics.shear': { label: 'A shearing box', of: (s) => !!s.physics?.shear },
  'physics.cosmology': { label: 'Comoving coordinates', of: (s) => !!s.physics?.cosmology },
  'parts.flow.whole': { label: 'A flow that fills the box', of: (s) => count(s, (p) => !!flowDef(p)?.wholeBox) > 0 },
};

/** A setup's facts, all of them. */
export function factsOf(spec: SetupSpec): Record<string, Value> {
  return Object.fromEntries(Object.entries(FACTS).map(([k, f]) => [k, f.of(spec)]));
}

const holds = (r: Rule, facts: Record<string, Value>): boolean => {
  const v = facts[r.fact];
  return r.is === undefined ? v === true : Array.isArray(r.is) ? r.is.includes(v) : v === r.is;
};

const sayRule = (r: Rule, negate: boolean): string => {
  if (r.reason) return r.reason;
  const f = FACTS[r.fact];
  const want = r.is === undefined ? '' : `: ${(Array.isArray(r.is) ? r.is : [r.is]).map((v) => f?.say?.(v) ?? String(v)).join(' or ')}`;
  return `${negate ? 'Not available with' : 'Needs'} ${f?.label ?? r.fact}${want}`;
};

/** Why an option, choice or kind of part with these rules is not available in a setup with these facts (empty: it is). */
export function unavailable(o: { requires?: Rule[]; excludes?: Rule[] }, facts: Record<string, Value>): string[] {
  const out: string[] = [];
  for (const r of o.requires ?? []) if (!holds(r, facts)) out.push(sayRule(r, false));
  for (const r of o.excludes ?? []) if (holds(r, facts)) out.push(sayRule(r, true));
  return out;
}

const unavailableIn = unavailable;

// ---- the table ----

const num = (min?: number, max?: number, extra: Partial<Extract<Control, { type: 'number' }>> = {}): Control => ({ type: 'number', ...(min !== undefined ? { min } : {}), ...(max !== undefined ? { max } : {}), ...extra });
const check: Control = { type: 'check' };
const pick = (...choices: (Choice | [Value, string, string?])[]): Control => ({ type: 'select', choices: choices.map((c) => (Array.isArray(c) ? { value: c[0], label: c[1], ...(c[2] ? { tip: c[2] } : {}) } : c)) });
/**
 * A select over a list of names kept elsewhere (the solver's Riemann solvers, the kinds of
 * star): a label, and any rules, for each name of the list and for no other. A name here that
 * the list lacks does not compile; one the list has and this lacks stops the page as it loads.
 */
function pickOf<T extends string>(kinds: readonly T[], ...choices: ((Choice & { value: T }) | [T, string, string?])[]): Control {
  const named = choices.map((c) => (Array.isArray(c) ? c[0] : c.value));
  const missing = kinds.filter((k) => !named.includes(k));
  if (missing.length || named.length !== kinds.length) throw new Error(`a menu of the option table does not match its list: ${missing.join(', ') || named.join(', ')}`);
  return pick(...choices);
}

const ENERGY: Rule[] = [{ fact: 'eos.energy' }];
const SELF: Rule[] = [{ fact: 'gravity.self' }];
const ADAPTIVE: Rule[] = [{ fact: 'mesh.adaptive' }];
const IN_3D: Rule[] = [{ fact: 'domain.ndim', is: 3 }];
/** 3D, or 2D in (R, z): bodies of revolution about the axis. */
const IN_SPACE: Rule[] = [{ fact: 'domain.space3' }];
/** The mirror sets a domain may name (src/domain.ts mirrorAxes). */
const MIRRORS = ['none', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'];
const NOT_PERIODIC: Rule[] = [{ fact: 'bc.periodic', reason: 'Not available with a periodic face: its gravity is not periodic' }];

export const BC_CHOICES: Choice[] = BOUNDARIES.map((b) => ({
  value: b.code,
  label: b.label,
  tip: b.tip,
  ...(b.periodic ? { excludes: [{ fact: 'physics.gravity', is: 'isolated' }, { fact: 'parts.body', reason: 'Not available with a point mass, a star or a disk: their gravity is not periodic' }] } : {}),
}));

export const OPTIONS: Option[] = [
  // ---- run: what the setup is called, and its clock ----
  { path: 'name', label: 'Name', tip: 'What the setup is called in the preset list and in its file', page: 'run', control: { type: 'text' }, apply: 'live' },
  { path: 'blurb', label: 'Description', tip: 'A sentence or two shown under the preset menu', page: 'run', control: { type: 'text', long: true }, apply: 'live' },
  { path: 'run.tEnd', label: 'End time', tip: 'The run pauses when it reaches this time. Empty runs on without end', page: 'run', control: num(0, undefined, { nullable: true, factor: 1.25, quantity: 'time' }), apply: 'live', default: null, linkKey: 'tend', cfg: 'tEnd' },
  { path: 'run.checkpointDt', label: 'Checkpoint every', tip: 'Time between checkpoints, when the page is asked to save them: the value the Time panel starts with', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'time' }), apply: 'live' },
  { path: 'run.tDyn', label: 'Dynamical time', tip: 'The time over which the flow visibly changes (a star’s sqrt(R³ / G M), a binary’s period): a movie takes a frame every 600th of it by default', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'time' }), apply: 'live' },
  { path: 'run.startPaused', label: 'Start paused', tip: 'The run waits, paused, for Run', page: 'run', control: check, apply: 'live', default: false },
  { path: 'physics.toolScale', label: 'Scale of the editing tools', tip: 'Least density and speed the editing tools (paint, push) measure their amounts against. They take the densest and fastest gas of the run, which leaves them nothing to go by in a box that holds next to nothing', page: 'run', apply: 'restart',
    control: { type: 'group', optional: true, fields: [
      { path: 'rho', label: 'Density', tip: 'The tools’ density scale is at least this', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'density' }), apply: 'restart', default: 1 },
      { path: 'v', label: 'Speed', tip: 'The tools’ speed scale is at least this', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'velocity' }), apply: 'restart', default: 1 },
    ] } },
  { path: 'view.mode', label: 'Opening view', tip: 'How the gas is drawn when the setup opens in 3D', page: 'run', control: pickOf(VIEW_MODES, ['volume', 'Volume'], ['slice', 'Slice'], ['projection', 'Projection']), apply: 'live', default: 'volume' },
  { path: 'view.field', label: 'Opening field', tip: 'The quantity shown when the setup opens (the View panel’s Field, by its number)', page: 'run', control: num(0, FIELDS.length - 1, { integer: true, step: 1 }), apply: 'live', default: 0 },
  { path: 'view.log', label: 'Logarithmic colors', tip: 'The color scale the setup opens with', page: 'run', control: check, apply: 'live', default: true },
  { path: 'view.axis', label: 'Slice normal', tip: 'The axis the opening slice is perpendicular to (0 x, 1 y, 2 z)', page: 'run', control: pick([2, 'z'], [1, 'y'], [0, 'x']), apply: 'live', default: 2 },
  { path: 'view.pitch', label: 'Camera elevation', tip: 'Elevation of the volume view’s camera when the setup opens, in radians', page: 'run', control: num(-1.5, 1.5, { step: 0.05 }), apply: 'live', default: 0.45 },

  // ---- domain and grid ----
  { path: 'domain.ndim', label: 'Dimensions', tip: 'Two or three dimensions. A 2D run is one layer of cells in x and y', page: 'domain', control: pick({ value: 3, label: '3D', excludes: [{ fact: 'domain.geometry', is: ['cylindrical', 'spherical'] }] }, { value: 2, label: '2D', requires: [{ fact: 'domain.supports2d', reason: 'This setup is three-dimensional (its file says it does not make sense in 2D)' }], excludes: [{ fact: 'gravity.selfFlat', reason: 'Not available with self-gravity (but an axisymmetric (R, z) or spherical run has it)' }, { fact: 'parts.starFlat', reason: 'Not available with a star (but an axisymmetric (R, z) run takes one on its axis)' }] }), apply: 'restart', linkKey: 'dim' },
  { path: 'domain.n', label: 'Resolution', tip: 'Cells across the box; on an adaptive mesh, at its finest level', page: 'domain', control: num(16, 16384, { integer: true, factor: 2 }), apply: 'restart', linkKey: 'n' },
  { path: 'domain.amr', label: 'Refinement', tip: 'Adaptive mesh: levels of refinement below the coarsest grid. 0 puts every cell at the full resolution', page: 'domain', control: num(0, 5, { integer: true, step: 1 }), apply: 'restart', linkKey: 'amr' },
  { path: 'domain.box', label: 'Box size', tip: 'Side of the box, in code units: the domain is a cube of this side centered on the origin', page: 'domain', control: num(1e-12, 1e30, { factor: 2, quantity: 'length' }), apply: 'restart', default: 1 },
  { path: 'domain.geometry', label: 'Geometry', tip: 'Cartesian, or a 2D body of revolution about the y axis, in (R, z): x is then R from the axis (the lower x face) to the box side, and cells are rings; or in (r, theta): x is then r from the inner radius to the box side, y theta from pole to pole', page: 'domain', apply: 'restart', default: 'cartesian',
    control: pick({ value: 'cartesian', label: 'Cartesian' }, { value: 'cylindrical', label: 'Axisymmetric (R, z)', requires: [{ fact: 'domain.ndim', is: 2, reason: 'An axisymmetric run is two-dimensional' }] }, { value: 'spherical', label: 'Spherical (r, theta)', requires: [{ fact: 'domain.ndim', is: 2, reason: 'A spherical run is two-dimensional' }] }) },
  { path: 'domain.rMin', label: 'Inner radius', tip: 'A spherical (r, theta) run: the radius of its inner face (above zero); the box side is then its outer radius', page: 'domain', control: num(1e-12, 1e30, { factor: 2, quantity: 'length' }), apply: 'restart', default: 0.01 },
  { path: 'domain.logR', label: 'Logarithmic in r', tip: 'A spherical (r, theta) run: cells spaced evenly in ln r, so that they stay about as long as they are wide, rather than evenly in r', page: 'domain', control: check, apply: 'restart', default: false },
  { path: 'domain.symmetric', label: 'Symmetric under', tip: 'The mirrors (planes through the origin, named by the axis normal to each) the setup is symmetric under: a run may then compute only the part above them', page: 'domain', apply: 'live', default: 'none',
    control: pick(...MIRRORS.map((m): [Value, string] => [m, m === 'none' ? 'No mirror' : m.split('').join(', ')])) },
  { path: 'domain.symmetricField', label: 'Field symmetric too', tip: 'The magnetic field is symmetric under those mirrors as well (a vector potential odd in its normal component and even in the others)', page: 'domain', control: check, apply: 'live', default: false },
  { path: 'domain.mirror', label: 'Mirror symmetry', tip: 'Compute only the part of a symmetric setup above its planes of symmetry, each mirrored axis’s lower face a plane of symmetry; the totals are those of the whole. With every axis mirrored (an octant in 3D, a quadrant in 2D) each runs from 0 to the box side: twice the resolution for the same cells. With only some (a half, or a quadrant in 3D) those run from 0 to half the box with half the cells: the same cube as without mirrors, at half or a quarter of the cost', page: 'domain', apply: 'restart', default: 'none', linkKey: 'sym', cfg: 'symmetry',
    control: pick(...MIRRORS.map((m): Choice => ({
      value: m,
      label: m === 'none' ? 'Whole box' : m === 'xyz' ? 'Octant (x, y, z)' : m === 'xy' ? 'Quadrant (x, y)' : `Mirrored in ${m.split('').join(', ')}`,
      requires: m === 'none' ? [] : [
        ...m.split('').map((a): Rule => ({ fact: `symmetric.${a}`, reason: `The setup is not symmetric under the ${a} mirror` })),
        ...(m.includes('z') ? IN_3D : []),
        { fact: 'symmetric.field', reason: 'The magnetic field is not symmetric under the mirrors' },
      ],
    }))) },
  { path: 'domain.supports2d', label: 'Makes sense in 2D', tip: 'The setup can also run on a 2D (x, y) grid. Without it the setup is three-dimensional only', page: 'domain', control: check, apply: 'live', default: false },
  { path: 'domain.units', label: 'Physical units', tip: 'What one code unit is, for a setup that stands for something real: a length and a mass (the time unit then makes G = 1), or a length, a density and a velocity. Without them the setup is scale free and shown in code units', page: 'domain', apply: 'restart',
    control: { type: 'group', optional: true,
      forms: { label: 'Defined by', tip: 'What fixes the unit system besides its length', choices: [
        { value: 'mass', label: 'Length and mass', tip: 'A length and a mass: the time unit is the one that makes G = 1, for gravity', keys: ['mass'] },
        { value: 'flow', label: 'Length, density and velocity', tip: 'A length, a density and a velocity: the time unit is the length over the velocity, for gas with no gravity to set the clock', keys: ['density', 'velocity'] },
      ] },
      fields: [
      { path: 'name', label: 'Name', tip: 'A name for the unit system', page: 'domain', control: { type: 'text' }, apply: 'live', default: 'custom' },
      { path: 'length.value', label: 'Length', tip: 'This many of the unit beside it are one code unit of length', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'length.unit', label: 'Length unit', tip: 'The unit of the length', page: 'domain', control: pick(...LENGTH_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'Rsun' },
      { path: 'mass.value', label: 'Mass', tip: 'This many of the unit beside it are one code unit of mass', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'mass.unit', label: 'Mass unit', tip: 'The unit of the mass', page: 'domain', control: pick(...MASS_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'Msun' },
      { path: 'density.value', label: 'Density', tip: 'This many of the unit beside it are one code unit of density', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'density.unit', label: 'Density unit', tip: 'The unit of the density', page: 'domain', control: pick(...DENSITY_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'g/cm3' },
      { path: 'velocity.value', label: 'Velocity', tip: 'This many of the unit beside it are one code unit of velocity', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'velocity.unit', label: 'Velocity unit', tip: 'The unit of the velocity', page: 'domain', control: pick(...VELOCITY_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'km/s' },
    ] } },

  // ---- boundaries ----
  { path: 'domain.bc', label: 'Faces of the box', tip: 'What the gas meets at each face of the box. Periodic faces come in pairs', page: 'boundaries', control: { type: 'faces' }, apply: 'live', linkKey: 'bc', cfg: 'bc' },
  { path: 'domain.wallVelocity.xlo', label: 'Wall velocity, lower x face', tip: 'Where the lower x face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.xlo' },
  { path: 'domain.wallVelocity.xhi', label: 'Wall velocity, upper x face', tip: 'Where the upper x face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.xhi' },
  { path: 'domain.wallVelocity.ylo', label: 'Wall velocity, lower y face', tip: 'Where the lower y face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.ylo' },
  { path: 'domain.wallVelocity.yhi', label: 'Wall velocity, upper y face', tip: 'Where the upper y face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.yhi' },
  { path: 'domain.wallVelocity.zlo', label: 'Wall velocity, lower z face', tip: 'Where the lower z face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.zlo' },
  { path: 'domain.wallVelocity.zhi', label: 'Wall velocity, upper z face', tip: 'Where the upper z face is a moving wall: its velocity along itself (the part normal to it is ignored)', page: 'boundaries', control: { type: 'vec3' }, apply: 'live', default: [0, 0, 0], cfg: 'wallVelocity.zhi' },

  // ---- gas ----
  { path: 'physics.eos', label: 'Equation of state', tip: 'How pressure follows from density and internal energy. Changing it carries each cell’s pressure over, and the run continues', page: 'gas', apply: 'shaders', default: 'ideal', linkKey: 'eos', cfg: 'eos',
    control: pick(...(Object.keys(EOS_LABELS) as (keyof typeof EOS_LABELS)[]).map((k): Choice => ({ value: k, label: EOS_LABELS[k], ...(!eosHasEnergy(k) ? { excludes: [{ fact: 'parts.star.polytrope', reason: 'Not available with a polytropic star: it needs an equation of state with an energy' }] } : {}), ...(k === 'aneos' || k === 'helmholtz' ? { requires: [{ fact: 'units', reason: 'Needs physical units: its tables are in grams, centimeters and kelvin' }] } : {}) }))) },
  { path: 'physics.eosParams.aneosA', label: 'Material', tip: 'The material of the ANEOS equation of state: rock, iron alloy or water, each from its published tables. A run whose two species are materials (a planet with a core) takes them instead', page: 'gas', control: pick(...ANEOS_MATERIALS.map((m, i): [Value, string] => [i, ANEOS_LABELS[m]])), apply: 'restart', default: 0, linkKey: 'material', cfg: 'eosParams.aneosA', show: { path: 'physics.eos', is: ['aneos'] } },
  { path: 'physics.eosParams.gamma', label: 'gamma', tip: 'Adiabatic index γ: p = (γ - 1) ρ e for an ideal gas, p = K ρ^γ for a barotropic one', page: 'gas', control: num(1, 100, { step: 0.05 }), apply: 'live', default: 5 / 3, linkKey: 'gamma', cfg: 'eosParams.gamma' },
  { path: 'physics.eosParams.K', label: 'K (p = K rho^gamma)', tip: 'Barotropic constant K: p = K ρ^γ. With γ = 1 it is the square of the sound speed', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', linkKey: 'K', cfg: 'eosParams.K', show: { path: 'physics.eos', is: ['barotropic'] } },
  { path: 'physics.eosParams.liqC', label: 'Sound speed', tip: 'The liquid’s sound speed c₀, a constant: p = c₀² (ρ - ρ₀). It is chosen, not physical: ten times the fastest flow keeps the density within 1% of ρ₀ (a Mach number of 0.1), and the step shrinks in proportion as it grows', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', default: 10, linkKey: 'lc', cfg: 'eosParams.liqC', show: { path: 'physics.eos', is: ['liquid'] } },
  { path: 'physics.eosParams.liqRho', label: 'Reference density', tip: 'The density ρ₀ at which the liquid’s pressure is zero: p = c₀² (ρ - ρ₀). The liquid should start near it', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', default: 1, linkKey: 'lrho', cfg: 'eosParams.liqRho', show: { path: 'physics.eos', is: ['liquid'] } },
  { path: 'physics.eosParams.arad', label: 'Radiation constant a', tip: 'Radiation pressure a T⁴ / 3 is added to the gas pressure', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', linkKey: 'arad', cfg: 'eosParams.arad', show: { path: 'physics.eos', is: ['gas_radiation', 'degenerate', 'ionizing'] } },
  { path: 'physics.eosParams.degRho', label: 'Degeneracy density', tip: 'Density at which the electrons’ Fermi momentum reaches mₑc. For a setup with physical units it is set from the composition (about 2×10⁶ g/cm³ for C/O); in code units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'degrho', cfg: 'eosParams.degRho', show: { path: 'physics.eos', is: ['degenerate'] } },
  { path: 'physics.eosParams.degP', label: 'Degeneracy pressure scale', tip: 'The scale π mₑ⁴ c⁵ / 3h³ of the degenerate electron pressure. Set from physical constants for a setup with physical units; in code units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'degp', cfg: 'eosParams.degP', show: { path: 'physics.eos', is: ['degenerate'] } },
  { path: 'physics.eosParams.helmCoulomb', label: 'Coulomb corrections', tip: 'The ions’ Coulomb energy, as a multiplier: 1 for all of it, 0 for none. It lowers the cold pressure by a percent or two in dense white dwarf matter (more in the envelope) and adds to the heat capacity of cold, strongly coupled ions', page: 'gas', control: num(0, 1, { step: 0.25 }), apply: 'live', default: 1, linkKey: 'coulomb', cfg: 'eosParams.helmCoulomb', show: { path: 'physics.eos', is: ['helmholtz'] } },
  { path: 'physics.eosParams.hheRad', label: 'Radiation term', tip: 'The radiation’s pressure a T⁴ / 3 and energy a T⁴ in the equation of state, as a multiplier: 1 for gas in equilibrium with its radiation (a star’s interior), 0 for gas alone (a cloud or thin gas, which its radiation leaves). Radiation transport does not run with this equation of state', page: 'gas', control: num(0, 1, { step: 1 }), apply: 'live', default: 1, linkKey: 'hherad', cfg: 'eosParams.hheRad', show: { path: 'physics.eos', is: ['hhe'] } },
  { path: 'physics.eosParams.ionT', label: 'Ionization temperature', tip: 'The ionization energy of hydrogen, 13.6 eV, over k, in code temperature. Set from physical constants for a setup with physical units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'iont', cfg: 'eosParams.ionT', show: { path: 'physics.eos', is: ['ionizing'] } },
  { path: 'physics.eosParams.ionRho', label: 'Saha density', tip: 'Density scale of the Saha equation in code density: denser gas needs a higher temperature to ionize. Set from physical constants for a setup with physical units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'ionrho', cfg: 'eosParams.ionRho', show: { path: 'physics.eos', is: ['ionizing'] } },
  { path: 'physics.composition', label: 'Composition', tip: 'What the gas is made of, which sets the kelvin scale and the physical constants of the equation of state: solar (fully ionized), solarNeutral, hydrogen or carbonOxygen. The hydrogen and helium equation of state has tables for the solar ones and for hydrogen', page: 'gas', control: pickOf(Object.keys(COMPOSITIONS) as (keyof typeof COMPOSITIONS)[], ['solar', 'Solar, ionized'], ['solarNeutral', 'Solar, neutral'], ['hydrogen', 'Hydrogen'], ['carbonOxygen', 'Carbon and oxygen']), apply: 'restart', default: 'solar' },
  { path: 'physics.species', label: 'Species', tip: 'Species whose mass fractions the gas carries, by name (he4, c12, ...), separated by commas; or aprox13 for the burning network’s thirteen. None by default', page: 'gas', control: { type: 'names' }, apply: 'restart' },
  { path: 'physics.speciesGamma', label: 'Gamma by species', tip: 'Adiabatic index of each species the gas carries, by name, where the species set the equation of state (a mixture of ideal gases): a species not listed takes the gas’s gamma', page: 'gas', control: { type: 'fractions', range: { min: 1, max: 100, step: 0.05 }, of: 'gamma' }, apply: 'restart' },
  { path: 'physics.burn', label: 'Burning', tip: 'Burning of the gas, wherever it is hot and dense enough: nuclear, by the trained network of the 13-isotope alpha chain, or a single fuel to ash reaction at an Arrhenius rate', page: 'gas', apply: 'restart',
    requires: [{ fact: 'eos.energy' }],
    control: { type: 'group', optional: true,
      forms: { label: 'Reaction', tip: 'What burns, and at what rate', choices: [
        { value: 'network', label: 'Network (aprox13)', tip: 'Nuclear burning of the 13-isotope alpha chain, by its trained network', keys: ['network', 'limiter'], requires: [{ fact: 'species.aprox13' }, { fact: 'units' }], excludes: [{ fact: 'physics.cosmology', reason: 'Not with comoving coordinates: the gas is in code units there, and the network would read it in the box’s physical ones (random test seed 10601)' }] },
        { value: 'arrhenius', label: 'Fuel to ash (Arrhenius)', tip: 'One reaction, fuel to ash: the fuel fraction f falls as df/dt = -k f exp(-Ea / T), and each unit of mass burned releases q', keys: ['fuel', 'ash', 'q', 'ea', 'k'], set: { inShocks: true }, excludes: [{ fact: 'species.nuclides', reason: 'Not available where every species is a nuclide (he4, c12, ...): those set the equation of state, which one reaction from fuel to ash cannot. Name the fuel and ash species otherwise' }] },
      ] },
      fields: [
      { path: 'network', label: 'Network', tip: 'The burning network', page: 'gas', control: pick(['aprox13', 'aprox13 (13 isotopes)']), apply: 'restart', default: 'aprox13' },
      { path: 'fuel', label: 'Fuel', tip: 'The species that burns: one the gas carries (Species)', page: 'gas', control: { type: 'text' }, apply: 'restart', default: 'fuel' },
      { path: 'ash', label: 'Ash', tip: 'The species it burns to: one the gas carries (Species)', page: 'gas', control: { type: 'text' }, apply: 'restart', default: 'ash' },
      { path: 'q', label: 'Heat release q', tip: 'Energy released per unit mass of fuel burned, in code units', page: 'gas', control: num(0, undefined, { factor: 1.25 }), apply: 'restart', default: 50 },
      { path: 'ea', label: 'Activation temperature Ea', tip: 'The rate goes as exp(-Ea / T), T the code temperature p / ρ', page: 'gas', control: num(0, undefined, { factor: 1.25 }), apply: 'restart', default: 25 },
      { path: 'k', label: 'Rate constant k', tip: 'The rate per unit time of fuel far above its activation temperature: df/dt = -k f exp(-Ea / T)', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'inShocks', label: 'Burn inside shocks', tip: 'Let cells the shock detector flags burn. Off, as in most supernova codes: a captured shock is a few cells of numerical smearing. Fuel to ash burning burns in them unless told not to', page: 'gas', control: check, apply: 'restart', default: false },
      { path: 'limiter', label: 'Burning limiter f', tip: 'No cell takes in burning energy faster than f times its internal energy per sound crossing of the cell (Kushnir et al. 2013). 0 switches it off', page: 'gas', control: num(0, 1, { step: 0.05 }), apply: 'restart', default: 0 },
    ] } },
  { path: 'physics.floors.rho', label: 'Density floor', tip: 'No cell is thinner than this', page: 'gas', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'restart', default: 1e-10 },
  { path: 'physics.floors.p', label: 'Pressure floor', tip: 'No cell has less pressure than this', page: 'gas', control: num(0, undefined, { factor: 10 }), apply: 'restart', default: 1e-14 },
  { path: 'physics.fluff.rho', label: 'Tenuous below density', tip: 'Gas thinner than this has its heat and speed capped (the next two fields), so a few hot or fast cells of ambient gas cannot set the timestep for the whole grid. Not conservative, but such gas carries next to no mass. 0 is off', page: 'gas', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'live', default: 0, linkKey: 'frho', cfg: 'fluff.rho' },
  { path: 'physics.fluff.u', label: 'Max specific energy', tip: 'Largest internal energy per unit mass tenuous gas may have. 0 is no cap', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'fu', cfg: 'fluff.u' },
  { path: 'physics.fluff.v', label: 'Max speed', tip: 'Largest speed tenuous gas may have. 0 is no cap', page: 'gas', control: num(0, undefined, { factor: 2, quantity: 'velocity' }), apply: 'live', default: 0, linkKey: 'fv', cfg: 'fluff.v' },
  { path: 'physics.fluff.va', label: 'Max Alfvén speed', tip: 'With a magnetic field: largest Alfvén speed |B| / √ρ anywhere. Where the field is stronger than that for the gas it threads, mass is added at rest until it is not. Not conservative. 0 is no cap', page: 'gas', control: num(0, undefined, { factor: 2, quantity: 'velocity' }), apply: 'live', default: 0, linkKey: 'fva', cfg: 'fluff.va', requires: [{ fact: 'magnetic' }] },
  { path: 'physics.cooling', label: 'Cooling function', tip: 'Optically thin radiative cooling (and heating) of the gas', page: 'gas', apply: 'shaders', default: 'off', linkKey: 'cool', cfg: 'cooling',
    control: pick(...(Object.keys(COOLING_LABELS) as (keyof typeof COOLING_LABELS)[]).map((k): Choice => ({ value: k, label: COOLING_LABELS[k], ...(k === 'off' ? {} : { requires: ENERGY }) }))) },
  { path: 'physics.coolParams.rate', label: 'Coefficient Λ₀', tip: 'Power law: the cooling rate per unit volume is ρ² Λ₀ T^α', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 1, linkKey: 'crate', cfg: 'coolParams.rate', show: { path: 'physics.cooling', is: ['power'] } },
  { path: 'physics.coolParams.alpha', label: 'Exponent of T', tip: 'Power law: the exponent α of T in the cooling rate (1/2 is bremsstrahlung)', page: 'gas', control: num(-4, 4, { step: 0.25 }), apply: 'live', default: 0.5, linkKey: 'calpha', cfg: 'coolParams.alpha', show: { path: 'physics.cooling', is: ['power'] } },
  { path: 'physics.coolParams.heat', label: 'Heating rate', tip: 'Heating per unit mass (power law), or the heating rate at unit density (interstellar)', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'cheat', cfg: 'coolParams.heat', show: { path: 'physics.cooling', is: ['power', 'ism'] } },
  { path: 'physics.coolParams.tUnit', label: 'Kelvin per unit T', tip: 'Interstellar: kelvin per code unit of temperature (code density is atoms per cm³)', page: 'gas', control: num(1e-6, undefined, { factor: 2 }), apply: 'live', default: 1000, linkKey: 'ctunit', cfg: 'coolParams.tUnit', show: { path: 'physics.cooling', is: ['ism'] } },
  { path: 'physics.coolParams.tFloor', label: 'Temperature floor', tip: 'Gas at or below this temperature does not cool', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'ctfloor', cfg: 'coolParams.tFloor', show: { path: 'physics.cooling', is: ['power', 'ism'] } },
  { path: 'physics.radiation', label: 'Radiation transport', tip: 'Gray radiation transport with the M1 closure: the radiation’s energy and flux, carried with the gas, streaming and diffusing between cells, and trading heat and momentum with the gas through its opacity (docs/numerics.md section 24). Cartesian coordinates, on the uniform grid or the adaptive mesh (all blocks then take one timestep), without a magnetic field', page: 'gas', apply: 'restart', default: 'off', linkKey: 'rad', cfg: 'radiation',
    control: pick(...(Object.keys(RADIATION_LABELS) as (keyof typeof RADIATION_LABELS)[]).map((k): Choice => ({ value: k, label: RADIATION_LABELS[k], ...(k === 'off' ? {} : { requires: ENERGY, excludes: [{ fact: 'magnetic', reason: 'Not available with a magnetic field yet' }, { fact: 'domain.curvilinear', reason: 'Cartesian grids only for now' }, { fact: 'eos.tabulated', reason: 'Not with the tabulated equation of state, whose radiation is in its table' }, { fact: 'eos.aneos', reason: 'Not with the ANEOS equation of state yet' }] }) }))) },
  { path: 'physics.opacity', label: 'Opacity', tip: 'How opaque the gas is to the radiation, per unit mass: a constant, electron scattering, or Kramers’ free-free absorption with electron scattering. With physical units the scattering and Kramers constants come from the composition', page: 'gas', apply: 'shaders', default: 'constant', linkKey: 'kap', cfg: 'opacity', show: { path: 'physics.radiation', is: ['m1'] },
    control: pick(...(Object.keys(OPACITY_LABELS) as (keyof typeof OPACITY_LABELS)[]).map((k): Choice => ({ value: k, label: OPACITY_LABELS[k] }))) },
  { path: 'physics.opacityParams.absorption', label: 'Absorption κ_a', tip: 'Absorption opacity per unit mass (the Planck mean), which exchanges heat between gas and radiation. With electron scattering, a small true absorption', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 1, linkKey: 'kapa', cfg: 'opacityParams.absorption', show: { path: 'physics.opacity', is: ['constant', 'electron'] } },
  { path: 'physics.opacityParams.scattering', label: 'Scattering κ_s', tip: 'Scattering opacity per unit mass: it drags on the radiation’s flux without heating the gas (electron scattering is 0.2 (1 + X) cm²/g)', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'kaps', cfg: 'opacityParams.scattering', show: { path: 'physics.radiation', is: ['m1'] } },
  { path: 'physics.opacityParams.kramers', label: 'Kramers K₀', tip: 'Kramers: the free-free absorption is K₀ ρ T^-3.5 per unit mass, in code units', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'kapk', cfg: 'opacityParams.kramers', show: { path: 'physics.opacity', is: ['kramers'] } },
  { path: 'physics.opacityParams.tFloor', label: 'Kramers least T', tip: 'Kramers: the opacity is taken at no lower temperature than this (it grows without bound as T falls)', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', default: 0, linkKey: 'kapt', cfg: 'opacityParams.tFloor', show: { path: 'physics.opacity', is: ['kramers'] } },
  { path: 'physics.radC', label: 'Speed of light c', tip: 'The speed of light in code units. With physical units it is theirs; a scale-free setup sets its own', page: 'gas', control: num(0, undefined, { factor: 2, quantity: 'velocity' }), apply: 'live', linkKey: 'clight', cfg: 'radC', show: { path: 'physics.radiation', is: ['m1'] } },
  { path: 'physics.radChat', label: 'Reduced speed of light', tip: 'The speed light crosses transparent gas at in the run (at most c): its step is about dx / chat. Ten times the fastest gas is enough; opaque cells take the true speed whatever this is', page: 'gas', control: num(0, undefined, { factor: 2, quantity: 'velocity' }), apply: 'live', linkKey: 'chat', cfg: 'radChat', show: { path: 'physics.radiation', is: ['m1'] } },
  { path: 'physics.radLocalC', label: 'Speed of light per cell', tip: 'Opaque cells take the true speed of light and only transparent ones the reduced one, so the radiation in a star has its true heat capacity. Off, one reduced speed everywhere', page: 'gas', control: check, apply: 'live', default: true, linkKey: 'clocal', cfg: 'radLocalC', show: { path: 'physics.radiation', is: ['m1'] } },
  { path: 'physics.radSolver', label: 'Radiation flux', tip: 'The flux between cells: HLL with the M1 closure’s wave speeds, or global Lax-Friedrichs (the speed of light: more diffusive, more isotropic around sources)', page: 'gas', apply: 'live', default: 'hll', linkKey: 'radflux', cfg: 'radSolver', show: { path: 'physics.radiation', is: ['m1'] },
    control: pick(['hll', 'HLL'], ['glf', 'Global Lax-Friedrichs']) },
  { path: 'physics.radSweeps', label: 'Radiation sweeps per step', tip: `Radiation sweeps per hydro step (1 to ${MAX_RAD_SWEEPS}). The hydro step is at most this many of the sweep’s own steps, so more sweeps let it be longer`, page: 'gas', control: num(1, MAX_RAD_SWEEPS, { integer: true, step: 1 }), apply: 'live', default: 8, linkKey: 'rsweeps', cfg: 'radSweeps', show: { path: 'physics.radiation', is: ['m1'] } },
  { path: 'physics.viscosity', label: 'Viscosity ν', tip: 'Kinematic viscosity: the stress is ρ ν times the shear. Obstacles hold the gas at their walls. 0 is off', page: 'gas', control: num(0, 1e30, { factor: 1.25 }), apply: 'live', default: 0, linkKey: 'nu', cfg: 'viscosity' },
  { path: 'physics.conductivity', label: 'Conductivity κ', tip: 'Thermal conduction: the heat flux is -κ ρ grad T, so the temperature of an ideal gas diffuses at (γ - 1) κ. 0 is off', page: 'gas', control: num(0, 1e30, { factor: 1.25 }), apply: 'live', default: 0, linkKey: 'kappa', cfg: 'conductivity', requires: ENERGY },
  { path: 'physics.resistivity', label: 'Resistivity η', tip: 'With a magnetic field: magnetic diffusivity, so the field diffuses as η ∇²B (the electric field gains η curl B) and the current it dissipates, η J², heats the gas. 0 is off (ideal MHD, with only the scheme’s own diffusion)', page: 'gas', control: num(0, 1e30, { factor: 1.25 }), apply: 'live', default: 0, linkKey: 'eta', cfg: 'resistivity', requires: [{ fact: 'magnetic' }] },
  { path: 'physics.driving.accel', label: 'Driving acceleration', tip: 'Turbulence driving: the root-mean-square acceleration of a random force that stirs the gas at large scales (a band of wavelengths, correlated over a time τ). The gas reaches speeds of about (accel × box / band)^½ where it balances the dissipation. 0 is off', page: 'gas', control: num(0, 1e30, { factor: 1.5 }), apply: 'restart', default: 0, linkKey: 'drive', cfg: 'driving.accel' },
  { path: 'physics.driving.tau', label: 'Driving correlation time τ', tip: 'How long the random force keeps its pattern (its Ornstein-Uhlenbeck correlation time)', page: 'gas', control: num(1e-6, 1e30, { factor: 1.5 }), apply: 'restart', default: 0.5, linkKey: 'dtau', cfg: 'driving.tau' },
  { path: 'physics.driving.kMin', label: 'Driving band, longest', tip: 'The longest wavelengths driven, as wavelengths per box side', page: 'gas', control: num(1, 16, { step: 0.5 }), apply: 'restart', default: 1, linkKey: 'dkmin', cfg: 'driving.kMin' },
  { path: 'physics.driving.kMax', label: 'Driving band, shortest', tip: 'The shortest wavelengths driven, as wavelengths per box side (at most 64 wavevectors are kept, the longest first)', page: 'gas', control: num(1, 16, { step: 0.5 }), apply: 'restart', default: 2, linkKey: 'dkmax', cfg: 'driving.kMax' },
  { path: 'physics.driving.zeta', label: 'Solenoidal share ζ', tip: 'What the force stirs: 1 swirls only (solenoidal, no compression), 0 compresses only, 0.5 the natural mix', page: 'gas', control: num(0, 1, { step: 0.05 }), apply: 'restart', default: 1, linkKey: 'dzeta', cfg: 'driving.zeta' },
  { path: 'physics.driving.seed', label: 'Driving seed', tip: 'Which random force: each seed gives another, and the same seed the same run', page: 'gas', control: num(0, 1e9, { integer: true, step: 1 }), apply: 'restart', default: 1, linkKey: 'dseed', cfg: 'driving.seed' },

  // ---- gravity ----
  { path: 'physics.gravity', label: 'Gravity', tip: 'Off, a uniform field, or the self-gravity of the gas. Point masses pull either way', page: 'gravity', apply: 'live', default: 'off', linkKey: 'gravity', cfg: 'gravity',
    control: pickOf(GRAVITY_MODES,
      { value: 'off', label: 'Off', tip: 'No gravity (point masses still pull)' },
      { value: 'constant', label: 'Constant', tip: 'A uniform field of strength G pointing down: -y in 2D, -z in 3D', excludes: [{ fact: 'domain.geometry', is: 'spherical', reason: 'Not available in a spherical (r, theta) run: a uniform field has no place on its shells' }] },
      { value: 'periodic', label: 'Self, periodic', tip: 'Self-gravity of the gas in a periodic box (3D only)', requires: [...IN_3D, { fact: 'bc.allPeriodic' }], excludes: [{ fact: 'parts.star', reason: 'Not available with a star: a star’s gravity is not periodic' }] },
      { value: 'isolated', label: 'Self, isolated', tip: 'Self-gravity of the gas, isolated: the potential falls off outside the box (3D, or an axisymmetric (R, z) or spherical run)', requires: IN_SPACE, excludes: [{ fact: 'bc.periodic' }] },
    ) },
  { path: 'physics.G', label: 'G', tip: 'Gravitational constant in code units, or the strength of the uniform field. A setup with physical units may leave it to them', page: 'gravity', control: num(0, undefined, { factor: 1.5 }), apply: 'live', default: 1, linkKey: 'G', cfg: 'G' },
  { path: 'physics.gext', label: 'Uniform acceleration', tip: 'A uniform acceleration on the gas, besides the Constant field', page: 'gravity', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0], excludes: [{ fact: 'domain.geometry', is: 'spherical', reason: 'Not available in a spherical (r, theta) run: a uniform field has no place on its shells' }] },
  { path: 'physics.vcycles', label: 'V-cycles', tip: 'Multigrid V-cycles per step for self-gravity. More converge the potential further, at more cost. On the adaptive mesh this is the most per step', page: 'gravity', control: num(1, 8, { integer: true, step: 1 }), apply: 'live', linkKey: 'vcycles', cfg: 'vcycles', requires: SELF },
  { path: 'physics.jeansFloor', label: 'Jeans floor', tip: 'Pressure floor that keeps the Jeans length resolved (Truelove)', page: 'gravity', control: check, apply: 'live', default: false, linkKey: 'jfloor', cfg: 'jeansFloor', requires: SELF },
  { path: 'physics.jeansCells', label: 'Cells per Jeans length', tip: 'The Jeans floor raises the pressure so the Jeans length spans at least this many cells', page: 'gravity', control: num(4, 32, { integer: true, step: 1 }), apply: 'live', default: 4, linkKey: 'jcells', cfg: 'jeansCells', requires: SELF },
  { path: 'physics.gravEnergyFlux', label: 'Conserve energy', tip: 'Flux-form gravitational work: conserves total energy', page: 'gravity', control: check, apply: 'live', default: true, linkKey: 'fluxwork', cfg: 'gravEnergyFlux', requires: SELF },
  { path: 'physics.rotation.omega', label: 'Rotating frame Ω', tip: 'Angular velocity of a frame turning about an axis through the box center: the gas feels the Coriolis force (and the centrifugal force, below). Velocities are those in the turning frame. In 2D only the z part turns the plane. Zero: an inertial frame', page: 'gravity', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] },
  { path: 'physics.rotation.centrifugal', label: 'Centrifugal force', tip: 'With a rotating frame: the centrifugal force as well as the Coriolis force. Off, the Coriolis force alone (an f-plane, as in a shearing box)', page: 'gravity', control: check, apply: 'restart', default: true },
  { path: 'physics.shear', label: 'Shearing box', tip: 'The local model of a disk: a frame turning at Ω about z with the tidal pull of the shear, x radial, y along the orbit. Its x faces are periodic with the shift of the background flow v_y = -q Ω x, which the initial state should hold. Hydrodynamics on a uniform grid, periodic in y (and z unless stratified)', page: 'gravity', apply: 'restart',
    control: { type: 'group', optional: true, fields: [
      { path: 'omega', label: 'Ω', tip: 'Angular velocity of the frame', page: 'gravity', control: num(0, 1e6, { factor: 1.5 }), apply: 'restart', default: 1 },
      { path: 'q', label: 'Shear q', tip: '-d ln Ω / d ln R: 1.5 for a Keplerian disk, 1 for a flat rotation curve', page: 'gravity', control: num(0, 2, { step: 0.1 }), apply: 'restart', default: 1.5 },
      { path: 'stratified', label: 'Stratified', tip: 'The vertical tidal pull, -Ω² z, toward the midplane', page: 'gravity', control: check, apply: 'restart', default: false },
    ] } },
  { path: 'physics.cosmology', label: 'Comoving coordinates', tip: 'An expanding universe: positions comoving, densities comoving (mean 1), velocities peculiar, times in 1/H0 from the starting redshift. G follows from the matter density. A periodic box of gas, with or without self-gravity', page: 'gravity', apply: 'restart',
    control: { type: 'group', optional: true, fields: [
      { path: 'omegaM', label: 'Matter density Ω_m', tip: 'Matter density today, in units of the critical density', page: 'gravity', control: num(0, 10, { step: 0.05 }), apply: 'restart', default: 1 },
      { path: 'omegaL', label: 'Cosmological constant Ω_Λ', tip: 'Its density today, in units of the critical density (the curvature makes up the rest)', page: 'gravity', control: num(0, 10, { step: 0.05 }), apply: 'restart', default: 0 },
      { path: 'zStart', label: 'Starting redshift', tip: 'Redshift at t = 0', page: 'gravity', control: num(0, 1e4, { factor: 2 }), apply: 'restart', default: 99 },
      { path: 'zEnd', label: 'Ending redshift', tip: 'Redshift the run stops at (it sets the end time)', page: 'gravity', control: num(0, 1e4, { factor: 2 }), apply: 'restart', default: 0 },
      { path: 'boxMpc', label: 'Box size (Mpc/h)', tip: 'The box’s comoving side in Mpc/h. With it the run has physical units, and the readouts show proper densities, pressures and lengths at the current scale factor. Left out, the run is scale free', page: 'gravity', control: num(0, 1e5, { factor: 2 }), apply: 'restart' },
      { path: 'h', label: 'Hubble constant h', tip: 'H0 in units of 100 km/s/Mpc, for the physical units a box size gives', page: 'gravity', control: num(0.1, 2, { step: 0.01 }), apply: 'restart', default: 0.7 },
    ] } },
  { path: 'physics.frame', label: 'Frame', tip: 'An inertial frame, or one that accelerates with the gas’s center of mass or the point masses’, so that what the run follows stays in the box', page: 'gravity', apply: 'restart', default: 'inertial',
    control: pickOf(FRAMES, ['inertial', 'Inertial'], { value: 'gas', label: 'Follows the gas', requires: [{ fact: 'parts.mass' }] }, { value: 'masses', label: 'Follows the masses', requires: [{ fact: 'parts.mass' }] }) },

  // ---- mesh ----
  { path: 'mesh.refineHi', label: 'Refine above', tip: 'Adaptive mesh: a block is split where the normalized second derivative of density or pressure exceeds this anywhere in it. The indicator is near 1 at a shock or contact and small where the gas varies smoothly. Lower refines more', page: 'mesh', control: num(0.05, 1, { step: 0.05 }), apply: 'live', default: 0.8, linkKey: 'refhi', cfg: 'refineHi', requires: ADAPTIVE },
  { path: 'mesh.refineLo', label: 'Merge below', tip: 'Adaptive mesh: four sibling blocks are merged where the indicator is below this in all of them. Kept below the refinement threshold', page: 'mesh', control: num(0, 1, { step: 0.05 }), apply: 'live', default: 0.2, linkKey: 'reflo', cfg: 'refineLo', requires: ADAPTIVE },
  { path: 'mesh.refineRho', label: 'Refine above density', tip: 'Adaptive mesh: gas thinner than this does not ask for refinement, so ripples in the ambient gas far from a star leave its blocks coarse. 0 is off', page: 'mesh', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'live', linkKey: 'refrho', cfg: 'refineRho', requires: ADAPTIVE },
  { path: 'mesh.refineDense', label: 'Always refine above density', tip: 'Adaptive mesh: gas denser than this is always at the finest level, as the smooth interior of a star must be. 0 is off', page: 'mesh', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'restart', default: 0, requires: ADAPTIVE },
  { path: 'mesh.refineMasses', label: 'Refine around point masses', tip: 'Adaptive mesh: keep the gas around every point mass at the finest level, so that softening and sinks can be as small as the finest cells allow', page: 'mesh', control: check, apply: 'restart', default: false, requires: [...ADAPTIVE, { fact: 'parts.mass' }] },
  { path: 'mesh.timeBins', label: 'Local timesteps', tip: 'Adaptive mesh: blocks of slow gas take longer steps than the fastest block, where that saves work', page: 'mesh', apply: 'shaders', default: 0, linkKey: 'tbins', cfg: 'timeBins', requires: ADAPTIVE, excludes: [{ fact: 'pressure.implicit', reason: 'Not with implicit pressure: one solve couples every cell at one time' }],
    control: pick([0, 'Off'], [1, 'Up to 2× longer'], [2, 'Up to 4× longer'], [3, 'Up to 8× longer']) },
  { path: 'mesh.jeansRefine', label: 'Refine the Jeans length', tip: 'With self-gravity, refine where the Jeans length is under this many cells. 0 is off', page: 'mesh', control: num(0, 64, { integer: true, step: 1 }), apply: 'restart', default: 16, requires: [...ADAPTIVE, ...SELF] },

  // ---- solver ----
  { path: 'physics.riemann', label: 'Riemann solver', tip: 'How the flux between neighboring cells is worked out. HLLC resolves contact waves; its low-Mach version cuts dissipation in slow flow; HLL is the most diffusive and the most robust', page: 'solver', control: pickOf(RIEMANN_KINDS, ['hllc', 'HLLC'], ['hllclm', 'HLLC, low Mach'], ['hll', 'HLL']), apply: 'shaders', default: 'hllc', linkKey: 'riemann', cfg: 'riemann' },
  { path: 'physics.limiter', label: 'Limiter', tip: 'How steeply the reconstruction may vary within a cell: Koren is sharpest, Minmod most diffusive and most robust', page: 'solver', control: pickOf(LIMITER_KINDS, ['koren', 'Third order (Koren)'], ['mc', 'Monotonized central'], ['vanleer', 'van Leer'], ['minmod', 'Minmod']), apply: 'shaders', default: 'koren', linkKey: 'limiter', cfg: 'limiter' },
  { path: 'physics.cfl', label: 'CFL number', tip: 'Timestep as a fraction of the time a signal takes to cross a cell. Smaller is more stable and slower', page: 'solver', control: num(0.05, 1, { step: 0.05 }), apply: 'live', default: 0.8, linkKey: 'cfl', cfg: 'cfl' },
  { path: 'physics.pressure', label: 'Pressure', tip: 'How the pressure acts. Explicit: in the fluxes between cells, with a timestep that follows the sound speed. Implicit: advection is explicit and the pressure is solved for each step, so the timestep follows the flow speed, however fast the sound: for the liquid and the ideal gas, in Cartesian coordinates, on the uniform grid or the adaptive mesh', page: 'solver', control: pick({ value: 'explicit', label: 'Explicit' }, { value: 'implicit', label: 'Implicit (solved each step)' }), apply: 'restart', default: 'explicit', linkKey: 'psplit', cfg: 'pressure' },
  { path: 'physics.cutCells', label: 'Partly filled cells', tip: 'Obstacles cut the cells their surfaces cross: each cell holds gas in a share of its volume and each face is open in a share of its area, so a body keeps its true shape below a cell (a thin blade, its pitch), where the plain grid makes it a staircase of whole cells. For the liquid with implicit pressure, on the uniform grid, with obstacles at rest', page: 'solver', control: check, apply: 'restart', default: false, linkKey: 'cut', cfg: 'cutCells', show: { path: 'physics.pressure', is: ['implicit'] },
    requires: [{ fact: 'pressure.implicit' }, { fact: 'physics.eos', is: 'liquid', reason: 'For the liquid, so far' }],
    excludes: [
      { fact: 'mesh.adaptive', reason: 'On the uniform grid only, so far' },
      { fact: 'parts.moving', reason: 'Obstacles at rest only, so far' },
      { fact: 'species.any', reason: 'Not with species, so far' },
      { fact: 'magnetic', reason: 'Not with a magnetic field' },
      { fact: 'radiation', reason: 'Not with radiation, so far' },
      { fact: 'domain.curvilinear', reason: 'In Cartesian coordinates only' },
      { fact: 'symmetric.x', reason: 'Not with mirror symmetry, so far' }, { fact: 'symmetric.y', reason: 'Not with mirror symmetry, so far' }, { fact: 'symmetric.z', reason: 'Not with mirror symmetry, so far' },
      { fact: 'physics.shear', reason: 'Not in a shearing box' },
      { fact: 'physics.cosmology', reason: 'Not with comoving coordinates' },
    ] },
  { path: 'physics.pressureCycles', label: 'Pressure V-cycles', tip: 'Implicit pressure: multigrid V-cycles per step (warm-started). More converge the pressure further, at more cost', page: 'solver', control: num(1, 16, { integer: true, step: 1 }), apply: 'restart', default: 3, linkKey: 'pcycles', cfg: 'pressureCycles', show: { path: 'physics.pressure', is: ['implicit'] } },
  { path: 'physics.pressureTol', label: 'Pressure tolerance', tip: 'Implicit pressure: the residual of the pressure equation after a step\u2019s V-cycles, relative to its largest right-hand side, above which the step is counted as unconverged', page: 'solver', control: num(1e-8, 1, { factor: 10 }), apply: 'live', default: 1e-4, linkKey: 'ptol', cfg: 'pressureTol', show: { path: 'physics.pressure', is: ['implicit'] } },
  { path: 'physics.acousticCfl', label: 'Acoustic steps per step', tip: 'Implicit pressure: the most steps of the explicit scheme that one step may span. It is what bounds the step of a liquid at rest; sound that a step spans many times over is damped', page: 'solver', control: num(1, 1e6, { factor: 2 }), apply: 'live', default: 1000, linkKey: 'acfl', cfg: 'acousticCfl', show: { path: 'physics.pressure', is: ['implicit'] } },

  // ---- the background the parts are painted on ----
  { path: 'background.rho', label: 'Ambient density', tip: 'Density of the gas that fills the box where no part puts anything else. With stars, the gas around them', page: 'parts', control: num(1e-300, undefined, { factor: 10, quantity: 'density' }), apply: 'restart', default: 1, excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.p', label: 'Ambient pressure', tip: 'Pressure of that gas. With stars and left empty, the pressure of the stars’ outermost layer', page: 'parts', control: num(0, undefined, { factor: 10 }), apply: 'restart', excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.v', label: 'Ambient velocity', tip: 'Velocity of that gas', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0], excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.s', label: 'Ambient dye', tip: 'Passive scalar of that gas: a dye carried with it', page: 'parts', control: num(0, 1, { step: 0.1 }), apply: 'restart', default: 0, excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.X', label: 'Ambient composition', tip: 'Mass fractions of that gas, by species name', page: 'parts', control: { type: 'fractions' }, apply: 'restart' },
  { path: 'background.cut', label: 'Stars end at density', tip: 'With stars: a cell thinner than this is ambient gas, not star. Left empty, a polytrope ends where the ambient gas would be as hot as its center', page: 'parts', control: num(0, undefined, { factor: 10 }), apply: 'restart', requires: [{ fact: 'parts.star' }] },
];

const POS: Option = { path: 'pos', label: 'Position', tip: 'Where it starts, relative to the box center', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] };
const VEL: Option = { path: 'vel', label: 'Velocity', tip: 'How it moves at the start', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] };
const P = <T extends Partial<Option>>(o: T & Pick<Option, 'path' | 'label' | 'tip' | 'control'>): Option => ({ page: 'parts', apply: 'restart', ...o });

const SHAPE: Option[] = [
  P({ path: 'shape.type', label: 'Shape', tip: 'A sphere (a cylinder in 2D), a box, the half of space on one side of a plane, or a mesh file (STL, OBJ, PLY, glTF; its cross-section through its center in 2D)', control: pickOf(SHAPE_TYPES, ['sphere', 'Sphere'], ['box', 'Box'], ['half', 'Half-space'], ['mesh', 'Mesh file']), default: 'sphere' }),
  P({ path: 'shape.center', label: 'Center', tip: 'Center of the sphere, or where the middle of the mesh\u2019s bounding box goes', control: { type: 'vec3' }, default: [0, 0, 0], show: { path: 'shape.type', is: ['sphere', 'mesh'] } }),
  P({ path: 'shape.file', label: 'File', tip: 'The mesh file: its name beside the setup document (the runner, the MCP server), a file dropped on the Setup dialog, or a URL. STL, OBJ, PLY, glTF or GLB. STEP and IGES are not read: export STL or glTF from the CAD program', control: { type: 'text' }, default: '', show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.anchor', label: 'Placed by', tip: 'What the center places: the middle of the mesh\u2019s bounding box, or the file\u2019s own origin (a rotor\u2019s hub, which a rotor\u2019s frame needs on its axis)', control: pickOf(MESH_ANCHORS, ['box', 'The middle of its box'], ['origin', 'The file\u2019s origin']), default: 'box', show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.node', label: 'Part of the file', tip: 'A glTF node or an OBJ group, with the parts below it. Empty: the whole file', control: { type: 'text' }, show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.fit', label: 'Fit', tip: 'The mesh\u2019s longest side, as a fraction of the box. STL has no units: give this or the scale', control: num(0, 1, { step: 0.05 }), show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.scale', label: 'Scale', tip: 'A factor on the file\u2019s lengths, instead of fitting it to the box (glTF\u2019s are meters)', control: num(0, undefined, { factor: 2 }), show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.up', label: 'Up axis', tip: 'Which of the file\u2019s axes is up: y for glTF and most 3D programs, z for most CAD. It becomes the box\u2019s z (y in 2D)', control: pickOf(MESH_UPS, ['x', 'x'], ['y', 'y'], ['z', 'z']), default: 'z', show: { path: 'shape.type', is: ['mesh'] } }),
  P({ path: 'shape.radius', label: 'Radius', tip: 'Radius of the sphere', control: num(0, undefined, { factor: 1.25, quantity: 'length' }), default: 0.1, show: { path: 'shape.type', is: ['sphere'] } }),
  P({ path: 'shape.radiusCells', label: 'Radius in cells', tip: 'Instead of the radius: this many finest cells, whatever the resolution (a point explosion)', control: num(0.5, 64, { step: 0.5 }), show: { path: 'shape.type', is: ['sphere'] } }),
  P({ path: 'shape.min', label: 'Lower corner', tip: 'Corner of the box nearest -x, -y, -z', control: { type: 'vec3' }, default: [-0.1, -0.1, -0.1], show: { path: 'shape.type', is: ['box'] } }),
  P({ path: 'shape.max', label: 'Upper corner', tip: 'Corner of the box nearest +x, +y, +z', control: { type: 'vec3' }, default: [0.1, 0.1, 0.1], show: { path: 'shape.type', is: ['box'] } }),
  P({ path: 'shape.rotate', label: 'Rotation', tip: 'Degrees the box or mesh is turned about its center: about x, then y, then z. In 2D only the turn about z counts for a box', control: { type: 'vec3' }, default: [0, 0, 0], show: { path: 'shape.type', is: ['box', 'mesh'] } }),
  P({ path: 'shape.normal', label: 'Normal', tip: 'The plane’s normal: the shape is the side the normal points away from', control: { type: 'vec3' }, default: [1, 0, 0], show: { path: 'shape.type', is: ['half'] } }),
  P({ path: 'shape.at', label: 'Plane at', tip: 'Distance of the plane from the box center, along its normal', control: num(undefined, undefined, { step: 0.05, quantity: 'length' }), default: 0, show: { path: 'shape.type', is: ['half'] } }),
];

export const PART_OPTIONS: Record<PartKind, PartOption> = {
  mass: {
    kind: 'mass', label: 'Point mass', tip: 'A point mass: softened gravity on the gas, with a sink that takes gas in. Point masses pull on one another and feel the gas', max: 32,
    excludes: NOT_PERIODIC,
    make: () => ({ kind: 'mass', gm: 1, soft: 0.05 }),
    fields: [
      P({ path: 'gm', label: 'G M', tip: 'Its mass times G, in code units', control: num(0, undefined, { factor: 2 }), default: 1 }),
      POS, VEL,
      P({ path: 'soft', label: 'Softening', tip: 'Plummer softening length of its gravity: at least two cells, and two cells if left empty. Changes take effect without a restart', control: num(0, undefined, { step: 0.01, quantity: 'length' }), apply: 'live' }),
      P({ path: 'sink', label: 'Sink radius', tip: 'Radius within which it takes gas in: the softening length if left empty, 0 for none. Changes take effect without a restart', control: num(0, undefined, { step: 0.01, quantity: 'length' }), apply: 'live' }),
      P({ path: 'sinkRate', label: 'Sink rate', tip: 'Removal rate at the sink’s center, in units of sqrt(G M / radius³)', control: num(0, undefined, { factor: 2 }), default: 1, apply: 'live' }),
      P({ path: 'onOrbit', label: 'Hold on its orbit', tip: 'The other point masses move it; the gas does not: neither the gas\u2019s pull nor what its sink takes changes its motion, and it does not grow. A binary with both masses held keeps the orbit it was given, as in most studies of circumbinary disks. What the sink takes is still counted. Changes take effect without a restart', control: check, default: false, apply: 'live' }),
      P({ path: 'massless', label: 'Massless', tip: 'A particle without a mass of its own: it pulls on nothing and moves as the gas and the masses pull it', control: check, default: false }),
      P({ path: 'fixed', label: 'Held in place', tip: 'Nothing moves it, whatever pulls on it or it takes in', control: check, default: false }),
    ],
  },
  orbit: {
    kind: 'orbit', label: 'Orbit', tip: 'Not an object: puts two bodies on a Kepler orbit', requires: [{ fact: 'parts.body' }],
    make: () => ({ kind: 'orbit', of: ['', ''], type: 'binary', separation: 1, ecc: 0 }),
    fields: [
      P({ path: 'of', label: 'Bodies', tip: 'The two bodies it places: point masses or stars (binary), a star and a point mass (flyby), or two stars (collision)', control: { type: 'ref', kinds: ['mass', 'star'], count: 2 } }),
      P({ path: 'type', label: 'Kind of orbit', tip: 'Binary: two bodies, point masses or stars, about their center of mass, at pericenter; on a circular orbit its stars turn with it and start in the balance of the Roche potential. Flyby: a point mass on a parabola about a star that stays at the origin (a tidal disruption in the star’s frame). Collision: two stars on a parabola about their center of mass', control: pickOf(ORBIT_TYPES, ['binary', 'Binary'], ['flyby', 'Flyby (star and mass)'], ['collision', 'Collision (two stars)']), default: 'binary' }),
      P({ path: 'q', label: 'Mass ratio', tip: 'Mass of the second body over the first. It sets their masses: in a binary of two point masses the total stays, in a binary with a star or a flyby the star’s', control: num(1e-6, 1e9, { factor: 1.25 }), show: { path: 'type', is: ['binary', 'flyby'] } }),
      P({ path: 'separation', label: 'Separation', tip: 'Semi-major axis of the binary’s orbit', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 1, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'ecc', label: 'Eccentricity', tip: 'Eccentricity of the binary’s orbit', control: num(0, 0.95, { step: 0.05 }), default: 0, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'incl', label: 'Inclination (°)', tip: 'Tilt of the binary’s orbit about the line through pericenter, in degrees (3D): 180 is retrograde', control: num(0, 180, { step: 1 }), default: 0, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'fill', label: 'Roche lobe fill', tip: 'In place of the separation: the first body, a star, is this many times the size of its Roche lobe (Eggleton 1983), which sets how far apart the two are. 1 just fills it; above 1 it pours gas through L1 onto its companion from the start', control: num(0.05, 2, { step: 0.01 }), show: { path: 'type', is: ['binary'] } }),
      P({ path: 'corotating', label: 'Corotating frame', tip: 'A circular binary: the run is in the frame turning with the orbit, so the bodies start at rest and a binary in balance stays still', control: check, default: false, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'track', label: 'Follow the orbit', tip: 'With the corotating frame: the frame turns faster or slower as the orbit changes (mass transfer, drag, gas lost through the faces), keeping the line between the two bodies where it started, and the gas and the point masses feel the Euler force of its changing turn. It follows the axis of the stars’ and point masses’ quadrupole, so any two bodies can be followed', control: check, default: false, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'retrograde', label: 'Retrograde', tip: 'In 2D: the binary orbits clockwise', control: check, default: false, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'beta', label: 'Impact parameter β', tip: 'Tidal radius over pericenter distance: above 1 the mass passes inside the star’s tidal radius', control: num(0.05, 20, { step: 0.1 }), default: 1, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'start', label: 'Starts at (tidal radii)', tip: 'Separation at the start, in tidal radii (never inside 1.5 pericenters)', control: num(1, 20, { step: 0.5 }), default: 3, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'debris', label: 'Debris energies', tip: 'Bin the star’s gas by its orbital energy about the mass, for the rate at which bound debris returns', control: check, default: false, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'impact', label: 'Impact parameter', tip: 'Pericenter distance over the sum of the two radii: 0 is head-on, 1 just grazes', control: num(0, 5, { step: 0.1 }), default: 0, show: { path: 'type', is: ['collision'] } }),
      P({ path: 'starts', label: 'Starts at (sums of radii)', tip: 'Separation at the start, in sums of the two radii (closer if they would not fit in the box)', control: num(1.05, 20, { step: 0.5 }), default: 4, show: { path: 'type', is: ['collision'] } }),
    ],
  },
  star: {
    kind: 'star', label: 'Star', tip: 'A polytrope, a cold white dwarf or a planet of ANEOS materials, settled into the grid’s own hydrostatic balance before the run', max: 4,
    requires: [...IN_SPACE, { fact: 'physics.gravity', is: 'isolated', reason: 'Needs isolated self-gravity: it is what holds a star together' }],
    excludes: NOT_PERIODIC,
    make: () => ({ kind: 'star', type: 'polytrope', n: 1.5, mass: 1, radius: 0.2 }),
    fields: [
      P({ path: 'type', label: 'Kind of star', tip: 'A polytrope of index n, a cold white dwarf held up by degenerate electrons, or a planet of an ANEOS material, with or without a core of another (physical units)', control: pickOf(STAR_TYPES,
        { value: 'polytrope', label: 'Polytrope', requires: [{ fact: 'eos.energy' }], excludes: [{ fact: 'parts.star.white-dwarf', reason: 'Not available with a white dwarf: the two are settled with different gases' }] },
        { value: 'white-dwarf', label: 'White dwarf', requires: [{ fact: 'eos.degenerate' }], excludes: [{ fact: 'parts.star.polytrope', reason: 'Not available with a polytropic star: the two are settled with different gases' }] },
        { value: 'planet', label: 'Planet (rock, iron, ice)', requires: [{ fact: 'eos.aneos' }] }), default: 'polytrope' }),
      P({ path: 'n', label: 'Polytropic index n', tip: 'The star’s structure, p = K ρ^(1 + 1/n): the larger n, the more centrally concentrated. Stable against convection only if the gas’s adiabatic index is at least 1 + 1/n', control: num(0.5, 4.5, { step: 0.5 }), default: 1.5, show: { path: 'type', is: ['polytrope'] } }),
      P({ path: 'mass', label: 'Mass', tip: 'Its mass, in code units (solar masses for a white dwarf in its usual units)', control: num(1e-12, undefined, { factor: 1.25, quantity: 'mass' }), default: 1 }),
      P({ path: 'radius', label: 'Radius', tip: 'A polytrope’s radius, in code units. A white dwarf’s follows from its mass', control: num(1e-12, undefined, { factor: 1.25, quantity: 'length' }), default: 0.2, show: { path: 'type', is: ['polytrope'] } }),
      P({ path: 'material', label: 'Material', tip: 'What a planet is made of; with a core, its mantle. A run holds two materials at most, and they are its species: the mantle’s first', control: pick(...ANEOS_MATERIALS.map((m): [Value, string] => [m, ANEOS_LABELS[m]])), default: 'pyrolite', show: { path: 'type', is: ['planet'] } }),
      P({ path: 'coreMaterial', label: 'Core material', tip: 'A planet’s core, if it has one: the inner part of its mass, of this material', control: pick(...ANEOS_MATERIALS.map((m): [Value, string] => [m, ANEOS_LABELS[m]])), show: { path: 'type', is: ['planet'] } }),
      P({ path: 'coreFraction', label: 'Core mass fraction', tip: 'The core’s share of a planet’s mass (the Earth’s is 0.32)', control: num(0.01, 0.99, { step: 0.05 }), default: 0.3, show: { path: 'type', is: ['planet'] } }),
      P({ path: 'surfaceT', label: 'Surface temperature (K)', tip: 'A planet’s temperature at its surface, in kelvin. Each of its layers lies along its adiabat from there: 2000 K is a body still hot from its accretion, as the published giant-impact calculations start', control: num(50, 20000, { factor: 1.25 }), default: 2000, show: { path: 'type', is: ['planet'] } }),
      P({ path: 'rhoc', label: 'Central density', tip: 'Instead of the mass: a polytrope’s central density', control: num(1e-300, undefined, { factor: 2, quantity: 'density' }), show: { path: 'type', is: ['polytrope'] } }),
      P({ path: 'core', label: 'Point-mass core', tip: 'A giant: part of the star’s mass is a point mass at its center, and the rest a polytropic envelope in balance with its own gravity and the core’s softened pull. The core moves with the star and then as the gas and the other masses pull it. By mass and radius only',
        show: { path: 'type', is: ['polytrope'] },
        control: { type: 'group', optional: true, fields: [
          P({ path: 'mass', label: 'Core mass', tip: 'The core’s share of the star’s mass, in code units: less than the star’s', control: num(1e-12, undefined, { factor: 1.25, quantity: 'mass' }), default: 0.4 }),
          P({ path: 'soft', label: 'Core softening', tip: 'Plummer softening length of the core’s gravity: a twentieth of the star’s radius if left empty. At least two finest cells, and the envelope’s steep inner part should span several', control: num(1e-12, undefined, { factor: 1.25, quantity: 'length' }) }),
        ] } }),
      POS, VEL,
      P({ path: 'dye', label: 'Dye', tip: 'Passive scalar of its gas, which tells the stars apart: spread over (0, 1] by their order if left empty', control: num(0, 1, { step: 0.1 }) }),
      P({ path: 'X', label: 'Composition', tip: 'Mass fractions of its gas, by species name (the ambient gas’s if left empty)', control: { type: 'fractions' } }),
      P({ path: 'field', label: 'Magnetic field inside', tip: 'A field confined to the star: a dipole along z or a tangled Kolmogorov field, with plasma beta at the center; or the same as Guillochon & McCourt (2017) set them, by their β_M',
        control: { type: 'group', optional: true, fields: [
          P({ path: 'type', label: 'Shape', tip: 'Dipole or random, by the plasma beta at the center; or Guillochon & McCourt’s, by β_M', control: pickOf(STAR_FIELD_TYPES, ['dipole', 'Dipole'], ['random', 'Random'], ['gm17-dipole', 'Dipole (G&M 2017)'], ['gm17-random', 'Random (G&M 2017)']), default: 'dipole' }),
          P({ path: 'beta', label: 'Plasma beta', tip: 'Gas pressure over magnetic pressure B² / 2 at the center (or β_M): the larger, the weaker the field', control: num(1, 1e12, { factor: 10 }), default: 1000 }),
          P({ path: 'seed', label: 'Random seed', tip: 'Which random field: each seed gives another', control: num(1, 9999, { integer: true, step: 1 }), default: 1 }),
        ] } }),
    ],
  },
  disk: {
    kind: 'disk', label: 'Disk', tip: 'A gas disk in the x-y plane about one or more bodies', requires: [{ fact: 'parts.body', reason: 'Needs a point mass or a star to orbit' }],
    excludes: NOT_PERIODIC,
    make: () => ({ kind: 'disk', about: [], profile: 'balanced', rIn: 0.3, rOut: 1, sigma0: 1e-3, slope: -1.5, cs: 0.1, floor: 1e-8 }),
    fields: [
      P({ path: 'about', label: 'Orbits', tip: 'The point masses or stars it orbits: about several, their total mass. The disk is centered on their center of mass and moves with it. About a star, keep the inner radius outside 1.3 of the star\u2019s radius: the star is settled within that', control: { type: 'ref', kinds: ['mass', 'star'], count: 'any' } }),
      P({ path: 'profile', label: 'Profile', tip: 'Thin: a surface density with a cavity and a Gaussian vertical profile, on circular orbits (the binary’s disk). Balanced: a midplane density in vertical balance with the softened potential, rotating as its pressure gradient asks (the magnetized disk); 3D only', control: pickOf(DISK_PROFILES, ['thin', 'Thin, with a cavity'], { value: 'balanced', label: 'Balanced', requires: IN_3D }), default: 'balanced' }),
      P({ path: 'rIn', label: 'Inner radius', tip: 'Where the disk is tapered away toward the center (the cavity’s radius)', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 0.3 }),
      P({ path: 'rOut', label: 'Outer radius', tip: 'Where the disk is tapered away outward', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 1 }),
      P({ path: 'sigma0', label: 'Density scale', tip: 'Thin: the surface density. Balanced: the midplane density at r = 1', control: num(1e-300, undefined, { factor: 2 }), default: 1e-3 }),
      P({ path: 'slope', label: 'Density slope', tip: 'Balanced: the midplane density goes as r to this power (-3/2 is a flat surface density)', control: num(-4, 2, { step: 0.25 }), default: -1.5, show: { path: 'profile', is: ['balanced'] } }),
      P({ path: 'radialFn', label: 'Radial profile (formula)', tip: 'A density profile of your own in radius, as a formula of r: the surface density (thin) or the midplane density (balanced). It replaces the density scale\u2019s law, the slope and the tapers. Example: sigma0 * (r / rIn)^-1 * exp(-(r / rOut)^2) * step(r - rIn). Names: r, rIn, rOut, sigma0, cs, H (the scale height cs / \u03a9 at r), pi, e. Functions: ' + Object.keys(FUNCTIONS).join(', ') + '; powers as x^y. Where it is not a positive number there is no disk. A file may give a table of points instead (radial)', control: { type: 'text' } }),
      P({ path: 'verticalFn', label: 'Vertical profile (formula)', tip: 'A density profile of your own in height, as a formula of z, the height above the midplane (the same on both sides), relative to the midplane: it replaces the Gaussian (thin) or the balance with the potential (balanced). Examples: exp(-z^2 / 2), 1 / cosh(z)^2, max(1 - z / 3, 0). Such a disk is in vertical balance only if the profile is, and settles as the run starts. 3D only. A file may give a table of points instead (vertical)', control: { type: 'text' }, requires: IN_3D }),
      P({ path: 'verticalIn', label: 'Heights in', tip: 'The z of the vertical profile: in scale heights cs / \u03a9 of each radius (the profile then flares with the disk), or in code lengths', control: pickOf(DISK_HEIGHTS, ['H', 'Scale heights'], ['length', 'Code lengths']), default: 'H' }),
      P({ path: 'cs', label: 'Sound speed', tip: 'Sound speed of the disk’s gas: its scale height is cs / Ω', control: num(1e-6, undefined, { factor: 1.25, quantity: 'velocity' }), default: 0.1 }),
      P({ path: 'floor', label: 'Gas outside the disk', tip: 'Thin: a fraction of the density scale. Balanced: the density of the ambient gas', control: num(0, undefined, { factor: 10 }), default: 1e-8 }),
      P({ path: 'damp', label: 'Damping zone', tip: 'Beyond a radius the gas is relaxed toward the disk as it started, more strongly toward the box’s edge, so the outer disk is a reservoir and the boundaries see only it',
        excludes: [{ fact: 'parts.damping', reason: 'A run has one damping zone: another disk already has it' }],
        control: { type: 'group', optional: true, fields: [
          P({ path: 'from', label: 'Damping from r', tip: 'Radius from the box center where the damping zone starts', control: num(0, 1e30, { step: 0.25, quantity: 'length' }), default: 1, apply: 'live', linkKey: 'dampr', cfg: 'dampRadius' }),
          P({ path: 'rate', label: 'Damping rate', tip: 'How many times per local orbital period the gas at the box edge is relaxed toward its initial state. 0 is off', control: num(0, 1e30, { step: 0.5 }), default: 3, apply: 'live', linkKey: 'damprate', cfg: 'dampRate' }),
        ] } }),
      P({ path: 'loops', label: 'Field loops', tip: 'Poloidal loops of magnetic field inside the disk, with no net flux: the seed of the magnetorotational instability',
        control: { type: 'group', optional: true, fields: [
          P({ path: 'beta', label: 'Plasma beta at the midplane', tip: 'Gas pressure over magnetic pressure where two loops meet: the larger, the weaker the field', control: num(1, 1e9, { factor: 2 }), default: 50 }),
          P({ path: 'span', label: 'Span of a loop', tip: 'Each loop spans this factor in radius', control: num(1.05, 4, { step: 0.05 }), default: 1.35 }),
          P({ path: 'count', label: 'Loops', tip: 'How many loops, outward from the inner radius', control: num(1, 16, { integer: true, step: 1 }), default: 4 }),
        ] } }),
      P({ path: 'stresses', label: 'Disk diagnostics', tip: 'Report the disk’s stresses and MRI quality factors about the first point mass', control: check, default: false }),
    ],
  },
  region: {
    kind: 'region', label: 'Gas region', tip: 'A shape with a state (density, pressure, velocity, dye, composition) that replaces what is under it. A value left empty keeps what is there',
    make: () => ({ kind: 'region', shape: { type: 'sphere', radius: 0.1 }, rho: 2 }),
    fields: [
      ...SHAPE,
      P({ path: 'rho', label: 'Density', tip: 'Density inside the shape', control: num(1e-300, undefined, { factor: 2, quantity: 'density' }) }),
      P({ path: 'p', label: 'Pressure', tip: 'Pressure inside the shape', control: num(0, undefined, { factor: 2 }) }),
      P({ path: 'energy', label: 'Total energy', tip: 'Instead of a pressure: this much energy in all, spread evenly over the sphere’s cells (a point explosion; per unit length in 2D)', control: num(0, undefined, { factor: 2 }), show: { path: 'shape.type', is: ['sphere'] }, requires: ENERGY }),
      P({ path: 'v', label: 'Velocity', tip: 'Velocity inside the shape', control: { type: 'vec3' } }),
      P({ path: 's', label: 'Dye', tip: 'Passive scalar inside the shape', control: num(0, 1, { step: 0.1 }) }),
      P({ path: 'X', label: 'Composition', tip: 'Mass fractions inside the shape, by species name', control: { type: 'fractions' } }),
      P({ path: 'erad', label: 'Radiation energy', tip: 'With radiation transport: the radiation’s energy density inside the shape. Empty: that in equilibrium with the gas, a T⁴', control: num(0, undefined, { factor: 2 }), requires: [{ fact: 'radiation' }] }),
      P({ path: 'frad', label: 'Radiation flux', tip: 'With radiation transport: the radiation’s flux inside the shape, at most c times its energy (a beam). Empty: the flux of diffusion. With an inflow over the same shape, a lamp that shines for the whole run', control: { type: 'vec3' }, requires: [{ fact: 'radiation' }] }),
      P({ path: 'supersample', label: 'Smooth edge (samples)', tip: 'A sphere’s edge cells take the share of their volume inside it, from this many samples a side. Empty: a sharp edge', control: num(2, 8, { integer: true, step: 1 }), show: { path: 'shape.type', is: ['sphere'] } }),
    ],
  },
  obstacle: {
    kind: 'obstacle', label: 'Obstacle', tip: 'A solid body the gas reflects off',
    make: () => ({ kind: 'obstacle', shape: { type: 'box', min: [-0.1, -0.1, -0.1], max: [0.1, 0.1, 0.1] } }),
    fields: [
      ...SHAPE,
      P({ path: 'refine', label: 'Refines', tip: 'On an adaptive mesh: Surface refines every block within a band of the obstacle’s surface, decided from its shape (so a part thinner than a coarse cell is whole at the finest level); Swept, everywhere its surface passes as it moves (a rotor’s disk; for a body at rest the same as Surface); Off leaves it to the flow’s own refinement', control: pickOf(REFINE_MODES, ['surface', 'Its surface'], ['swept', 'All it sweeps'], ['off', 'Off']), default: 'surface', requires: ADAPTIVE }),
      P({ path: 'refineBand', label: 'Refined band (cells)', tip: 'How far from the surface the mesh is kept at the obstacle’s level, in finest cells', control: num(0, 64, { step: 1 }), default: 4, requires: ADAPTIVE, show: { path: 'refine', is: ['surface', 'swept'] } }),
      P({ path: 'refineLevel', label: 'Refined to level', tip: 'The level its band is kept at, 0 the coarsest. Empty: the finest', control: num(0, 5, { integer: true, step: 1 }), requires: ADAPTIVE, show: { path: 'refine', is: ['surface', 'swept'] } }),
    ],
  },
  tunnel: {
    kind: 'tunnel', label: 'Wind tunnel', max: 1,
    tip: 'A stream through the box from an inlet to an outlet with a sponge, and the box\u2019s faces, its gas and its background set to match: put obstacles in it, and the diagnostics give their drag and lift coefficients. Below Mach 0.3 the gas is the liquid with implicit pressure; above, an ideal gas',
    excludes: [{ fact: 'magnetic', reason: 'Obstacles take no magnetic field' }, { fact: 'domain.curvilinear', reason: 'The tunnel is Cartesian' }],
    make: () => ({ kind: 'tunnel', speed: 1, axis: 'x', sponge: 0.1, mach: 0.1 }),
    fields: [
      P({ path: 'speed', label: 'Speed', tip: 'The stream\u2019s speed', control: num(0, undefined, { factor: 1.25, quantity: 'velocity' }), default: 1 }),
      P({ path: 'axis', label: 'Along', tip: 'The axis the stream flows along, toward +. Yaw and angle of attack turn the body, not the stream', control: pickOf(AXES, ['x', 'x'], ['y', 'y'], ['z', 'z']), default: 'x' }),
      P({ path: 'mach', label: 'Mach number', tip: 'The stream\u2019s speed over the gas\u2019s sound speed. Below 0.3 the gas is the liquid with implicit pressure (incompressible, in effect); above, an ideal gas with the explicit scheme (shocks)', control: num(0.01, 5, { factor: 1.25 }), default: 0.1 }),
      P({ path: 'rho', label: 'Density', tip: 'The gas\u2019s density', control: num(0, undefined, { factor: 2, quantity: 'density' }), default: 1 }),
      P({ path: 'sides', label: 'Sides and roof', tip: 'Slip: walls the gas slides along. Open: faces it may leave by. Periodic: the box repeats sideways', control: pickOf(TUNNEL_SIDES, ['slip', 'Slip walls'], ['open', 'Open'], ['periodic', 'Periodic']), default: 'slip' }),
      P({ path: 'floor', label: 'Floor', tip: 'The lower face along the box\u2019s up axis (z in 3D, y in 2D): as the sides, a slip wall, a no-slip wall, or a wall moving with the stream (a car on a road)', control: pickOf(TUNNEL_FLOORS, ['none', 'As the sides'], ['slip', 'Slip wall'], ['fixed', 'No-slip wall'], ['moving', 'Moving with the stream']), default: 'none' }),
      P({ path: 'sponge', label: 'Sponge', tip: 'The outlet\u2019s damping zone, as a fraction of the box\u2019s length: it pulls the gas back to the stream so that the wake leaves without reflecting. 0 for none', control: num(0, 0.5, { step: 0.05 }), default: 0.1 }),
      P({ path: 'inlet', label: 'Inlet cells', tip: 'Cells held at the stream\u2019s state at the upstream face: finest cells on a uniform grid, the coarsest level\u2019s on blocks', control: num(1, 16, { step: 1, integer: true }), default: 2 }),
      P({ path: 'rotor', label: 'Rotor frame', tip: 'An angular velocity about the stream\u2019s axis through the box\u2019s center: the run is in the frame that turns at it, so that a rotor centered there is at rest (the stream is U - Omega x r, the corners of the box held at it). Torque and thrust then come steady. 0: none. 3D only', control: num(undefined, undefined, { step: 0.5 }), default: 0, requires: [{ fact: 'domain.ndim', is: 3, reason: 'A rotor turns about the stream, out of the plane of a 2D run' }] }),
      P({ path: 'area', label: 'Reference area', tip: 'The area the force coefficients are over (a length in 2D). Empty: the first obstacle\u2019s shadow along the stream', control: num(0, undefined, { factor: 2 }) }),
      P({ path: 'length', label: 'Reference length', tip: 'The length the moment coefficients are over. Empty: the square root of the area (the area itself in 2D)', control: num(0, undefined, { factor: 2, quantity: 'length' }) }),
    ],
  },
  inflow: {
    kind: 'inflow', label: 'Inflow', tip: 'Cells held at their initial state for the whole run: an inlet',
    make: () => ({ kind: 'inflow', shape: { type: 'box', min: [-0.5, -0.5, -0.5], max: [-0.45, 0.5, 0.5] } }),
    fields: [...SHAPE],
  },
  field: {
    kind: 'field', label: 'Uniform magnetic field', tip: 'A uniform field through the whole box, in units where the magnetic pressure is B² / 2',
    make: () => ({ kind: 'field', b: [0, 0, 0.1] }),
    fields: [
      P({ path: 'b', label: 'Field', tip: 'The field’s vector', control: { type: 'vec3' } }),
      P({ path: 'up', label: 'Strength, upward', tip: 'Instead of a vector: a strength along the axis drawn upward (y in 2D, z in 3D)', control: num(undefined, undefined, { factor: 2 }) }),
    ],
  },
  flow: {
    kind: 'flow', label: 'Flow', tip: 'A named, built-in initial condition with its parameters, for what shapes cannot say',
    make: () => ({ kind: 'flow', name: Object.keys(FLOWS)[0] }),
    fields: [],
  },
};

/** A flow part's fields, from the flow's own definition. */
export function flowOptions(name: string): Option[] {
  const def = FLOWS[name];
  if (!def) return [];
  return Object.entries(def.fields).map(([key, f]) =>
    P({ path: key, label: f.label, tip: f.tip ?? f.label, default: f.default, apply: f.apply ?? 'restart',
      control: typeof f.default === 'boolean' ? check : num(f.min, f.max, { ...(f.step !== undefined ? { step: f.step } : {}), ...(f.factor !== undefined ? { factor: f.factor } : {}) }) }));
}

/** The fields of a part: its kind's, and for a flow the named flow's. */
export function partFields(part: Part): Option[] {
  if (part.kind !== 'flow') return PART_OPTIONS[part.kind].fields;
  return [
    P({ path: 'name', label: 'Flow', tip: 'Which built-in initial condition', default: part.name,
      control: pick(...Object.values(FLOWS).map((f): Choice => ({ value: f.name, label: f.label, tip: f.tip, ...(f.requires ? { requires: f.requires } : {}), excludes: [
        ...(f.wholeBox ? [{ fact: 'parts.flow.whole', reason: 'Only one flow can fill the whole box' }, ...(f.liquid ? [] : [{ fact: 'physics.eos', is: 'liquid', reason: 'Not for the liquid: its densities stray far from the liquid’s one reference density' }])] : []),
        // A field given on the faces, not by a potential, is divergence-free in Cartesian
        // coordinates only: in (R, z) a radial field has B_R / R besides (random test seed 10747).
        ...(f.face && !f.potential ? [{ fact: 'domain.curvilinear', reason: 'Not in an axisymmetric or spherical run: its field is divergence-free in Cartesian coordinates only' }] : []),
      ] }))) }),
    ...flowOptions(part.name),
  ];
}

/** Options with their groups' fields laid out flat, each path whole ('damp.rate'). */
export function flatten(options: Option[], prefix = ''): Option[] {
  return options.flatMap((o) => {
    const path = prefix + o.path;
    return o.control.type === 'group' ? [{ ...o, path }, ...flatten(o.control.fields, `${path}.`)] : [{ ...o, path }];
  });
}

/** Every general option by its path, groups' fields included. */
export const OPTION_BY_PATH: Record<string, Option> = Object.fromEntries(flatten(OPTIONS).map((o) => [o.path, o]));

/** The option a path of a document leads to: a general one, or a part's field ('parts.2.ecc'). */
export function optionAt(spec: SetupSpec, path: string): Option | undefined {
  const m = /^parts\.(\d+)\.(.+)$/.exec(path);
  if (!m) return OPTION_BY_PATH[path];
  const part = spec.parts?.[+m[1]];
  return part ? flatten(partFields(part)).find((o) => o.path === m[2]) : undefined;
}

/** An option's value in a document, or its default. */
export function valueAt(spec: SetupSpec, path: string): unknown {
  const v = getPath(spec, path);
  return v !== undefined ? v : optionAt(spec, path)?.default;
}

/** What is wrong with a setup that a list of options cannot say: checks on values, each a sentence. */
export interface Issue {
  level: 'error' | 'warning';
  text: string;
  /** the part or option it is about */
  path?: string;
}

/** Every rule a setup breaks: an option set, a choice made or a part present that its rules do not allow. */
export function ruleIssues(spec: SetupSpec): Issue[] {
  const out: Issue[] = [];
  // A setting of the adaptive mesh on a uniform grid is idle, not wrong: the grid may change.
  const unavailable = (o: { requires?: Rule[]; excludes?: Rule[] }, facts: Record<string, Value>) => unavailableIn({ requires: o.requires?.filter((r) => r.fact !== 'mesh.adaptive'), excludes: o.excludes }, facts);
  const add = (what: string, path: string, why: string[]) => why.forEach((w) => out.push({ level: 'error', path, text: `${what}: ${w.charAt(0).toLowerCase()}${w.slice(1)}` }));
  const visit = (options: Option[], base: unknown, prefix: string, name: string, facts: Record<string, Value>) => {
    for (const o of options) {
      const v = getPath(base, o.path);
      if (v === undefined || v === false || v === 0 || v === null) continue;
      add(`${name}${o.label}`, prefix + o.path, unavailable(o, facts));
      if (o.control.type === 'select') {
        const c = o.control.choices.find((k) => k.value === v);
        if (c) add(`${name}${o.label}: ${c.label}`, prefix + o.path, unavailable(c, facts));
      }
      if (o.control.type === 'group') {
        if (o.control.forms) {
          const f = formOf(o.control.forms, v);
          add(`${name}${o.label}: ${f.label}`, prefix + o.path, unavailable(f, facts));
        }
        visit(o.control.fields, v, `${prefix}${o.path}.`, name, facts);
      }
    }
  };
  visit(OPTIONS, spec, '', '', factsOf(spec));
  (spec.parts ?? []).forEach((p, i) => {
    // A part is judged against the setup without it: what it excludes is a second of its kind.
    const facts = factsWithout(spec, i);
    const k = PART_OPTIONS[p.kind];
    add(k.label, `parts.${i}`, unavailable(k, facts));
    visit(partFields(p), p, `parts.${i}.`, `${k.label}: `, facts);
  });
  // The periodic choice of a face is one option six times: judged once.
  const facts = factsOf(spec);
  if (facts['bc.periodic']) add('A periodic face', 'domain.bc', unavailable(BC_CHOICES[0], facts));
  return out;
}

/** The facts of a setup without one of its parts: what that part's own rules are judged against. */
export function factsWithout(spec: SetupSpec, index: number): Record<string, Value> {
  return factsOf({ ...spec, parts: (spec.parts ?? []).filter((_, i) => i !== index) });
}
