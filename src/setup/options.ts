// The option table: every setting of a setup, listed once. The dialog's forms, the conflict
// checks, the defaults, the codec's ranges, the link keys and the scripting interface's list of
// controls are all read from it.
//
// Which options cannot be combined is data too: each option (or one choice of a select, or a
// kind of part) may name what it `requires` and what it `excludes`, as rules over the setup's
// facts (FACTS below). The dialog greys out an option whose rules do not hold and makes its
// tooltip from the rule that disables it.
import { COOLING_LABELS } from '../cooling/cooling';
import { EOS_LABELS } from '../eos/eos';
import { FLOWS } from './parts/flow';
import { getPath, type Part, type PartKind, type SetupSpec } from './spec';
import { LENGTH_UNITS, MASS_UNITS } from './units';


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
  /** mass fractions by species name */
  | { type: 'fractions' }
  /** names, comma separated */
  | { type: 'names' }
  /** fields of their own, present or not as a whole */
  | { type: 'group'; optional: boolean; fields: Option[] };

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
const faces = (s: SetupSpec): number[] => [...(s.domain?.bc?.lo ?? [1, 1, 1]), ...(s.domain?.bc?.hi ?? [1, 1, 1])];
const flowDef = (p: Part) => (p.kind === 'flow' ? FLOWS[p.name] : undefined);

/** What rules are about: each a named property of a setup, with how to say it. */
export const FACTS: Record<string, { label: string; of: (s: SetupSpec) => Value; say?: (v: Value) => string }> = {
  'domain.ndim': { label: 'Dimensions', of: (s) => s.domain?.ndim ?? 3, say: (v) => `${v}D` },
  'domain.supports2d': { label: 'A setup that makes sense in 2D', of: (s) => s.domain?.supports2d ?? true },
  'mesh.adaptive': { label: 'An adaptive mesh', of: (s) => (s.domain?.amr ?? 0) > 0 },
  'physics.eos': { label: 'Equation of state', of: (s) => s.physics?.eos ?? 'ideal', say: (v) => EOS_LABELS[v as keyof typeof EOS_LABELS] ?? String(v) },
  'eos.energy': { label: 'An equation of state with an energy', of: (s) => (s.physics?.eos ?? 'ideal') !== 'barotropic' },
  'eos.degenerate': { label: 'The degenerate or tabulated (Helmholtz) equation of state', of: (s) => ['degenerate', 'helmholtz'].includes(s.physics?.eos ?? 'ideal') },
  'physics.gravity': { label: 'Gravity', of: (s) => s.physics?.gravity ?? 'off', say: (v) => ({ off: 'Off', constant: 'Constant', periodic: 'Self, periodic', isolated: 'Self, isolated' })[String(v)] ?? String(v) },
  'gravity.self': { label: 'Self-gravity', of: (s) => ['periodic', 'isolated'].includes(s.physics?.gravity ?? 'off') },
  'bc.periodic': { label: 'A periodic face', of: (s) => faces(s).some((b) => b === 0) },
  'bc.allPeriodic': { label: 'Every face periodic', of: (s) => faces(s).every((b) => b === 0) },
  'units': { label: 'Physical units', of: (s) => !!s.domain?.units },
  'species.aprox13': { label: 'The 13 species of the burning network', of: (s) => s.physics?.species === 'aprox13' },
  'burn.network': { label: 'Network burning', of: (s) => !!s.physics?.burn && 'network' in s.physics.burn },
  'magnetic': { label: 'A magnetic field', of: (s) => parts(s).some((p) => p.kind === 'field' || (p.kind === 'star' && !!p.field) || (p.kind === 'disk' && !!p.loops) || !!flowDef(p)?.potential || !!flowDef(p)?.face) },
  'parts.mass': { label: 'A point mass', of: (s) => count(s, (p) => p.kind === 'mass') > 0 },
  'parts.star': { label: 'A star', of: (s) => count(s, (p) => p.kind === 'star') > 0 },
  'parts.star.polytrope': { label: 'A polytropic star', of: (s) => count(s, (p) => p.kind === 'star' && p.type === 'polytrope') > 0 },
  'parts.star.white-dwarf': { label: 'A white dwarf', of: (s) => count(s, (p) => p.kind === 'star' && p.type === 'white-dwarf') > 0 },
  'parts.disk': { label: 'A disk', of: (s) => count(s, (p) => p.kind === 'disk') > 0 },
  'parts.body': { label: 'A point mass or a star', of: (s) => count(s, (p) => p.kind === 'mass' || p.kind === 'star') > 0 },
  'parts.damping': { label: 'A damping zone', of: (s) => count(s, (p) => p.kind === 'disk' && !!p.damp) > 0 },
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

const ENERGY: Rule[] = [{ fact: 'eos.energy' }];
const SELF: Rule[] = [{ fact: 'gravity.self' }];
const ADAPTIVE: Rule[] = [{ fact: 'mesh.adaptive' }];
const IN_3D: Rule[] = [{ fact: 'domain.ndim', is: 3 }];
const NOT_PERIODIC: Rule[] = [{ fact: 'bc.periodic', reason: 'Not available with a periodic face: its gravity is not periodic' }];

export const BC_CHOICES: Choice[] = [
  { value: 0, label: 'Periodic', tip: 'Gas leaving through this face comes back in through the opposite one (both faces of an axis)', excludes: [{ fact: 'physics.gravity', is: 'isolated' }, { fact: 'parts.body', reason: 'Not available with a point mass, a star or a disk: their gravity is not periodic' }] },
  { value: 1, label: 'Outflow', tip: 'Open: the state at the face continues beyond it, so gas may leave or enter' },
  { value: 2, label: 'Reflecting', tip: 'A solid wall' },
  { value: 3, label: 'Outflow only', tip: 'Open for gas leaving; nothing comes in' },
];

export const OPTIONS: Option[] = [
  // ---- run: what the setup is called, and its clock ----
  { path: 'name', label: 'Name', tip: 'What the setup is called in the preset list and in its file', page: 'run', control: { type: 'text' }, apply: 'live' },
  { path: 'blurb', label: 'Description', tip: 'A sentence or two shown under the preset menu', page: 'run', control: { type: 'text', long: true }, apply: 'live' },
  { path: 'run.tEnd', label: 'End time', tip: 'The run pauses when it reaches this time. Empty runs on without end', page: 'run', control: num(0, undefined, { nullable: true, factor: 1.25, quantity: 'time' }), apply: 'live', default: null, linkKey: 'tend', cfg: 'tEnd' },
  { path: 'run.checkpointDt', label: 'Checkpoint every', tip: 'Time between checkpoints, when the page is asked to save them: the value the Time panel starts with', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'time' }), apply: 'live' },
  { path: 'run.tDyn', label: 'Dynamical time', tip: 'The time over which the flow visibly changes (a star’s sqrt(R³ / G M), a binary’s period): a movie takes a frame every 600th of it by default', page: 'run', control: num(0, undefined, { factor: 2, quantity: 'time' }), apply: 'live' },
  { path: 'run.startPaused', label: 'Start paused', tip: 'The run waits, paused, for Run', page: 'run', control: check, apply: 'live', default: false },
  { path: 'view.mode', label: 'Opening view', tip: 'How the gas is drawn when the setup opens in 3D', page: 'run', control: pick(['volume', 'Volume'], ['slice', 'Slice'], ['projection', 'Projection']), apply: 'live', default: 'volume' },
  { path: 'view.field', label: 'Opening field', tip: 'The quantity shown when the setup opens (the View panel’s Field, by its number)', page: 'run', control: num(0, 29, { integer: true, step: 1 }), apply: 'live', default: 0 },
  { path: 'view.log', label: 'Logarithmic colors', tip: 'The color scale the setup opens with', page: 'run', control: check, apply: 'live', default: true },
  { path: 'view.axis', label: 'Slice normal', tip: 'The axis the opening slice is perpendicular to (0 x, 1 y, 2 z)', page: 'run', control: pick([2, 'z'], [1, 'y'], [0, 'x']), apply: 'live', default: 2 },
  { path: 'view.pitch', label: 'Camera elevation', tip: 'Elevation of the volume view’s camera when the setup opens, in radians', page: 'run', control: num(-1.5, 1.5, { step: 0.05 }), apply: 'live', default: 0.45 },

  // ---- domain and grid ----
  { path: 'domain.ndim', label: 'Dimensions', tip: 'Two or three dimensions. A 2D run is one layer of cells in x and y', page: 'domain', control: pick({ value: 3, label: '3D' }, { value: 2, label: '2D', requires: [{ fact: 'domain.supports2d', reason: 'This setup is three-dimensional (its file says it does not make sense in 2D)' }], excludes: [{ fact: 'gravity.self' }, { fact: 'parts.star' }] }), apply: 'restart', linkKey: 'dim' },
  { path: 'domain.n', label: 'Resolution', tip: 'Cells across the box; on an adaptive mesh, at its finest level', page: 'domain', control: num(16, 16384, { integer: true, factor: 2 }), apply: 'restart', linkKey: 'n' },
  { path: 'domain.amr', label: 'Refinement', tip: 'Adaptive mesh: levels of refinement below the coarsest grid. 0 puts every cell at the full resolution', page: 'domain', control: num(0, 5, { integer: true, step: 1 }), apply: 'restart', linkKey: 'amr' },
  { path: 'domain.box', label: 'Box size', tip: 'Side of the box, in code units: the domain is a cube of this side centered on the origin', page: 'domain', control: num(1e-12, 1e30, { factor: 2, quantity: 'length' }), apply: 'restart', default: 1 },
  { path: 'domain.supports2d', label: 'Makes sense in 2D', tip: 'The setup can also run on a 2D (x, y) grid', page: 'domain', control: check, apply: 'live', default: true },
  { path: 'domain.units', label: 'Physical units', tip: 'What one code unit is, for a setup that stands for something real: a length and a mass (the time unit then makes G = 1), or a length, a density and a velocity. Without them the setup is scale free and shown in code units', page: 'domain', apply: 'restart',
    control: { type: 'group', optional: true, fields: [
      { path: 'name', label: 'Name', tip: 'A name for the unit system', page: 'domain', control: { type: 'text' }, apply: 'live', default: 'custom' },
      { path: 'length.value', label: 'Length', tip: 'This many of the unit beside it are one code unit of length', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'length.unit', label: 'Length unit', tip: 'The unit of the length', page: 'domain', control: pick(...LENGTH_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'Rsun' },
      { path: 'mass.value', label: 'Mass', tip: 'This many of the unit beside it are one code unit of mass', page: 'domain', control: num(1e-300, 1e300, { factor: 2 }), apply: 'restart', default: 1 },
      { path: 'mass.unit', label: 'Mass unit', tip: 'The unit of the mass', page: 'domain', control: pick(...MASS_UNITS.map((u): [Value, string] => [u, u])), apply: 'restart', default: 'Msun' },
    ] } },

  // ---- boundaries ----
  { path: 'domain.bc', label: 'Faces of the box', tip: 'What the gas meets at each face of the box. Periodic faces come in pairs', page: 'boundaries', control: { type: 'faces' }, apply: 'live', linkKey: 'bc', cfg: 'bc' },

  // ---- gas ----
  { path: 'physics.eos', label: 'Equation of state', tip: 'How pressure follows from density and internal energy. Changing it carries each cell’s pressure over, and the run continues', page: 'gas', apply: 'shaders', default: 'ideal', linkKey: 'eos', cfg: 'eos',
    control: pick(...(Object.keys(EOS_LABELS) as (keyof typeof EOS_LABELS)[]).map((k): Choice => ({ value: k, label: EOS_LABELS[k], ...(k === 'barotropic' ? { excludes: [{ fact: 'parts.star.polytrope', reason: 'Not available with a polytropic star: it needs an equation of state with an energy' }] } : {}) }))) },
  { path: 'physics.eosParams.gamma', label: 'gamma', tip: 'Adiabatic index γ: p = (γ - 1) ρ e for an ideal gas, p = K ρ^γ for a barotropic one', page: 'gas', control: num(1, 100, { step: 0.05 }), apply: 'live', default: 5 / 3, linkKey: 'gamma', cfg: 'eosParams.gamma' },
  { path: 'physics.eosParams.K', label: 'K (p = K rho^gamma)', tip: 'Barotropic constant K: p = K ρ^γ. With γ = 1 it is the square of the sound speed', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', linkKey: 'K', cfg: 'eosParams.K', show: { path: 'physics.eos', is: ['barotropic'] } },
  { path: 'physics.eosParams.arad', label: 'Radiation constant a', tip: 'Radiation pressure a T⁴ / 3 is added to the gas pressure', page: 'gas', control: num(0, undefined, { factor: 2 }), apply: 'live', linkKey: 'arad', cfg: 'eosParams.arad', show: { path: 'physics.eos', is: ['gas_radiation', 'degenerate', 'ionizing'] } },
  { path: 'physics.eosParams.degRho', label: 'Degeneracy density', tip: 'Density at which the electrons’ Fermi momentum reaches mₑc. For a setup with physical units it is set from the composition (about 2×10⁶ g/cm³ for C/O); in code units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'degrho', cfg: 'eosParams.degRho', show: { path: 'physics.eos', is: ['degenerate'] } },
  { path: 'physics.eosParams.degP', label: 'Degeneracy pressure scale', tip: 'The scale π mₑ⁴ c⁵ / 3h³ of the degenerate electron pressure. Set from physical constants for a setup with physical units; in code units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'degp', cfg: 'eosParams.degP', show: { path: 'physics.eos', is: ['degenerate'] } },
  { path: 'physics.eosParams.ionT', label: 'Ionization temperature', tip: 'The ionization energy of hydrogen, 13.6 eV, over k, in code temperature. Set from physical constants for a setup with physical units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'iont', cfg: 'eosParams.ionT', show: { path: 'physics.eos', is: ['ionizing'] } },
  { path: 'physics.eosParams.ionRho', label: 'Saha density', tip: 'Density scale of the Saha equation in code density: denser gas needs a higher temperature to ionize. Set from physical constants for a setup with physical units', page: 'gas', control: num(1e-30, undefined, { factor: 2 }), apply: 'live', linkKey: 'ionrho', cfg: 'eosParams.ionRho', show: { path: 'physics.eos', is: ['ionizing'] } },
  { path: 'physics.composition', label: 'Composition', tip: 'What the gas is made of, which sets the kelvin scale and the physical constants of the equation of state: solar (fully ionized), solarNeutral, hydrogen or carbonOxygen', page: 'gas', control: pick(['solar', 'Solar, ionized'], ['solarNeutral', 'Solar, neutral'], ['hydrogen', 'Hydrogen'], ['carbonOxygen', 'Carbon and oxygen']), apply: 'restart', default: 'solar' },
  { path: 'physics.species', label: 'Species', tip: 'Species whose mass fractions the gas carries, by name (he4, c12, ...), separated by commas; or aprox13 for the burning network’s thirteen. None by default', page: 'gas', control: { type: 'names' }, apply: 'restart' },
  { path: 'physics.burn', label: 'Burning', tip: 'Nuclear burning by the trained network of the 13-isotope alpha chain (aprox13), wherever the gas is hot and dense enough', page: 'gas', apply: 'restart',
    requires: [{ fact: 'eos.energy' }, { fact: 'species.aprox13' }, { fact: 'units' }],
    control: { type: 'group', optional: true, fields: [
      { path: 'network', label: 'Network', tip: 'The burning network', page: 'gas', control: pick(['aprox13', 'aprox13 (13 isotopes)']), apply: 'restart', default: 'aprox13' },
      { path: 'inShocks', label: 'Burn inside shocks', tip: 'Let cells the shock detector flags burn. Off, as in most supernova codes: a captured shock is a few cells of numerical smearing', page: 'gas', control: check, apply: 'restart', default: false },
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
  { path: 'physics.viscosity', label: 'Viscosity ν', tip: 'Kinematic viscosity: the stress is ρ ν times the shear. Obstacles hold the gas at their walls. 0 is off', page: 'gas', control: num(0, 1e30, { factor: 1.25 }), apply: 'live', default: 0, linkKey: 'nu', cfg: 'viscosity' },
  { path: 'physics.conductivity', label: 'Conductivity κ', tip: 'Thermal conduction: the heat flux is -κ ρ grad T, so the temperature of an ideal gas diffuses at (γ - 1) κ. 0 is off', page: 'gas', control: num(0, 1e30, { factor: 1.25 }), apply: 'live', default: 0, linkKey: 'kappa', cfg: 'conductivity', requires: ENERGY },

  // ---- gravity ----
  { path: 'physics.gravity', label: 'Gravity', tip: 'Off, a uniform field, or the self-gravity of the gas. Point masses pull either way', page: 'gravity', apply: 'live', default: 'off', linkKey: 'gravity', cfg: 'gravity',
    control: pick(
      { value: 'off', label: 'Off', tip: 'No gravity (point masses still pull)' },
      { value: 'constant', label: 'Constant', tip: 'A uniform field of strength G pointing down: -y in 2D, -z in 3D' },
      { value: 'periodic', label: 'Self, periodic', tip: 'Self-gravity of the gas in a periodic box (3D only)', requires: [...IN_3D, { fact: 'bc.allPeriodic' }], excludes: [{ fact: 'parts.star', reason: 'Not available with a star: a star’s gravity is not periodic' }] },
      { value: 'isolated', label: 'Self, isolated', tip: 'Self-gravity of the gas, isolated: the potential falls off outside the box (3D only)', requires: IN_3D, excludes: [{ fact: 'bc.periodic' }] },
    ) },
  { path: 'physics.G', label: 'G', tip: 'Gravitational constant in code units, or the strength of the uniform field. A setup with physical units may leave it to them', page: 'gravity', control: num(0, undefined, { factor: 1.5 }), apply: 'live', default: 1, linkKey: 'G', cfg: 'G' },
  { path: 'physics.gext', label: 'Uniform acceleration', tip: 'A uniform acceleration on the gas, besides the Constant field', page: 'gravity', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] },
  { path: 'physics.vcycles', label: 'V-cycles', tip: 'Multigrid V-cycles per step for self-gravity. More converge the potential further, at more cost. On the adaptive mesh this is the most per step', page: 'gravity', control: num(1, 8, { integer: true, step: 1 }), apply: 'live', linkKey: 'vcycles', cfg: 'vcycles', requires: SELF },
  { path: 'physics.jeansFloor', label: 'Jeans floor', tip: 'Pressure floor that keeps the Jeans length resolved (Truelove)', page: 'gravity', control: check, apply: 'live', default: false, linkKey: 'jfloor', cfg: 'jeansFloor', requires: SELF },
  { path: 'physics.jeansCells', label: 'Cells per Jeans length', tip: 'The Jeans floor raises the pressure so the Jeans length spans at least this many cells', page: 'gravity', control: num(4, 32, { integer: true, step: 1 }), apply: 'live', default: 4, linkKey: 'jcells', cfg: 'jeansCells', requires: SELF },
  { path: 'physics.gravEnergyFlux', label: 'Conserve energy', tip: 'Flux-form gravitational work: conserves total energy', page: 'gravity', control: check, apply: 'live', default: true, linkKey: 'fluxwork', cfg: 'gravEnergyFlux', requires: SELF },
  { path: 'physics.frame', label: 'Frame', tip: 'An inertial frame, or one that accelerates with the gas’s center of mass or the point masses’, so that what the run follows stays in the box', page: 'gravity', apply: 'restart', default: 'inertial',
    control: pick(['inertial', 'Inertial'], { value: 'gas', label: 'Follows the gas', requires: [{ fact: 'parts.mass' }] }, { value: 'masses', label: 'Follows the masses', requires: [{ fact: 'parts.mass' }] }) },

  // ---- mesh ----
  { path: 'mesh.refineHi', label: 'Refine above', tip: 'Adaptive mesh: a block is split where the normalized second derivative of density or pressure exceeds this anywhere in it. The indicator is near 1 at a shock or contact and small where the gas varies smoothly. Lower refines more', page: 'mesh', control: num(0.05, 1, { step: 0.05 }), apply: 'live', default: 0.8, linkKey: 'refhi', cfg: 'refineHi', requires: ADAPTIVE },
  { path: 'mesh.refineLo', label: 'Merge below', tip: 'Adaptive mesh: four sibling blocks are merged where the indicator is below this in all of them. Kept below the refinement threshold', page: 'mesh', control: num(0, 1, { step: 0.05 }), apply: 'live', default: 0.2, linkKey: 'reflo', cfg: 'refineLo', requires: ADAPTIVE },
  { path: 'mesh.refineRho', label: 'Refine above density', tip: 'Adaptive mesh: gas thinner than this does not ask for refinement, so ripples in the ambient gas far from a star leave its blocks coarse. 0 is off', page: 'mesh', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'live', linkKey: 'refrho', cfg: 'refineRho', requires: ADAPTIVE },
  { path: 'mesh.refineDense', label: 'Always refine above density', tip: 'Adaptive mesh: gas denser than this is always at the finest level, as the smooth interior of a star must be. 0 is off', page: 'mesh', control: num(0, undefined, { factor: 10, quantity: 'density' }), apply: 'restart', default: 0, requires: ADAPTIVE },
  { path: 'mesh.refineMasses', label: 'Refine around point masses', tip: 'Adaptive mesh: keep the gas around every point mass at the finest level, so that softening and sinks can be as small as the finest cells allow', page: 'mesh', control: check, apply: 'restart', default: false, requires: [...ADAPTIVE, { fact: 'parts.mass' }] },
  { path: 'mesh.timeBins', label: 'Local timesteps', tip: 'Adaptive mesh: blocks of slow gas take longer steps than the fastest block, where that saves work', page: 'mesh', apply: 'shaders', default: 0, linkKey: 'tbins', cfg: 'timeBins', requires: ADAPTIVE,
    control: pick([0, 'Off'], [1, 'Up to 2× longer'], [2, 'Up to 4× longer'], [3, 'Up to 8× longer']) },
  { path: 'mesh.jeansRefine', label: 'Refine the Jeans length', tip: 'With self-gravity, refine where the Jeans length is under this many cells. 0 is off', page: 'mesh', control: num(0, 64, { integer: true, step: 1 }), apply: 'restart', default: 16, requires: [...ADAPTIVE, ...SELF] },

  // ---- solver ----
  { path: 'physics.riemann', label: 'Riemann solver', tip: 'How the flux between neighboring cells is worked out. HLLC resolves contact waves; its low-Mach version cuts dissipation in slow flow; HLL is the most diffusive and the most robust', page: 'solver', control: pick(['hllc', 'HLLC'], ['hllclm', 'HLLC, low Mach'], ['hll', 'HLL']), apply: 'shaders', default: 'hllc', linkKey: 'riemann', cfg: 'riemann' },
  { path: 'physics.limiter', label: 'Limiter', tip: 'How steeply the reconstruction may vary within a cell: Koren is sharpest, Minmod most diffusive and most robust', page: 'solver', control: pick(['koren', 'Third order (Koren)'], ['mc', 'Monotonized central'], ['vanleer', 'van Leer'], ['minmod', 'Minmod']), apply: 'shaders', default: 'koren', linkKey: 'limiter', cfg: 'limiter' },
  { path: 'physics.cfl', label: 'CFL number', tip: 'Timestep as a fraction of the time a signal takes to cross a cell. Smaller is more stable and slower', page: 'solver', control: num(0.05, 1, { step: 0.05 }), apply: 'live', default: 0.8, linkKey: 'cfl', cfg: 'cfl' },

  // ---- the background the parts are painted on ----
  { path: 'background.rho', label: 'Ambient density', tip: 'Density of the gas that fills the box where no part puts anything else. With stars, the gas around them', page: 'parts', control: num(1e-300, undefined, { factor: 10, quantity: 'density' }), apply: 'restart', default: 1, excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.p', label: 'Ambient pressure', tip: 'Pressure of that gas. With stars and left empty, the pressure of the stars’ outermost layer', page: 'parts', control: num(0, undefined, { factor: 10 }), apply: 'restart', excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.v', label: 'Ambient velocity', tip: 'Velocity of that gas', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0], excludes: [{ fact: 'parts.flow.whole', reason: 'Not used with a flow that fills the box' }] },
  { path: 'background.X', label: 'Ambient composition', tip: 'Mass fractions of that gas, by species name', page: 'parts', control: { type: 'fractions' }, apply: 'restart' },
  { path: 'background.cut', label: 'Stars end at density', tip: 'With stars: a cell thinner than this is ambient gas, not star. Left empty, a polytrope ends where the ambient gas would be as hot as its center', page: 'parts', control: num(0, undefined, { factor: 10 }), apply: 'restart', requires: [{ fact: 'parts.star' }] },
];

const POS: Option = { path: 'pos', label: 'Position', tip: 'Where it starts, relative to the box center', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] };
const VEL: Option = { path: 'vel', label: 'Velocity', tip: 'How it moves at the start', page: 'parts', control: { type: 'vec3' }, apply: 'restart', default: [0, 0, 0] };
const P = <T extends Partial<Option>>(o: T & Pick<Option, 'path' | 'label' | 'tip' | 'control'>): Option => ({ page: 'parts', apply: 'restart', ...o });

const SHAPE: Option[] = [
  P({ path: 'shape.type', label: 'Shape', tip: 'A sphere (a cylinder in 2D), a box, or the half of space on one side of a plane', control: pick(['sphere', 'Sphere'], ['box', 'Box'], ['half', 'Half-space']), default: 'sphere' }),
  P({ path: 'shape.center', label: 'Center', tip: 'Center of the sphere', control: { type: 'vec3' }, default: [0, 0, 0], show: { path: 'shape.type', is: ['sphere'] } }),
  P({ path: 'shape.radius', label: 'Radius', tip: 'Radius of the sphere', control: num(0, undefined, { factor: 1.25, quantity: 'length' }), default: 0.1, show: { path: 'shape.type', is: ['sphere'] } }),
  P({ path: 'shape.radiusCells', label: 'Radius in cells', tip: 'Instead of the radius: this many finest cells, whatever the resolution (a point explosion)', control: num(0.5, 64, { step: 0.5 }), show: { path: 'shape.type', is: ['sphere'] } }),
  P({ path: 'shape.min', label: 'Lower corner', tip: 'Corner of the box nearest -x, -y, -z', control: { type: 'vec3' }, default: [-0.1, -0.1, -0.1], show: { path: 'shape.type', is: ['box'] } }),
  P({ path: 'shape.max', label: 'Upper corner', tip: 'Corner of the box nearest +x, +y, +z', control: { type: 'vec3' }, default: [0.1, 0.1, 0.1], show: { path: 'shape.type', is: ['box'] } }),
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
      P({ path: 'massless', label: 'Massless', tip: 'A particle without a mass of its own: it pulls on nothing and moves as the gas and the masses pull it', control: check, default: false }),
      P({ path: 'fixed', label: 'Held in place', tip: 'Nothing moves it, whatever pulls on it or it takes in', control: check, default: false }),
    ],
  },
  orbit: {
    kind: 'orbit', label: 'Orbit', tip: 'Not an object: puts two bodies on a Kepler orbit', requires: [{ fact: 'parts.body' }],
    make: () => ({ kind: 'orbit', of: ['', ''], type: 'binary', separation: 1, ecc: 0 }),
    fields: [
      P({ path: 'of', label: 'Bodies', tip: 'The two bodies it places: two point masses (binary), a star and a point mass (flyby), or two stars (collision)', control: { type: 'ref', kinds: ['mass', 'star'], count: 2 } }),
      P({ path: 'type', label: 'Kind of orbit', tip: 'Binary: two point masses about their center of mass, at pericenter. Flyby: a point mass on a parabola about a star that stays at the origin (a tidal disruption in the star’s frame). Collision: two stars on a parabola about their center of mass', control: pick(['binary', 'Binary (two masses)'], ['flyby', 'Flyby (star and mass)'], ['collision', 'Collision (two stars)']), default: 'binary' }),
      P({ path: 'q', label: 'Mass ratio', tip: 'Mass of the second body over the first. It sets their masses: in a binary the total stays, in a flyby the star’s', control: num(1e-6, 1e9, { factor: 1.25 }), show: { path: 'type', is: ['binary', 'flyby'] } }),
      P({ path: 'separation', label: 'Separation', tip: 'Semi-major axis of the binary’s orbit', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 1, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'ecc', label: 'Eccentricity', tip: 'Eccentricity of the binary’s orbit', control: num(0, 0.95, { step: 0.05 }), default: 0, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'incl', label: 'Inclination (°)', tip: 'Tilt of the binary’s orbit about the line through pericenter, in degrees (3D): 180 is retrograde', control: num(0, 180, { step: 1 }), default: 0, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'retrograde', label: 'Retrograde', tip: 'In 2D: the binary orbits clockwise', control: check, default: false, show: { path: 'type', is: ['binary'] } }),
      P({ path: 'beta', label: 'Impact parameter β', tip: 'Tidal radius over pericenter distance: above 1 the mass passes inside the star’s tidal radius', control: num(0.05, 20, { step: 0.1 }), default: 1, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'start', label: 'Starts at (tidal radii)', tip: 'Separation at the start, in tidal radii (never inside 1.5 pericenters)', control: num(1, 20, { step: 0.5 }), default: 3, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'debris', label: 'Debris energies', tip: 'Bin the star’s gas by its orbital energy about the mass, for the rate at which bound debris returns', control: check, default: false, show: { path: 'type', is: ['flyby'] } }),
      P({ path: 'impact', label: 'Impact parameter', tip: 'Pericenter distance over the sum of the two radii: 0 is head-on, 1 just grazes', control: num(0, 5, { step: 0.1 }), default: 0, show: { path: 'type', is: ['collision'] } }),
      P({ path: 'starts', label: 'Starts at (sums of radii)', tip: 'Separation at the start, in sums of the two radii (closer if they would not fit in the box)', control: num(1.05, 20, { step: 0.5 }), default: 4, show: { path: 'type', is: ['collision'] } }),
    ],
  },
  star: {
    kind: 'star', label: 'Star', tip: 'A polytrope or a cold white dwarf, settled into the grid’s own hydrostatic balance before the run', max: 4,
    requires: [...IN_3D, { fact: 'physics.gravity', is: 'isolated', reason: 'Needs isolated self-gravity: it is what holds a star together' }],
    excludes: NOT_PERIODIC,
    make: () => ({ kind: 'star', type: 'polytrope', n: 1.5, mass: 1, radius: 0.2 }),
    fields: [
      P({ path: 'type', label: 'Kind of star', tip: 'A polytrope of index n, or a cold white dwarf held up by degenerate electrons', control: pick(
        { value: 'polytrope', label: 'Polytrope', requires: [{ fact: 'eos.energy' }], excludes: [{ fact: 'parts.star.white-dwarf', reason: 'Not available with a white dwarf: the two are settled with different gases' }] },
        { value: 'white-dwarf', label: 'White dwarf', requires: [{ fact: 'eos.degenerate' }], excludes: [{ fact: 'parts.star.polytrope', reason: 'Not available with a polytropic star: the two are settled with different gases' }] }), default: 'polytrope' }),
      P({ path: 'n', label: 'Polytropic index n', tip: 'The star’s structure, p = K ρ^(1 + 1/n): the larger n, the more centrally concentrated. Stable against convection only if the gas’s adiabatic index is at least 1 + 1/n', control: num(0.5, 4.5, { step: 0.5 }), default: 1.5, show: { path: 'type', is: ['polytrope'] } }),
      P({ path: 'mass', label: 'Mass', tip: 'Its mass, in code units (solar masses for a white dwarf in its usual units)', control: num(1e-12, undefined, { factor: 1.25, quantity: 'mass' }), default: 1 }),
      P({ path: 'radius', label: 'Radius', tip: 'A polytrope’s radius, in code units. A white dwarf’s follows from its mass', control: num(1e-12, undefined, { factor: 1.25, quantity: 'length' }), default: 0.2, show: { path: 'type', is: ['polytrope'] } }),
      P({ path: 'rhoc', label: 'Central density', tip: 'Instead of the mass: a polytrope’s central density', control: num(1e-300, undefined, { factor: 2, quantity: 'density' }), show: { path: 'type', is: ['polytrope'] } }),
      POS, VEL,
      P({ path: 'dye', label: 'Dye', tip: 'Passive scalar of its gas, which tells the stars apart: spread over (0, 1] by their order if left empty', control: num(0, 1, { step: 0.1 }) }),
      P({ path: 'X', label: 'Composition', tip: 'Mass fractions of its gas, by species name (the ambient gas’s if left empty)', control: { type: 'fractions' } }),
      P({ path: 'field', label: 'Magnetic field inside', tip: 'A field confined to the star: a dipole along z or a tangled Kolmogorov field, with plasma beta at the center; or the same as Guillochon & McCourt (2017) set them, by their β_M',
        control: { type: 'group', optional: true, fields: [
          P({ path: 'type', label: 'Shape', tip: 'Dipole or random, by the plasma beta at the center; or Guillochon & McCourt’s, by β_M', control: pick(['dipole', 'Dipole'], ['random', 'Random'], ['gm17-dipole', 'Dipole (G&M 2017)'], ['gm17-random', 'Random (G&M 2017)']), default: 'dipole' }),
          P({ path: 'beta', label: 'Plasma beta', tip: 'Gas pressure over magnetic pressure B² / 2 at the center (or β_M): the larger, the weaker the field', control: num(1, 1e12, { factor: 10 }), default: 1000 }),
          P({ path: 'seed', label: 'Random seed', tip: 'Which random field: each seed gives another', control: num(1, 9999, { integer: true, step: 1 }), default: 1 }),
        ] } }),
    ],
  },
  disk: {
    kind: 'disk', label: 'Disk', tip: 'A gas disk in the x-y plane about one or more bodies', requires: [{ fact: 'parts.mass', reason: 'Needs a point mass to orbit' }],
    excludes: NOT_PERIODIC,
    make: () => ({ kind: 'disk', about: [], profile: 'balanced', rIn: 0.3, rOut: 1, sigma0: 1e-3, slope: -1.5, cs: 0.1, floor: 1e-8 }),
    fields: [
      P({ path: 'about', label: 'Orbits', tip: 'The point masses it orbits: about several, their total mass at their center of mass', control: { type: 'ref', kinds: ['mass'], count: 'any' } }),
      P({ path: 'profile', label: 'Profile', tip: 'Thin: a surface density with a cavity and a Gaussian vertical profile, on circular orbits (the binary’s disk). Balanced: a midplane density in vertical balance with the softened potential, rotating as its pressure gradient asks (the magnetized disk); 3D only', control: pick(['thin', 'Thin, with a cavity'], { value: 'balanced', label: 'Balanced', requires: IN_3D }), default: 'balanced' }),
      P({ path: 'rIn', label: 'Inner radius', tip: 'Where the disk is tapered away toward the center (the cavity’s radius)', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 0.3 }),
      P({ path: 'rOut', label: 'Outer radius', tip: 'Where the disk is tapered away outward', control: num(1e-6, undefined, { factor: 1.25, quantity: 'length' }), default: 1 }),
      P({ path: 'sigma0', label: 'Density scale', tip: 'Thin: the surface density. Balanced: the midplane density at r = 1', control: num(1e-300, undefined, { factor: 2 }), default: 1e-3 }),
      P({ path: 'slope', label: 'Density slope', tip: 'Balanced: the midplane density goes as r to this power (-3/2 is a flat surface density)', control: num(-4, 2, { step: 0.25 }), default: -1.5, show: { path: 'profile', is: ['balanced'] } }),
      P({ path: 'cs', label: 'Sound speed', tip: 'Sound speed of the disk’s gas: its scale height is cs / Ω', control: num(1e-6, undefined, { factor: 1.25, quantity: 'velocity' }), default: 0.1 }),
      P({ path: 'floor', label: 'Gas outside the disk', tip: 'Thin: a fraction of the density scale. Balanced: the density of the ambient gas', control: num(0, undefined, { factor: 10 }), default: 1e-8 }),
      P({ path: 'damp', label: 'Damping zone', tip: 'Beyond a radius the gas is relaxed toward the disk as it started, more strongly toward the box’s edge, so the outer disk is a reservoir and the boundaries see only it',
        excludes: [{ fact: 'parts.damping', reason: 'A run has one damping zone: another disk already has it' }],
        control: { type: 'group', optional: true, fields: [
          P({ path: 'from', label: 'Damping from r', tip: 'Radius from the box center where the damping zone starts', control: num(0, 1e30, { step: 0.25, quantity: 'length' }), default: 1, apply: 'live', linkKey: 'dampr', cfg: 'dampRadius' }),
          P({ path: 'rate', label: 'Damping rate', tip: 'How many times per local orbital period the gas at the box edge is relaxed toward its initial state. 0 is off', control: num(0, 1e30, { step: 0.5 }), default: 3, apply: 'live', linkKey: 'damprate', cfg: 'dampRate' }),
        ] } }),
      P({ path: 'loops', label: 'Field loops', tip: 'Poloidal loops of magnetic field inside the disk, with no net flux: the seed of the magnetorotational instability (balanced profile)',
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
      P({ path: 'supersample', label: 'Smooth edge (samples)', tip: 'A sphere’s edge cells take the share of their volume inside it, from this many samples a side. Empty: a sharp edge', control: num(2, 8, { integer: true, step: 1 }), show: { path: 'shape.type', is: ['sphere'] } }),
    ],
  },
  obstacle: {
    kind: 'obstacle', label: 'Obstacle', tip: 'A solid body the gas reflects off',
    make: () => ({ kind: 'obstacle', shape: { type: 'box', min: [-0.1, -0.1, -0.1], max: [0.1, 0.1, 0.1] } }),
    fields: [...SHAPE],
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
      control: pick(...Object.values(FLOWS).map((f): Choice => ({ value: f.name, label: f.label, tip: f.tip, ...(f.wholeBox ? { excludes: [{ fact: 'parts.flow.whole', reason: 'Only one flow can fill the whole box' }] } : {}) }))) }),
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
      if (o.control.type === 'group') visit(o.control.fields, v, `${prefix}${o.path}.`, name, facts);
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
