// Reading and writing setup documents: files, the browser's library and the `?setup=` of a
// link. Reading never trusts: unknown keys are dropped and listed, values out of range are held
// to it and listed, a wrong type is an error (the discipline of ui/link.ts decodeLink).
//
// `version` counts changes to the document's shape. MIGRATIONS keeps one step per version and
// applies them in order; a file from a newer version loads with a warning. The rule of
// PROBLEM_ALIASES applies: add, never remove.
import { BOUNDARY_CODES } from '../problems/boundaries';
import { resolveExtends } from './extends';
import { FLOWS } from './parts/flow';
import { flatten, OPTION_BY_PATH, PART_OPTIONS, type Option } from './options';
import { SETUP_FORMAT, SETUP_VERSION, type SetupSpec } from './spec';
import { UNIT_SI } from './units';

/** What reading a document could not use as written. */
export interface SetupReport {
  /** why the document cannot be read at all */
  error?: string;
  /** keys dropped, values held to their range, a newer version */
  notes: string[];
}

/** One step per version: MIGRATIONS[v] takes a document of version v to v + 1. */
export const MIGRATIONS: Record<number, (doc: Record<string, unknown>) => Record<string, unknown>> = {};

const MAX_TEXT = 4000;
const MAX_PARTS = 256;
/** Where the JSON Schema of a setup document is served (public/shaderous-setup.schema.json): files this build writes name it, so an editor checks them and completes their keys. */
export const SETUP_SCHEMA_URL = 'https://guillochon.github.io/shaderous/shaderous-setup.schema.json';
const MAX_DOCUMENT = 2_000_000;

class Bad extends Error {}

type Schema =
  | 'number' | 'int' | 'string' | 'boolean' | 'vec3' | 'pair' | 'fractions' | 'numbers'
  | { enum: readonly (string | number)[] }
  | { object: Record<string, Schema>; open?: Schema }
  | { array: Schema; max: number }
  | { any: Schema[] }
  | { nullable: Schema };

const obj = (object: Record<string, Schema>): Schema => ({ object });
/** The choices of a select in the option table: the one list of what each such key may hold. */
function choicesOf(o: Option | undefined, name: string): Schema {
  if (o?.control.type !== 'select') throw new Error(`${name} is not a choice in the option table`);
  return { enum: o.control.choices.map((c) => c.value as string | number) };
}
const choice = (path: string): Schema => choicesOf(OPTION_BY_PATH[path], path);
const partChoice = (kind: keyof typeof PART_OPTIONS, path: string): Schema => choicesOf(flatten(PART_OPTIONS[kind].fields).find((o) => o.path === path), `${kind}.${path}`);
const BC: Schema = { array: { enum: [...BOUNDARY_CODES] }, max: 3 };
const FACES = obj({ lo: BC, hi: BC });
const UNIT = obj({ unit: { enum: Object.keys(UNIT_SI) }, value: 'number', per: { any: ['number', 'string'] } });
const SHAPE: Schema = obj({
  type: partChoice('region', 'shape.type'), center: 'vec3', radius: 'number', radiusCells: 'number',
  min: { array: { nullable: 'number' }, max: 3 }, max: { array: { nullable: 'number' }, max: 3 }, normal: 'vec3', at: 'number', rotate: 'vec3',
  // A mesh file (geometry/store.ts).
  file: 'string', node: 'string', scale: 'number', fit: 'number', up: partChoice('region', 'shape.up'), anchor: partChoice('region', 'shape.anchor'),
});
const ID: Schema = 'string';
/** A number or a formula of t. */
const AMOUNT: Schema = { any: ['number', 'string'] };
const AMOUNTS: Schema = { array: AMOUNT, max: 3 };
/** An obstacle's motion (geometry/motion.ts). */
const MOTION: Schema = obj({
  spin: obj({ axis: 'vec3', about: 'vec3', rate: AMOUNT }),
  slide: obj({ velocity: AMOUNTS }),
  swing: obj({ axis: 'vec3', about: 'vec3', amplitude: 'number', period: 'number', phase: 'number' }),
  path: obj({ pos: AMOUNTS, rotate: AMOUNTS, about: 'vec3' }),
});
/** A tabulated profile: up to 32 points (x, value). */
const PROFILE: Schema = { array: 'pair', max: 32 };

const PART_SCHEMAS: Record<string, Record<string, Schema>> = {
  mass: { gm: 'number', pos: 'vec3', vel: 'vec3', soft: 'number', sink: 'number', sinkRate: 'number', massless: 'boolean', fixed: 'boolean', onOrbit: 'boolean' },
  orbit: { of: { array: 'string', max: 2 }, type: partChoice('orbit', 'type'), q: 'number', separation: 'number', ecc: 'number', incl: 'number', retrograde: 'boolean', fill: 'number', corotating: 'boolean', track: 'boolean', release: 'boolean', beta: 'number', start: 'number', impact: 'number', starts: 'number', debris: 'boolean' },
  star: { type: partChoice('star', 'type'), n: 'number', radius: 'number', mass: 'number', rhoc: 'number', core: obj({ mass: 'number', soft: 'number' }), material: partChoice('star', 'material'), coreMaterial: partChoice('star', 'coreMaterial'), coreFraction: 'number', surfaceT: 'number', pos: 'vec3', vel: 'vec3', dye: 'number', X: 'fractions', field: obj({ type: partChoice('star', 'field.type'), beta: 'number', seed: 'number' }) },
  disk: { about: { array: 'string', max: 32 }, profile: partChoice('disk', 'profile'), radial: PROFILE, vertical: PROFILE, radialFn: 'string', verticalFn: 'string', verticalIn: partChoice('disk', 'verticalIn'), rIn: 'number', rOut: 'number', sigma0: 'number', slope: 'number', cs: 'number', floor: 'number', damp: obj({ from: 'number', rate: 'number' }), loops: obj({ beta: 'number', span: 'number', count: 'int' }), stresses: 'boolean' },
  region: { shape: SHAPE, rho: 'number', p: 'number', energy: 'number', v: 'vec3', s: 'number', X: 'fractions', erad: 'number', frad: 'vec3', supersample: 'int' },
  obstacle: { shape: SHAPE, motion: MOTION, parent: 'string', refine: partChoice('obstacle', 'refine'), refineBand: 'number', refineLevel: 'int' },
  inflow: { shape: SHAPE },
  field: { b: 'vec3', up: 'number' },
  tunnel: { speed: 'number', axis: partChoice('tunnel', 'axis'), floor: partChoice('tunnel', 'floor'), sides: partChoice('tunnel', 'sides'), sponge: 'number', mach: 'number', rho: 'number', inlet: 'int', area: 'number', length: 'number', rotor: 'number' },
};

const FEATURED: Schema = { any: ['string', obj({
  path: { any: ['string', { array: 'string', max: 8 }] }, id: 'string', label: 'string', tip: 'string', min: 'number', max: 'number', step: 'number', factor: 'number',
  control: { enum: ['number', 'slider', 'checkbox'] }, dims: { enum: [2, 3] }, unit: obj({ label: 'string', times: 'string' }),
  unitChoices: { array: obj({ label: 'string', value: 'number' }), max: 16 },
})] };

const DOCUMENT: Record<string, Schema> = {
  format: 'string', version: 'int', id: 'string', name: 'string', group: 'string', family: 'string', familyName: 'string', variant: 'string', blurb: 'string', extends: 'string',
  domain: obj({ ndim: choice('domain.ndim'), n: 'int', amr: 'int', box: 'number', supports2d: 'boolean', symmetric: choice('domain.symmetric'), symmetricField: 'boolean', mirror: choice('domain.mirror'), geometry: choice('domain.geometry'), rMin: 'number', logR: 'boolean', bc: FACES, bc2d: FACES, wallVelocity: obj({ xlo: 'vec3', xhi: 'vec3', ylo: 'vec3', yhi: 'vec3', zlo: 'vec3', zhi: 'vec3' }), units: obj({ name: 'string', length: UNIT, mass: UNIT, density: UNIT, velocity: UNIT }) }),
  physics: obj({
    eos: choice('physics.eos'),
    eosParams: { object: { aneosA: choice('physics.eosParams.aneosA') }, open: 'number' },
    gravity: choice('physics.gravity'), G: { any: ['number', { enum: ['units'] }] }, gext: { any: ['vec3', obj({ down: 'number' })] },
    cooling: choice('physics.cooling'), coolParams: { object: {}, open: 'number' },
    radiation: choice('physics.radiation'), opacity: choice('physics.opacity'), opacityParams: { object: {}, open: 'number' }, radC: 'number', radChat: 'number', radLocalC: 'boolean', radSolver: choice('physics.radSolver'), radSweeps: 'int',
    viscosity: 'number', conductivity: 'number', resistivity: 'number', driving: obj({ accel: 'number', tau: 'number', kMin: 'number', kMax: 'number', zeta: 'number', seed: 'number' }),
    riemann: choice('physics.riemann'), limiter: choice('physics.limiter'), cfl: 'number', pressure: choice('physics.pressure'), cutCells: 'boolean', storedPrims: choice('physics.storedPrims'), pressureCycles: 'int', pressureTol: 'number', acousticCfl: 'number',
    floors: obj({ rho: 'number', p: 'number' }), fluff: obj({ rho: 'number', u: 'number', v: 'number', va: 'number' }),
    frame: choice('physics.frame'), rotation: obj({ omega: 'vec3', centrifugal: 'boolean' }), cosmology: obj({ omegaM: 'number', omegaL: 'number', zStart: 'number', zEnd: 'number', h: 'number', boxMpc: 'number' }), shear: obj({ omega: 'number', q: 'number', stratified: 'boolean' }), jeansFloor: 'boolean', jeansCells: 'int', vcycles: 'int', gravEnergyFlux: 'boolean',
    composition: { any: [choice('physics.composition'), obj({ name: 'string', X: 'fractions', ionization: { enum: ['full', 'neutral'] } })] },
    species: { any: [{ enum: ['aprox13'] }, { array: 'string', max: 64 }] }, speciesGamma: 'fractions',
    burn: { any: [obj({ network: choice('physics.burn.network'), inShocks: 'boolean', limiter: 'number' }), obj({ fuel: 'string', ash: 'string', q: 'number', ea: 'number', k: 'number', inShocks: 'boolean' })] },
    toolScale: obj({ rho: 'number', v: 'number' }),
  }),
  mesh: obj({ refineHi: 'number', refineLo: 'number', refineMasses: 'boolean', refineDense: 'number', refineRho: 'number', timeBins: choice('mesh.timeBins'), jeansRefine: 'number' }),
  background: obj({ rho: 'number', p: 'number', v: 'vec3', s: 'number', X: 'fractions', cut: 'number', relativeTo: 'string' }),
  run: obj({ tEnd: { nullable: 'number' }, checkpointDt: 'number', tDyn: 'number', startPaused: 'boolean' }),
  view: obj({ field: 'int', log: 'boolean', mode: choice('view.mode'), axis: choice('view.axis'), pitch: 'number' }),
  featured: { array: FEATURED, max: 32 },
};

const kindOf = (v: unknown) => (v === null ? 'null' : Array.isArray(v) ? 'an array' : typeof v);

/** A number held to an option's range, noting it if it moved. */
function ranged(v: number, o: Option | undefined, path: string, notes: string[]): number {
  if (!o || o.control.type !== 'number') return v;
  const { min = -Infinity, max = Infinity } = o.control;
  const held = Math.min(max, Math.max(min, v));
  if (held !== v) notes.push(`${path}: ${v} held to ${held}`);
  return held;
}

/** A value made to fit a schema: throws Bad on a wrong type, drops and notes what it does not know. */
function conform(v: unknown, s: Schema, path: string, notes: string[], option: (path: string) => Option | undefined, depth = 0): unknown {
  if (depth > 12) throw new Bad(`${path}: nested too deep`);
  const wrong = (want: string) => new Bad(`${path}: expected ${want}, found ${kindOf(v)}`);
  if (s === 'number' || s === 'int') {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw wrong('a number');
    if (s === 'int' && !Number.isInteger(v)) throw wrong('a whole number');
    return ranged(v, option(path), path, notes);
  }
  if (s === 'string') {
    if (typeof v !== 'string') throw wrong('text');
    if (v.length > MAX_TEXT) throw new Bad(`${path}: text too long`);
    return v;
  }
  if (s === 'boolean') {
    if (typeof v !== 'boolean') throw wrong('true or false');
    return v;
  }
  if (s === 'vec3') {
    if (!Array.isArray(v) || v.length !== 3 || v.some((c) => typeof c !== 'number' || !Number.isFinite(c))) throw wrong('three numbers');
    return [...v];
  }
  if (s === 'pair') {
    if (!Array.isArray(v) || v.length !== 2 || v.some((c) => typeof c !== 'number' || !Number.isFinite(c))) throw wrong('two numbers');
    return [...v];
  }
  if (s === 'numbers') {
    if (!Array.isArray(v) || v.length > 64 || v.some((c) => typeof c !== 'number' || !Number.isFinite(c))) throw wrong('numbers');
    return [...v];
  }
  if (s === 'fractions') {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) throw wrong('numbers by name');
    const out: Record<string, number> = {};
    const entries = Object.entries(v);
    if (entries.length > 64) throw new Bad(`${path}: too many entries`);
    for (const [k, x] of entries) {
      if (typeof x !== 'number' || !Number.isFinite(x)) throw new Bad(`${path}.${k}: expected a number`);
      if (/^[A-Za-z0-9_+-]{1,16}$/.test(k)) out[k] = x;
      else notes.push(`${path}.${k}: dropped (not a name)`);
    }
    return out;
  }
  if ('enum' in s) {
    if (!s.enum.includes(v as string | number)) throw new Bad(`${path}: ${JSON.stringify(v)?.slice(0, 40)} is not one of ${s.enum.join(', ')}`);
    return v;
  }
  if ('nullable' in s) return v === null ? null : conform(v, s.nullable, path, notes, option, depth + 1);
  if ('any' in s) {
    // The alternative that reads the value with the fewest notes: an object that fits one shape
    // whole is not read as another that would drop half its keys.
    let first: Bad | undefined;
    let best: { out: unknown; trial: string[] } | undefined;
    for (const alt of s.any) {
      const trial: string[] = [];
      try {
        const out = conform(v, alt, path, trial, option, depth + 1);
        if (!best || trial.length < best.trial.length) best = { out, trial };
        if (!trial.length) break;
      } catch (e) {
        if (!(e instanceof Bad)) throw e;
        first ??= e;
      }
    }
    if (best) {
      notes.push(...best.trial);
      return best.out;
    }
    throw first ?? wrong('another kind of value');
  }
  if ('array' in s) {
    if (!Array.isArray(v)) throw wrong('a list');
    if (v.length > s.max) throw new Bad(`${path}: more than ${s.max} entries`);
    return v.map((x, i) => conform(x, s.array, `${path}.${i}`, notes, option, depth + 1));
  }
  if (v === null || typeof v !== 'object' || Array.isArray(v)) throw wrong('an object');
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    const sub = s.object[k] ?? s.open;
    if (!sub || x === undefined) {
      if (x !== undefined) notes.push(`${path ? `${path}.` : ''}${k}: dropped (not a key this build reads)`);
      continue;
    }
    out[k] = conform(x, sub, path ? `${path}.${k}` : k, notes, option, depth + 1);
  }
  return out;
}

function conformPart(v: unknown, path: string, notes: string[]): unknown {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) throw new Bad(`${path}: expected a part`);
  const raw = v as Record<string, unknown>;
  const kind = raw.kind;
  if (typeof kind !== 'string') throw new Bad(`${path}: a part needs a kind`);
  const id = raw.id === undefined ? {} : { id: conform(raw.id, ID, `${path}.id`, notes, () => undefined) };
  const { kind: _k, id: _i, ...rest } = raw;
  if (kind === 'flow') {
    const def = typeof raw.name === 'string' ? FLOWS[raw.name] : undefined;
    if (!def) throw new Bad(`${path}: unknown flow ${JSON.stringify(raw.name)?.slice(0, 40)}`);
    const out: Record<string, unknown> = { kind, ...id, name: def.name };
    for (const [k, x] of Object.entries(rest)) {
      if (k === 'name') continue;
      const f = def.fields[k];
      if (!f) {
        notes.push(`${path}.${k}: dropped (not a parameter of this flow)`);
        continue;
      }
      if (typeof x !== typeof f.default || (typeof x === 'number' && !Number.isFinite(x))) throw new Bad(`${path}.${k}: expected ${typeof f.default === 'boolean' ? 'true or false' : 'a number'}`);
      const held = typeof x === 'number' ? Math.min(f.max ?? Infinity, Math.max(f.min ?? -Infinity, x)) : x;
      if (held !== x) notes.push(`${path}.${k}: ${x} held to ${held}`);
      out[k] = held;
    }
    return out;
  }
  const schema = PART_SCHEMAS[kind];
  if (!schema) throw new Bad(`${path}: unknown kind of part "${kind.slice(0, 40)}"`);
  const fields = flatten(PART_OPTIONS[kind as keyof typeof PART_OPTIONS].fields);
  const option = (p: string) => fields.find((o) => `${path}.${o.path}` === p);
  return { kind, ...id, ...(conform(rest, obj(schema), path, notes, option) as object) };
}

// ---- the format as JSON Schema ----

type Json = Record<string, unknown>;

/** What an option of the table adds to its key's schema: its name, its tooltip and its range. */
function described(o: Option | undefined): Json {
  if (!o) return {};
  const c = o.control;
  return { title: o.label, description: o.tip, ...(c.type === 'number' && c.min !== undefined ? { minimum: c.min } : {}), ...(c.type === 'number' && c.max !== undefined ? { maximum: c.max } : {}) };
}

function jsonSchema(s: Schema, path: string, option: (path: string) => Option | undefined): Json {
  const about = described(option(path));
  const list = (n: number): Json => ({ type: 'array', items: { type: 'number' }, minItems: n, maxItems: n });
  if (s === 'number') return { type: 'number', ...about };
  if (s === 'int') return { type: 'integer', ...about };
  if (s === 'string') return { type: 'string', maxLength: MAX_TEXT, ...about };
  if (s === 'boolean') return { type: 'boolean', ...about };
  if (s === 'vec3') return { ...list(3), ...about };
  if (s === 'pair') return list(2);
  if (s === 'numbers') return { type: 'array', items: { type: 'number' }, maxItems: 64, ...about };
  if (s === 'fractions') return { type: 'object', additionalProperties: { type: 'number' }, ...about };
  const { title, description } = about as { title?: string; description?: string };
  const named = { ...(title ? { title } : {}), ...(description ? { description } : {}) };
  if ('enum' in s) return { enum: [...s.enum], ...named };
  if ('nullable' in s) return { anyOf: [{ type: 'null' }, jsonSchema(s.nullable, path, () => undefined)], ...named };
  if ('any' in s) return { anyOf: s.any.map((alt) => jsonSchema(alt, path, () => undefined)), ...named };
  if ('array' in s) return { type: 'array', items: jsonSchema(s.array, `${path}.0`, option), maxItems: s.max, ...named };
  const properties = Object.fromEntries(Object.entries(s.object).map(([k, sub]) => [k, jsonSchema(sub, path ? `${path}.${k}` : k, option)]));
  return { type: 'object', properties, additionalProperties: s.open ? jsonSchema(s.open, '', () => undefined) : false, ...named };
}

/**
 * The format as JSON Schema (draft 2020-12), made from the same tables this file reads documents
 * with: an editor that knows it completes a setup file's keys and marks a value that is not a
 * choice. It is stricter than readSetup in one way: a key this build does not know is an error
 * there, where reading drops it with a note.
 */
export function setupJsonSchema(): Json {
  const top = jsonSchema(obj(DOCUMENT), '', (path) => OPTION_BY_PATH[path]) as { properties: Record<string, Json> };
  const part = (kind: string, fields: Json, title: string): Json => ({
    type: 'object', title,
    properties: { kind: { const: kind }, id: { type: 'string', description: 'A name other parts refer to it by (an orbit\u2019s `of`, a disk\u2019s `about`)' }, ...fields },
    required: ['kind'], additionalProperties: false,
  });
  const kinds = Object.entries(PART_SCHEMAS).map(([kind, schema]) => {
    const k = PART_OPTIONS[kind as keyof typeof PART_OPTIONS];
    const fields = flatten(k.fields);
    return part(kind, (jsonSchema(obj(schema), '', (path) => fields.find((o) => o.path === path)) as { properties: Json }).properties, k.label);
  });
  const flows = Object.values(FLOWS).map((def) => {
    const fields = Object.fromEntries(Object.entries(def.fields).map(([k, f]) => [k, typeof f.default === 'boolean'
      ? { type: 'boolean', title: f.label, default: f.default }
      : { type: 'number', title: f.label, default: f.default, ...(f.min !== undefined ? { minimum: f.min } : {}), ...(f.max !== undefined ? { maximum: f.max } : {}) }]));
    return { ...part('flow', { name: { const: def.name }, ...fields }, `Flow: ${def.label}`), description: def.tip, required: ['kind', 'name'] };
  });
  return {
    $schema: 'https://json-schema.org/draft/2020-12/schema',
    $id: SETUP_SCHEMA_URL,
    title: 'Shaderous setup',
    description: 'A simulation setup for Shaderous: the domain, the physics, the mesh, the background gas and the parts that make the initial conditions (docs/setups.md). Generated from the option table by `npm run gen`.',
    type: 'object',
    required: ['format', 'version'],
    additionalProperties: false,
    properties: {
      $schema: { type: 'string' },
      ...top.properties,
      format: { const: SETUP_FORMAT },
      version: { type: 'integer', minimum: 1, maximum: SETUP_VERSION },
      parts: { type: 'array', maxItems: MAX_PARTS, items: { oneOf: [...kinds, ...flows] } },
    },
  };
}

/**
 * Read a setup document: text or an already parsed value. `find` returns a preset by id, for a
 * document that `extends` one. The result is whole (no `extends` left) and holds only what this
 * build reads; or null, with the reason in the report.
 */
export function readSetup(input: unknown, find: (id: string) => SetupSpec | undefined = () => undefined): { spec: SetupSpec | null; report: SetupReport } {
  const report: SetupReport = { notes: [] };
  const fail = (error: string) => ({ spec: null, report: { ...report, error } });
  let doc = input;
  if (typeof input === 'string') {
    if (input.length > MAX_DOCUMENT) return fail('the file is too large to be a setup');
    try {
      doc = JSON.parse(input);
    } catch (e) {
      return fail(`not JSON: ${(e as Error).message}`);
    }
  }
  if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) return fail('not a setup: expected an object');
  let raw = doc as Record<string, unknown>;
  if (raw.format !== SETUP_FORMAT) return fail(`not a setup: its format is ${JSON.stringify(raw.format)?.slice(0, 40)}, not "${SETUP_FORMAT}"`);
  const version = raw.version;
  if (typeof version !== 'number' || !Number.isInteger(version) || version < 1) return fail('not a setup: it has no version');
  if (version > SETUP_VERSION) report.notes.push(`written by a newer build (version ${version}; this build reads ${SETUP_VERSION}): what it does not know is listed here and left out`);
  try {
    for (let v = version; v < SETUP_VERSION; v++) {
      const step = MIGRATIONS[v];
      if (!step) return fail(`no way to read version ${v} of the format`);
      raw = step(raw);
    }
    // `$schema` (what an editor checks the file against) is not part of the setup.
    const { parts, $schema: _schema, ...rest } = raw;
    const out = conform(rest, obj(DOCUMENT), '', report.notes, (p) => OPTION_BY_PATH[p]) as Record<string, unknown>;
    if (parts !== undefined) {
      if (!Array.isArray(parts)) throw new Bad('parts: expected a list');
      if (parts.length > MAX_PARTS) throw new Bad(`parts: more than ${MAX_PARTS}`);
      out.parts = parts.map((p, i) => conformPart(p, `parts.${i}`, report.notes));
    }
    out.format = SETUP_FORMAT;
    out.version = SETUP_VERSION;
    const spec = resolveExtends(out as unknown as SetupSpec, find);
    const ids = (spec.parts ?? []).map((p) => p.id).filter((x) => x !== undefined);
    if (new Set(ids).size !== ids.length) throw new Bad('parts: two have the same id');
    return { spec, report };
  } catch (e) {
    if (e instanceof Bad || (e instanceof Error && e.message.startsWith('setup:'))) return fail(e.message);
    throw e;
  }
}

/** A setup as the text of its file: pretty-printed, with short lists and small objects on one line. */
export function formatSetup(spec: SetupSpec): string {
  const flat = (v: unknown): boolean => v === null || typeof v !== 'object';
  const inline = (v: unknown): string => {
    if (flat(v)) return JSON.stringify(v);
    if (Array.isArray(v)) return `[${v.map(inline).join(', ')}]`;
    const entries = Object.entries(v as object).filter(([, x]) => x !== undefined);
    return entries.length ? `{ ${entries.map(([k, x]) => `${JSON.stringify(k)}: ${inline(x)}`).join(', ')} }` : '{}';
  };
  const write = (v: unknown, indent: string, key: number): string => {
    if (flat(v)) return JSON.stringify(v);
    const one = inline(v);
    // Lists of numbers and small objects of plain values stay on one line.
    const plain = (x: unknown) => flat(x) || (Array.isArray(x) && x.every(flat));
    const simple = Array.isArray(v) ? v.every(flat) : Object.values(v as object).every(plain);
    if (simple && one.length + indent.length + key <= 120) return one;
    const inner = `${indent}  `;
    if (Array.isArray(v)) return `[\n${v.map((x) => inner + write(x, inner, 0)).join(',\n')}\n${indent}]`;
    const entries = Object.entries(v as object).filter(([, x]) => x !== undefined);
    return `{\n${entries.map(([k, x]) => `${inner}${JSON.stringify(k)}: ${write(x, inner, k.length + 4)}`).join(',\n')}\n${indent}}`;
  };
  return `${write({ $schema: SETUP_SCHEMA_URL, ...spec }, '', 0)}\n`;
}

// ---- links ----

const b64url = (bytes: Uint8Array): string => {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const unb64url = (text: string): Uint8Array => Uint8Array.from(atob(text.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0));

async function pipe(bytes: Uint8Array, through: CompressionStream | DecompressionStream, limit = Infinity): Promise<Uint8Array> {
  const reader = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(through).getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > limit) throw new Error('setup: the link unpacks to more than a setup can be');
    chunks.push(value);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** A value with the keys of its objects in order, so that one document always makes one link. */
function sorted(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sorted);
  if (v === null || typeof v !== 'object') return v;
  return Object.fromEntries(Object.keys(v).sort().map((k) => [k, sorted((v as Record<string, unknown>)[k])]));
}

/** A setup as the value of a link's `setup=`: minified, deflated and base64url-encoded. */
export async function encodeSetupParam(spec: SetupSpec): Promise<string> {
  return b64url(await pipe(new TextEncoder().encode(JSON.stringify(sorted(spec))), new CompressionStream('deflate-raw')));
}

/** The text of the document in a link's `setup=` (to be read with readSetup). Throws if it is not one. */
export async function decodeSetupParam(value: string): Promise<string> {
  if (value.length > 100_000) throw new Error('setup: the link is too long');
  return new TextDecoder().decode(await pipe(unb64url(value), new DecompressionStream('deflate-raw'), MAX_DOCUMENT));
}

/** Past this many characters a link is refused by some servers and chat programs: Copy link offers the file instead. */
export const MAX_LINK = 8000;
