// Reading and writing setup documents: files, the browser's library and the `?setup=` of a
// link. Reading never trusts: unknown keys are dropped and listed, values out of range are held
// to it and listed, a wrong type is an error (the discipline of ui/link.ts decodeLink).
//
// `version` counts changes to the document's shape. MIGRATIONS keeps one step per version and
// applies them in order; a file from a newer version loads with a warning. The rule of
// PROBLEM_ALIASES applies: add, never remove.
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
const BC: Schema = { array: { enum: [0, 1, 2, 3] }, max: 3 };
const FACES = obj({ lo: BC, hi: BC });
const UNIT = obj({ unit: { enum: Object.keys(UNIT_SI) }, value: 'number', per: { any: ['number', 'string'] } });
const SHAPE: Schema = obj({
  type: { enum: ['sphere', 'box', 'half'] }, center: 'vec3', radius: 'number', radiusCells: 'number',
  min: { array: { nullable: 'number' }, max: 3 }, max: { array: { nullable: 'number' }, max: 3 }, normal: 'vec3', at: 'number',
});
const ID: Schema = 'string';
/** A tabulated profile: up to 32 points (x, value). */
const PROFILE: Schema = { array: 'pair', max: 32 };

const PART_SCHEMAS: Record<string, Record<string, Schema>> = {
  mass: { gm: 'number', pos: 'vec3', vel: 'vec3', soft: 'number', sink: 'number', sinkRate: 'number', massless: 'boolean', fixed: 'boolean', onOrbit: 'boolean' },
  orbit: { of: { array: 'string', max: 2 }, type: { enum: ['binary', 'flyby', 'collision'] }, q: 'number', separation: 'number', ecc: 'number', incl: 'number', retrograde: 'boolean', beta: 'number', start: 'number', impact: 'number', starts: 'number', debris: 'boolean' },
  star: { type: { enum: ['polytrope', 'white-dwarf'] }, n: 'number', radius: 'number', mass: 'number', rhoc: 'number', pos: 'vec3', vel: 'vec3', dye: 'number', X: 'fractions', field: obj({ type: { enum: ['dipole', 'random', 'gm17-dipole', 'gm17-random'] }, beta: 'number', seed: 'number' }) },
  disk: { about: { array: 'string', max: 32 }, profile: { enum: ['thin', 'balanced'] }, radial: PROFILE, vertical: PROFILE, radialFn: 'string', verticalFn: 'string', verticalIn: { enum: ['H', 'length'] }, rIn: 'number', rOut: 'number', sigma0: 'number', slope: 'number', cs: 'number', floor: 'number', damp: obj({ from: 'number', rate: 'number' }), loops: obj({ beta: 'number', span: 'number', count: 'int' }), stresses: 'boolean' },
  region: { shape: SHAPE, rho: 'number', p: 'number', energy: 'number', v: 'vec3', s: 'number', X: 'fractions', supersample: 'int' },
  obstacle: { shape: SHAPE },
  inflow: { shape: SHAPE },
  field: { b: 'vec3', up: 'number' },
};

const FEATURED: Schema = { any: ['string', obj({
  path: { any: ['string', { array: 'string', max: 8 }] }, id: 'string', label: 'string', tip: 'string', min: 'number', max: 'number', step: 'number', factor: 'number',
  control: { enum: ['number', 'slider', 'checkbox'] }, dims: { enum: [2, 3] }, unit: obj({ label: 'string', times: 'string' }),
  unitChoices: { array: obj({ label: 'string', value: 'number' }), max: 16 },
})] };

const DOCUMENT: Record<string, Schema> = {
  format: 'string', version: 'int', id: 'string', name: 'string', group: 'string', family: 'string', familyName: 'string', variant: 'string', blurb: 'string', extends: 'string',
  domain: obj({ ndim: { enum: [2, 3] }, n: 'int', amr: 'int', box: 'number', supports2d: 'boolean', bc: FACES, bc2d: FACES, units: obj({ name: 'string', length: UNIT, mass: UNIT, density: UNIT, velocity: UNIT }) }),
  physics: obj({
    eos: { enum: ['ideal', 'barotropic', 'gas_radiation', 'degenerate', 'ionizing', 'helmholtz'] },
    eosParams: { object: {}, open: 'number' },
    gravity: { enum: ['off', 'constant', 'periodic', 'isolated'] }, G: { any: ['number', { enum: ['units'] }] }, gext: { any: ['vec3', obj({ down: 'number' })] },
    cooling: { enum: ['off', 'power', 'ism'] }, coolParams: { object: {}, open: 'number' }, viscosity: 'number', conductivity: 'number',
    riemann: { enum: ['hllc', 'hllclm', 'hll'] }, limiter: { enum: ['koren', 'mc', 'vanleer', 'minmod'] }, cfl: 'number',
    floors: obj({ rho: 'number', p: 'number' }), fluff: obj({ rho: 'number', u: 'number', v: 'number', va: 'number' }),
    frame: { enum: ['inertial', 'gas', 'masses'] }, jeansFloor: 'boolean', jeansCells: 'int', vcycles: 'int', gravEnergyFlux: 'boolean',
    composition: { any: ['string', obj({ name: 'string', X: 'fractions', ionization: { enum: ['full', 'neutral'] } })] },
    species: { any: [{ enum: ['aprox13'] }, { array: 'string', max: 64 }] }, speciesGamma: 'fractions',
    burn: { any: [obj({ network: { enum: ['aprox13'] }, inShocks: 'boolean', limiter: 'number' }), obj({ fuel: 'string', ash: 'string', q: 'number', ea: 'number', k: 'number', inShocks: 'boolean' })] },
    toolScale: obj({ rho: 'number', v: 'number' }),
  }),
  mesh: obj({ refineHi: 'number', refineLo: 'number', refineMasses: 'boolean', refineDense: 'number', refineRho: 'number', timeBins: 'int', jeansRefine: 'number' }),
  background: obj({ rho: 'number', p: 'number', v: 'vec3', s: 'number', X: 'fractions', cut: 'number', relativeTo: 'string' }),
  run: obj({ tEnd: { nullable: 'number' }, checkpointDt: 'number', tDyn: 'number', startPaused: 'boolean' }),
  view: obj({ field: 'int', log: 'boolean', mode: { enum: ['volume', 'slice', 'projection'] }, axis: { enum: [0, 1, 2] }, pitch: 'number' }),
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
    let first: Bad | undefined;
    for (const alt of s.any) {
      const trial: string[] = [];
      try {
        const out = conform(v, alt, path, trial, option, depth + 1);
        notes.push(...trial);
        return out;
      } catch (e) {
        if (!(e instanceof Bad)) throw e;
        first ??= e;
      }
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
    const { parts, ...rest } = raw;
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
  return `${write(spec, '', 0)}\n`;
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
