// Random setup documents for Shaderous, drawn from the option table (src/setup/options.ts), so
// that every option, choice and kind of part the table lists is a target the generator aims at,
// and a new one becomes a target without a change here.
//
// A document is built in one of two ways (odd seeds the first, even the second):
//   compose  a small box of gas, then a few features drawn at random from the whole list
//   preset   a preset's document on a random grid, then up to three features drawn the same way
//
// A feature is one setting of the table: an option set (`physics.viscosity`), a choice made
// (`physics.eos=hhe`), a group present in one of its forms (`domain.units:flow`), a kind of part
// (`part:disk`), a field or choice of a part (`part:orbit.type=flyby`), a built-in flow
// (`flow:zeldovich`), or a kind of box face (`domain.bc=4`). Applying one sets it, then satisfies
// the rules it needs (its `requires` and `excludes`, through FIXERS: a self-gravity it needs is
// switched on, a periodic face it cannot have made open) and repairs what that broke elsewhere
// (whatever the page's own check then flags is taken out). A feature whose document still fails
// the check is dropped, and the document stays as it was before it.
//
// The check is the page's (setup/check.ts checkSetup, with the simulators' refusal): what comes
// out loads, unless it needs more GPU memory than the page has. What it then does when run is
// the test.
//
// Numbers are drawn from the table's ranges, around its defaults, or from HINTS where those
// ranges are too wide to give a run that does anything in a few hundred steps (a viscosity of
// 0 to 1e30, a point mass of any G M).
import { checkSetup } from '../../src/setup/check';
import { compiledFrom } from '../../src/setup/compile';
import { factsOf, factsWithout, flatten, formOf, OPTIONS, PART_OPTIONS, partFields, type Choice, type Option, type Rule, type Value } from '../../src/setup/options';
import { partIds } from '../../src/setup/parts/bodies';
import { FLOWS } from '../../src/setup/parts/flow';
import { cloneSetup, getPath, setPath, SETUP_FORMAT, SETUP_VERSION, type Part, type PartKind, type SetupSpec, type Vec3 } from '../../src/setup/spec';
import { BOUNDARIES, FACE_KEYS, MOVING_WALL, OUTFLOW, PERIODIC } from '../../src/problems/boundaries';
import { PRESET_IDS, presetSpec } from '../../src/presets';
import { ANEOS_MATERIALS } from '../../src/eos/aneos';
import { configFor, refusal } from '../../src/sim';

// ---- random numbers ----

export interface Rng {
  rnd(): number;
  pick<T>(a: readonly T[]): T;
  U(a: number, b: number): number;
  /** log-uniform between a and b */
  L(a: number, b: number): number;
  int(a: number, b: number): number;
}

export function rng(seed: number): Rng {
  // Mixed first, so that neighboring seeds give unrelated setups.
  let s = Math.imul((seed >>> 0) ^ 0x9e3779b9, 2654435761) >>> 0;
  s = Math.imul(s ^ (s >>> 15), 2246822519) >>> 0;
  const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  return {
    rnd,
    pick: (a) => a[Math.floor(rnd() * a.length)],
    U: (a, b) => a + (b - a) * rnd(),
    L: (a, b) => a * (b / a) ** rnd(),
    int: (a, b) => a + Math.floor(rnd() * (b - a + 1)),
  };
}

const r3 = (v: number) => +v.toPrecision(3);

// ---- the features ----

export interface Feature {
  id: string;
  type: 'option' | 'part' | 'flow' | 'face';
  /** the option's path in the document, or the field's in its part */
  path?: string;
  /** the option (or the part's field) */
  option?: Option;
  /** the choice made, for a select */
  choice?: Choice;
  /** the form, for a group that has several */
  form?: string;
  kind?: PartKind;
  flow?: string;
  /** a face's kind */
  code?: number;
  /** a check whose default is on: the feature is turning it off */
  off?: boolean;
}

/** Options that are not about the run: names, the opening view, the clock, the editing tools. */
const NOT_RUN = /^(name|blurb|run\.|view\.|physics\.toolScale|domain\.supports2d|domain\.symmetric$|domain\.bc$)/;
/** Fields of a mesh shape: no mesh file is at hand. */
const MESH_ONLY = /^shape\.(file|anchor|node|fit|scale|up)$/;

/**
 * What the generator never aims at, and why: printed with the coverage, and kept out of the
 * total. (Obstacles' motions and parents are not in the option table at all.)
 */
export const EXCLUDED: Record<string, string> = {
  'shape.type=mesh': 'needs a mesh file beside the document',
  'run, view, name, blurb, toolScale, supports2d, symmetric': 'say nothing about the run',
};

const isDefault = (o: Option, v: Value) => o.default !== undefined && String(o.default) === String(v);

/** The features of a list of options, flattened: `prefix` is put before each id. */
function optionFeatures(options: Option[], idPrefix: string, type: 'option' | 'part', kind?: PartKind): Feature[] {
  const out: Feature[] = [];
  for (const o of flatten(options)) {
    if (type === 'option' && NOT_RUN.test(o.path)) continue;
    if (type === 'part' && MESH_ONLY.test(o.path)) continue;
    const c = o.control;
    const base = { type, path: o.path, option: o, kind } as const;
    if (c.type === 'select') {
      for (const ch of c.choices) {
        if (isDefault(o, ch.value) || (o.path === 'shape.type' && ch.value === 'mesh')) continue;
        if (c.choices.length === 1) continue;
        out.push({ ...base, id: `${idPrefix}${o.path}=${ch.value}`, choice: ch });
      }
    } else if (c.type === 'group' && c.forms) {
      for (const f of c.forms.choices) out.push({ ...base, id: `${idPrefix}${o.path}:${f.value}`, form: f.value });
    } else if (c.type === 'check') {
      out.push({ ...base, id: `${idPrefix}${o.path}${o.default === true ? '=false' : ''}`, off: o.default === true });
    } else if (c.type === 'ref') {
      continue;
    } else out.push({ ...base, id: `${idPrefix}${o.path}` });
  }
  return out;
}

let universe: Feature[] | undefined;

/** Every feature the generator aims at. */
export function features(): Feature[] {
  if (universe) return universe;
  const out: Feature[] = optionFeatures(OPTIONS, '', 'option');
  for (const b of BOUNDARIES) out.push({ id: `domain.bc=${b.code}`, type: 'face', code: b.code });
  for (const [kind, k] of Object.entries(PART_OPTIONS) as [PartKind, (typeof PART_OPTIONS)[PartKind]][]) {
    if (kind === 'flow') continue;
    out.push({ id: `part:${kind}`, type: 'part', kind });
    out.push(...optionFeatures(k.fields, `part:${kind}.`, 'part', kind));
  }
  for (const [name, def] of Object.entries(FLOWS)) {
    out.push({ id: `flow:${name}`, type: 'flow', flow: name, kind: 'flow' });
    for (const key of Object.keys(def.fields)) out.push({ id: `flow:${name}.${key}`, type: 'flow', flow: name, kind: 'flow', path: key });
  }
  return (universe = out);
}

const trivial = (v: unknown) => v === undefined || v === null || v === false || v === 0 || v === '';
const facesOf = (s: SetupSpec): number[] => {
  const k = (s.domain?.ndim ?? 3) === 2 ? 2 : 3;
  return [...(s.domain?.bc?.lo ?? [1, 1, 1]).slice(0, k), ...(s.domain?.bc?.hi ?? [1, 1, 1]).slice(0, k)];
};

/** Whether a value at a feature's path makes the feature present. */
function holdsValue(f: Feature, at: unknown): boolean {
  const c = f.option!.control;
  if (f.choice) return at !== undefined && String(at) === String(f.choice.value);
  if (f.form) return at !== undefined && at !== null && c.type === 'group' && formOf(c.forms!, at).value === f.form;
  if (c.type === 'check') return f.off ? at === false : at === true;
  if (c.type === 'group') return at !== undefined && at !== null;
  if (c.type === 'vec3') return Array.isArray(at) && at.some((x) => x !== 0);
  if (c.type === 'names' || c.type === 'fractions') return at !== undefined && (typeof at === 'string' || Object.keys(at as object).length > 0);
  return !trivial(at);
}

/** Whether a document has a feature. */
export function hasFeature(doc: SetupSpec, f: Feature): boolean {
  const parts = doc.parts ?? [];
  switch (f.type) {
    case 'face':
      return facesOf(doc).includes(f.code!);
    case 'option':
      return holdsValue(f, getPath(doc, f.path!));
    case 'part':
      return parts.some((p) => p.kind === f.kind && (!f.path || holdsValue(f, getPath(p, f.path))));
    case 'flow':
      return parts.some((p) => p.kind === 'flow' && p.name === f.flow && (!f.path || (p as Record<string, unknown>)[f.path] !== undefined));
  }
}

/** The ids of the features a document has. */
export const featuresOf = (doc: SetupSpec): string[] => features().filter((f) => hasFeature(doc, f)).map((f) => f.id);

// ---- the check ----

/** What the page would refuse as it loads the document (ui/app.ts setupHost.refusal). */
function refusalOf(spec: SetupSpec): string | null {
  try {
    const d = spec.domain ?? {};
    const problem = compiledFrom(spec).problem;
    return refusal(configFor(problem, d.n ?? 128, { ndim: d.ndim ?? 3, amrLevels: problem.shear ? 0 : d.amr ?? 0 }));
  } catch {
    return null;
  }
}

/** The page's errors for a document: its check and the simulators' refusal. */
export function errorsOf(spec: SetupSpec): { text: string; path?: string }[] {
  return checkSetup(spec, { refusal: refusalOf(spec) }).filter((i) => i.level === 'error');
}

// ---- building a document ----

interface Ctx {
  g: Rng;
  /** a preset's document: its ndim may not change to 2 unless it supports 2D */
  preset?: string;
}

const curved = (d: SetupSpec) => ['cylindrical', 'spherical'].includes(d.domain?.geometry ?? 'cartesian');
const ndimOf = (d: SetupSpec) => (d.domain?.ndim ?? 3) as 2 | 3;

/** A resolution and refinement that run in a few seconds and fit in about 4 GB. */
function setGrid(doc: SetupSpec, g: Rng, amr?: number) {
  const d = (doc.domain ??= {});
  d.amr = amr ?? g.pick([0, 0, 1, 2]);
  d.n = ndimOf(doc) === 2 ? g.pick([128, 256, 512]) : d.amr ? g.pick([128, 256]) : g.pick([64, 64, 128]);
  clampGrid(doc);
}

/** A run that cannot refine (a shearing box) takes the uniform grid: keep 3D ones at 128^3. */
function clampGrid(doc: SetupSpec) {
  const d = doc.domain!;
  if (ndimOf(doc) === 3 && (!d.amr || doc.physics?.shear)) d.n = Math.min(d.n ?? 128, 128);
  if (ndimOf(doc) === 3 && d.amr) d.n = Math.min(d.n ?? 128, 256);
  if (ndimOf(doc) === 2) d.n = Math.min(d.n ?? 256, 1024);
}

/** A place for a body or a shape: on the axis in (R, z) and (r, theta) runs. */
function centre(doc: SetupSpec, g: Rng, spread = 0.25): Vec3 {
  const b = (doc.domain?.box ?? 1) * spread;
  if (doc.domain?.geometry === 'spherical') return [0, 0, 0];
  if (curved(doc)) return [0, r3(g.U(-b, b)), 0];
  return [r3(g.U(-b, b)), r3(g.U(-b, b)), ndimOf(doc) === 3 ? r3(g.U(-b, b)) : 0];
}

const vec = (g: Rng, a: number, doc: SetupSpec): Vec3 => [r3(g.U(-a, a)), r3(g.U(-a, a)), ndimOf(doc) === 3 || curved(doc) ? r3(g.U(-a, a)) : 0];

function newId(doc: SetupSpec, kind: string): string {
  const ids = new Set(partIds(doc.parts ?? []));
  for (let i = 1; ; i++) if (!ids.has(`${kind}${i}`)) return `${kind}${i}`;
}

/** Remove a part, with the orbits that place it and the disks left orbiting nothing. */
function removePart(doc: SetupSpec, index: number) {
  const parts = doc.parts ?? [];
  const id = partIds(parts)[index];
  parts.splice(index, 1);
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === 'orbit' && (p.of ?? []).includes(id)) removePart(doc, i);
    else if (p.kind === 'disk' && (p.about ?? []).includes(id)) {
      p.about = p.about.filter((a) => a !== id);
      if (!p.about.length) removePart(doc, i);
    } else if (p.kind === 'obstacle' && p.parent === id) delete p.parent;
  }
  if (doc.background?.relativeTo === id) {
    delete doc.background.relativeTo;
    delete doc.background.cut;
  }
}

const removeWhere = (doc: SetupSpec, which: (p: Part) => boolean) => {
  for (let i = (doc.parts ?? []).length - 1; i >= 0; i--) if (i < doc.parts!.length && which(doc.parts![i])) removePart(doc, i);
};

/** Gas so thin around bodies that it does not weigh on them, as the star presets have. */
function thinBackground(doc: SetupSpec) {
  if ((doc.parts ?? []).some((p) => p.kind === 'flow' && FLOWS[p.name]?.wholeBox)) return;
  doc.background = { ...doc.background, rho: 1e-6, p: 1e-8 };
}

function setFaces(doc: SetupSpec, lo: number[], hi: number[]) {
  doc.domain ??= {};
  doc.domain.bc = { lo: lo as [number, number, number], hi: hi as [number, number, number] };
}

/** Faces with no periodic one, for bodies whose gravity is not periodic. */
function openFaces(doc: SetupSpec) {
  const bc = doc.domain?.bc ?? { lo: [1, 1, 1], hi: [1, 1, 1] };
  setFaces(doc, bc.lo.map((b) => (b === PERIODIC ? OUTFLOW : b)), bc.hi.map((b) => (b === PERIODIC ? OUTFLOW : b)));
}

/** The faces of an (R, z) or (r, theta) run: the axis (and the poles) reflecting, the rest open or reflecting. */
function curvedFaces(doc: SetupSpec, g: Rng) {
  if (doc.domain?.geometry === 'spherical') setFaces(doc, [2, 2, 0], [g.pick([1, 2]), 2, 0]);
  else setFaces(doc, [2, g.pick([1, 2]), 0], [g.pick([1, 2]), g.pick([1, 2]), 0]);
}

function setGeometry(doc: SetupSpec, geometry: 'cartesian' | 'cylindrical' | 'spherical', g: Rng) {
  const d = (doc.domain ??= {});
  const was = d.geometry ?? 'cartesian';
  if (geometry === 'cartesian') {
    delete d.geometry;
    delete d.rMin;
    delete d.logR;
    if (was !== 'cartesian') {
      const b = g.pick([0, 1, 2]);
      setFaces(doc, [b, b, b], [b, b, b]);
    }
    return;
  }
  d.geometry = geometry;
  if (d.ndim !== 2) {
    d.ndim = 2;
    setGrid(doc, g, d.amr);
  }
  // A spherical run has both poles in the grid; an axisymmetric one is mirrored across z = 0 only
  // (code y: R already starts at the axis).
  if (geometry === 'spherical' || d.mirror?.includes('x')) delete d.mirror;
  if (geometry === 'spherical') {
    d.rMin ??= g.pick([0.005, 0.02, 0.05]);
    d.box = 0.5;
  } else delete d.rMin;
  curvedFaces(doc, g);
  // Bodies sit on the axis.
  for (const p of doc.parts ?? []) if ((p.kind === 'star' || p.kind === 'mass') && p.pos) p.pos = [0, geometry === 'spherical' ? 0 : p.pos[1], 0];
}

function setNdim(doc: SetupSpec, ndim: 2 | 3, ctx: Ctx): boolean {
  const d = (doc.domain ??= {});
  if (ndim === 2 && !d.supports2d) return false;
  if (ndim === 3 && curved(doc)) setGeometry(doc, 'cartesian', ctx.g);
  if (ndimOf(doc) === ndim) return true;
  d.ndim = ndim;
  setGrid(doc, ctx.g, d.amr);
  return true;
}

/** Units a body of rock or a star can be measured in. */
function addUnits(doc: SetupSpec, g: Rng, form?: string) {
  const d = (doc.domain ??= {});
  if (d.units) return;
  if ((form ?? (g.rnd() < 0.7 ? 'mass' : 'flow')) === 'mass') {
    const [l, m] = g.pick([['Rearth', 'Mearth'], ['Rsun', 'Msun'], ['au', 'Msun'], ['km', 'Msun']] as const);
    d.units = { length: { unit: l, value: 1 }, mass: { unit: m, value: 1 } };
  } else d.units = { length: { unit: g.pick(['pc', 'au', 'km', 'm']), value: 1 }, density: { unit: 'g/cm3', value: g.pick([1, 1e-20, 1e-3]) }, velocity: { unit: 'km/s', value: g.pick([1, 10]) } };
}

function setEos(doc: SetupSpec, eos: string, g: Rng) {
  const ph = (doc.physics ??= {});
  ph.eos = eos as never;
  let gamma = ph.eosParams?.gamma ?? 5 / 3;
  // (Only the barotropic gas takes gamma 1, as an isothermal gas.)
  if (gamma <= 1 && eos !== 'barotropic') gamma = 5 / 3;
  ph.eosParams = { gamma };
  if (eos === 'gas_radiation') ph.eosParams = { gamma: 5 / 3, arad: r3(g.L(0.01, 1)), R: 1 };
  if (eos === 'barotropic') ph.eosParams = { gamma: g.pick([1, 1.4, 5 / 3]), K: r3(g.L(0.1, 1)) };
  if (eos === 'liquid') ph.eosParams = { liqC: g.pick([5, 10, 20]), liqRho: 1 };
  // As the white dwarf presets: 10^4 km and a solar mass, carbon and oxygen.
  if (eos === 'helmholtz' && !doc.domain?.units) {
    doc.domain = { ...doc.domain, units: { length: { unit: 'm', value: 1e7 }, mass: { unit: 'Msun', value: 1 } } };
    ph.composition ??= 'carbonOxygen';
    if (typeof ph.G === 'number') ph.G = 'units';
  }
  // As the planet preset: Earth radii and masses, G from them, a box four radii across.
  if (eos === 'aneos' && !doc.domain?.units) {
    doc.domain = { ...doc.domain, units: { length: { unit: 'Rearth', value: 1 }, mass: { unit: 'Mearth', value: 1 } }, box: 4 };
    ph.G = 'units';
  }
}

// ---- making parts ----

interface PartOpts {
  /** a star's or an orbit's type */
  type?: string;
  /** a region's, obstacle's or inflow's shape */
  shape?: string;
  flow?: string;
}

/** A new part of a kind, placed and sized so that it does something in the box. */
function makePart(doc: SetupSpec, kind: PartKind, ctx: Ctx, opts: PartOpts = {}): Part {
  const { g } = ctx;
  const id = newId(doc, kind);
  switch (kind) {
    case 'mass':
      return { kind, id, gm: r3(g.L(0.02, 0.5)), pos: centre(doc, g), soft: 0.03 };
    case 'star': {
      const type = opts.type ?? 'polytrope';
      if (type === 'white-dwarf') {
        // As the white dwarf preset: 10^4 km and a solar mass, G from them, a box 3 units across.
        if (!doc.domain?.units) {
          doc.domain = { ...doc.domain, units: { length: { unit: 'm', value: 1e7 }, mass: { unit: 'Msun', value: 1 } }, box: 3 };
          doc.physics = { ...doc.physics, G: 'units', composition: 'carbonOxygen' };
        }
        return { kind, id, type, mass: r3(g.U(0.4, 1)), pos: centre(doc, g, 0.1) };
      }
      if (type === 'planet') return { kind, id, type, mass: r3(g.U(0.1, 1)), pos: centre(doc, g, 0.1) };
      return { kind, id, type: 'polytrope', n: g.pick([1, 1.5, 3]), mass: r3(g.U(0.05, 0.2)), radius: r3(g.U(0.06, 0.12)), pos: centre(doc, g, 0.15) };
    }
    case 'orbit': {
      const type = (opts.type ?? g.pick(['binary', 'flyby', 'collision'])) as 'binary' | 'flyby' | 'collision';
      if (type === 'binary') return { kind, id, of: ['', ''], type, separation: r3(g.U(0.2, 0.35)), ecc: g.pick([0, 0, 0.3]) };
      if (type === 'flyby') return { kind, id, of: ['', ''], type, beta: r3(g.U(0.5, 2)), start: 3 };
      return { kind, id, of: ['', ''], type, impact: r3(g.U(0, 1)), separation: 0.3 };
    }
    case 'disk':
      return { kind, id, about: [], profile: ndimOf(doc) === 3 && g.rnd() < 0.5 ? 'balanced' : 'thin', rIn: 0.08, rOut: 0.4, sigma0: 1, slope: -1, cs: g.pick([0.05, 0.1]), floor: 1e-4 };
    case 'region':
      if (liquidRho(doc)) return { kind, id, shape: randomShape(doc, g, true, opts.shape), rho: r3(liquidRho(doc)! * g.U(0.95, 1.05)), v: vec(g, 0.5, doc), s: 1 };
      return { kind, id, shape: randomShape(doc, g, true, opts.shape), ...g.pick([{ rho: r3(g.L(3, 30)) }, { rho: r3(g.L(0.1, 0.5)) }, { p: r3(g.L(1, 30)) }]), s: 1 };
    case 'obstacle':
      return { kind, id, shape: randomShape(doc, g, false, opts.shape) };
    case 'inflow': {
      if (opts.shape) return { kind, id, shape: randomShape(doc, g, true, opts.shape) };
      const a = g.int(0, ndimOf(doc) - 1);
      const min = [-0.5, -0.5, -0.5];
      const max = [0.5, 0.5, 0.5];
      max[a] = -0.45;
      return { kind, id, shape: { type: 'box', min: min as never, max: max as never } };
    }
    case 'tunnel':
      return { kind, id, speed: 1, axis: 'x', mach: g.pick([0.1, 0.5, 2]), sponge: 0.1 };
    case 'field':
      return { kind, id, b: curved(doc) ? [0, r3(g.U(0.05, 0.3)), 0] : (vec(g, 0.3, doc).map(Math.abs) as never) };
    case 'flow':
      return { kind, id, name: opts.flow ?? g.pick(Object.keys(FLOWS)) };
  }
}

function randomShape(doc: SetupSpec, g: Rng, half = true, type?: string) {
  const t = type ?? g.pick(half ? ['sphere', 'sphere', 'box', 'half'] : ['sphere', 'box']);
  const c = centre(doc, g);
  const h = r3(g.U(0.03, 0.1));
  if (t === 'sphere') return { type: 'sphere' as const, center: c as never, radius: h };
  if (t === 'box') return { type: 'box' as const, min: c.map((x) => r3(x - h)) as never, max: c.map((x) => r3(x + h)) as never };
  return { type: 'half' as const, normal: (ndimOf(doc) === 3 ? [0, 0, 1] : [0, 1, 0]) as never, at: r3(g.U(-0.4, -0.2)) };
}

/** Add a part and what it needs: its kind's rules, the bodies an orbit or a disk names. */
function addPart(doc: SetupSpec, kind: PartKind, ctx: Ctx, opts: PartOpts = {}, depth = 0): Part | null {
  if (depth > 3) return null;
  const part = makePart(doc, kind, ctx, opts);
  (doc.parts ??= []).push(part);
  if (kind === 'star' || kind === 'disk') thinBackground(doc);
  if (kind === 'star' && !(typeof doc.physics?.G === 'string' || (doc.physics?.G ?? 1) > 0)) doc.physics!.G = 1;
  if (kind === 'orbit') {
    const o = part as Extract<Part, { kind: 'orbit' }>;
    const want: PartKind[] = o.type === 'collision' ? ['star', 'star'] : o.type === 'flyby' ? ['star', 'mass'] : [ctx.g.pick(['mass', 'star']), 'mass'];
    for (let k = 0; k < 2; k++) {
      const body = addPart(doc, want[k], ctx, {}, depth + 1);
      if (!body) return null;
      o.of[k] = body.id!;
    }
  }
  if (kind === 'disk') {
    const d = part as Extract<Part, { kind: 'disk' }>;
    const m = addPart(doc, 'mass', ctx, {}, depth + 1);
    if (!m) return null;
    Object.assign(m, { pos: [0, 0, 0], gm: 1, soft: 0.02, sink: 0.02 });
    d.about = [m.id!];
  }
  if (kind === 'tunnel' && !doc.parts!.some((p) => p.kind === 'obstacle')) doc.parts!.push({ kind: 'obstacle', id: newId(doc, 'obstacle'), shape: { type: 'sphere', center: [0, 0, 0], radius: 0.05 } });
  if (kind === 'flow') {
    const name = (part as { name: string }).name;
    if (name.startsWith('shearing')) applyShear(doc, ctx);
    if (name === 'zeldovich') applyCosmology(doc, ctx);
  }
  const k = PART_OPTIONS[kind];
  const index = doc.parts!.indexOf(part);
  // A flow's own needs are the rules of its choice in the menu of flows.
  const named = kind === 'flow' ? partFields(part).find((o) => o.path === 'name') : undefined;
  const choice = named?.control.type === 'select' ? named.control.choices.find((c) => c.value === (part as { name: string }).name) : undefined;
  if (!satisfy(doc, [k, ...(choice ? [choice] : [])], ctx, () => factsWithout(doc, doc.parts!.indexOf(part)))) return null;
  return index >= 0 && doc.parts!.includes(part) ? part : null;
}

// ---- what some features need besides their rules ----

function applyShear(doc: SetupSpec, ctx: Ctx) {
  const ph = (doc.physics ??= {});
  ph.shear ??= { omega: 1, q: 1.5 };
  setFaces(doc, [0, 0, 0], [0, 0, 0]);
  doc.domain!.amr = 0;
  clampGrid(doc);
  ph.gravity = 'off';
  delete ph.gext;
  delete ph.rotation;
  delete ph.cosmology;
  if (!(doc.parts ?? []).some((p) => p.kind === 'flow' && p.name.startsWith('shearing'))) (doc.parts ??= []).unshift({ kind: 'flow', id: newId(doc, 'flow'), name: ctx.g.pick(['shearing-noise', 'shearing-wave']) });
}

function applyCosmology(doc: SetupSpec, ctx: Ctx) {
  const ph = (doc.physics ??= {});
  ph.cosmology ??= { omegaM: 1, omegaL: 0, zStart: ctx.g.pick([10, 50]), zEnd: 0 };
  setFaces(doc, [0, 0, 0], [0, 0, 0]);
  ph.G = 1;
  if (ph.gravity === 'isolated' || ph.gravity === 'constant') ph.gravity = ndimOf(doc) === 3 ? 'periodic' : 'off';
  delete ph.gext;
  delete ph.rotation;
  delete ph.shear;
}

/** Species for fractions to be of, and their names. */
function ensureSpecies(doc: SetupSpec, g: Rng): string[] {
  const ph = (doc.physics ??= {});
  if (ph.species === 'aprox13') return ['he4', 'c12', 'o16'];
  if (!Array.isArray(ph.species) || !ph.species.length) ph.species = g.pick([['fuel', 'ash'], ['he4', 'c12', 'o16'], ['h1', 'he4']]);
  return ph.species as string[];
}

function fractions(names: string[], g: Rng): Record<string, number> {
  const w = names.map(() => g.rnd() + 0.05);
  const sum = w.reduce((a, b) => a + b, 0);
  return Object.fromEntries(names.map((n, i) => [n, r3(w[i] / sum)]));
}

/** A liquid's reference density (a wind tunnel's below Mach 0.3, or the liquid's own), or undefined for a gas. */
function liquidRho(doc: SetupSpec): number | undefined {
  const t = (doc.parts ?? []).find((p) => p.kind === 'tunnel');
  if (t && t.kind === 'tunnel') return (t.mach ?? 0.1) < 0.3 ? (t.rho ?? 1) : undefined;
  return doc.physics?.eos === 'liquid' ? (doc.physics.eosParams?.liqRho ?? 1) : undefined;
}

/** Values for numbers whose table range is too wide, by path ('physics.viscosity', 'mass.gm'). */
const HINTS: Record<string, (g: Rng, doc: SetupSpec, part?: Part) => unknown> = {
  'domain.amr': (g) => g.int(1, 5),
  'domain.box': (g) => g.pick([0.5, 2]),
  'domain.rMin': (g) => g.pick([0.005, 0.02, 0.05]),
  'physics.eosParams.gamma': (g) => g.pick([1.1, 1.3, 1.4, 5 / 3]),
  'physics.eosParams.K': (g) => r3(g.L(0.1, 1)),
  'physics.eosParams.arad': (g) => r3(g.L(1e-3, 0.1)),
  'physics.eosParams.degRho': () => 1,
  'physics.eosParams.degP': () => 1,
  'physics.eosParams.ionT': (g) => r3(g.L(0.1, 1)),
  'physics.eosParams.ionRho': (g) => r3(g.L(0.1, 10)),
  'physics.viscosity': (g) => r3(g.L(1e-4, 1e-3)),
  'physics.conductivity': (g) => r3(g.L(1e-4, 1e-2)),
  'physics.resistivity': (g) => r3(g.L(1e-4, 1e-3)),
  'physics.driving.accel': (g) => r3(g.L(0.1, 1)),
  'physics.fluff.rho': (g) => r3(g.L(1e-5, 1e-3)),
  'physics.fluff.u': (g) => r3(g.L(1, 100)),
  'physics.fluff.v': (g) => r3(g.L(1, 10)),
  'physics.fluff.va': (g) => g.pick([3, 10]),
  'physics.floors.rho': (g) => r3(g.L(1e-10, 1e-6)),
  'physics.floors.p': (g) => r3(g.L(1e-14, 1e-9)),
  'physics.coolParams.heat': (g) => r3(g.L(1e-3, 0.1)),
  'physics.coolParams.tFloor': (g) => r3(g.L(0.01, 0.3)),
  'physics.opacityParams.absorption': (g) => g.pick([0.1, 10, 1000]),
  'physics.opacityParams.scattering': (g) => g.pick([0.1, 1, 10]),
  'physics.opacityParams.kramers': (g) => r3(g.L(0.1, 10)),
  'physics.opacityParams.tFloor': (g) => r3(g.L(0.01, 0.1)),
  'physics.radC': (g) => g.pick([100, 1000]),
  'physics.radChat': (g) => g.pick([20, 50]),
  'physics.G': (g) => r3(g.U(0.5, 2)),
  'physics.gext': (g, d) => (ndimOf(d) === 3 ? [0, 0, -r3(g.U(0.1, 1))] : [0, -r3(g.U(0.1, 1)), 0]),
  'physics.rotation.omega': (g, d) => [ndimOf(d) === 3 && g.rnd() < 0.3 ? r3(g.U(0.1, 0.5)) : 0, 0, r3(g.U(0.1, 2))],
  'physics.cosmology.omegaL': (g) => r3(g.U(0.5, 0.7)),
  'physics.cosmology.zStart': (g) => g.pick([10, 50]),
  'physics.cosmology.zEnd': (g) => g.pick([1, 3]),
  'physics.cosmology.boxMpc': (g) => g.pick([10, 50, 100]),
  'mesh.refineRho': (g) => r3(g.L(1e-3, 0.1)),
  'mesh.refineDense': (g) => r3(g.L(1, 10)),
  'background.rho': (g) => r3(g.L(0.1, 10)),
  'background.p': (g) => r3(g.L(1e-3, 1)),
  'background.v': (g, d) => vec(g, 0.3, d),
  'background.s': (g) => r3(g.rnd()),
  'background.cut': (g) => r3(g.L(1e-5, 1e-3)),
  'mass.gm': (g) => r3(g.L(0.02, 0.5)),
  'mass.pos': (g, d) => centre(d, g),
  'mass.vel': (g, d) => vec(g, 0.3, d),
  'mass.soft': (g) => r3(g.U(0.02, 0.05)),
  'mass.sink': (g) => r3(g.U(0.01, 0.03)),
  'star.mass': (g, _d, p) => ((p as { type?: string })?.type === 'polytrope' ? r3(g.U(0.05, 0.2)) : r3(g.U(0.4, 1))),
  'star.radius': (g) => r3(g.U(0.06, 0.12)),
  'star.rhoc': (g) => r3(g.L(1, 30)),
  'star.pos': (g, d) => centre(d, g, 0.1),
  'star.vel': (g, d) => vec(g, 0.2, d),
  'star.core.mass': (g, _d, p) => r3(((p as { mass?: number })?.mass ?? 0.1) * g.U(0.1, 0.5)),
  'star.core.soft': (g, _d, p) => r3(((p as { radius?: number })?.radius ?? 0.1) * g.U(0.1, 0.2)),
  'orbit.separation': (g) => r3(g.U(0.2, 0.35)),
  'orbit.q': (g) => r3(g.L(0.1, 1)),
  'orbit.fill': (g) => r3(g.U(0.6, 1.1)),
  'orbit.incl': (g) => g.pick([30, 90, 180]),
  'disk.rIn': (g) => r3(g.U(0.05, 0.1)),
  'disk.rOut': (g) => r3(g.U(0.3, 0.45)),
  'disk.sigma0': (g, _d, p) => ((p as { profile?: string })?.profile === 'balanced' ? r3(g.L(0.1, 1)) : 1),
  'disk.cs': (g) => g.pick([0.05, 0.1]),
  'disk.floor': (g) => r3(g.L(1e-5, 1e-3)),
  'disk.radialFn': (g) => g.pick(['sigma0 * (r / rIn)^-1 * exp(-(r / rOut)^2) * step(r - rIn)', 'sigma0 * exp(-((r - 0.25) / 0.05)^2)']),
  'disk.verticalFn': (g) => g.pick(['exp(-z^2 / 2)', '1 / cosh(z)^2', 'max(1 - z / 3, 0)']),
  'region.rho': (g, d) => (liquidRho(d) ? r3(liquidRho(d)! * g.U(0.95, 1.05)) : r3(g.L(0.1, 30))),
  'region.p': (g) => r3(g.L(1e-3, 10)),
  'region.energy': (g) => r3(g.L(0.1, 2)),
  'region.v': (g, d) => vec(g, 1, d),
  'region.erad': (g) => r3(g.L(1e-3, 1)),
  'region.frad': (g, d) => vec(g, 0.5, d),
  'shape.center': (g, d) => centre(d, g),
  'shape.radius': (g) => r3(g.U(0.03, 0.12)),
  'shape.radiusCells': (g) => g.pick([2, 4, 8]),
  'shape.min': (g, d) => centre(d, g).map((x) => r3(x - 0.08)),
  'shape.max': (g, d) => centre(d, g).map((x) => r3(x + 0.08)),
  'shape.rotate': (g) => [0, 0, g.int(10, 80)],
  'shape.normal': (g, d) => (ndimOf(d) === 3 ? [r3(g.U(-1, 1)), r3(g.U(-1, 1)), 1] : [r3(g.U(-1, 1)), 1, 0]),
  'shape.at': (g) => r3(g.U(-0.3, 0)),
  'tunnel.rotor': (g) => r3(g.U(0.5, 2)),
  'tunnel.area': (g) => r3(g.L(0.005, 0.05)),
  'tunnel.length': (g) => r3(g.L(0.05, 0.2)),
  'field.b': (g, d) => (curved(d) ? [0, r3(g.U(0.05, 0.3)), 0] : vec(g, 0.3, d).map(Math.abs)),
  'field.up': (g) => r3(g.U(0.05, 0.3)),
};

/** A value for a number, a vector or a name from the table alone. */
function tableValue(o: Option, g: Rng, doc: SetupSpec): unknown {
  const c = o.control;
  if (c.type === 'vec3') return vec(g, 0.3, doc);
  if (c.type === 'names') return ensureSpecies(doc, g);
  if (c.type === 'fractions') {
    const names = ensureSpecies(doc, g);
    if (c.range) return Object.fromEntries(names.map((n) => [n, r3(g.U(Math.max(c.range!.min, 1.1), Math.min(c.range!.max ?? 1.67, 1.67)))]));
    return fractions(names, g);
  }
  if (c.type === 'check') return !o.default;
  if (c.type === 'text') return undefined;
  if (c.type !== 'number') return undefined;
  const d = typeof o.default === 'number' ? o.default : undefined;
  let v: number;
  if (d && c.factor) v = d * c.factor ** g.int(-2, 2);
  else if (c.min !== undefined && c.max !== undefined) v = c.step ? c.min + c.step * g.int(0, Math.floor((c.max - c.min) / c.step)) : g.U(c.min, c.max);
  else if (d) v = d * g.U(0.5, 1.5);
  else v = (c.min ?? 0) + (c.step ?? 0.1) * g.int(1, 5);
  if (c.min !== undefined) v = Math.max(c.min, v);
  if (c.max !== undefined) v = Math.min(c.max, v);
  return c.integer ? Math.round(v) : r3(v);
}

function valueFor(o: Option, key: string, g: Rng, doc: SetupSpec, part?: Part): unknown {
  const shape = key.includes('.shape.') ? `shape.${key.split('.shape.')[1]}` : undefined;
  const h = HINTS[key] ?? (shape ? HINTS[shape] : undefined);
  return h ? h(g, doc, part) : tableValue(o, g, doc);
}

/** A group's object: its fields' defaults (the ones that do something), in one of its forms. */
function groupValue(o: Option, key: string, g: Rng, doc: SetupSpec, form?: string): Record<string, unknown> {
  const c = o.control as Extract<Option['control'], { type: 'group' }>;
  const f = c.forms ? (c.forms.choices.find((x) => x.value === form) ?? g.pick(c.forms.choices)) : undefined;
  const v: Record<string, unknown> = {};
  for (const field of flatten(c.fields)) {
    if (field.control.type === 'group') continue;
    const k = field.path.split('.')[0];
    if (f && !f.keys.includes(k) && c.forms!.choices.some((x) => x.keys.includes(k))) continue;
    const hint = HINTS[`${key}.${field.path}`];
    const dv = hint ? hint(g, doc) : field.default;
    if (!trivial(dv) && !(field.control.type === 'check' && dv === false)) setPath(v, field.path, dv);
  }
  if (f?.set) Object.assign(v, f.set);
  return v;
}

// ---- rules ----

const holds = (r: Rule, facts: Record<string, Value>) => {
  const v = facts[r.fact];
  return r.is === undefined ? v === true : Array.isArray(r.is) ? r.is.includes(v) : v === r.is;
};

/**
 * How to make a fact of the setup hold (want) or not: a fact a feature's rules ask for. Each
 * returns false where it cannot.
 */
const FIXERS: Record<string, (doc: SetupSpec, want: boolean, value: Value | undefined, ctx: Ctx) => boolean> = {
  'domain.ndim': (doc, want, value, ctx) => setNdim(doc, (want ? value : value === 3 ? 2 : 3) as 2 | 3, ctx),
  'domain.supports2d': (doc, want) => !want || !!doc.domain?.supports2d,
  'domain.space3': (doc, want, _v, ctx) => (want ? (ctx.g.rnd() < 0.6 || !doc.domain?.supports2d ? setNdim(doc, 3, ctx) : (setGeometry(doc, 'cylindrical', ctx.g), true)) : setNdim(doc, 2, ctx) && (setGeometry(doc, 'cartesian', ctx.g), true)),
  'gravity.selfFlat': (doc, want) => {
    if (want) return false;
    if (ndimOf(doc) === 2 && !curved(doc)) (doc.physics ??= {}).gravity = 'off';
    return true;
  },
  'parts.starFlat': (doc, want) => (want ? false : (removeWhere(doc, (p) => p.kind === 'star'), true)),
  'domain.geometry': (doc, want, value, ctx) => (setGeometry(doc, (want ? value : 'cartesian') as 'cartesian', ctx.g), true),
  'symmetric.x': (doc, want) => symmetric(doc, 'x', want),
  'symmetric.y': (doc, want) => symmetric(doc, 'y', want),
  'symmetric.z': (doc, want) => symmetric(doc, 'z', want),
  'symmetric.field': (doc, want) => {
    if (want) doc.domain!.symmetricField = true;
    else delete doc.domain!.symmetricField;
    return true;
  },
  'mesh.adaptive': (doc, want, _v, ctx) => {
    if (want && doc.physics?.shear) return false;
    setGrid(doc, ctx.g, want ? ctx.g.pick([1, 2]) : 0);
    return true;
  },
  'physics.eos': (doc, want, value, ctx) => (setEos(doc, want ? String(value) : 'ideal', ctx.g), true),
  'eos.energy': (doc, want, _v, ctx) => (setEos(doc, want ? 'ideal' : ctx.g.pick(['barotropic', 'liquid']), ctx.g), true),
  'eos.tabulated': (doc, want, _v, ctx) => (setEos(doc, want ? ctx.g.pick(['helmholtz', 'hhe']) : 'ideal', ctx.g), true),
  'eos.degenerate': (doc, want, _v, ctx) => (setEos(doc, want ? 'degenerate' : 'ideal', ctx.g), true),
  'eos.aneos': (doc, want, _v, ctx) => (setEos(doc, want ? 'aneos' : 'ideal', ctx.g), true),
  'domain.curvilinear': (doc, want, _v, ctx) => (want ? doc.domain?.supports2d && (setGeometry(doc, ctx.g.pick(['cylindrical', 'spherical']), ctx.g), true) : (setGeometry(doc, 'cartesian', ctx.g), true)) === true,
  'physics.gravity': (doc, want, value, ctx) => {
    const ph = (doc.physics ??= {});
    ph.gravity = (want ? value : 'off') as never;
    if (want && (value === 'periodic' || value === 'isolated') && !(typeof ph.G === 'string' || (ph.G ?? 1) > 0)) ph.G = 1;
    return !!ctx;
  },
  'gravity.self': (doc, want, _v, ctx) => {
    const ph = (doc.physics ??= {});
    if (!want) return (ph.gravity = 'off'), true;
    ph.gravity = facesOf(doc).every((b) => b === PERIODIC) && ndimOf(doc) === 3 ? 'periodic' : 'isolated';
    if (ph.gravity === 'isolated') openFaces(doc);
    if (!(typeof ph.G === 'string' || (ph.G ?? 1) > 0)) ph.G = 1;
    return !!ctx;
  },
  'bc.periodic': (doc, want) => {
    if (want) {
      const bc = doc.domain?.bc ?? { lo: [1, 1, 1], hi: [1, 1, 1] };
      setFaces(doc, [0, ...bc.lo.slice(1)], [0, ...bc.hi.slice(1)]);
    } else openFaces(doc);
    return true;
  },
  'bc.allPeriodic': (doc, want) => (want ? (setFaces(doc, [0, 0, 0], [0, 0, 0]), true) : (openFaces(doc), true)),
  units: (doc, want, _v, ctx) => {
    const k = doc.physics?.cosmology;
    // (Comoving coordinates take theirs from the box's size in Mpc/h.)
    if (k) return want ? ((k.boxMpc = ctx.g.pick([10, 50, 100])), true) : (delete k.boxMpc, true);
    if (want) addUnits(doc, ctx.g);
    else delete doc.domain!.units;
    return true;
  },
  'species.nuclides': (doc, want, _v, ctx) => ((doc.physics!.species = want ? ['he4', 'c12', 'o16'] : ['fuel', 'ash']), !!ctx),
  'units.own': (doc, want, _v, ctx) => {
    if (!want) return false;
    if (doc.physics?.cosmology) return false;
    addUnits(doc, ctx.g);
    return true;
  },
  'pressure.implicit': (doc, want) => {
    if (want) return ((doc.physics ??= {}).pressure = 'implicit'), true;
    if ((doc.parts ?? []).some((p) => p.kind === 'tunnel')) return false;
    delete doc.physics?.pressure;
    return true;
  },
  'species.fuelAsh': (doc, want) => {
    if (!want) return false;
    doc.physics!.species = ['fuel', 'ash'];
    doc.background = { ...doc.background, X: { fuel: 1 } };
    return true;
  },
  'species.aprox13': (doc, want) => ((doc.physics!.species = want ? 'aprox13' : undefined), true),
  // The network burns all thirteen species, or watches he4, c12 and o16 for burning.
  'species.network': (doc, want, _v, ctx) => ((doc.physics!.species = want ? ctx.g.pick<'aprox13' | string[]>(['aprox13', ['he4', 'c12', 'o16']]) : undefined), true),
  radiation: (doc, want, _v, ctx) => {
    const ph = (doc.physics ??= {});
    if (!want) {
      delete ph.radiation;
      removeWhere(doc, (p) => p.kind === 'region' && (p.erad !== undefined || p.frad !== undefined));
      return true;
    }
    ph.radiation = 'm1';
    ph.radC ??= ctx.g.pick([100, 1000]);
    ph.radChat ??= ctx.g.pick([20, 50]);
    if (ph.eos === undefined || ph.eos === 'ideal') (ph.eosParams ??= {}).arad ??= r3(ctx.g.L(1e-3, 0.1));
    return true;
  },
  magnetic: (doc, want, _v, ctx) => {
    if (want) return !!addPart(doc, 'field', ctx);
    removeWhere(doc, (p) => p.kind === 'field' || (p.kind === 'flow' && (!!FLOWS[p.name]?.potential || !!FLOWS[p.name]?.face)));
    for (const p of doc.parts ?? []) {
      if (p.kind === 'star') delete p.field;
      if (p.kind === 'disk') delete p.loops;
    }
    return true;
  },
  'parts.mass': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'mass', ctx) : (removeWhere(doc, (p) => p.kind === 'mass' || (p.kind === 'star' && !!p.core)), true)),
  'parts.star': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'star', ctx) : (removeWhere(doc, (p) => p.kind === 'star'), true)),
  'parts.star.polytrope': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'star', ctx) : (removeWhere(doc, (p) => p.kind === 'star' && p.type === 'polytrope'), true)),
  'parts.star.white-dwarf': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'star', ctx, { type: 'white-dwarf' }) : (removeWhere(doc, (p) => p.kind === 'star' && p.type === 'white-dwarf'), true)),
  'parts.star.planet': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'star', ctx, { type: 'planet' }) : (removeWhere(doc, (p) => p.kind === 'star' && p.type === 'planet'), true)),
  'parts.body': (doc, want, _v, ctx) => (want ? !!addPart(doc, 'mass', ctx) : (removeWhere(doc, (p) => p.kind === 'mass' || p.kind === 'star'), true)),
  'parts.damping': (doc, want) => {
    if (want) return false;
    for (const p of doc.parts ?? []) {
      if (p.kind === 'disk') delete p.damp;
      if (p.kind === 'tunnel') p.sponge = 0;
    }
    return true;
  },
  'physics.shear': (doc, want, _v, ctx) => (want ? applyShear(doc, ctx) : (removeWhere(doc, (p) => p.kind === 'flow' && p.name.startsWith('shearing')), delete doc.physics!.shear), true),
  'physics.cosmology': (doc, want, _v, ctx) => (want ? applyCosmology(doc, ctx) : (removeWhere(doc, (p) => p.kind === 'flow' && p.name === 'zeldovich'), delete doc.physics!.cosmology), true),
  'parts.flow.whole': (doc, want) => (want ? false : (removeWhere(doc, (p) => p.kind === 'flow' && !!FLOWS[p.name]?.wholeBox), true)),
};

function symmetric(doc: SetupSpec, axis: string, want: boolean): boolean {
  const d = (doc.domain ??= {});
  const s = d.symmetric && d.symmetric !== 'none' ? d.symmetric : '';
  const next = want ? (s.includes(axis) ? s : s + axis) : s.replace(axis, '');
  d.symmetric = ['x', 'y', 'z'].filter((a) => next.includes(a)).join('') || 'none';
  if (!want && d.mirror?.includes(axis)) delete d.mirror;
  return true;
}

/** Make the rules of these options, choices or kinds hold, through the fixers. */
function satisfy(doc: SetupSpec, rulesets: { requires?: Rule[]; excludes?: Rule[] }[], ctx: Ctx, facts: () => Record<string, Value> = () => factsOf(doc)): boolean {
  for (let round = 0; round < 6; round++) {
    const f = facts();
    const bad: [Rule, boolean][] = [];
    for (const rs of rulesets) {
      for (const r of rs.requires ?? []) if (!holds(r, f)) bad.push([r, true]);
      for (const r of rs.excludes ?? []) if (holds(r, f)) bad.push([r, false]);
    }
    if (!bad.length) return true;
    const [r, want] = bad[0];
    const fix = FIXERS[r.fact];
    if (!fix) return false;
    const value = r.is === undefined ? true : Array.isArray(r.is) ? ctx.g.pick(r.is) : r.is;
    if (!fix(doc, want, value, ctx)) return false;
  }
  return false;
}

// ---- applying a feature ----

/** Set a feature in the document; false where it cannot be. Returns the part it is on, if any. */
function apply(doc: SetupSpec, f: Feature, ctx: Ctx): { ok: boolean; part?: Part } {
  const { g } = ctx;
  switch (f.type) {
    case 'face': {
      const k = ndimOf(doc);
      if (curved(doc) && f.code === PERIODIC) return { ok: false };
      const bc = doc.domain?.bc ?? { lo: [1, 1, 1], hi: [1, 1, 1] };
      const lo = [...bc.lo];
      const hi = [...bc.hi];
      if (f.code === PERIODIC) {
        const a = g.int(0, k - 1);
        lo[a] = hi[a] = PERIODIC;
      } else {
        // Faces of an axis are periodic together: the other face of an axis made not periodic is too.
        const a = g.int(0, k - 1);
        const side = curved(doc) && a === 0 ? hi : g.pick([lo, hi]);
        side[a] = f.code!;
        if (lo[a] === PERIODIC) lo[a] = OUTFLOW;
        if (hi[a] === PERIODIC) hi[a] = OUTFLOW;
        if (f.code === MOVING_WALL) {
          const key = FACE_KEYS[2 * a + (side === hi ? 1 : 0)];
          const v = [0, 0, 0];
          v[(a + 1) % k] = r3(g.U(0.2, 1));
          (doc.domain!.wallVelocity ??= {})[key] = v as never;
        }
      }
      setFaces(doc, lo, hi);
      return { ok: true };
    }
    case 'flow': {
      let part = (doc.parts ?? []).find((p) => p.kind === 'flow' && p.name === f.flow);
      if (!part) part = addPart(doc, 'flow', ctx, { flow: f.flow }) ?? undefined;
      if (!part) return { ok: false };
      if (f.path) {
        const def = FLOWS[f.flow!].fields[f.path];
        const o = flatten(partFields(part)).find((x) => x.path === f.path);
        (part as Record<string, unknown>)[f.path] = typeof def.default === 'boolean' ? !def.default : o ? tableValue(o, g, doc) : def.default;
      }
      return { ok: true, part };
    }
    case 'part': {
      const o = f.option;
      // The type or shape this field belongs to: a part is made with it, or one that has it is used.
      const typed = (p: string) => (f.path === p && f.choice ? String(f.choice.value) : o?.show?.path === p ? String(g.pick(o.show.is)) : undefined);
      const type = typed('type');
      const shape = typed('shape.type');
      const fits = (p: Part) => p.kind === f.kind && (!o?.show || o.show.is.includes(getPath(p, o.show.path) as Value)) && !(f.path === 'type' || f.path === 'shape.type');
      let part = g.rnd() < 0.5 ? g.pick((doc.parts ?? []).filter(fits)) : undefined;
      if (!part) part = addPart(doc, f.kind!, ctx, { type, shape }) ?? undefined;
      if (!part) return { ok: false };
      if (!f.path) return { ok: true, part };
      const key = `${f.kind}.${f.path}`;
      // A field of a group: the group first, with its defaults.
      const top = f.path.split('.')[0];
      if (top !== f.path && getPath(part, top) === undefined) {
        const go = PART_OPTIONS[f.kind!].fields.find((x) => x.path === top);
        if (go) setPath(part, top, groupValue(go, `${f.kind}.${top}`, g, doc));
      }
      if (f.choice) setPath(part, f.path, f.choice.value);
      else if (o!.control.type === 'group') setPath(part, f.path, groupValue(o!, key, g, doc, f.form));
      else {
        const v = f.off ? false : valueFor(o!, key, g, doc, part);
        if (v === undefined) return { ok: false };
        setPath(part, f.path, v);
      }
      partHooks(doc, part, f, ctx);
      const idx = () => doc.parts!.indexOf(part!);
      // The rules of the choice that shows this field, and of the part's kind of star or orbit.
      const shownBy = o!.show ? flatten(PART_OPTIONS[f.kind!].fields).find((x) => x.path === o!.show!.path) : undefined;
      const shownChoice = shownBy?.control.type === 'select' ? shownBy.control.choices.find((c) => c.value === getPath(part, o!.show!.path)) : undefined;
      const rules = [o!, ...(f.choice ? [f.choice] : []), ...(shownChoice ? [shownChoice] : [])];
      return { ok: satisfy(doc, rules, ctx, () => factsWithout(doc, idx())) && idx() >= 0, part };
    }
    case 'option': {
      const o = f.option!;
      const path = f.path!;
      if (o.show) {
        const cur = getPath(doc, o.show.path) as Value;
        if (!o.show.is.includes(cur)) {
          const want = g.pick(o.show.is);
          const showOpt = flatten(OPTIONS).find((x) => x.path === o.show!.path);
          const choice = showOpt?.control.type === 'select' ? showOpt.control.choices.find((c) => c.value === want) : undefined;
          optionHooks(doc, o.show.path, want, ctx);
          if (!satisfy(doc, [showOpt ?? {}, choice ?? {}], ctx)) return { ok: false };
        }
      }
      // A field of a group: the group first, in a form that has the field.
      const groupPath = flatten(OPTIONS).find((x) => x.control.type === 'group' && path.startsWith(`${x.path}.`));
      if (groupPath && getPath(doc, groupPath.path) === undefined) {
        const gc = groupPath.control as Extract<Option['control'], { type: 'group' }>;
        const key = path.slice(groupPath.path.length + 1).split('.')[0];
        const form = gc.forms?.choices.find((c) => c.keys.includes(key))?.value;
        const gv = groupValue(groupPath, groupPath.path, g, doc, form);
        if (!optionHooks(doc, groupPath.path, gv, ctx)) setPath(doc, groupPath.path, gv);
        const formRules = gc.forms ? [formOf(gc.forms, getPath(doc, groupPath.path))] : [];
        if (!satisfy(doc, [groupPath, ...formRules], ctx)) return { ok: false };
      }
      let value: unknown;
      if (f.choice) value = f.choice.value;
      else if (o.control.type === 'group') value = groupValue(o, path, g, doc, f.form);
      else if (f.off) value = false;
      else value = valueFor(o, path, g, doc);
      // (A name its group already set, as the fuel and ash of burning are.)
      if (value === undefined && getPath(doc, path) !== undefined) value = getPath(doc, path);
      if (value === undefined) return { ok: false };
      if (!optionHooks(doc, path, value as Value, ctx)) setPath(doc, path, value);
      const rules: { requires?: Rule[]; excludes?: Rule[] }[] = [o, ...(f.choice ? [f.choice] : [])];
      if (f.form && o.control.type === 'group') rules.push(o.control.forms!.choices.find((c) => c.value === f.form)!);
      return { ok: satisfy(doc, rules, ctx) };
    }
  }
}

/** What some options need besides their rules. Returns true where the hook has set the option itself. */
function optionHooks(doc: SetupSpec, path: string, value: Value | Record<string, unknown> | unknown, ctx: Ctx): boolean {
  const { g } = ctx;
  switch (path) {
    case 'domain.geometry':
      setGeometry(doc, value as 'cartesian', g);
      return true;
    case 'domain.ndim':
      return setNdim(doc, value as 2 | 3, ctx) || true;
    case 'domain.amr':
    case 'domain.n':
      setPath(doc, path, value);
      clampGrid(doc);
      return true;
    case 'domain.rMin':
    case 'domain.logR':
      if (doc.domain?.geometry !== 'spherical') setGeometry(doc, 'spherical', g);
      return false;
    case 'domain.mirror': {
      const m = String(value);
      for (const a of m) symmetric(doc, a, true);
      if (FIXERS.magnetic && (doc.parts ?? []).some((p) => p.kind === 'field')) doc.domain!.symmetricField = true;
      return false;
    }
    case 'physics.eos':
      setEos(doc, String(value), g);
      return true;
    case 'physics.shear':
      applyShear(doc, ctx);
      if (value && typeof value === 'object') Object.assign(doc.physics!.shear!, value);
      return true;
    case 'physics.cosmology':
      applyCosmology(doc, ctx);
      if (value && typeof value === 'object') Object.assign(doc.physics!.cosmology!, value);
      return true;
    case 'physics.burn':
      if (value && typeof value === 'object' && 'fuel' in value) {
        const sp = doc.physics?.species;
        if (!Array.isArray(sp) || sp.some((s) => /^[a-z]+\d+$/.test(s))) doc.physics!.species = ['fuel', 'ash'];
        const names = doc.physics!.species as string[];
        Object.assign(value, { fuel: names[0], ash: names[1] });
        doc.background = { ...doc.background, X: { [names[0]]: 1 } };
      }
      return false;
    case 'physics.radiation':
      return FIXERS.radiation(doc, value !== 'off', true, ctx);
    case 'physics.pressure':
      if (value === 'implicit') delete doc.mesh?.timeBins;
      return false;
    case 'physics.gravity':
      if (value === 'periodic' || value === 'isolated' || value === 'constant') (doc.physics ??= {}).G ??= 1;
      if (value === 'isolated') openFaces(doc);
      if (value === 'periodic') setFaces(doc, [0, 0, 0], [0, 0, 0]);
      return false;
    case 'physics.frame':
      if (!(doc.parts ?? []).some((p) => p.kind === 'mass')) addPart(doc, 'mass', ctx);
      return false;
  }
  if (path.startsWith('domain.wallVelocity.')) {
    const key = path.split('.')[2] as (typeof FACE_KEYS)[number];
    const a = Math.floor(FACE_KEYS.indexOf(key) / 2);
    if (a >= ndimOf(doc)) return false;
    const bc = doc.domain?.bc ?? { lo: [1, 1, 1], hi: [1, 1, 1] };
    const lo = [...bc.lo];
    const hi = [...bc.hi];
    (key.endsWith('lo') ? lo : hi)[a] = MOVING_WALL;
    if (lo[a] === PERIODIC) lo[a] = OUTFLOW;
    if (hi[a] === PERIODIC) hi[a] = OUTFLOW;
    setFaces(doc, lo, hi);
    const v = [0, 0, 0];
    v[(a + 1) % ndimOf(doc)] = r3(g.U(0.2, 1));
    setPath(doc, path, v);
    return true;
  }
  return false;
}

/** A planet's materials are the run's: its mantle's and core's as the two species, or its one material as the equation of state's. */
function planetMaterials(doc: SetupSpec, part: Extract<Part, { kind: 'star' }>) {
  const ph = (doc.physics ??= {});
  // A core of the mantle's own material: the mantle becomes another.
  if (part.coreMaterial && part.coreMaterial === (part.material ?? 'pyrolite')) part.material = ANEOS_MATERIALS.find((m) => m !== part.coreMaterial);
  const mantle = part.material ?? 'pyrolite';
  if (part.coreMaterial) ph.species = [mantle, part.coreMaterial];
  else {
    if (Array.isArray(ph.species) && ph.species.includes(mantle)) return;
    delete ph.species;
    (ph.eosParams ??= {}).aneosA = ANEOS_MATERIALS.indexOf(mantle as never);
  }
}

/** What some parts' fields need besides their rules. */
function partHooks(doc: SetupSpec, part: Part, f: Feature, ctx: Ctx) {
  if (part.kind === 'star' && part.type === 'planet') planetMaterials(doc, part);
  if (part.kind === 'star' && f.path === 'rhoc') delete part.mass;
  if (part.kind === 'star' && f.path?.startsWith('core')) delete part.rhoc;
  if (part.kind === 'star' && f.path === 'type') {
    const fresh = makePart(doc, 'star', ctx, { type: part.type });
    for (const k of Object.keys(part)) if (!['kind', 'id', 'type', 'pos'].includes(k)) delete (part as unknown as Record<string, unknown>)[k];
    Object.assign(part, { ...fresh, id: part.id, pos: part.pos ?? (fresh as typeof part).pos });
  }
  if (part.kind === 'orbit' && f.path === 'type' && f.choice) {
    // A new orbit of this type is made with its bodies; an orbit changed in place keeps them if they fit.
    const fresh = makePart(doc, 'orbit', ctx, { type: String(f.choice.value) });
    Object.assign(part, { ...fresh, id: part.id, of: part.of });
  }
  if (part.kind === 'orbit' && f.path === 'fill') delete part.separation;
  if (part.kind === 'disk' && f.path === 'profile' && part.profile === 'balanced') part.sigma0 = 0.3;
  if (part.kind === 'disk' && f.path === 'loops') {
    if (part.profile !== 'balanced' && ndimOf(doc) === 3) part.profile = 'balanced';
    (doc.physics ??= {}).fluff = { ...doc.physics.fluff, va: 3 };
  }
  if (part.kind === 'disk' && f.path === 'damp' && part.damp) part.damp = { from: r3(ctx.g.U(0.3, 0.45)), rate: 3 };
  if (part.kind === 'star' && part.type === 'planet') planetMaterials(doc, part);
}

/**
 * Take out what the page's check flags, other than the feature itself. False where the target
 * itself is wrong, or what is wrong cannot be taken out.
 */
function repair(doc: SetupSpec, target: { path?: string; part?: Part }): boolean {
  for (let round = 0; round < 8; round++) {
    const errs = errorsOf(doc);
    if (!errs.length) return true;
    const e = errs[0];
    if (!e.path) return false;
    const parts = doc.parts ?? [];
    const tIndex = target.part ? parts.indexOf(target.part) : -1;
    const m = /^parts\.(\d+)(?:\.(.+))?$/.exec(e.path);
    if (m) {
      const i = +m[1];
      if (i === tIndex) return false;
      if (!m[2] || /^(of|about|shape|name|type)/.test(m[2])) removePart(doc, i);
      else setPath(parts[i], m[2], undefined);
      continue;
    }
    if (target.path && (e.path === target.path || e.path.startsWith(`${target.path}.`) || target.path.startsWith(`${e.path}.`))) return false;
    if (e.path === 'domain.bc') openFaces(doc);
    else if (e.path === 'domain.n') return false;
    else if (e.path === 'background.rho' && doc.background?.relativeTo) delete doc.background.relativeTo;
    else setPath(doc, e.path, undefined);
  }
  return errorsOf(doc).length === 0;
}

/** Try a feature on a copy of the document: the copy if it worked, else why not. */
function tryFeature(doc: SetupSpec, f: Feature, ctx: Ctx): { doc: SetupSpec } | { why: string; doc: SetupSpec } {
  const trial = cloneSetup(doc);
  let out: { ok: boolean; part?: Part };
  try {
    out = apply(trial, f, ctx);
  } catch (e) {
    return { why: `threw: ${(e as Error).message}`, doc: trial };
  }
  if (!out.ok) return { why: 'its rules could not be met', doc: trial };
  clampGrid(trial);
  // (After the rules: an equation of state set for them takes the planet's material with it.)
  for (const p of trial.parts ?? []) if (p.kind === 'star' && p.type === 'planet') planetMaterials(trial, p);
  if (!repair(trial, { path: f.type === 'option' ? f.path : undefined, part: out.part })) return { why: `the check: ${errorsOf(trial)[0]?.text ?? '(repair gave up)'}`, doc: trial };
  return hasFeature(trial, f) ? { doc: trial } : { why: 'the repair took it out', doc: trial };
}

/**
 * Why a feature does or does not make it into documents: it is tried on the compose base of
 * each seed, and each attempt's outcome returned. For finding out why the coverage misses one.
 */
export function attempt(id: string, seeds: number[] = [1, 3, 5, 7, 9]): { seed: number; ok: boolean; why?: string; doc: SetupSpec }[] {
  const f = features().find((x) => x.id === id);
  if (!f) throw new Error(`no feature ${id}`);
  return seeds.map((seed) => {
    const g = rng(seed);
    const r = tryFeature(baseDoc(seed, g), f, { g });
    return { seed, ok: !('why' in r), why: 'why' in r ? r.why : undefined, doc: r.doc };
  });
}

// ---- documents ----

export interface Generated {
  seed: number;
  mode: 'compose' | 'preset';
  /** the preset it started from */
  preset?: string;
  doc: SetupSpec;
  /** the features drawn, and whether each made it in */
  tried: { id: string; ok: boolean }[];
  /** every feature the document has */
  features: string[];
  /** no face lets gas through and nothing adds or takes it: the mass should not change */
  closed: boolean;
}

function baseDoc(seed: number, g: Rng): SetupSpec {
  const doc: SetupSpec = {
    format: SETUP_FORMAT,
    version: SETUP_VERSION,
    name: `random ${seed}`,
    domain: { ndim: g.pick([2, 2, 3]), box: 1, supports2d: true },
    physics: { eos: 'ideal', eosParams: { gamma: g.pick([5 / 3, 1.4, 1.1]) }, gravity: 'off' },
    background: { rho: 1, p: g.pick([1e-4, 0.01, 1]) },
    parts: [],
    run: { tEnd: null },
  };
  setGrid(doc, g);
  const b = g.pick([0, 1, 2]);
  setFaces(doc, [b, b, b], [b, b, b]);
  return doc;
}

/** Something that moves the gas, so that a run of plain gas is not a box at rest. */
function stirs(doc: SetupSpec): boolean {
  const ph = doc.physics ?? {};
  return (doc.parts ?? []).some((p) => p.kind !== 'field' && p.kind !== 'obstacle') || !!ph.driving?.accel || !!ph.cosmology || !!ph.shear || !!doc.background?.v?.some((x) => x);
}

function dress(doc: SetupSpec, ctx: Ctx): SetupSpec {
  if (stirs(doc)) return doc;
  const { g } = ctx;
  for (let i = 0; i < 3; i++) {
    const trial = cloneSetup(doc);
    const shape = { type: 'sphere' as const, center: centre(trial, g) as never, radius: r3(g.U(0.03, 0.08)) };
    const blast = factsOf(trial)['eos.energy'] && g.rnd() < 0.5;
    trial.parts!.push(blast ? { kind: 'region', id: newId(trial, 'region'), shape, energy: r3(g.L(0.1, 2)), s: 1 } : { kind: 'region', id: newId(trial, 'region'), shape, rho: r3(g.L(3, 30)), v: vec(g, 1, trial) as never, s: 1 });
    if (repair(trial, {})) return trial;
  }
  return doc;
}

/** Whether a document's mass should stay what it was: walls and periodic faces, and no source or sink. */
export function closedBox(doc: SetupSpec): boolean {
  const shut = new Set(BOUNDARIES.filter((b) => b.periodic || b.wall).map((b) => b.code));
  if (!facesOf(doc).every((b) => shut.has(b))) return false;
  if ((doc.physics?.fluff?.va ?? 0) > 0) return false;
  return !(doc.parts ?? []).some((p) => (p.kind === 'mass' && (p.sink ?? 1) > 0) || (p.kind === 'star' && !!p.core) || p.kind === 'inflow' || p.kind === 'tunnel' || (p.kind === 'disk' && !!p.damp));
}

/** A random setup document: a deterministic function of the seed (and of the build's option table). */
export function generate(seed: number, opts: { mode?: 'compose' | 'preset'; features?: number } = {}): Generated {
  const g = rng(seed);
  const mode = opts.mode ?? (seed % 2 ? 'compose' : 'preset');
  const all = features();
  let doc: SetupSpec;
  let preset: string | undefined;
  let count: number;
  if (mode === 'preset') {
    preset = g.pick(PRESET_IDS);
    doc = cloneSetup(presetSpec(preset)!);
    delete doc.featured;
    delete doc.id;
    doc.name = `random ${seed} (${preset})`;
    doc.domain ??= {};
    const own = cloneSetup(doc);
    if (!curved(doc)) doc.domain.ndim = doc.domain.supports2d && g.rnd() < 0.5 ? 2 : 3;
    setGrid(doc, g);
    if (errorsOf(doc).length && !repair(doc, {})) doc = own;
    count = g.int(0, 3);
  } else {
    doc = baseDoc(seed, g);
    count = g.int(2, 6);
  }
  const ctx: Ctx = { g, preset };
  const tried: Generated['tried'] = [];
  for (let i = 0; i < count; i++) {
    const f = g.pick(all);
    const next = tryFeature(doc, f, ctx);
    const ok = !('why' in next);
    tried.push({ id: f.id, ok });
    if (ok) doc = next.doc;
  }
  if (mode === 'compose') doc = dress(doc, ctx);
  return { seed, mode, preset, doc, tried, features: featuresOf(doc), closed: closedBox(doc) };
}

// ---- coverage ----

export interface Coverage {
  docs: number;
  /** features in the list, and how many documents had each */
  counts: Record<string, number>;
  missing: string[];
  /** tries of each feature that the check turned away */
  refused: Record<string, number>;
  /** pairs of the main features (choices, groups, kinds of part and flow) seen together, of all such pairs */
  pairs: { seen: number; of: number };
}

/** The features worth pairing: choices and groups of the physics and the grid, kinds of part, flows. */
const major = (f: Feature) => f.type === 'face' || (f.type === 'part' && !f.path) || (f.type === 'flow' && !f.path) || (f.type === 'option' && (!!f.choice || f.option!.control.type === 'group'));

export function coverage(gens: Generated[]): Coverage {
  const all = features();
  const counts: Record<string, number> = Object.fromEntries(all.map((f) => [f.id, 0]));
  const refused: Record<string, number> = {};
  const majors = new Set(all.filter(major).map((f) => f.id));
  const pairs = new Set<string>();
  for (const gen of gens) {
    for (const id of gen.features) counts[id] = (counts[id] ?? 0) + 1;
    for (const t of gen.tried) if (!t.ok) refused[t.id] = (refused[t.id] ?? 0) + 1;
    const m = gen.features.filter((id) => majors.has(id));
    for (let i = 0; i < m.length; i++) for (let j = i + 1; j < m.length; j++) pairs.add(`${m[i]} & ${m[j]}`);
  }
  const n = majors.size;
  return { docs: gens.length, counts, missing: all.filter((f) => counts[f.id] === 0).map((f) => f.id), refused, pairs: { seen: pairs.size, of: (n * (n - 1)) / 2 } };
}
