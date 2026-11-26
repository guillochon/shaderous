// Shareable links: the simulator's settings as URL query parameters, and back.
//
// A link names the preset, dimensionality, resolution and refinement, and then only what
// differs from that preset's defaults, so most links stay short. Anything unrecognized or
// out of range in a link is ignored rather than trusted.
import { MAX_AMR_LEVELS } from '../amr/tree';
import { BOUNDARY_CODES, boundaryList } from '../problems/boundaries';
import { configFor, unitConstants, type SimConfig } from '../sim';
import { mirrorAxes, mirrorName } from '../domain';
import { physicalEosParams, type EosParams } from '../eos/eos';
import type { CoolingParams } from '../cooling/cooling';
import { FIELDS, MASS_FRACTION_FIELD, problemUnits, type Bc3, type Problem } from '../problems';
import { CUSTOM_ID } from '../presets/custom';
import { hasProblem, problemOf } from '../presets/lookup';
import { flatten, OPTIONS, PART_OPTIONS, type Option } from '../setup/options';
import { defaultView, type ViewState } from '../render/renderer';
import { MAX_CONTOURS, MAX_DIST, MAX_ZOOM, MIN_DIST, VIEW_MODES, VIEW_OPTIONS, type ViewOption } from '../render/view_options';
import { COLORMAPS, defaultOpacity, type OpacityPoint } from '../render/transfer';
import { SNAPSHOT_FORMAT, type SnapshotMeta } from '../io/snapshot';
import { decodeProfileLines, encodeProfileLines, MAX_PROFILES, type Vec3 } from './profile';

export { MAX_CONTOURS, MAX_DIST, MAX_ZOOM, MIN_DIST };

/** What a fresh visit shows. */
export const DEFAULT_SETUP = { problem: 'kelvin-helmholtz', ndim: 2 as 2 | 3, n: 512, amr: 3 };

/**
 * The settings a link carries under keys of the option table (setup/options.ts): each option
 * with a `linkKey` and the SimConfig field it sets. The grid's keys and the faces' are written
 * out below; `wb` is a switch of the tests, not an option.
 */
const LINKED: Option[] = [...flatten(OPTIONS), ...flatten(PART_OPTIONS.disk.fields)].filter((o) => o.linkKey && o.cfg && !['bc', 'tEnd'].includes(o.cfg));
/** Options that only mean something on an adaptive mesh, or with a cooling function: left out of a link otherwise. */
const adaptiveOnly = (o: Option) => !!o.requires?.some((r) => r.fact === 'mesh.adaptive');
const coolingOnly = (o: Option) => o.cfg!.startsWith('coolParams.');
/** Options that only mean something with radiation transport: left out of a link without it. */
const radiationOnly = (o: Option) => o.cfg!.startsWith('opacityParams.') || ['opacity', 'radC', 'radChat', 'radLocalC', 'radSolver', 'radSweeps'].includes(o.cfg!);
const at = (c: object, path: string): unknown => path.split('.').reduce<unknown>((v, k) => (v as Record<string, unknown> | undefined)?.[k], c);

/**
 * Profile lines on show (the Profile tool, app.ts): their ends in fractions of the box, and the
 * field plotted along them (default: the one the view shows) with its log axis (null: as the
 * field's default).
 */
export interface LinkProfiles {
  lines: [Vec3, Vec3][];
  /** the fields plotted along them, in the panel's order (left out: the one the view shows) */
  fields?: { field: number; species?: number; log?: boolean | null }[];
}

export interface LinkState {
  /** the custom setup's document, packed for a link (setup/codec.ts encodeSetupParam), when the run is one */
  setup?: string;
  cfg: SimConfig;
  view: ViewState;
  /** the displayed range follows the data */
  auto: boolean;
  colormap: string;
  opacity: OpacityPoint[];
  profiles?: LinkProfiles;
}

export interface LinkOverrides {
  problem: string;
  ndim: 2 | 3;
  n: number; // 0: the default for that dimensionality
  amr: number;
  cfg: Partial<Pick<SimConfig, 'symmetry' | 'tEnd' | 'eos' | 'riemann' | 'limiter' | 'cfl' | 'timeBins' | 'refineHi' | 'refineLo' | 'refineRho' | 'gravity' | 'G' | 'vcycles' | 'jeansFloor' | 'jeansCells' | 'gravEnergyFlux' | 'wellBalanced' | 'bcLo' | 'bcHi' | 'cooling' | 'viscosity' | 'conductivity' | 'resistivity' | 'dampRadius' | 'dampRate' | 'radiation' | 'opacity' | 'radC' | 'radChat' | 'radLocalC' | 'radSolver' | 'radSweeps'>>;
  eosParams: Partial<EosParams>;
  coolParams: Partial<CoolingParams>;
  opacityParams: Partial<SimConfig['opacityParams']>;
  fluff: Partial<SimConfig['fluff']>;
  startDamping: Partial<SimConfig['startDamping']>;
  driving: Partial<SimConfig['driving']>;
  params: Record<string, number>;
  view: Partial<ViewState>;
  auto?: boolean;
  colormap?: string;
  opacity?: OpacityPoint[];
  profiles?: LinkProfiles;
}

/** The view a preset opens with. */
export function viewDefaults(problem: Problem, ndim: 2 | 3): ViewState {
  const v = defaultView();
  v.field = problem.view?.field ?? 0;
  v.log = problem.view?.log ?? true;
  v.mode = ndim === 2 ? 'slice' : (problem.view?.mode ?? 'volume');
  v.sliceAxis = ndim === 2 ? 2 : (problem.view?.axis ?? 2);
  v.pitch = problem.view?.pitch ?? 0.45;
  // A magnetized run shows its field lines.
  v.fieldLines = !!problem.mhd;
  // 2D: contours of the flux function, fewer than the tubes of 3D.
  v.fieldLineCount = ndim === 2 ? 24 : 160;
  return v;
}

const fmt = (v: number) => (Number.isFinite(v) ? String(+v.toPrecision(6)) : 'inf');
const flag = (b: boolean) => (b ? '1' : '0');
// The view's settings come from one table (render/view_options.ts). The ends of the color
// range travel as a pair, and only when the range is not automatic.
const VIEW_LINKED = VIEW_OPTIONS.filter((o) => o.key !== 'lo' && o.key !== 'hi');

export interface LinkKey {
  key: string;
  /** 'run': which preset and grid; 'settings': what differs from the preset's own; 'view': what is drawn */
  group: 'run' | 'settings' | 'view';
  about: string;
  /** more, for the docs: an option's tooltip */
  detail?: string;
}

/**
 * Every key a link (or a setup given to the scripting API as a query string or a flat object)
 * may carry. The docs and the headless tools list these, so a key added to the option table or
 * to the view's table shows up there without being written again.
 */
export function linkKeys(): LinkKey[] {
  const settings = LINKED.map((o): LinkKey => {
    const c = o.control;
    const range = c.type === 'select' ? `: ${c.choices.map((k) => k.value).join(', ')}` : c.type === 'check' ? ' (0 or 1)' : '';
    return { key: o.linkKey!, group: 'settings', about: `${o.label}${range}`, detail: o.tip };
  });
  return [
    { key: 'problem', group: 'run', about: 'Preset id' },
    { key: 'setup', group: 'run', about: 'A setup document of one’s own, packed (with problem=custom)' },
    { key: 'dim', group: 'run', about: '2 or 3 dimensions' },
    { key: 'n', group: 'run', about: 'Cells per side (the finest level’s on an adaptive mesh). Left out: the preset’s own grid' },
    { key: 'amr', group: 'run', about: 'Levels of refinement, 0 to 7. With n but no amr, the grid is uniform' },
    { key: 'tend', group: 'settings', about: 'End time (inf for none)' },
    ...settings,
    { key: 'wb', group: 'settings', about: 'Well-balanced reconstruction (0 or 1)' },
    { key: 'bc', group: 'settings', about: `Six digits: the boundary at the lower x, y, z faces and then the upper ones (${boundaryList()})` },
    { key: 'p.<id>', group: 'settings', about: 'One of the preset’s own parameters, by its id' },
    ...VIEW_OPTIONS.map((o): LinkKey => ({ key: o.link, group: 'view', about: o.doc })),
    { key: 'cmap', group: 'view', about: 'Colormap' },
    { key: 'tf', group: 'view', about: 'Volume view: the opacity curve, as x:a points separated by commas' },
    { key: 'prof', group: 'view', about: `Profile lines on show, at most ${MAX_PROFILES}: each x0,y0,x1,y1 (2D) or x0,y0,z0,x1,y1,z1 (3D) in fractions of the box, ; between lines` },
    { key: 'proff', group: 'view', about: 'The fields plotted along the profile lines, indices into the fields separated by commas, the first against the left axis (default: the one shown)' },
    { key: 'profsp', group: 'view', about: 'For each of proff, the species of a mass fraction, by index (commas; empty or 0 for other fields)' },
    { key: 'proflog', group: 'view', about: 'For each of proff, its axis logarithmic (1) or linear (0), or empty for the field\u2019s default (commas)' },
  ];
}

export function encodeLink(s: LinkState): URLSearchParams {
  const { cfg, view } = s;
  const q = new URLSearchParams();
  q.set('problem', cfg.problem.id);
  if (cfg.problem.id === CUSTOM_ID && s.setup) q.set('setup', s.setup);
  q.set('dim', String(cfg.ndim));
  q.set('n', String(cfg.n));
  if (cfg.amrLevels) q.set('amr', String(cfg.amrLevels));

  // (With the run's mirrors: their planes' reflecting faces are theirs, not the link's to say.)
  const def = configFor(cfg.problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels, symmetry: cfg.symmetry });
  // Units that follow the preset's parameters set the equation of state's physical constants: the
  // defaults to compare with are those of the run's parameters, which the link carries.
  if (typeof cfg.problem.units === 'function') unitConstants(def);
  const num = (key: string, a: number, b: number) => {
    if (fmt(a) !== fmt(b)) q.set(key, fmt(a));
  };
  const str = (key: string, a: string, b: string) => {
    if (a !== b) q.set(key, a);
  };
  num('tend', cfg.tEnd, def.tEnd);
  for (const o of LINKED) {
    if ((adaptiveOnly(o) && !cfg.amrLevels) || (coolingOnly(o) && cfg.cooling === 'off') || (radiationOnly(o) && cfg.radiation === 'off')) continue;
    const [a, b] = [at(cfg, o.cfg!) ?? 0, at(def, o.cfg!) ?? 0];
    if (o.cfg === 'symmetry') str(o.linkKey!, mirrorName(cfg.symmetry), mirrorName(configFor(cfg.problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels }).symmetry));
    else if (typeof a === 'boolean') str(o.linkKey!, flag(a), flag(!!b));
    else if (typeof a === 'number' && o.control.type === 'number') num(o.linkKey!, a, b as number);
    else str(o.linkKey!, String(a), String(b));
  }
  str('wb', flag(cfg.wellBalanced), flag(def.wellBalanced));
  // Boundaries: one digit per face, the lower faces of x, y, z and then the upper ones.
  str('bc', [...cfg.bcLo, ...cfg.bcHi].join(''), [...def.bcLo, ...def.bcHi].join(''));
  for (const p of cfg.problem.params ?? []) num(`p.${p.id}`, cfg.params[p.id], p.value);

  const dv = viewDefaults(cfg.problem, cfg.ndim);
  // In the order links have always been written: the mode, the numbers, then the switches.
  for (const o of VIEW_LINKED) if (o.type === 'mode') str(o.link, view[o.key], dv[o.key]);
  for (const o of VIEW_LINKED) if (o.type === 'number' || o.type === 'int') num(o.link, view[o.key], dv[o.key]);
  for (const o of VIEW_LINKED) if (o.type === 'boolean') str(o.link, flag(view[o.key]), flag(dv[o.key]));
  if (!s.auto) {
    q.set('lo', fmt(view.lo));
    q.set('hi', fmt(view.hi));
  }
  str('cmap', s.colormap, 'inferno');
  const pts = (p: OpacityPoint[]) => p.map((o) => `${+o.x.toFixed(3)}:${+o.a.toFixed(3)}`).join(',');
  str('tf', pts(s.opacity), pts(defaultOpacity()));
  const pr = s.profiles;
  if (pr?.lines.length) {
    q.set('prof', encodeProfileLines(pr.lines.map(([a, b]) => ({ a, b })), cfg.ndim));
    const fs = pr.fields ?? [];
    if (fs.length) {
      q.set('proff', fs.map((f) => f.field).join(','));
      const sp = fs.map((f) => (f.field === MASS_FRACTION_FIELD && f.species ? String(f.species) : ''));
      if (sp.some(Boolean)) q.set('profsp', sp.join(','));
      const logs = fs.map((f) => (f.log === undefined || f.log === null ? '' : flag(f.log)));
      if (logs.some(Boolean)) q.set('proflog', logs.join(','));
    }
  }
  return q;
}

/**
 * Names that have changed, old on the left and today's on the right, so links and files made
 * before a rename still open the same preset with the same settings. Add to these, never
 * remove, whenever a preset id or a link key is renamed.
 */
export const PROBLEM_ALIASES: Record<string, string> = {};
export const KEY_ALIASES: Record<string, string> = {};

/** What decoding a link could not use. */
export interface LinkReport {
  /** the preset the link names, if this build has no such preset (or the link names none) */
  unknownProblem?: string;
  /** keys this build does not read, values it rejects, and values held to its limits */
  ignored: string[];
}

/** The link with old names replaced by today's. A key given under both names keeps today's. */
function renamed(q: URLSearchParams): URLSearchParams {
  const out = new URLSearchParams();
  for (const [k, v] of q) if (!(k in KEY_ALIASES)) out.append(k, v);
  for (const [k, v] of q) if (k in KEY_ALIASES && !out.has(KEY_ALIASES[k])) out.append(KEY_ALIASES[k], v);
  const p = out.get('problem');
  if (p !== null && p in PROBLEM_ALIASES) out.set('problem', PROBLEM_ALIASES[p]);
  return out;
}

/** The link's parameters, noting which keys were looked at. */
class Reading extends URLSearchParams {
  readonly seen = new Set<string>();
  override get(name: string) {
    this.seen.add(name);
    return super.get(name);
  }
  override has(name: string, value?: string) {
    this.seen.add(name);
    return value === undefined ? super.has(name) : super.has(name, value);
  }
}

/**
 * The settings a file of ours was written with, and what this build reads differently. Throws
 * if this build does not have the file's preset: its gas would land in another problem.
 */
export function snapshotSettings(meta: SnapshotMeta, build: string): { link: LinkOverrides; warnings: string[] } {
  const report: LinkReport = { ignored: [] };
  const link = decodeLink(new URLSearchParams(meta.query), report);
  const by = `build ${meta.build}`;
  if (report.unknownProblem !== undefined) {
    throw new Error(report.unknownProblem ? `this file is from ${by}, for the preset "${report.unknownProblem}", which this build does not have` : `this file is from ${by} and does not name its preset`);
  }
  const warnings: string[] = [];
  const format = meta.format ?? 1;
  if (format > SNAPSHOT_FORMAT) warnings.push(`this file is from ${by}, which writes a newer format (${format}) than this build reads (${SNAPSHOT_FORMAT}): some settings may be read wrongly`);
  else if (meta.build !== build) warnings.push(`this file is from ${by}; this is build ${build}. Any setting whose meaning or default has changed between them takes this build's`);
  if (report.ignored.length) warnings.push(`settings this build could not use as written, so it uses its own: ${report.ignored.join(', ')}`);
  return { link, warnings };
}

export function decodeLink(link: URLSearchParams, report?: LinkReport): LinkOverrides {
  const q = new Reading(renamed(link));
  const out = decodeRenamed(q, report);
  if (report) {
    // Without its preset nothing in a link is read.
    const unread = [...new Set(q.keys())].filter((k) => (report.unknownProblem === undefined ? !q.seen.has(k) : k !== 'problem'));
    report.ignored.push(...unread);
  }
  return out;
}

function decodeRenamed(q: URLSearchParams, report?: LinkReport): LinkOverrides {
  const named = q.get('problem');
  // A custom setup's document is the page's to unpack, before this is called (ui/app.ts).
  q.get('setup');
  const known = hasProblem(named);
  if (!known && report) report.unknownProblem = named ?? '';
  // A bare visit gets the default setup. A link that names a preset but no grid gets the
  // preset's own grid (Problem.grid), or else 3D at the default size, as links always have. A
  // link that gives the cells but no levels is uniform: links leave out amr=0.
  const problem = known ? named! : DEFAULT_SETUP.problem;
  const grid = known ? problemOf(problem).grid : undefined;
  const dim = q.get('dim');
  const out: LinkOverrides = {
    problem,
    ndim: known ? (dim ? (dim === '2' ? 2 : 3) : (grid?.ndim ?? 3)) : DEFAULT_SETUP.ndim,
    n: known ? (q.has('n') ? Math.max(0, Math.round(+q.get('n')! || 0)) : (grid?.n ?? 0)) : DEFAULT_SETUP.n,
    amr: known ? Math.min(MAX_AMR_LEVELS, Math.max(0, Math.round(q.has('amr') ? +q.get('amr')! || 0 : q.has('n') ? 0 : (grid?.amr ?? 0)))) : DEFAULT_SETUP.amr,
    cfg: {},
    eosParams: {},
    coolParams: {},
    opacityParams: {},
    driving: {},
    fluff: {},
    startDamping: {},
    params: {},
    view: {},
  };
  if (!known) return out;

  const rejected = (key: string, raw: string, note = 'not a value this build accepts') => report?.ignored.push(`${key}=${raw} (${note})`);
  const num = (key: string, lo = -Infinity, hi = Infinity): number | undefined => {
    const raw = q.get(key);
    if (raw === null || raw === '') return undefined;
    const v = raw === 'inf' ? Infinity : +raw;
    if (Number.isNaN(v)) return void rejected(key, raw);
    const held = Math.min(hi, Math.max(lo, v));
    if (held !== v) rejected(key, raw, `held to ${fmt(held)}`);
    return held;
  };
  const pick = <T extends string>(key: string, allowed: readonly T[]): T | undefined => {
    const raw = q.get(key);
    if (raw === null) return undefined;
    if ((allowed as readonly string[]).includes(raw)) return raw as T;
    return void rejected(key, raw);
  };
  const bool = (key: string): boolean | undefined => {
    const raw = q.get(key);
    if (raw === null) return undefined;
    if (raw === '1' || raw === '0') return raw === '1';
    return void rejected(key, raw);
  };
  const put = <O extends object, K extends keyof O>(o: O, k: K, v: O[K] | undefined) => {
    if (v !== undefined) o[k] = v;
  };

  put(out.cfg, 'tEnd', num('tend', 0));
  for (const o of LINKED) {
    const c = o.control;
    let v: unknown;
    if (c.type === 'number') {
      const x = num(o.linkKey!, c.min, c.max);
      v = x === undefined ? undefined : c.integer ? Math.round(x) : x;
    } else if (c.type === 'select') {
      const raw = pick(o.linkKey!, c.choices.map((k) => String(k.value)));
      v = raw === undefined ? undefined : c.choices.find((k) => String(k.value) === raw)!.value;
    } else if (c.type === 'check') v = bool(o.linkKey!);
    if (v === undefined) continue;
    if (o.cfg === 'symmetry') v = mirrorAxes(v as string);
    const [head, key] = o.cfg!.split('.');
    if (key === undefined) (out.cfg as Record<string, unknown>)[head] = v;
    else (out[head as 'eosParams' | 'coolParams' | 'opacityParams' | 'fluff' | 'startDamping' | 'driving'] as Record<string, unknown>)[key] = v;
  }
  put(out.cfg, 'wellBalanced', bool('wb'));
  const bc = q.get('bc');
  if (bc && bc.length === 6 && [...bc].every((d) => BOUNDARY_CODES.includes(Number(d)) && /\d/.test(d))) {
    const d = [...bc].map(Number);
    // Periodic faces come in pairs.
    if ([0, 1, 2].every((a) => (d[a] === 0) === (d[a + 3] === 0))) {
      out.cfg.bcLo = d.slice(0, 3) as Bc3;
      out.cfg.bcHi = d.slice(3) as Bc3;
    } else rejected('bc', bc);
  } else if (bc) rejected('bc', bc);
  for (const p of problemOf(problem).params ?? []) {
    const v = num(`p.${p.id}`, p.min, p.max);
    if (v !== undefined) out.params[p.id] = v;
  }

  // The run sets the largest index of a field and of a species; the table sets every other limit.
  const most: Partial<Record<string, number>> = { field: FIELDS.length - 1, species: (problemOf(problem).species?.length ?? 1) - 1 };
  for (const o of VIEW_LINKED) {
    if (o.type === 'mode') put(out.view, o.key, pick(o.link, VIEW_MODES));
    else if (o.type === 'boolean') put(out.view, o.key, bool(o.link));
    else {
      const limits: ViewOption = o;
      const v = num(o.link, limits.min, most[o.key] ?? limits.max);
      put(out.view, o.key, v === undefined || o.type === 'number' ? v : Math.round(v));
    }
  }
  const lo = num('lo');
  const hi = num('hi');
  if (lo !== undefined && hi !== undefined && Number.isFinite(lo) && Number.isFinite(hi)) {
    out.auto = false;
    out.view.lo = lo;
    out.view.hi = hi;
  }
  out.colormap = pick('cmap', Object.keys(COLORMAPS));
  const tf = q.get('tf');
  if (tf) {
    const pts = tf.split(',').map((s) => s.split(':').map(Number));
    const ok = pts.length >= 2 && pts.length <= 16 && pts.every((p) => p.length === 2 && p.every((v) => v >= 0 && v <= 1));
    if (ok) out.opacity = pts.map(([x, a]) => ({ x, a })).sort((a, b) => a.x - b.x);
    else rejected('tf', tf);
  }
  // The profile lines, and the fields plotted along them: read whether or not there are lines
  // to plot them along, so that what comes without them is reported, not passed over.
  const prof = q.get('prof');
  const lines = prof ? decodeProfileLines(prof, out.ndim) : null;
  if (prof && !lines) rejected('prof', prof, `not ${out.ndim === 2 ? 'x0,y0,x1,y1' : 'x0,y0,z0,x1,y1,z1'} lines in fractions of the box, at most ${MAX_PROFILES}`);
  /** A list of the profile keys, each entry a whole number from 0 to `hi` or empty; null if malformed. */
  const list = (key: string, hi: number): (number | null)[] | undefined => {
    const raw = q.get(key);
    if (raw === null || raw === '') return undefined;
    const v = raw.split(',').map((x) => (x.trim() === '' ? null : Number(x)));
    if (v.length > FIELDS.length + most.species! + 1 || !v.every((x) => x === null || (Number.isInteger(x) && x >= 0 && x <= hi))) return void rejected(key, raw);
    return v;
  };
  const pf = list('proff', FIELDS.length - 1);
  const psp = list('profsp', most.species!);
  const plog = list('proflog', 1);
  // (proff2, profsp2 and proflog2: a second field, as links had it briefly. Read on after proff's.)
  const pf2 = num('proff2', 0, FIELDS.length - 1);
  const psp2 = num('profsp2', 0, most.species);
  const plog2 = bool('proflog2');
  if (lines) {
    out.profiles = { lines };
    const fields = (pf ?? []).flatMap((f, k) => (f === null ? [] : [{
      field: f,
      ...(f === MASS_FRACTION_FIELD && psp?.[k] ? { species: psp[k]! } : {}),
      ...(plog?.[k] !== undefined && plog[k] !== null ? { log: plog[k] === 1 } : {}),
    }]));
    if (pf2 !== undefined && fields.length) {
      const f = Math.round(pf2);
      fields.push({ field: f, ...(f === MASS_FRACTION_FIELD && psp2 ? { species: Math.round(psp2) } : {}), ...(plog2 !== undefined ? { log: plog2 } : {}) });
    }
    if (fields.length) out.profiles.fields = fields;
  } else {
    for (const k of ['proff', 'profsp', 'proflog', 'proff2', 'profsp2', 'proflog2']) {
      const raw = q.get(k);
      if (raw !== null) rejected(k, raw, 'goes with prof');
    }
  }
  return out;
}
