// Shareable links: the simulator's settings as URL query parameters, and back.
//
// A link names the preset, dimensionality, resolution and refinement, and then only what
// differs from that preset's defaults, so most links stay short. Anything unrecognized or
// out of range in a link is ignored rather than trusted.
import { configFor, type SimConfig } from '../sim';
import { physicalEosParams, type EosParams } from '../eos/eos';
import type { CoolingParams } from '../cooling/cooling';
import { FIELDS, problemUnits, type Bc3, type Problem } from '../problems';
import { CUSTOM_ID } from '../presets/custom';
import { hasProblem, problemOf } from '../presets/lookup';
import { flatten, OPTIONS, PART_OPTIONS, type Option } from '../setup/options';
import { defaultView, type ViewState } from '../render/renderer';
import { MAX_FIELD_LINES } from '../render/fieldlines';
import { COLORMAPS, defaultOpacity, type OpacityPoint } from '../render/transfer';
import { SNAPSHOT_FORMAT, type SnapshotMeta } from '../io/snapshot';

/** Largest magnification of the slice view. */
export const MAX_ZOOM = 256;
/** Volume view: the nearest and farthest the camera comes to the point it orbits, in box lengths. */
export const MIN_DIST = 0.225;
export const MAX_DIST = 6;
/** Most contour lines the slice and projection views draw. */
export const MAX_CONTOURS = 40;

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
const at = (c: object, path: string): unknown => path.split('.').reduce<unknown>((v, k) => (v as Record<string, unknown> | undefined)?.[k], c);

export interface LinkState {
  /** the custom setup's document, packed for a link (setup/codec.ts encodeSetupParam), when the run is one */
  setup?: string;
  cfg: SimConfig;
  view: ViewState;
  /** the displayed range follows the data */
  auto: boolean;
  colormap: string;
  opacity: OpacityPoint[];
}

export interface LinkOverrides {
  problem: string;
  ndim: 2 | 3;
  n: number; // 0: the default for that dimensionality
  amr: number;
  cfg: Partial<Pick<SimConfig, 'tEnd' | 'eos' | 'riemann' | 'limiter' | 'cfl' | 'timeBins' | 'refineHi' | 'refineLo' | 'refineRho' | 'gravity' | 'G' | 'vcycles' | 'jeansFloor' | 'jeansCells' | 'gravEnergyFlux' | 'wellBalanced' | 'bcLo' | 'bcHi' | 'cooling' | 'viscosity' | 'conductivity' | 'dampRadius' | 'dampRate'>>;
  eosParams: Partial<EosParams>;
  coolParams: Partial<CoolingParams>;
  fluff: Partial<SimConfig['fluff']>;
  params: Record<string, number>;
  view: Partial<ViewState>;
  auto?: boolean;
  colormap?: string;
  opacity?: OpacityPoint[];
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
const VIEW_NUMBERS = ['field', 'species', 'sliceAxis', 'slicePos', 'minRho', 'opacity', 'yaw', 'pitch', 'dist', 'tx', 'ty', 'tz', 'zoom', 'cx', 'cy', 'nContours', 'surfaceOpacity', 'fieldLineCount'] as const;
const VIEW_KEYS: Record<(typeof VIEW_NUMBERS)[number], string> = {
  field: 'field', species: 'species', sliceAxis: 'axis', slicePos: 'slice', minRho: 'rhocut', opacity: 'opacity', yaw: 'yaw', pitch: 'pitch', dist: 'dist', tx: 'tx', ty: 'ty', tz: 'tz', zoom: 'zoom', cx: 'cx', cy: 'cy', nContours: 'ncont', surfaceOpacity: 'isoa', fieldLineCount: 'nflines',
};

export function encodeLink(s: LinkState): URLSearchParams {
  const { cfg, view } = s;
  const q = new URLSearchParams();
  q.set('problem', cfg.problem.id);
  if (cfg.problem.id === CUSTOM_ID && s.setup) q.set('setup', s.setup);
  q.set('dim', String(cfg.ndim));
  q.set('n', String(cfg.n));
  if (cfg.amrLevels) q.set('amr', String(cfg.amrLevels));

  const def = configFor(cfg.problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels });
  // Units that follow the preset's parameters set the equation of state's physical constants: the
  // defaults to compare with are those of the run's parameters, which the link carries.
  if (typeof cfg.problem.units === 'function') Object.assign(def.eosParams, physicalEosParams(problemUnits(cfg.problem, cfg.params), cfg.problem.composition), cfg.problem.eosParams);
  const num = (key: string, a: number, b: number) => {
    if (fmt(a) !== fmt(b)) q.set(key, fmt(a));
  };
  const str = (key: string, a: string, b: string) => {
    if (a !== b) q.set(key, a);
  };
  num('tend', cfg.tEnd, def.tEnd);
  for (const o of LINKED) {
    if ((adaptiveOnly(o) && !cfg.amrLevels) || (coolingOnly(o) && cfg.cooling === 'off')) continue;
    const [a, b] = [at(cfg, o.cfg!) ?? 0, at(def, o.cfg!) ?? 0];
    if (typeof a === 'boolean') str(o.linkKey!, flag(a), flag(!!b));
    else if (typeof a === 'number' && o.control.type === 'number') num(o.linkKey!, a, b as number);
    else str(o.linkKey!, String(a), String(b));
  }
  str('wb', flag(cfg.wellBalanced), flag(def.wellBalanced));
  // Boundaries: one digit per face, the lower faces of x, y, z and then the upper ones.
  str('bc', [...cfg.bcLo, ...cfg.bcHi].join(''), [...def.bcLo, ...def.bcHi].join(''));
  for (const p of cfg.problem.params ?? []) num(`p.${p.id}`, cfg.params[p.id], p.value);

  const dv = viewDefaults(cfg.problem, cfg.ndim);
  str('mode', view.mode, dv.mode);
  for (const k of VIEW_NUMBERS) num(VIEW_KEYS[k], view[k], dv[k]);
  str('log', flag(view.log), flag(dv.log));
  str('blocks', flag(view.showBlocks), flag(dv.showBlocks));
  str('contours', flag(view.contours), flag(dv.contours));
  str('fog', flag(view.fog), flag(dv.fog));
  str('light', flag(view.light), flag(dv.light));
  str('box', flag(view.box), flag(dv.box));
  str('flines', flag(view.fieldLines), flag(dv.fieldLines));
  if (!s.auto) {
    q.set('lo', fmt(view.lo));
    q.set('hi', fmt(view.hi));
  }
  str('cmap', s.colormap, 'inferno');
  const pts = (p: OpacityPoint[]) => p.map((o) => `${+o.x.toFixed(3)}:${+o.a.toFixed(3)}`).join(',');
  str('tf', pts(s.opacity), pts(defaultOpacity()));
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
    amr: known ? Math.min(5, Math.max(0, Math.round(q.has('amr') ? +q.get('amr')! || 0 : q.has('n') ? 0 : (grid?.amr ?? 0)))) : DEFAULT_SETUP.amr,
    cfg: {},
    eosParams: {},
    coolParams: {},
    fluff: {},
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
    const [head, key] = o.cfg!.split('.');
    if (key === undefined) (out.cfg as Record<string, unknown>)[head] = v;
    else (out[head as 'eosParams' | 'coolParams' | 'fluff'] as Record<string, unknown>)[key] = v;
  }
  put(out.cfg, 'wellBalanced', bool('wb'));
  const bc = q.get('bc');
  if (bc && /^[0-3]{6}$/.test(bc)) {
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

  put(out.view, 'mode', pick<ViewState['mode']>('mode', ['volume', 'slice', 'projection']));
  const field = num('field', 0, FIELDS.length - 1);
  put(out.view, 'field', field === undefined ? undefined : Math.round(field));
  const species = num('species', 0, (problemOf(problem).species?.length ?? 1) - 1);
  put(out.view, 'species', species === undefined ? undefined : Math.round(species));
  const axis = num('axis', 0, 2);
  put(out.view, 'sliceAxis', axis === undefined ? undefined : Math.round(axis));
  put(out.view, 'slicePos', num('slice', 0, 0.999));
  put(out.view, 'minRho', num('rhocut', 0, 1));
  put(out.view, 'opacity', num('opacity', 1, 300));
  put(out.view, 'yaw', num('yaw'));
  put(out.view, 'pitch', num('pitch', -1.5, 1.5));
  put(out.view, 'dist', num('dist', MIN_DIST, MAX_DIST));
  put(out.view, 'tx', num('tx', 0, 1));
  put(out.view, 'ty', num('ty', 0, 1));
  put(out.view, 'tz', num('tz', 0, 1));
  put(out.view, 'zoom', num('zoom', 1, MAX_ZOOM));
  put(out.view, 'cx', num('cx', 0, 1));
  put(out.view, 'cy', num('cy', 0, 1));
  put(out.view, 'log', bool('log'));
  put(out.view, 'showBlocks', bool('blocks'));
  put(out.view, 'contours', bool('contours'));
  put(out.view, 'fog', bool('fog'));
  put(out.view, 'light', bool('light'));
  put(out.view, 'box', bool('box'));
  put(out.view, 'fieldLines', bool('flines'));
  const nfl = num('nflines', 1, MAX_FIELD_LINES);
  put(out.view, 'fieldLineCount', nfl === undefined ? undefined : Math.round(nfl));
  put(out.view, 'surfaceOpacity', num('isoa', 0.01, 1));
  const ncont = num('ncont', 1, MAX_CONTOURS);
  put(out.view, 'nContours', ncont === undefined ? undefined : Math.round(ncont));
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
  return out;
}
