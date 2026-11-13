// Shareable links: the simulator's settings as URL query parameters, and back.
//
// A link names the preset, dimensionality, resolution and refinement, and then only what
// differs from that preset's defaults, so most links stay short. Anything unrecognized or
// out of range in a link is ignored rather than trusted.
import { configFor, type LimiterKind, type RiemannKind, type SimConfig } from '../sim';
import { EOS_LABELS, type EosKind, type EosParams } from '../eos/eos';
import { COOLING_LABELS, type CoolingKind, type CoolingParams } from '../cooling/cooling';
import { FIELDS, GRAVITY_MODES, PROBLEMS, problemById, type Bc3, type GravityMode, type Problem } from '../problems';
import { defaultView, type ViewState } from '../render/renderer';
import { COLORMAPS, defaultOpacity, type OpacityPoint } from '../render/transfer';

/** Largest magnification of the slice view. */
export const MAX_ZOOM = 64;
/** Most contour lines the slice and projection views draw. */
export const MAX_CONTOURS = 40;

/** What a fresh visit shows. */
export const DEFAULT_SETUP = { problem: 'kelvin-helmholtz', ndim: 2 as 2 | 3, n: 1024 };

export interface LinkState {
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
  return v;
}

const fmt = (v: number) => (Number.isFinite(v) ? String(+v.toPrecision(6)) : 'inf');
const flag = (b: boolean) => (b ? '1' : '0');
const VIEW_NUMBERS = ['field', 'species', 'sliceAxis', 'slicePos', 'minRho', 'opacity', 'yaw', 'pitch', 'dist', 'zoom', 'cx', 'cy', 'nContours', 'surfaceOpacity'] as const;
const VIEW_KEYS: Record<(typeof VIEW_NUMBERS)[number], string> = {
  field: 'field', species: 'species', sliceAxis: 'axis', slicePos: 'slice', minRho: 'rhocut', opacity: 'opacity', yaw: 'yaw', pitch: 'pitch', dist: 'dist', zoom: 'zoom', cx: 'cx', cy: 'cy', nContours: 'ncont', surfaceOpacity: 'isoa',
};

export function encodeLink(s: LinkState): URLSearchParams {
  const { cfg, view } = s;
  const q = new URLSearchParams();
  q.set('problem', cfg.problem.id);
  q.set('dim', String(cfg.ndim));
  q.set('n', String(cfg.n));
  if (cfg.amrLevels) q.set('amr', String(cfg.amrLevels));

  const def = configFor(cfg.problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels });
  const num = (key: string, a: number, b: number) => {
    if (fmt(a) !== fmt(b)) q.set(key, fmt(a));
  };
  const str = (key: string, a: string, b: string) => {
    if (a !== b) q.set(key, a);
  };
  num('tend', cfg.tEnd, def.tEnd);
  str('eos', cfg.eos, def.eos);
  num('gamma', cfg.eosParams.gamma, def.eosParams.gamma);
  num('K', cfg.eosParams.K, def.eosParams.K);
  num('arad', cfg.eosParams.arad, def.eosParams.arad);
  num('degrho', cfg.eosParams.degRho, def.eosParams.degRho);
  num('degp', cfg.eosParams.degP, def.eosParams.degP);
  num('iont', cfg.eosParams.ionT, def.eosParams.ionT);
  num('ionrho', cfg.eosParams.ionRho, def.eosParams.ionRho);
  str('cool', cfg.cooling, def.cooling);
  if (cfg.cooling !== 'off') {
    num('crate', cfg.coolParams.rate, def.coolParams.rate);
    num('calpha', cfg.coolParams.alpha, def.coolParams.alpha);
    num('cheat', cfg.coolParams.heat, def.coolParams.heat);
    num('ctfloor', cfg.coolParams.tFloor, def.coolParams.tFloor);
    num('ctunit', cfg.coolParams.tUnit, def.coolParams.tUnit);
  }
  num('nu', cfg.viscosity, def.viscosity);
  num('kappa', cfg.conductivity, def.conductivity);
  num('dampr', cfg.dampRadius, def.dampRadius);
  num('damprate', cfg.dampRate, def.dampRate);
  str('riemann', cfg.riemann, def.riemann);
  str('limiter', cfg.limiter, def.limiter);
  num('cfl', cfg.cfl, def.cfl);
  if (cfg.amrLevels) num('tbins', cfg.timeBins, def.timeBins);
  if (cfg.amrLevels) num('refhi', cfg.refineHi, def.refineHi);
  if (cfg.amrLevels) num('reflo', cfg.refineLo, def.refineLo);
  if (cfg.amrLevels) num('refrho', cfg.refineRho, def.refineRho);
  num('frho', cfg.fluff.rho, def.fluff.rho);
  num('fu', cfg.fluff.u, def.fluff.u);
  num('fv', cfg.fluff.v, def.fluff.v);
  str('gravity', cfg.gravity, def.gravity);
  num('G', cfg.G, def.G);
  num('vcycles', cfg.vcycles, def.vcycles);
  str('jfloor', flag(cfg.jeansFloor), flag(def.jeansFloor));
  num('jcells', cfg.jeansCells, def.jeansCells);
  str('fluxwork', flag(cfg.gravEnergyFlux), flag(def.gravEnergyFlux));
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
  if (!s.auto) {
    q.set('lo', fmt(view.lo));
    q.set('hi', fmt(view.hi));
  }
  str('cmap', s.colormap, 'inferno');
  const pts = (p: OpacityPoint[]) => p.map((o) => `${+o.x.toFixed(3)}:${+o.a.toFixed(3)}`).join(',');
  str('tf', pts(s.opacity), pts(defaultOpacity()));
  return q;
}

export function decodeLink(q: URLSearchParams): LinkOverrides {
  const named = q.get('problem');
  const known = PROBLEMS.some((p) => p.id === named);
  // A bare visit gets the default setup. A link that names a preset but no grid gets the
  // preset's own grid (Problem.grid), or else 3D at the default size, as links always have. A
  // link that gives the cells but no levels is uniform: links leave out amr=0.
  const problem = known ? named! : DEFAULT_SETUP.problem;
  const grid = known ? problemById(problem).grid : undefined;
  const dim = q.get('dim');
  const out: LinkOverrides = {
    problem,
    ndim: known ? (dim ? (dim === '2' ? 2 : 3) : (grid?.ndim ?? 3)) : DEFAULT_SETUP.ndim,
    n: known ? (q.has('n') ? Math.max(0, Math.round(+q.get('n')! || 0)) : (grid?.n ?? 0)) : DEFAULT_SETUP.n,
    amr: known ? Math.min(5, Math.max(0, Math.round(q.has('amr') ? +q.get('amr')! || 0 : q.has('n') ? 0 : (grid?.amr ?? 0)))) : 0,
    cfg: {},
    eosParams: {},
    coolParams: {},
    fluff: {},
    params: {},
    view: {},
  };
  if (!known) return out;

  const num = (key: string, lo = -Infinity, hi = Infinity): number | undefined => {
    const raw = q.get(key);
    if (raw === null || raw === '') return undefined;
    const v = raw === 'inf' ? Infinity : +raw;
    return Number.isNaN(v) ? undefined : Math.min(hi, Math.max(lo, v));
  };
  const pick = <T extends string>(key: string, allowed: readonly T[]): T | undefined => {
    const raw = q.get(key);
    return raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : undefined;
  };
  const bool = (key: string): boolean | undefined => (q.get(key) === '1' ? true : q.get(key) === '0' ? false : undefined);
  const put = <O extends object, K extends keyof O>(o: O, k: K, v: O[K] | undefined) => {
    if (v !== undefined) o[k] = v;
  };

  put(out.cfg, 'tEnd', num('tend', 0));
  put(out.cfg, 'eos', pick('eos', Object.keys(EOS_LABELS) as EosKind[]));
  put(out.cfg, 'riemann', pick<RiemannKind>('riemann', ['hllc', 'hllclm', 'hll']));
  put(out.cfg, 'limiter', pick<LimiterKind>('limiter', ['koren', 'mc', 'vanleer', 'minmod']));
  put(out.cfg, 'cfl', num('cfl', 0.05, 1));
  put(out.cfg, 'refineHi', num('refhi', 0.05, 1));
  put(out.cfg, 'refineLo', num('reflo', 0, 1));
  put(out.cfg, 'refineRho', num('refrho', 0));
  put(out.fluff, 'rho', num('frho', 0));
  put(out.fluff, 'u', num('fu', 0));
  put(out.fluff, 'v', num('fv', 0));
  const tb = num('tbins', 0, 3);
  if (tb !== undefined) out.cfg.timeBins = Math.round(tb);
  put(out.cfg, 'gravity', pick<GravityMode>('gravity', GRAVITY_MODES));
  put(out.cfg, 'G', num('G'));
  const vc = num('vcycles', 1, 8);
  put(out.cfg, 'vcycles', vc === undefined ? undefined : Math.round(vc));
  put(out.cfg, 'jeansFloor', bool('jfloor'));
  const jc = num('jcells', 4, 32);
  put(out.cfg, 'jeansCells', jc === undefined ? undefined : Math.round(jc));
  put(out.cfg, 'gravEnergyFlux', bool('fluxwork'));
  put(out.cfg, 'wellBalanced', bool('wb'));
  const bc = q.get('bc');
  if (bc && /^[0-3]{6}$/.test(bc)) {
    const d = [...bc].map(Number);
    // Periodic faces come in pairs.
    if ([0, 1, 2].every((a) => (d[a] === 0) === (d[a + 3] === 0))) {
      out.cfg.bcLo = d.slice(0, 3) as Bc3;
      out.cfg.bcHi = d.slice(3) as Bc3;
    }
  }
  put(out.eosParams, 'gamma', num('gamma', 1, 100));
  put(out.eosParams, 'K', num('K', 0));
  put(out.eosParams, 'arad', num('arad', 0));
  put(out.eosParams, 'degRho', num('degrho', 1e-30));
  put(out.eosParams, 'degP', num('degp', 1e-30));
  put(out.eosParams, 'ionT', num('iont', 1e-30));
  put(out.eosParams, 'ionRho', num('ionrho', 1e-30));
  put(out.cfg, 'cooling', pick('cool', Object.keys(COOLING_LABELS) as CoolingKind[]));
  put(out.coolParams, 'rate', num('crate', 0));
  put(out.coolParams, 'alpha', num('calpha', -4, 4));
  put(out.coolParams, 'heat', num('cheat', 0));
  put(out.coolParams, 'tFloor', num('ctfloor', 0));
  put(out.coolParams, 'tUnit', num('ctunit', 1e-6));
  put(out.cfg, 'viscosity', num('nu', 0, 1e30));
  put(out.cfg, 'conductivity', num('kappa', 0, 1e30));
  put(out.cfg, 'dampRadius', num('dampr', 0, 1e30));
  put(out.cfg, 'dampRate', num('damprate', 0, 1e30));
  for (const p of problemById(problem).params ?? []) {
    const v = num(`p.${p.id}`, p.min, p.max);
    if (v !== undefined) out.params[p.id] = v;
  }

  put(out.view, 'mode', pick<ViewState['mode']>('mode', ['volume', 'slice', 'projection']));
  const field = num('field', 0, FIELDS.length - 1);
  put(out.view, 'field', field === undefined ? undefined : Math.round(field));
  const species = num('species', 0, (problemById(problem).species?.length ?? 1) - 1);
  put(out.view, 'species', species === undefined ? undefined : Math.round(species));
  const axis = num('axis', 0, 2);
  put(out.view, 'sliceAxis', axis === undefined ? undefined : Math.round(axis));
  put(out.view, 'slicePos', num('slice', 0, 0.999));
  put(out.view, 'minRho', num('rhocut', 0, 1));
  put(out.view, 'opacity', num('opacity', 1, 300));
  put(out.view, 'yaw', num('yaw'));
  put(out.view, 'pitch', num('pitch', -1.5, 1.5));
  put(out.view, 'dist', num('dist', 0.9, 6));
  put(out.view, 'zoom', num('zoom', 1, MAX_ZOOM));
  put(out.view, 'cx', num('cx', 0, 1));
  put(out.view, 'cy', num('cy', 0, 1));
  put(out.view, 'log', bool('log'));
  put(out.view, 'showBlocks', bool('blocks'));
  put(out.view, 'contours', bool('contours'));
  put(out.view, 'fog', bool('fog'));
  put(out.view, 'surfaceOpacity', num('isoa', 0.05, 1));
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
  }
  return out;
}
