// A run and its setup document, both ways: the document a run's configuration stands for, and
// what it takes to get from one document to another (a uniform write, new shaders, or a fresh
// start). Free of the DOM: the page (ui/app.ts), the dialog and the scripting interface share it.
import { expandTunnel, tunnelOf } from './parts/tunnel';
import { configFor, withSymmetry, type SimConfig } from '../sim';
import { mirrorAxes, mirrorName } from '../domain';
import { compileSetup, withParams } from './compile';
import { flatten, OPTIONS, PART_OPTIONS, type Apply, type Option } from './options';
import { cloneSetup, emptySetup, getPath, setPath, type SetupSpec } from './spec';

/** An option the run holds as a field of its SimConfig, at its whole path in a document. */
export interface Setting {
  path: string;
  option: Option;
  /** the SimConfig field ('eosParams.gamma'), or 'bc' for the faces */
  cfg: string;
}

/** The settings of a document that a SimConfig holds directly: the general ones, and the damping zone of its first disk that has one. */
export function settingsOf(spec: SetupSpec): Setting[] {
  const out: Setting[] = flatten(OPTIONS).filter((o) => o.cfg).map((o) => ({ path: o.path, option: o, cfg: o.cfg! }));
  const i = (spec.parts ?? []).findIndex((p) => p.kind === 'disk' && p.damp);
  if (i >= 0) for (const o of flatten(PART_OPTIONS.disk.fields)) if (o.cfg) out.push({ path: `parts.${i}.${o.path}`, option: o, cfg: o.cfg });
  return out;
}

const same = (a: unknown, b: unknown): boolean => {
  if (typeof a === 'number' && typeof b === 'number') return a === b || Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b));
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
};

/** A setting's value in a configuration, as a document holds it. */
export function settingValue(cfg: SimConfig, s: Setting): unknown {
  if (s.cfg === 'bc') return { lo: [...cfg.bcLo], hi: [...cfg.bcHi] };
  if (s.cfg === 'symmetry') return mirrorName(cfg.symmetry);
  const v = getPath(cfg, s.cfg);
  if (s.cfg === 'tEnd') return Number.isFinite(v as number) ? v : null;
  return v;
}

/** Write a setting's value from a document into a configuration. */
export function setSetting(cfg: SimConfig, s: Setting, value: unknown): void {
  if (s.cfg === 'bc') {
    const bc = value as { lo: number[]; hi: number[] };
    // (The faces a run does not resolve stay periodic, but a spherical run's poles: configFor.)
    const keepY = cfg.ndim > 1 || cfg.problem.geometry === 'spherical';
    cfg.bcLo = [bc.lo[0], keepY ? bc.lo[1] : 0, cfg.ndim < 3 ? 0 : bc.lo[2]];
    cfg.bcHi = [bc.hi[0], keepY ? bc.hi[1] : 0, cfg.ndim < 3 ? 0 : bc.hi[2]];
  } else if (s.cfg === 'symmetry') {
    // The mirrors the problem allows, their planes' faces reflecting.
    Object.assign(cfg, withSymmetry(cfg, mirrorAxes(value as string)));
  } else if (s.cfg === 'tEnd') cfg.tEnd = value === null || value === undefined ? Infinity : (value as number);
  else if (s.cfg.startsWith('fluff.')) cfg.fluff = { ...cfg.fluff, [s.cfg.slice(6)]: value as number };
  else if (s.cfg.startsWith('driving.')) cfg.driving = { ...cfg.driving, [s.cfg.slice(8)]: value as number };
  else if (s.cfg.startsWith('wallVelocity.')) cfg.wallVelocity = { ...cfg.wallVelocity, [s.cfg.slice(13)]: value as [number, number, number] };
  else setPath(cfg, s.cfg, value);
}

/**
 * The document a run stands for: the one its problem was compiled from, with the run's grid,
 * its parameter values, and every setting changed since written in. Compiled on the same grid
 * it gives the run's configuration back.
 */
export function runningSpec(cfg: SimConfig): SetupSpec {
  const from = cfg.problem.setup ?? emptySetup();
  const spec = cloneSetup(withParams(from, cfg.params));
  spec.domain = { ...spec.domain, ndim: cfg.ndim, n: cfg.n, amr: cfg.amrLevels };
  // A setting written in may move another's default (the tenuous-gas density is the default
  // of the density refinement starts at): again, until the document gives the run's values.
  for (let pass = 0; pass < 4; pass++) {
    // Compiled from a copy: a document is not changed once something has been compiled from it.
    // (With the run's mirrors, whose planes' faces are then not settings of their own.)
    const problem = compileSetup(cloneSetup(spec)).problem;
    const def = configFor(problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels, symmetry: cfg.symmetry });
    const plain = configFor(problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels });
    let changed = false;
    // What a wind tunnel sets (parts/tunnel.ts), the run holding it still: the tunnel's, not written in.
    const tunnel = tunnelOf(spec) ? expandTunnel(spec) : null;
    for (const s of settingsOf(spec)) {
      const now = settingValue(cfg, s);
      if (same(now, settingValue(s.cfg === 'symmetry' ? plain : def, s))) continue;
      if (tunnel && getPath(spec, s.path) === undefined && getPath(tunnel, s.path) !== undefined && same(now, getPath(tunnel, s.path))) continue;
      changed = true;
      // The faces of a 2D run are the document's 2D faces, where it keeps them apart.
      if (s.cfg === 'bc' && cfg.ndim === 2) spec.domain = { ...spec.domain, bc2d: now as { lo: [number, number, number]; hi: [number, number, number] } };
      else setPath(spec, s.path, now);
    }
    if (!changed) break;
  }
  return spec;
}

/** Whether two documents are the same but for their grid (dimensions, resolution, refinement). */
export function sameSetup(a: SetupSpec, b: SetupSpec): boolean {
  const bare = (s: SetupSpec) => {
    const c = cloneSetup(s);
    if (c.domain) {
      delete c.domain.ndim;
      delete c.domain.n;
      delete c.domain.amr;
    }
    return JSON.stringify(prune(c));
  };
  return bare(a) === bare(b);
}

/** A value without its empty objects and absent keys, keys in order: for comparing documents. */
function prune(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(prune);
  if (v === null || typeof v !== 'object') return v;
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(v).sort()) {
    const x = prune((v as Record<string, unknown>)[k]);
    if (x === undefined || (x !== null && typeof x === 'object' && !Array.isArray(x) && Object.keys(x).length === 0)) continue;
    out[k] = x;
  }
  return out;
}

/** Fields of each kind of part that a run takes up where it is: the option table's `live` ones that are not settings of the run. */
const LIVE_FIELDS: Record<string, string[]> = Object.fromEntries(
  Object.values(PART_OPTIONS).map((k) => [k.kind, k.fields.filter((o) => o.apply === 'live' && !o.cfg && o.control.type !== 'group').map((o) => o.path)]),
);

/** Keys of a document that describe it and change nothing of a run. */
const DESCRIPTIVE = ['name', 'blurb', 'group', 'family', 'familyName', 'variant', 'featured', 'id', 'view', 'extends'];

export interface SetupChange {
  /** settings the run holds directly whose values differ, with the new value */
  settings: { setting: Setting; value: unknown }[];
  /** the grid differs */
  grid: boolean;
  /** something else differs: the parts, the box, the units, what the gas is made of */
  structure: boolean;
  /**
   * The parts differ only in fields a run takes up where it is (a point mass's softening and
   * sink): no new run is needed for them, and `structure` is false.
   */
  retune: boolean;
  /** only names and descriptions differ, besides the above */
  described: boolean;
  /** the most any of it costs a running simulation */
  cost: Apply | 'none';
}

/** What it takes to get a run of document `from` to document `to`. */
export function changeBetween(from: SetupSpec, to: SetupSpec): SetupChange {
  const [a, b] = [cloneSetup(from), cloneSetup(to)];
  const settings: SetupChange['settings'] = [];
  const paths = new Map<string, Setting>();
  for (const s of [...settingsOf(a), ...settingsOf(b)]) paths.set(s.path, s);
  for (const s of paths.values()) {
    const [x, y] = [getPath(a, s.path) ?? s.option.default, getPath(b, s.path) ?? s.option.default];
    if (!same(x, y)) settings.push({ setting: s, value: getPath(b, s.path) ?? s.option.default });
    setPath(a, s.path, undefined);
    setPath(b, s.path, undefined);
  }
  const grid = (['ndim', 'n', 'amr'] as const).some((k) => (a.domain?.[k] ?? null) !== (b.domain?.[k] ?? null));
  for (const d of [a, b]) {
    if (d.domain) {
      delete d.domain.ndim;
      delete d.domain.n;
      delete d.domain.amr;
    }
  }
  const text = (d: SetupSpec, drop: string[]) => JSON.stringify(prune(Object.fromEntries(Object.entries(d).filter(([k]) => !drop.includes(k)))));
  // The clock's starting values are the bar's to change: no run starts again for them.
  const quiet = (d: SetupSpec) => ({ ...d, run: undefined });
  let structure = text(quiet(a) as SetupSpec, DESCRIPTIVE) !== text(quiet(b) as SetupSpec, DESCRIPTIVE);
  // Without the fields of parts that a run takes up where it is, are they the same?
  let retune = false;
  if (structure) {
    const bare = (d: SetupSpec) => {
      const c = cloneSetup(quiet(d) as SetupSpec);
      for (const p of c.parts ?? []) for (const f of LIVE_FIELDS[p.kind] ?? []) delete (p as unknown as Record<string, unknown>)[f];
      return text(c, DESCRIPTIVE);
    };
    if ((a.parts ?? []).length === (b.parts ?? []).length && bare(a) === bare(b)) {
      structure = false;
      retune = true;
    }
  }
  const described = (text(a, []) !== text(b, []) && !structure) || retune;
  let cost: SetupChange['cost'] = settings.length || described ? 'live' : 'none';
  if (settings.some((s) => s.setting.option.apply === 'shaders')) cost = 'shaders';
  if (grid || structure || settings.some((s) => s.setting.option.apply === 'restart')) cost = 'restart';
  return { settings, grid, structure, retune, described, cost };
}
