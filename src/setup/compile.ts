// A setup document compiled to the solver's existing input: a Problem and the overrides of a
// SimConfig. Deterministic and free of the DOM, so the page, a worker and Node get the same run
// from the same document.
//
// The compiled initial state paints in layers: a whole-box flow or the background, then disks,
// then stars (also left to settling), then regions, obstacles and inflows in the order of the
// parts, and last the flows that only change what is there.
import { NET_SPECIES } from '../burn/network';
import type { AnyBurnSpec } from '../burn/burn';
import { defaultEosParams, makeEos, physicalEosParams, type Eos } from '../eos/eos';
import type { Settling } from '../problems/settle';
import { OUTFLOW, type Bc3, type MhdSpec, type PointMass, type Prim, type Problem, type ProblemParam } from '../problems/types';
import type { SimConfig } from '../sim';
import { COMPOSITIONS, composition, type Composition } from '../units/composition';
import { codeG, type UnitSystem } from '../units/units';
import { buildBodies, partIds, partQuantity, starPotential, type Bodies } from './parts/bodies';
import { buildDisk, checkDiskFormulas, sumDisks, type BuiltDisk } from './parts/disk';
import { flowValues, type FaceFn, type FlowContext, type FlowDef, type FlowInitContext, type InitFn, type PotentialFn } from './parts/flow';
import { obstaclePainter, regionPainter, type PaintContext } from './parts/region';
import { cloneSetup, featuredId, featuredOf, featuredPaths, getPath, setPath, type CompositionSpec, type DiskPart, type Featured, type FlowPart, type SetupSpec } from './spec';
import { unitSystem, unitsFollowParts } from './units';

export interface Compiled {
  spec: SetupSpec;
  problem: Problem;
  /** Settings of the run that are not the Problem's: the solver's and the mesh's. */
  overrides: Partial<SimConfig>;
}

/**
 * Paths whose options are parameters of the setup's parts: the run holds their values in
 * SimConfig.params, by the featured option's id. Every other path is a setting SimConfig has a
 * field for.
 */
const PARAM_ROOTS = ['parts.', 'background.', 'domain.box', 'domain.units.', 'physics.burn.'];
export const isParamPath = (path: string): boolean => PARAM_ROOTS.some((r) => path === r || path.startsWith(r));

/** The featured options that are parameters, in order. */
export function paramFeatures(spec: SetupSpec): Featured[] {
  return (spec.featured ?? []).map(featuredOf).filter((f) => featuredPaths(f).every(isParamPath));
}

/** The setup with these parameter values written into it (the setup itself if there are none). */
export function withParams(spec: SetupSpec, params: Record<string, number> | undefined): SetupSpec {
  const feats = paramFeatures(spec);
  if (!params || !feats.some((f) => params[featuredId(f)] !== undefined)) return spec;
  const out = cloneSetup(spec);
  for (const f of feats) {
    const v = params[featuredId(f)];
    if (v === undefined) continue;
    for (const path of featuredPaths(f)) setPath(out, path, typeof getPath(spec, path) === 'boolean' ? v > 0.5 : v);
  }
  return out;
}

const namedComposition = (c: CompositionSpec | undefined): Composition | undefined => {
  if (c === undefined) return undefined;
  if (typeof c === 'string') {
    const hit = (COMPOSITIONS as Record<string, Composition>)[c];
    if (!hit) throw new Error(`unknown composition: ${c}`);
    return hit;
  }
  return composition(c.name, c.X, c.ionization ?? 'full');
};

const speciesOf = (spec: SetupSpec): readonly string[] | undefined => {
  const s = spec.physics?.species;
  return s === 'aprox13' ? NET_SPECIES : s;
};

const boxOf = (spec: SetupSpec): number => spec.domain?.box ?? 1;

/** The setup's units and G. Units that follow a part's quantity need the bodies, which need G. */
function scales(spec: SetupSpec): { units?: UnitSystem; G: number } {
  const u = spec.domain?.units;
  const g = spec.physics?.G;
  if (!u) return { G: typeof g === 'number' ? g : 1 };
  if (g === 'units') {
    const units = unitSystem(u, () => {
      throw new Error('units that follow a part need G given as a number');
    });
    return { units, G: codeG(units) };
  }
  const G = g ?? 1;
  let bodies: Bodies | null = null;
  const units = unitSystem(u, (name) => partQuantity((bodies ??= buildBodies(spec, { G, dx: 0, ndim: 3, box: boxOf(spec) })), G, name));
  return { units, G };
}

/** The equation of state a setup opens with, for what is asked without one (the masses, the field). */
function specEos(spec: SetupSpec, units: UnitSystem | undefined): Eos {
  const ph = spec.physics ?? {};
  return makeEos(ph.eos ?? 'ideal', { ...defaultEosParams(), ...physicalEosParams(units, namedComposition(ph.composition)), ...ph.eosParams });
}

const hasWhiteDwarf = (spec: SetupSpec) => (spec.parts ?? []).some((p) => p.kind === 'star' && p.type === 'white-dwarf');

/** Mass fractions in the run's species order from fractions by name. */
function fractionsOf(species: readonly string[] | undefined) {
  return (X: Record<string, number>): number[] => (species ?? []).map((s) => X[s] ?? 0);
}

interface FlowUse {
  part: FlowPart;
  def: FlowDef;
  values: Record<string, number | boolean>;
}
const flowsOf = (spec: SetupSpec): FlowUse[] => (spec.parts ?? []).flatMap((p) => (p.kind === 'flow' ? [{ part: p, ...flowValues(p) }] : []));

/** The disks of a setup, built: each about the bodies it names. */
function disksOf(spec: SetupSpec, bodies: Bodies, G: number, n: number, ndim: 2 | 3): BuiltDisk[] {
  const box = boxOf(spec);
  const dx = n > 0 ? box / n : 0;
  return (spec.parts ?? []).flatMap((p) => {
    if (p.kind !== 'disk') return [];
    const d = p as DiskPart;
    const ms = d.about.map((id) => bodies.massIds.indexOf(id)).filter((i) => i >= 0).map((i) => bodies.masses[i]);
    const stars = d.about.flatMap((id) => bodies.stars.filter((s) => s.id === id));
    if (ms.length + stars.length === 0) throw new Error('a disk needs a point mass or a star to orbit');
    const gm = ms.reduce((s, m) => s + m.gm, 0) + stars.reduce((s, st) => s + G * st.M, 0);
    // The disk is centered on the center of mass of what it orbits, and moves with it. A center
    // that is the box's to rounding (a binary about the origin) is the box's exactly.
    const bodies_ = [...ms.map((m) => ({ gm: m.gm, pos: m.pos, vel: m.vel })), ...stars.map((st) => ({ gm: G * st.M, pos: st.center, vel: st.vel }))];
    const mean = (key: 'pos' | 'vel', tiny: number) => [0, 1, 2].map((a) => {
      const c = bodies_.reduce((s, q) => s + q.gm * (q[key][a] ?? 0), 0) / gm;
      return Math.abs(c) <= tiny ? 0 : c;
    });
    const center = mean('pos', 1e-12 * box);
    const drift = mean('vel', 1e-12 * Math.sqrt(gm / box));
    const sep = d.about.length === 2 ? bodies.separations.get(`${d.about[0]}+${d.about[1]}`) ?? bodies.separations.get(`${d.about[1]}+${d.about[0]}`) : undefined;
    const soft = ms[0]?.soft ?? 2 * dx;
    return [buildDisk(d, { box, n: Math.max(n, 1), ndim, gm, soft: Math.max(soft, 2 * dx), flat: sep !== undefined ? 0.5 * sep : undefined, center, vel: drift })];
  });
}

/** How far from its center, in radii, a star's settling may reach where other gas is about (as a star the star tool places). */
const OWN_REACH = 1.3;

/** The ambient gas around stars: where a star ends, the gas outside, and its pressure. */
function ambient(spec: SetupSpec, bodies: Bodies) {
  const bg = spec.background ?? {};
  const ref = bg.relativeTo ? bodies.stars.find((s) => s.id === bg.relativeTo) : undefined;
  if (bg.relativeTo && !ref) throw new Error(`background: no star "${bg.relativeTo}"`);
  const rho = ref ? (bg.rho ?? 1e-6) * ref.rhoc : (bg.rho ?? 1e-6);
  // Without a cut, a polytrope ends where its pressure over the ambient density equals the
  // center's p / rho: the ambient gas, which carries the pressure at the cut, is then as hot as
  // the center and no hotter.
  const first = bodies.stars[0];
  const own = first?.part.type === 'polytrope' ? first.rhoc * Math.pow(rho / first.rhoc, 1 / (1 + 1 / (first.part.n ?? 1.5))) : 1e-4;
  const cut = bg.cut === undefined ? own : ref ? bg.cut * ref.rhoc : bg.cut;
  return { cut, rho };
}

/** The initial state of a setup on a grid of n cells a side. */
function initOf(spec: SetupSpec, n: number, eos: Eos, ndim: 2 | 3): InitFn {
  const { units, G } = scales(spec);
  const box = boxOf(spec);
  const species = speciesOf(spec);
  const fractions = fractionsOf(species);
  const bodies = buildBodies(spec, { G, dx: box / n, ndim, box, eos });
  const fctx: FlowInitContext = { spec, box, ndim, G, units, species, n, eos, bodies };
  const flows = flowsOf(spec);
  const whole = flows.filter((f) => f.def.wholeBox && f.def.init);
  if (whole.length > 1) throw new Error('only one flow can fill the whole box');
  const bg = spec.background ?? {};
  const bgX = bg.X ? fractions(bg.X) : species ? species.map((_, i) => (i === 0 ? 1 : 0)) : undefined;
  const stars = bodies.stars;
  const amb = stars.length ? ambient(spec, bodies) : null;

  let at: InitFn;
  if (whole.length) at = whole[0].def.init!(fctx, whole[0].values);
  else {
    const v = bg.v ?? [0, 0, 0];
    const w: Prim = { rho: amb ? amb.rho : (bg.rho ?? 1), vx: v[0], vy: v[1], vz: v[2], p: bg.p ?? (amb ? stars[0].pressure(amb.cut) : 1), s: bg.s ?? 0 };
    if (bgX) w.X = bgX;
    at = () => w;
  }

  const disks = disksOf(spec, bodies, G, n, ndim);
  const base = at;
  if (disks.length) {
    const one = disks.length === 1 ? disks[0] : null;
    at = one ? (x, y, z) => one.state(x, y, z) : (x, y, z) => sumDisks(disks.map((d) => d.state(x, y, z)));
    // A disk brings its own ambient gas. Where the setup names one too (a background density, or
    // the gas around its stars), that shows wherever the disk is thinner than it: a disk with a
    // profile of its own may end altogether.
    if (amb || bg.rho !== undefined) {
      const [disk, under] = [at, base];
      const least = amb ? amb.rho : bg.rho!;
      at = (x, y, z) => {
        const w = disk(x, y, z);
        return w.rho > least ? w : under(x, y, z);
      };
    }
    if (bgX) {
      const inner = at;
      at = (x, y, z) => ({ ...inner(x, y, z), X: bgX });
    }
  }

  if (amb) {
    const under = at;
    const placed = stars.map((st) => ({ st, X: st.part.X ? fractions(st.part.X) : bgX }));
    at = (x, y, z) => {
      for (const { st, X } of placed) {
        const rho = st.rho(Math.hypot(x - st.center[0], y - st.center[1], z - st.center[2]));
        if (rho > amb.cut) {
          // Cold: the star's own pressure at that density.
          const [vx, vy, vz] = st.vel;
          const w: Prim = { rho, vx, vy, vz, p: st.pressure(rho), s: st.scalar };
          if (X) w.X = X;
          return w;
        }
        // A polytrope thins into the ambient gas: its outermost layer, below the cut, is ambient
        // gas at the star's density, carrying the pressure at the cut.
        if (st.part.type === 'polytrope' && rho > amb.rho) {
          const [vx, vy, vz] = st.vel;
          const w: Prim = { rho, vx, vy, vz, p: st.pressure(amb.cut), s: 0 };
          if (X) w.X = X;
          return w;
        }
      }
      return under(x, y, z);
    };
  }

  const pctx: PaintContext = { box, n, ndim, eos, fractions };
  for (const p of spec.parts ?? []) {
    const paint = p.kind === 'region' ? regionPainter(p, pctx) : p.kind === 'obstacle' || p.kind === 'inflow' ? obstaclePainter(p, pctx) : null;
    if (!paint) continue;
    const under = at;
    at = (x, y, z) => paint(under(x, y, z), x, y, z);
  }
  for (const f of flows) if (f.def.decorate) at = f.def.decorate(fctx, f.values, at);
  return at;
}

function settlingOfSetup(spec: SetupSpec, eos: Eos): Settling {
  const { G } = scales(spec);
  const bodies = buildBodies(spec, { G, dx: 0, ndim: 3, box: boxOf(spec), eos });
  const amb = ambient(spec, bodies);
  const dwarfs = bodies.stars.some((s) => s.part.type === 'white-dwarf');
  // (A flow that only changes what the stars painted, a hot spot, brings no gas of its own.)
  const others = (spec.parts ?? []).some((p) => p.kind === 'disk' || p.kind === 'region' || (p.kind === 'flow' && flowValues(p).def.wholeBox));
  return {
    stars: bodies.stars.map((s) => ({ barotrope: s.barotrope, center: s.center, vel: s.vel, R: s.R, ...(s.reach !== undefined ? { reach: s.reach } : others ? { reach: OWN_REACH * s.R } : {}), scalar: s.scalar })),
    rhoCut: amb.cut,
    rhoAmb: amb.rho,
    pAmb: spec.background?.p ?? bodies.stars[0].barotrope.pressure(amb.cut),
    G,
    // A 0.6 M_sun dwarf settled to 1e-3 (12 rounds) stays as still as one settled to 1e-5 (26):
    // its central density swings by about 1% either way, from what the settling cannot reach.
    ...(dwarfs ? { tol: 1e-3, relax: 1.6 } : {}),
    // Gas that is not the stars' (a disk, a region, a flow) is not theirs to settle: a cell a
    // star does not take keeps the gas it has, and a star reaches no farther than just past
    // its own radius.
    ...(others ? { keep: true } : {}),
  };
}

/** The initial magnetic field's sources: potentials (stars, disk loops, flows) and face values (uniform fields, flows). */
function fieldOf(spec: SetupSpec, ndim: 2 | 3): { potential: PotentialFn | null; face: FaceFn | null } {
  const { units, G } = scales(spec);
  const box = boxOf(spec);
  const needsEos = hasWhiteDwarf(spec);
  const bodies = buildBodies(spec, { G, dx: 0, ndim, box, eos: needsEos ? specEos(spec, units) : undefined });
  const fctx: FlowContext = { spec, box, ndim, G, units, species: speciesOf(spec) };
  const pots: PotentialFn[] = [];
  const faces: FaceFn[] = [];
  for (const st of bodies.stars) {
    const A = starPotential(st);
    if (A) pots.push(A);
  }
  for (const d of disksOf(spec, bodies, G, 0, ndim)) {
    const loop = d.loop;
    if (loop)
      pots.push((x, y, z) => {
        const R = Math.hypot(x, y);
        if (R < 1e-9) return [0, 0, 0];
        const a = loop(R, z);
        return [(-a * y) / R, (a * x) / R, 0];
      });
  }
  for (const f of flowsOf(spec)) {
    if (f.def.potential) pots.push(f.def.potential(fctx, f.values));
    if (f.def.face) faces.push(f.def.face(fctx, f.values));
  }
  for (const p of spec.parts ?? []) {
    if (p.kind !== 'field') continue;
    const b = p.b ?? [0, 0, 0];
    const up = ndim === 2 ? 1 : 2;
    const strength = p.up;
    faces.push(strength !== undefined ? (axis) => (axis === up ? strength : 0) : (axis) => b[axis]);
  }
  const sum = <T extends unknown[]>(fs: ((...a: T) => number)[]) => (fs.length === 1 ? fs[0] : (...a: T) => fs.reduce((s, f) => s + f(...a), 0));
  return {
    potential: pots.length === 0 ? null : pots.length === 1 ? pots[0] : (x, y, z) => pots.reduce<[number, number, number]>((s, A) => { const a = A(x, y, z); return [s[0] + a[0], s[1] + a[1], s[2] + a[2]]; }, [0, 0, 0]),
    face: faces.length === 0 ? null : sum(faces),
  };
}

/** Which kinds of field source a setup has, without building anything. */
function fieldKinds(spec: SetupSpec): { potential: boolean; face: boolean } {
  let potential = false;
  let face = false;
  for (const p of spec.parts ?? []) {
    if (p.kind === 'star' && p.field) potential = true;
    if (p.kind === 'disk' && p.loops) potential = true;
    if (p.kind === 'field') face = true;
    if (p.kind === 'flow') {
      const { def } = flowValues(p);
      if (def.potential) potential = true;
      if (def.face) face = true;
    }
  }
  return { potential, face };
}

/** A featured parameter as the side bar shows it. */
function problemParam(spec: SetupSpec, f: Featured): ProblemParam {
  const path = featuredPaths(f)[0];
  const raw = getPath(spec, path);
  const value = typeof raw === 'boolean' ? (raw ? 1 : 0) : typeof raw === 'number' ? raw : 0;
  const id = featuredId(f);
  const leaf = path.split('.').pop()!;
  const p: ProblemParam = { id, label: f.label ?? id, value, min: f.min ?? -Infinity, max: f.max ?? Infinity };
  if (f.tip !== undefined) p.title = f.tip;
  if (f.step !== undefined) p.step = f.step;
  if (f.factor !== undefined) p.factor = f.factor;
  // A point mass's softening and sink are taken up by the run where the masses are now.
  if (/^parts\.\d+\.(soft|sink|sinkRate|onOrbit)$/.test(path) && (getPath(spec, path.replace(/\.\w+$/, '.kind')) === 'mass')) p.live = true;
  // A value of the units only sets the physical scale.
  if (path.startsWith('domain.units.') && leaf === 'value') p.scale = true;
  if (f.unit) {
    const times = f.unit.times;
    p.unit = { label: f.unit.label, factor: (v) => v[times] };
  }
  if (f.unitChoices) p.unitChoices = f.unitChoices;
  if (f.control) p.kind = f.control;
  if (f.dims) p.dims = f.dims;
  return p;
}

/** Compile a setup. The document must be whole: `extends` resolved (setup/codec.ts). */
export function compileSetup(spec: SetupSpec): Compiled {
  const d = spec.domain ?? {};
  const ph = spec.physics ?? {};
  const mesh = spec.mesh ?? {};
  const run = spec.run ?? {};
  const parts = spec.parts ?? [];
  const ids = partIds(parts);
  if (new Set(ids).size !== ids.length) throw new Error('two parts have the same id');
  // A formula that does not parse is found here, not at the first cell.
  parts.forEach((p, i) => {
    if (p.kind !== 'disk') return;
    try {
      checkDiskFormulas(p);
    } catch (e) {
      throw new Error(`${ids[i]}: ${(e as Error).message}`);
    }
  });
  const feats = paramFeatures(spec);
  const base = scales(spec);

  // The setup with a run's parameter values, kept for the last few sets asked for: init, the
  // masses and the field all ask with the same ones.
  const memo = new Map<string, SetupSpec>();
  const at = (params: Record<string, number> | undefined): SetupSpec => {
    if (!feats.length || !params) return spec;
    const key = feats.map((f) => params[featuredId(f)]).join('/');
    let s = memo.get(key);
    if (!s) {
      if (memo.size > 8) memo.clear();
      memo.set(key, (s = withParams(spec, params)));
    }
    return s;
  };

  const all = (b: number): Bc3 => [b, b, b];
  const flows = flowsOf(spec);
  const kinds = fieldKinds(spec);
  const hasMasses = parts.some((p) => p.kind === 'mass');
  const hasStars = parts.some((p) => p.kind === 'star');
  const debris = parts.some((p) => p.kind === 'orbit' && p.debris);
  const unitsFollow = !!d.units && (unitsFollowParts(d.units) || feats.some((f) => featuredPaths(f).some((p) => p.startsWith('domain.units.'))));
  const boxFeature = feats.find((f) => featuredPaths(f).includes('domain.box'));
  const burnFlow = flows.find((f) => f.def.burn);
  const heatFlow = flows.find((f) => f.def.heat);
  const flowCtx = (s: SetupSpec, ndim: 2 | 3): FlowContext => ({ spec: s, box: boxOf(s), ndim, ...scales(s), species: speciesOf(s) });

  // The damping zone is the first disk's that has one, as the setup opens.
  let damping: Problem['damping'];
  if (parts.some((p) => p.kind === 'disk' && p.damp)) {
    const bodies = buildBodies(spec, { G: base.G, dx: 0, ndim: 3, box: boxOf(spec) });
    damping = disksOf(spec, bodies, base.G, 0, 3).find((k) => k.damping)?.damping ?? undefined;
  }

  const mhd: MhdSpec | undefined =
    kinds.potential || kinds.face
      ? {
          ...(kinds.potential ? { potential: (params: Record<string, number>, ndim: 2 | 3) => fieldOf(at(params), ndim).potential! } : {}),
          ...(kinds.face ? { face: (params: Record<string, number>, ndim: 2 | 3) => fieldOf(at(params), ndim).face! } : {}),
        }
      : undefined;

  const grid = d.ndim !== undefined || d.n !== undefined || d.amr !== undefined ? { ...(d.ndim !== undefined ? { ndim: d.ndim } : {}), ...(d.n !== undefined ? { n: d.n } : {}), ...(d.amr !== undefined ? { amr: d.amr } : {}) } : undefined;
  const gext = ph.gext;
  const burn = ph.burn;
  const fl = ph.fluff;

  const optional: Partial<Problem> = {
    units: base.units === undefined ? undefined : unitsFollow ? (params: Record<string, number>) => scales(at(params)).units! : base.units,
    composition: namedComposition(ph.composition),
    species: speciesOf(spec),
    speciesGamma: ph.speciesGamma,
    boxParam: boxFeature ? featuredId(boxFeature) : undefined,
    checkpointDt: run.checkpointDt,
    tDyn: run.tDyn,
    rhoFloor: ph.floors?.rho,
    pFloor: ph.floors?.p,
    jeansFloor: ph.jeansFloor,
    jeansCells: ph.jeansCells,
    supports2d: d.supports2d,
    bc2d: d.bc2d,
    fluff: fl ? { rho: fl.rho ?? 0, u: fl.u ?? 0, v: fl.v ?? 0, ...(fl.va !== undefined ? { va: fl.va } : {}) } : undefined,
    params: feats.length ? feats.map((f) => problemParam(spec, f)) : undefined,
    masses: hasMasses
      ? (params, G, dx, ndim): PointMass[] => {
          const s = at(params);
          return buildBodies(s, { G, dx, ndim, box: boxOf(s), eos: hasWhiteDwarf(s) ? specEos(s, scales(s).units) : undefined }).masses;
        }
      : undefined,
    grid,
    debris: debris
      ? (params, G) => {
          const s = at(params);
          return buildBodies(s, { G, dx: 0, ndim: 3, box: boxOf(s), eos: hasWhiteDwarf(s) ? specEos(s, scales(s).units) : undefined }).debris!;
        }
      : undefined,
    refineMasses: mesh.refineMasses,
    frame: ph.frame === 'gas' || ph.frame === 'masses' ? ph.frame : undefined,
    refineRho: mesh.refineRho,
    refineDense: mesh.refineDense,
    cooling: ph.cooling,
    coolParams: ph.coolParams,
    burn: burnFlow
      ? (params) => burnFlow.def.burn!(flowCtx(at(params), 3), flowValues(at(params).parts!.find((p) => p.kind === 'flow' && p.name === burnFlow.part.name) as FlowPart).values)
      : burn
        ? (params): AnyBurnSpec => {
            const b = at(params).physics!.burn!;
            return 'network' in b ? { network: b.network, inShocks: !!b.inShocks, limiter: b.limiter ?? 0 } : { fuel: b.fuel, ash: b.ash, q: b.q, ea: b.ea, k: b.k, inShocks: b.inShocks ?? true };
          }
        : undefined,
    damping,
    viscosity: ph.viscosity,
    conductivity: ph.conductivity,
    gext: gext === undefined ? undefined : Array.isArray(gext) ? () => [...gext] as [number, number, number] : (ndim) => (ndim === 2 ? [0, -gext.down, 0] : [0, 0, -gext.down]),
    settle: hasStars ? (params, eos) => settlingOfSetup(at(params), eos) : undefined,
    heat: heatFlow
      ? (params, eos) => {
          const s = at(params);
          const sc = scales(s);
          const bodies = buildBodies(s, { G: sc.G, dx: 0, ndim: 3, box: boxOf(s), eos });
          const f = flowsOf(s).find((k) => k.def.heat)!;
          return f.def.heat!({ spec: s, box: boxOf(s), ndim: 3, ...sc, species: speciesOf(s), n: 0, eos, bodies }, f.values);
        }
      : undefined,
    startPaused: run.startPaused,
    toolScale: ph.toolScale,
    view: spec.view,
    mhd,
    disk: parts.some((p) => p.kind === 'disk' && p.stresses) ? true : undefined,
  };

  const problem: Problem = {
    id: spec.id ?? 'custom',
    label: spec.name ?? 'Custom setup',
    blurb: spec.blurb ?? '',
    eos: ph.eos ?? 'ideal',
    eosParams: { ...ph.eosParams },
    bcLo: d.bc?.lo ?? all(OUTFLOW),
    bcHi: d.bc?.hi ?? all(OUTFLOW),
    gravity: ph.gravity ?? 'off',
    G: base.G,
    box: boxOf(spec),
    tEnd: run.tEnd ?? Infinity,
    init: (n, eos, ndim, params) => initOf(at(params), n, eos, ndim),
    setup: spec,
  };
  // Only what the setup gives: an absent key is the Problem's own default (sim.ts configFor).
  for (const [k, v] of Object.entries(optional)) if (v !== undefined) (problem as unknown as Record<string, unknown>)[k] = v;

  const overrides: Partial<SimConfig> = {};
  const over = <K extends keyof SimConfig>(k: K, v: SimConfig[K] | undefined) => {
    if (v !== undefined) overrides[k] = v;
  };
  over('riemann', ph.riemann);
  over('limiter', ph.limiter);
  over('cfl', ph.cfl);
  over('vcycles', ph.vcycles);
  over('gravEnergyFlux', ph.gravEnergyFlux);
  over('refineHi', mesh.refineHi);
  over('refineLo', mesh.refineLo);
  over('timeBins', mesh.timeBins);
  over('jeansRefine', mesh.jeansRefine);
  return { spec, problem, overrides };
}

const compiledOnce = new WeakMap<SetupSpec, Compiled>();
/** A setup compiled, once per document: the same object each time it is asked for. */
export function compiledFrom(spec: SetupSpec): Compiled {
  let c = compiledOnce.get(spec);
  if (!c) compiledOnce.set(spec, (c = compileSetup(spec)));
  return c;
}
