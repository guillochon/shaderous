// What is wrong with a setup, as the dialog's issues list shows it: the rules of the option table
// (options.ts ruleIssues) and the checks on values a list of options cannot express, each a plain
// sentence with the path of the part or option it is about. Free of the DOM, and it never throws:
// a document that cannot even be built is itself an issue.
import { extensionOf } from '../geometry';
import { hasMeshFile, placeMesh, type MeshShape } from '../geometry/store';
import { defaultEosParams, makeEos, physicalEosParams, type Eos } from '../eos/eos';
import type { PointMass } from '../problems/types';
import { COMPOSITIONS, composition, type Composition } from '../units/composition';
import { codeG, type UnitSystem } from '../units/units';
import { compileSetup } from './compile';
import { FACTS, PART_OPTIONS, ruleIssues, type Issue } from './options';
import { buildBodies, partIds, type Bodies, type BuiltStar } from './parts/bodies';
import { FLOWS } from './parts/flow';
import { eggleton } from '../problems/roche';
import type { DiskPart, OrbitPart, Part, SetupSpec } from './spec';
import { unitSystem } from './units';

export interface CheckEnv {
  /** GPU memory the document's grid needs and what the page may take, in bytes (SetupHost.memory) */
  memory?: { need: number; budget: number } | null;
  /** What the simulators would refuse in it as the page loads it (sim.ts refusal): an error, since it would not load */
  refusal?: string | null;
}

/** A setup's bodies as far as they can be built from the document alone. */
export interface BuiltSetup {
  G: number;
  box: number;
  ndim: 2 | 3;
  /** cell size of the document's grid; 0 where it names no resolution */
  dx: number;
  /** the id each part goes by */
  ids: string[];
  /** stars and point masses, placed by the orbits that could be followed; null if none could be built */
  bodies: Bodies | null;
  /** parts left out of `bodies`, by their place: orbits that name what is not there, stars that need more than the document gives */
  skipped: number[];
  /** why the last attempt failed, where parts were left out for it */
  error?: string;
}

const partsOf = (spec: SetupSpec): Part[] => spec.parts ?? [];
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** G as the compiler has it, without the units that follow a part (those need the bodies first). */
function gravConst(spec: SetupSpec, units: UnitSystem | undefined): number {
  const g = spec.physics?.G;
  if (typeof g === 'number') return g;
  if (g === 'units' && units) return codeG(units);
  return 1;
}

/** The setup's units where they stand on their own (not on a part's mass or radius). */
function plainUnits(spec: SetupSpec): UnitSystem | undefined {
  const u = spec.domain?.units;
  if (!u) return undefined;
  try {
    return unitSystem(u, () => {
      throw new Error('follows a part');
    });
  } catch {
    return undefined;
  }
}

/** The equation of state the setup opens with, for a white dwarf's structure; undefined if it cannot be made. */
function eosOf(spec: SetupSpec, units: UnitSystem | undefined): Eos | undefined {
  try {
    const ph = spec.physics ?? {};
    const c = ph.composition;
    let comp: Composition | undefined;
    if (typeof c === 'string') comp = (COMPOSITIONS as Record<string, Composition>)[c];
    else if (c) comp = composition(c.name, c.X, c.ionization ?? 'full');
    return makeEos(ph.eos ?? 'ideal', { ...defaultEosParams(), ...physicalEosParams(units, comp), ...ph.eosParams });
  } catch {
    return undefined;
  }
}

/** What an orbit's two bodies may be, by its type. */
const ORBIT_KINDS: Record<OrbitPart['type'], [Part['kind'][], Part['kind'][]]> = {
  binary: [['mass', 'star'], ['mass', 'star']],
  flyby: [['star'], ['mass']],
  collision: [['star'], ['star']],
};
const KIND_NAMES: Partial<Record<Part['kind'], string>> = { mass: 'a point mass', star: 'a star' };

/** What is wrong with an orbit's two names (empty: nothing). */
function orbitFaults(o: OrbitPart, parts: Part[], ids: string[]): string[] {
  const out: string[] = [];
  const want = ORBIT_KINDS[o.type] ?? ORBIT_KINDS.binary;
  const of = Array.isArray(o.of) ? o.of : [];
  for (let k = 0; k < 2; k++) {
    const id = of[k];
    const which = k === 0 ? 'first' : 'second';
    if (!id) {
      out.push(`its ${which} body is not chosen`);
      continue;
    }
    const at = ids.indexOf(id);
    if (at < 0) out.push(`it names "${id}", which is not there`);
    else if (!want[k].includes(parts[at].kind)) out.push(`its ${which} body must be ${want[k].map((w) => KIND_NAMES[w]).join(' or ')} for this kind of orbit, and "${id}" is ${KIND_NAMES[parts[at].kind] ?? `a ${parts[at].kind}`}`);
  }
  if (of[0] && of[0] === of[1]) out.push('it names the same body twice');
  if (o.type === 'binary' && o.fill !== undefined && of[0] && parts[ids.indexOf(of[0])]?.kind !== 'star') out.push('its fill is of its first body, which must be a star');
  return out;
}

/** What is wrong with a binary's other fields: a frame that cannot turn with it, a tilt its stars cannot take. */
function binaryIssues(spec: SetupSpec, parts: Part[], ids: string[], label: (i: number) => string, out: Issue[]) {
  const binaries = parts.flatMap((p, i) => (p.kind === 'orbit' && p.type === 'binary' ? [{ o: p, i }] : []));
  const withStar = (o: OrbitPart) => (o.of ?? []).some((id) => parts[ids.indexOf(id)]?.kind === 'star');
  const framed = binaries.filter(({ o }) => !o.ecc && (o.corotating || withStar(o)));
  if (framed.length > 1) out.push({ level: 'error', path: `parts.${framed[1].i}`, text: `${label(framed[1].i)}: only one circular binary may hold stars or turn the frame, and ${label(framed[0].i)} already does` });
  for (const { o, i } of binaries) {
    if (o.corotating && o.ecc) out.push({ level: 'error', path: `parts.${i}.corotating`, text: `${label(i)}: only a circular orbit has a frame that turns with it (its eccentricity is ${fmt(o.ecc)})` });
    if (o.corotating && spec.physics?.rotation?.omega?.some((w) => w !== 0)) out.push({ level: 'error', path: `parts.${i}.corotating`, text: `${label(i)}: it turns the frame with the orbit, and the setup has a rotating frame of its own. Clear one` });
    if (o.corotating && parts.some((p) => p.kind === 'disk')) out.push({ level: 'error', path: `parts.${i}.corotating`, text: `${label(i)}: a disk is set up moving in an inertial frame, not in one turning with the binary` });
    if (o.track && !o.corotating) out.push({ level: 'warning', path: `parts.${i}.track`, text: `${label(i)}: it follows the orbit only in the corotating frame, which it does not turn` });
    if (withStar(o) && (o.incl ?? 0) !== 0) out.push({ level: 'error', path: `parts.${i}.incl`, text: `${label(i)}: a binary with a star orbits in the x-y plane (its inclination must be 0)` });
    if (o.q !== undefined && (o.of ?? []).every((id) => parts[ids.indexOf(id)]?.kind === 'star')) out.push({ level: 'warning', path: `parts.${i}.q`, text: `${label(i)}: the mass ratio of two stars is their masses’: q is not used` });
  }
}

/** How full a binary's stars are of their Roche lobes (Eggleton 1983): above 1 one pours gas onto its companion from the start. */
function rocheIssues(parts: Part[], ids: string[], built: BuiltSetup, label: (i: number) => string, out: Issue[]) {
  const b = built.bodies;
  if (!b?.binary) return;
  parts.forEach((p, i) => {
    if (p.kind !== 'orbit' || p.type !== 'binary' || p.ecc) return;
    const [A, B] = p.of;
    const mass = (id: string) => b.stars.find((s) => s.id === id)?.M ?? (b.masses[b.massIds.indexOf(id)]?.gm ?? 0) / built.G;
    const sep = b.binary!.separation;
    for (const [id, other] of [[A, B], [B, A]]) {
      const st = b.stars.find((s) => s.id === id);
      if (!st || !(mass(other) > 0)) continue;
      const fill = st.R / (eggleton(st.M / mass(other)) * sep);
      if (fill > 1.001 && p.fill === undefined)
        out.push({ level: 'warning', path: `parts.${i}.separation`, text: `${label(i)}: star "${id}" is ${fmt(fill)} times the size of its Roche lobe: it overflows onto "${other}" from the start` });
      else if (fill > 1.001 && id !== A) out.push({ level: 'warning', path: `parts.${i}.fill`, text: `${label(i)}: star "${id}" overfills its Roche lobe too, ${fmt(fill)} times its size` });
    }
  });
}

/**
 * The stars and point masses of a document, built as far as they can be. An orbit that names
 * what is not there is left out, and so is a star that cannot be built from the document alone
 * (a white dwarf without its equation of state): the rest is still placed, for the checks and
 * for the dialog's preview.
 */
export function buildSetupBodies(spec: SetupSpec): BuiltSetup {
  const parts = partsOf(spec);
  const ids = partIds(parts);
  const box = spec.domain?.box ?? 1;
  const ndim: 2 | 3 = spec.domain?.ndim === 2 ? 2 : 3;
  const n = spec.domain?.n ?? 0;
  const dx = n > 0 ? box / n : 0;
  const units = plainUnits(spec);
  const G = gravConst(spec, units);
  const out: BuiltSetup = { G, box, ndim, dx, ids, bodies: null, skipped: [] };
  const eos = parts.some((p) => p.kind === 'star' && p.type === 'white-dwarf') ? eosOf(spec, units) : undefined;

  const dropped = new Set<number>();
  parts.forEach((p, i) => {
    if (p.kind === 'orbit' && orbitFaults(p, parts, ids).length) dropped.add(i);
  });
  // Every part keeps the id it has in the whole document, whatever is left out before it.
  const attempt = (): Bodies => {
    const kept = parts.flatMap((p, i) => (dropped.has(i) ? [] : [{ ...p, id: ids[i] } as Part]));
    return buildBodies({ ...spec, parts: kept }, { G, dx, ndim, box, eos });
  };
  // Leaving a star out takes the orbits that place it out too.
  const drop = (which: (p: Part) => boolean) => {
    parts.forEach((p, i) => {
      if (which(p)) dropped.add(i);
    });
    parts.forEach((p, i) => {
      if (p.kind === 'orbit' && (p.of ?? []).some((id) => dropped.has(ids.indexOf(id)))) dropped.add(i);
    });
  };
  const steps: (() => void)[] = [
    () => undefined,
    () => drop((p) => p.kind === 'star' && p.type === 'white-dwarf'),
    () => drop((p) => p.kind === 'star'),
    () => drop((p) => p.kind === 'orbit'),
  ];
  for (const step of steps) {
    step();
    try {
      out.bodies = attempt();
      break;
    } catch (e) {
      out.error ??= message(e);
    }
  }
  out.skipped = [...dropped].sort((a, b) => a - b);
  return out;
}

/** A disk's mass, by integrating its surface density over radius (per unit length in 2D). */
export function diskMass(d: DiskPart, gm: number, ndim: 2 | 3, box: number): number {
  const { rIn, rOut, sigma0, cs } = d;
  if (!(rIn > 0) || !(rOut > 0) || !(sigma0 > 0)) return 0;
  const slope = d.slope ?? -1.5;
  const surface = (r: number): number => {
    if (d.profile === 'thin') return sigma0 * Math.exp(-((rIn / r) ** 4)) * Math.exp(-((r / rOut) ** 8));
    // Balanced: the midplane density times sqrt(2 pi) scale heights, H = cs / Omega.
    const mid = sigma0 * r ** slope * Math.exp(-((rIn / r) ** 8) - (r / rOut) ** 8);
    const h = ndim === 3 && gm > 0 ? (cs * r ** 1.5) / Math.sqrt(gm) : 1;
    return mid * (ndim === 3 ? Math.sqrt(2 * Math.PI) * h : 1);
  };
  // Even steps in ln r from well inside the cavity to past the outer taper (or the box's corner).
  const lo = Math.log(Math.min(rIn, rOut) / 4);
  const hi = Math.log(Math.min(2 * Math.max(rIn, rOut), Math.max(box, Math.min(rIn, rOut))));
  if (!(hi > lo)) return 0;
  const steps = 400;
  const h = (hi - lo) / steps;
  let sum = 0;
  for (let i = 0; i <= steps; i++) {
    const r = Math.exp(lo + i * h);
    const f = 2 * Math.PI * r * surface(r) * r;
    sum += (i === 0 || i === steps ? 0.5 : 1) * f;
  }
  const m = sum * h;
  return Number.isFinite(m) ? m : 0;
}

const fmt = (v: number): string => String(+v.toPrecision(3));
const gb = (bytes: number): string => (bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`);

interface Body {
  index: number;
  id: string;
  star?: BuiltStar;
  mass?: PointMass;
  pos: number[];
  /** how far it reaches: a star's radius; nothing for a point */
  radius: number;
  /** placed by a flyby, which starts wherever the parabola says */
  flyby: boolean;
}

function bodiesOf(spec: SetupSpec, built: BuiltSetup): Body[] {
  const parts = partsOf(spec);
  const b = built.bodies;
  if (!b) return [];
  const flyby = new Set(parts.flatMap((p) => (p.kind === 'orbit' && p.type === 'flyby' ? p.of ?? [] : [])));
  const out: Body[] = [];
  for (const s of b.stars) out.push({ index: built.ids.indexOf(s.id), id: s.id, star: s, pos: s.center, radius: s.R, flyby: flyby.has(s.id) });
  // (A star's core, "<star>.core", is its star's part.)
  const partOf = (id: string) => built.ids.indexOf(id.endsWith('.core') && !built.ids.includes(id) ? id.slice(0, -5) : id);
  b.masses.forEach((m, k) => out.push({ index: partOf(b.massIds[k]), id: b.massIds[k], mass: m, pos: m.pos, radius: 0, flyby: flyby.has(b.massIds[k]) }));
  return out;
}

function checkBodies(spec: SetupSpec, built: BuiltSetup, out: Issue[]): void {
  const bodies = bodiesOf(spec, built);
  const half = built.box / 2;
  const axes = built.ndim === 2 ? [0, 1] : [0, 1, 2];
  const name = (b: Body) => `${b.star ? 'Star' : 'Point mass'} "${b.id}"`;
  for (const b of bodies) {
    const beyond = Math.max(...axes.map((a) => Math.abs(b.pos[a]) + b.radius - half));
    if (!(beyond > 1e-9 * built.box)) continue;
    const path = `parts.${b.index}`;
    if (b.star) out.push({ level: 'error', path, text: `${name(b)} leaves the box: it reaches ${fmt(beyond)} past a face` });
    // A flyby starts where its parabola says, often outside the box: the mass still pulls from there.
    else if (!b.flyby) out.push({ level: 'warning', path, text: `${name(b)} starts outside the box` });
  }
  for (let i = 0; i < bodies.length; i++)
    for (let j = i + 1; j < bodies.length; j++) {
      const [a, b] = [bodies[i], bodies[j]];
      const d = Math.hypot(...axes.map((k) => a.pos[k] - b.pos[k]));
      const path = `parts.${b.index}`;
      if (a.star && b.star) {
        if (d < a.radius + b.radius) out.push({ level: 'error', path, text: `Stars "${a.id}" and "${b.id}" overlap: their centers are ${fmt(d)} apart and their radii add to ${fmt(a.radius + b.radius)}` });
      } else if (a.star || b.star) {
        const [star, mass] = a.star ? [a, b] : [b, a];
        if (d < star.radius && !mass.mass?.massless && mass.id !== `${star.id}.core`) out.push({ level: 'warning', path: `parts.${mass.index}`, text: `Point mass "${mass.id}" starts inside star "${star.id}"` });
      } else {
        if (a.mass?.massless || b.mass?.massless) continue;
        const reach = Math.max(a.mass?.soft ?? 2 * built.dx, b.mass?.soft ?? 2 * built.dx);
        if (d <= reach) out.push({ level: d === 0 ? 'error' : 'warning', path, text: d === 0 ? `Point masses "${a.id}" and "${b.id}" start at the same place: move one, or add an orbit that places them` : `Point masses "${a.id}" and "${b.id}" overlap: they start ${fmt(d)} apart, within their softening of ${fmt(reach)}` });
      }
    }
}

/**
 * Everything wrong with a setup: errors stop Apply, warnings do not. `env.memory` is what the
 * page says the grid needs and may take.
 */
export function checkSetup(spec: SetupSpec, env: CheckEnv = {}): Issue[] {
  const out: Issue[] = [];
  const guard = (what: () => void) => {
    try {
      what();
    } catch (e) {
      out.push({ level: 'error', text: `This setup could not be checked: ${message(e)}` });
    }
  };
  guard(() => out.push(...ruleIssues(spec)));
  if (env.refusal) out.push({ level: 'error', text: env.refusal });
  // A shearing box's equilibrium is the background flow v_y = -q Omega x: gas started at rest
  // is flung about by the tidal and Coriolis forces (densities 0.4 to 6 in a box of uniform gas).
  guard(() => {
    if (!spec.physics?.shear) return;
    const flows = partsOf(spec).filter((p) => p.kind === 'flow').map((p) => (p as { name: string }).name);
    if (!flows.some((n) => n === 'shearing-wave' || n === 'shearing-noise'))
      out.push({ level: 'warning', path: 'physics.shear', text: 'Shearing box: the initial state should hold the background flow v_y = -q Omega x (the shearing-noise or shearing-wave flow), or the gas starts far from equilibrium' });
  });
  // A sink empties its neighborhood to the density floor, where a field's Alfven speed sets the
  // step: without a cap on it (Hydro: tenuous-gas caps, va) the run crawls (a disk with B = 1e-3
  // around a sink advanced 50 times slower than without the field).
  guard(() => {
    const sink = partsOf(spec).some((p) => p.kind === 'mass' && ((p as { sink?: number }).sink ?? 0) > 0);
    if (sink && FACTS['magnetic'].of(spec) && !((spec.physics?.fluff?.va ?? 0) > 0))
      out.push({ level: 'warning', path: 'physics.fluff.va', text: 'A magnetic field around a sink: the gas the sink empties sets the timestep by its Alfven speed. Set an Alfven speed cap (Hydro, tenuous-gas caps), as the magnetized disk does' });
  });

  // Fuel to ash burning: both are species the gas carries, and a flow with its own burning wins.
  guard(() => {
    const burn = spec.physics?.burn;
    if (!burn || 'network' in burn) return;
    const species = spec.physics?.species;
    const carried = Array.isArray(species) ? species : [];
    for (const [what, name] of [['fuel', burn.fuel], ['ash', burn.ash]] as const)
      if (!carried.includes(name)) out.push({ level: 'error', path: `physics.burn.${what}`, text: `Burning: its ${what}, "${name}", is not a species the gas carries. Add it to Species` });
    if (burn.fuel === burn.ash) out.push({ level: 'error', path: 'physics.burn.ash', text: 'Burning: its fuel and its ash are the same species' });
    const flow = partsOf(spec).find((p) => p.kind === 'flow' && FLOWS[p.name]?.burn);
    if (flow) out.push({ level: 'warning', path: 'physics.burn', text: `Burning: the flow "${(flow as { name: string }).name}" brings its own, which is used instead` });
  });

  guard(() => {
    const parts = partsOf(spec);
    const ids = partIds(parts);
    const label = (i: number) => `${PART_OPTIONS[parts[i].kind]?.label ?? parts[i].kind} "${ids[i]}"`;

    // What the parts name.
    parts.forEach((p, i) => {
      if (p.kind === 'orbit') for (const f of orbitFaults(p, parts, ids)) out.push({ level: 'error', path: `parts.${i}.of`, text: `${label(i)}: ${f}` });
      if (p.kind === 'disk') {
        const about = Array.isArray(p.about) ? p.about : [];
        const found = about.filter((id) => ids.includes(id));
        for (const id of about) {
          const at = ids.indexOf(id);
          if (at < 0) out.push({ level: 'error', path: `parts.${i}.about`, text: `${label(i)}: it names "${id}", which is not there` });
          else if (parts[at].kind !== 'mass' && parts[at].kind !== 'star') out.push({ level: 'error', path: `parts.${i}.about`, text: `${label(i)}: it cannot orbit "${id}", which is a ${parts[at].kind}` });
        }
        if (!found.length) out.push({ level: 'error', path: `parts.${i}.about`, text: `${label(i)}: it orbits nothing. Choose the point masses it goes around` });
        if (p.rOut <= p.rIn) out.push({ level: 'warning', path: `parts.${i}.rOut`, text: `${label(i)}: its outer radius is not beyond its inner radius` });
      }
    });
    if (spec.background?.relativeTo !== undefined && !parts.some((p, i) => p.kind === 'star' && ids[i] === spec.background!.relativeTo))
      out.push({ level: 'error', path: 'background.rho', text: `The ambient gas is given relative to star "${spec.background.relativeTo}", which is not there` });

    // More than one flow that fills the box: the second would paint over the first.
    const whole = parts.flatMap((p, i) => (p.kind === 'flow' && FLOWS[p.name]?.wholeBox ? [i] : []));
    if (whole.length > 1 && !out.some((o) => /only one flow/i.test(o.text)))
      out.push({ level: 'error', path: `parts.${whole[1]}`, text: `${whole.length} flows each fill the whole box (${whole.map((i) => `"${ids[i]}"`).join(', ')}): only one can` });
    // A shape is read as its kind says: a box by its corners, a half-space by its normal.
    parts.forEach((p, i) => {
      if (p.kind !== 'region' && p.kind !== 'obstacle' && p.kind !== 'inflow') return;
      const s = p.shape as Partial<Record<'type' | 'min' | 'max' | 'normal', unknown>> | undefined;
      const path = `parts.${i}.shape.type`;
      if (!s || typeof s !== 'object') out.push({ level: 'error', path, text: `${label(i)}: it has no shape` });
      else if (s.type === 'box' && !(Array.isArray(s.min) && Array.isArray(s.max))) out.push({ level: 'error', path, text: `${label(i)}: a box needs its lower and upper corners` });
      else if (s.type === 'half' && !(Array.isArray(s.normal) && Math.hypot(...(s.normal as number[])) > 0)) out.push({ level: 'error', path, text: `${label(i)}: a half-space needs a normal that is not zero` });
      else if (s.type === 'mesh') out.push(...meshIssues(p.shape as MeshShape, i, label(i), spec));
    });
    parts.forEach((p, i) => {
      if (p.kind === 'flow' && !FLOWS[p.name]) out.push({ level: 'error', path: `parts.${i}.name`, text: `${label(i)}: this build has no flow named "${p.name}"` });
    });

    const built = buildSetupBodies(spec);
    const gamma = spec.physics?.eosParams?.gamma ?? 5 / 3;
    parts.forEach((p, i) => {
      if (p.kind === 'mass' && p.soft !== undefined && built.dx > 0 && p.soft < 2 * built.dx)
        out.push({ level: 'warning', path: `parts.${i}.soft`, text: `${label(i)}: its softening of ${fmt(p.soft)} is under two cells of the grid (${fmt(2 * built.dx)}), and is raised to that` });
      if (p.kind === 'star' && p.type === 'polytrope' && p.core) {
        const soft = p.core.soft ?? (p.radius ?? 1) / 20;
        if (p.rhoc !== undefined) out.push({ level: 'error', path: `parts.${i}.core`, text: `${label(i)}: a star with a core is given by its mass and radius, not its central density` });
        else if (!(p.core.mass > 0 && p.core.mass < (p.mass ?? 1))) out.push({ level: 'error', path: `parts.${i}.core.mass`, text: `${label(i)}: its core’s mass of ${fmt(p.core.mass)} must be above zero and below the star’s ${fmt(p.mass ?? 1)}` });
        if (built.dx > 0 && soft < 2 * built.dx)
          out.push({ level: 'warning', path: `parts.${i}.core.soft`, text: `${label(i)}: its core’s softening of ${fmt(soft)} is under two cells of the grid (${fmt(2 * built.dx)}): the envelope’s inner part, shaped by it, is not resolved` });
      }
      if (p.kind === 'star' && p.type === 'polytrope') {
        const n = p.n ?? 1.5;
        if (n > 0 && gamma < 1 + 1 / n - 1e-9)
          out.push({ level: 'warning', path: `parts.${i}.n`, text: `${label(i)}: a polytrope of index ${fmt(n)} is unstable to convection in gas of gamma ${fmt(gamma)}, which is below 1 + 1/n = ${fmt(1 + 1 / n)}` });
      }
      if (p.kind === 'star' && built.skipped.includes(i) && !out.some((o) => o.level === 'error' && o.path?.startsWith(`parts.${i}`)))
        out.push({ level: 'warning', path: `parts.${i}`, text: `${label(i)} could not be built from the document alone${built.error ? ` (${built.error})` : ''}: its place and size are not checked` });
    });

    checkBodies(spec, built, out);
    binaryIssues(spec, parts, ids, label, out);
    rocheIssues(parts, ids, built, label, out);

    // A disk heavier than a tenth of what it orbits pulls on itself more than the setup accounts for.
    if (built.bodies) {
      const b = built.bodies;
      parts.forEach((p, i) => {
        if (p.kind !== 'disk') return;
        const about = Array.isArray(p.about) ? p.about : [];
        const gm =
          about.reduce((s, id) => s + (b.masses[b.massIds.indexOf(id)]?.gm ?? 0), 0) +
          about.reduce((s, id) => s + built.G * (b.stars.find((st) => st.id === id)?.M ?? 0), 0);
        if (!(gm > 0)) return;
        const m = built.G * diskMass(p, gm, built.ndim, built.box);
        if (m > 0.1 * gm) out.push({ level: 'warning', path: `parts.${i}.sigma0`, text: `${label(i)}: its mass is ${fmt(m / gm)} of what it orbits. Above a tenth its own gravity matters, and the disk is set up without it` });
      });
    }
  });

  const mem = env.memory;
  if (mem && mem.need > mem.budget) out.push({ level: 'error', path: 'domain.n', text: `The grid needs ${gb(mem.need)} of GPU memory and ${gb(mem.budget)} is available: lower the resolution or add refinement` });

  try {
    compileSetup(spec);
  } catch (e) {
    const text = message(e);
    if (!out.some((o) => o.text.includes(text))) out.push({ level: 'error', text: `This setup cannot be built: ${text}` });
  }
  return out;
}

/**
 * A mesh shape (geometry/store.ts): a file that is named and loaded, a size for STL (which has
 * no units), and, once it is loaded, its thinnest part against the finest cell (plans/wind-tunnel.md
 * section 1: a wall thinner than two cells is not a wall).
 */
function meshIssues(sh: MeshShape, i: number, label: string, spec: SetupSpec): Issue[] {
  const path = `parts.${i}.shape`;
  if (typeof sh.file !== 'string' || !sh.file) return [{ level: 'error', path: `${path}.file`, text: `${label}: a mesh shape needs its file` }];
  const out: Issue[] = [];
  if (extensionOf(sh.file) === 'stl' && !(sh.scale! > 0) && !(sh.fit! > 0)) out.push({ level: 'error', path: `${path}.fit`, text: `${label}: an STL file has no units: give it a fit (its longest side as a fraction of the box) or a scale` });
  if (!hasMeshFile(sh.file)) {
    out.push({ level: 'warning', path: `${path}.file`, text: `${label}: the file "${sh.file}" is not loaded yet: the run needs it beside the setup document, or dropped on the Setup dialog` });
    return out;
  }
  const d = spec.domain ?? {};
  const box = typeof d.box === 'number' ? d.box : 1;
  const n = typeof d.n === 'number' ? d.n : 256;
  const ndim = d.ndim === 3 ? 3 : 2;
  try {
    const m = placeMesh(sh, { box, ndim, dx: box / n });
    if (ndim === 3 && m.report.thinnest < 2) out.push({ level: 'warning', path, text: `${label}: its thinnest part is ${m.report.thinnest.toFixed(1)} finest cells across: a wall thinner than two cells is not a wall` });
    if (m.report.stray > 0.1) out.push({ level: 'warning', path, text: `${label}: the mesh is open or broken (its winding number strays ${m.report.stray.toFixed(2)} from 0 or 1 away from its surface): what is inside is a guess near the holes` });
    if (m.min.some((x, a) => (ndim === 2 && a === 2 ? false : x < -box / 2)) || m.max.some((x, a) => (ndim === 2 && a === 2 ? false : x > box / 2))) out.push({ level: 'warning', path, text: `${label}: it reaches past the box` });
  } catch (e) {
    out.push({ level: 'error', path: `${path}.file`, text: `${label}: ${(e as Error).message}` });
  }
  return out;
}
