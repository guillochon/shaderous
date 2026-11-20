// The bodies of a setup: its stars and point masses, placed by its orbits.
//
// The star and orbit builders are those of the presets they came from (the polytrope, the tidal
// disruption, the white dwarf collision, the binary), kept operation for operation: a preset
// compiled from its JSON file starts bit for bit as its hand-written problem did.
import { aneosKeyOf, type Eos } from '../../eos/eos';
import { aneosFromT, aneosMaterialsOf, aneosTable, type AneosMaterial, type AneosTable } from '../../eos/aneos';
import { buildPlanet } from '../../problems/planet';
import { SI } from '../../units/constants';
import { dipoleField, gm17Dipole, gm17Random, randomField, type StarProfile } from '../../mhd/star_fields';
import { coredPolytrope } from '../../problems/cored_polytrope';
import { laneEmden, laneEmdenCached, polytropeK, polytropeMass, polytropeScales } from '../../problems/lane_emden';
import { polytropeBarotrope, type Barotrope } from '../../problems/settle';
import type { PointMass } from '../../problems/types';
import { binaryOmega, fillSeparation, l1Distance } from '../../problems/roche';
import { coldPressure, degenerateBarotrope, scalesOf, whiteDwarf } from '../../problems/white_dwarf';
import type { MassPart, OrbitPart, Part, SetupSpec, StarPart, Vec3 } from '../spec';

/** The id a part goes by: its own, or its kind and place among parts of that kind ("mass1"). */
export function partIds(parts: readonly Part[]): string[] {
  const count: Record<string, number> = {};
  return parts.map((p) => p.id ?? `${p.kind}${(count[p.kind] = (count[p.kind] ?? 0) + 1)}`);
}

export interface BuiltStar {
  id: string;
  part: StarPart;
  /** mass (with its core's, if it has one), radius and central density, in code units */
  M: number;
  R: number;
  rhoc: number;
  /** density at radius r from its center, and the cold pressure of its gas at a density */
  rho(r: number): number;
  pressure(rho: number): number;
  barotrope: Barotrope;
  /** a point-mass core at its center: its mass and softening (StarPart.core) */
  core?: { mass: number; soft: number };
  /** farthest from its center its settling may reach */
  reach?: number;
  center: number[];
  vel: number[];
  /**
   * In a circular binary, not in its corotating frame: the star turns with the orbit, at this
   * angular velocity about z through its center (tidally locked), on top of `vel`.
   */
  spin?: number;
  scalar: number;
  /** its pressure profile, for a field inside it */
  profile(): StarProfile;
  /**
   * A planet (problems/planet.ts): its gas at a radius, with the pressure it has there (warm,
   * not on one cold curve) and which of the run's materials it is (the mass fraction of the
   * second). A planet is not settled: it starts on its own integrated structure.
   */
  planet?: { at(r: number): { rho: number; p: number; xb: number } };
}

const starCache = new Map<string, Omit<BuiltStar, 'id' | 'part' | 'center' | 'vel' | 'scalar'>>();

/** A polytrope given by its central density: the n = 1 star in closed form, any other from Lane-Emden. */
function polytropeByDensity(n: number, R: number, rhoc: number, G: number) {
  const g = 1 + 1 / n;
  if (n === 1) {
    const K = (2 * G * R ** 2) / Math.PI;
    const density = (r: number) => {
      if (r >= R) return 0;
      const xi = (Math.PI * r) / R;
      return xi < 1e-6 ? rhoc : (rhoc * Math.sin(xi)) / xi;
    };
    return { K, M: (4 / Math.PI) * rhoc * R ** 3, density, pressure: (rho: number) => K * rho * rho };
  }
  const le = laneEmdenCached(n);
  const K = polytropeK(le, rhoc, R, G);
  return { K, M: polytropeMass(le, rhoc, R), density: (r: number) => rhoc * Math.pow(le.theta(r / R), n), pressure: (rho: number) => K * Math.pow(rho, g) };
}

/** The table a planet is built from, if the run's equation of state is ANEOS and its table has come. */
function planetTable(eos: Eos | undefined): { table: AneosTable; names: AneosMaterial[] } | null {
  if (eos?.kind !== 'aneos') return null;
  try {
    const key = aneosKeyOf(eos.params);
    return { table: aneosTable(key), names: aneosMaterialsOf(key) };
  } catch {
    return null;
  }
}

/**
 * A planet (StarPart.material and the fields after it). Without the equation of state's table
 * (a document being checked, or drawn, before a run has loaded it) it is a uniform ball of
 * about the right size, R = M^0.27 in Earth units, which is enough to place it.
 */
function buildPlanetStar(part: StarPart, G: number, eos: Eos | undefined): Omit<BuiltStar, 'id' | 'part' | 'center' | 'vel' | 'scalar'> {
  const M = part.mass ?? 1;
  const tab = planetTable(eos);
  if (!tab || !eos) {
    const R = Math.pow(M, 0.27);
    const rho = M / ((4 / 3) * Math.PI * R ** 3);
    return { M, R, rhoc: rho, rho: (r) => (r < R ? rho : 0), pressure: () => 0, barotrope: { pressure: () => 0, rhoOfH: () => 0 }, profile: () => ({ pressure: () => 0, pc: 0, pcut: 0, radius: R }) };
  }
  const xbOf = (m: string): number => {
    const i = tab.names.indexOf(m as AneosMaterial);
    if (i < 0) throw new Error(`a planet of ${m} needs it among the run's materials (${tab.names.join(', ')}): its species are its materials, the mantle's first`);
    return i;
  };
  const mantle = xbOf(part.material ?? tab.names[0]);
  const cf = part.coreFraction ?? 0.3;
  const layers = part.coreMaterial ? [{ xb: xbOf(part.coreMaterial), fraction: cf }, { xb: mantle, fraction: 1 - cf }] : [{ xb: mantle, fraction: 1 }];
  const Ts = part.surfaceT ?? 2000;
  // The code's units in cgs: density and specific energy from the equation of state's
  // conversions, and the length from G (G_cgs = G v^2 / (rho L^2) in those units).
  const { helmRho: cr, helmE: ce } = eos.params;
  const L = Math.sqrt((G * ce) / (cr * SI.G * 1e3));
  const pl = buildPlanet(tab.table, layers, M * cr * L ** 3, Ts);
  const cp = cr * ce;
  const at = (r: number) => {
    const s = pl.at(r * L);
    return { rho: s.rho / cr, p: s.p / cp, xb: layers[s.layer].xb };
  };
  // Outside itself: its mantle's material at the surface's temperature (the vapor over it).
  const pressure = (rho: number) => aneosFromT(tab.table, rho * cr, Ts, mantle).p / cp;
  return {
    M, R: pl.R / L, rhoc: pl.rhoc / cr,
    rho: (r) => at(r).rho,
    pressure,
    barotrope: { pressure, rhoOfH: () => 0 },
    profile: () => ({ pressure: (r) => at(r).p, pc: pl.p[0] / cp, pcut: 0, radius: pl.R / L }),
    planet: { at },
  };
}

function buildStar(part: StarPart, G: number, eos: Eos | undefined) {
  // A white dwarf is built on its equation of state's cold curve: the electrons' scales, and the Coulomb term if it has one.
  const key = JSON.stringify([part.type, part.n, part.radius, part.mass, part.rhoc, part.core, G, part.type === 'white-dwarf' && eos ? scalesOf(eos) : 0,
    part.type === 'planet' ? [part.material, part.coreMaterial, part.coreFraction, part.surfaceT, eos?.params.helmRho, eos?.params.helmE, planetTable(eos)?.table.key ?? null] : 0]);
  const hit = starCache.get(key);
  if (hit) return hit;
  let out: Omit<BuiltStar, 'id' | 'part' | 'center' | 'vel' | 'scalar'>;
  if (part.type === 'planet') out = buildPlanetStar(part, G, eos);
  else if (part.type === 'white-dwarf') {
    if (!eos) throw new Error('a white dwarf needs the equation of state');
    const s = scalesOf(eos);
    const wd = whiteDwarf(part.mass ?? 0.6, s, G);
    out = {
      M: wd.M, R: wd.R, rhoc: wd.rhoc, rho: (r) => wd.rho(r), pressure: (rho) => coldPressure(s, rho), barotrope: degenerateBarotrope(s),
      // A settled dwarf stays within a few cells of its analytic radius: 1.5 radii bounds the work.
      reach: 1.5 * wd.R,
      profile: () => ({ pressure: (r) => coldPressure(s, wd.rho(r)), pc: coldPressure(s, wd.rhoc), pcut: 0.01 * coldPressure(s, wd.rhoc), radius: wd.R }),
    };
  } else {
    const n = part.n ?? 1.5;
    const R = part.radius ?? 1;
    const g = 1 + 1 / n;
    if (part.core) {
      // An envelope around a point-mass core (problems/cored_polytrope.ts). Its gas is settled
      // on the CPU, which adds the core's potential (no `gpu` kind: problems/settle.ts).
      const M = part.mass ?? 1;
      if (part.rhoc !== undefined) throw new Error('a star with a core is given by its mass and radius, not its central density');
      if (!(part.core.mass > 0 && part.core.mass < M)) throw new Error('a star’s core must have a mass above zero and below the star’s');
      const soft = part.core.soft ?? R / 20;
      const env = coredPolytrope(n, M - part.core.mass, R, part.core.mass, soft, G);
      const { rhoc, K } = env;
      const { gpu: _gpu, ...barotrope } = polytropeBarotrope(n, K);
      out = {
        M, R, rhoc, core: { mass: part.core.mass, soft },
        rho: (r) => rhoc * Math.pow(env.theta(r / R), n),
        pressure: (rho) => K * Math.pow(rho, g),
        barotrope,
        profile: () => ({ pressure: (r) => (r >= R ? 0 : K * Math.pow(rhoc * Math.pow(env.theta(r / R), n), g)), pc: K * Math.pow(rhoc, g), pcut: 0, radius: R }),
      };
    } else if (part.mass !== undefined && part.rhoc === undefined) {
      // By mass and radius (the tidal disruption's star).
      const le = laneEmden(n);
      const { rhoc, K } = polytropeScales(le, part.mass, R, G);
      out = {
        M: part.mass, R, rhoc,
        rho: (r) => rhoc * Math.pow(le.theta(r / R), n),
        pressure: (rho) => K * Math.pow(rho, g),
        barotrope: polytropeBarotrope(n, K),
        profile: () => ({ pressure: (r) => (r >= R ? 0 : K * Math.pow(rhoc * Math.pow(Math.max(le.theta(r / R), 0), n), g)), pc: K * Math.pow(rhoc, g), pcut: 0, radius: R }),
      };
    } else {
      const rhoc = part.rhoc ?? 1;
      const st = polytropeByDensity(n, R, rhoc, G);
      const pc = st.pressure(rhoc);
      out = {
        M: st.M, R, rhoc, rho: st.density, pressure: st.pressure, barotrope: polytropeBarotrope(n, st.K),
        profile: () => ({ pressure: (r) => st.pressure(st.density(r)), pc, pcut: 0.01 * pc, radius: R }),
      };
    }
  }
  starCache.set(key, out);
  return out;
}

/** The field inside a star, as a vector potential about the star's center (mhd/star_fields.ts). */
export function starPotential(st: BuiltStar): ((x: number, y: number, z: number) => [number, number, number]) | null {
  const f = st.part.field;
  if (!f) return null;
  const seed = Math.round(f.seed ?? 1);
  const prof = st.profile();
  // Guillochon & McCourt's fields follow the pressure to the surface: no cut.
  const A =
    f.type === 'dipole' ? dipoleField(prof, f.beta)
    : f.type === 'random' ? randomField(prof, f.beta, seed)
    : f.type === 'gm17-dipole' ? gm17Dipole({ ...prof, pcut: 0 }, f.beta)
    : gm17Random({ ...prof, pcut: 0 }, f.beta, seed);
  const [cx, cy, cz] = st.center;
  return cx === 0 && cy === 0 && cz === 0 ? A : (x, y, z) => A(x - cx, y - cy, z - cz);
}

/**
 * The two masses of a binary at pericenter, on the x axis. With `incl` 0 they orbit
 * counter-clockwise about the origin. Otherwise the orbit is tilted about the x axis (the line of
 * nodes, through pericenter) by `incl` degrees: 180 is a retrograde orbit in the same plane.
 * In units of G M = 1 and semi-major axis 1.
 */
export function binaryMasses(q: number, ecc: number, soft: number, incl = 0): PointMass[] {
  const m1 = 1 / (1 + q);
  const m2 = q / (1 + q);
  const r = 1 - ecc;
  // Pericenter speed of the Kepler orbit, lowered to match the softened pull between the two
  // (so that a circular orbit stays circular).
  const v = Math.sqrt((1 + ecc) / (1 - ecc)) * Math.pow((r * r) / (r * r + soft * soft), 0.75);
  // Exact at the ends, so that a flat orbit has no vertical velocity at all.
  const a = (incl * Math.PI) / 180;
  const cy = incl === 0 ? 1 : incl === 180 ? -1 : Math.cos(a);
  const cz = incl === 0 || incl === 180 ? 0 : Math.sin(a);
  return [
    { gm: m1, pos: [-m2 * r, 0, 0], vel: [0, -m2 * v * cy, -m2 * v * cz], soft, sinkRadius: soft },
    { gm: m2, pos: [m1 * r, 0, 0], vel: [0, m1 * v * cy, m1 * v * cz], soft, sinkRadius: soft },
  ];
}

/**
 * A body of mass q M on a parabola about a star of mass M and radius R at the origin: its state
 * at separation `r` before pericenter, which is at r_t / beta on the +x axis (r_t = R q^(1/3)).
 */
export function parabolicFlyby(q: number, beta: number, r: number, M = 1, R = 1, G = 1) {
  const rt = R * Math.cbrt(q);
  const rp = rt / beta;
  const mu = G * M * (1 + q);
  const f = -Math.acos(Math.min(1, Math.max(-1, (2 * rp) / r - 1))); // true anomaly, negative on the way in
  const speed = Math.sqrt(mu / (2 * rp));
  // Time until pericenter, from Barker's equation.
  const D = Math.tan(-f / 2);
  const tPeri = Math.sqrt((2 * rp ** 3) / mu) * (D + D ** 3 / 3);
  return { rt, rp, tPeri, pos: [r * Math.cos(f), r * Math.sin(f), 0], vel: [-speed * Math.sin(f), speed * (1 + Math.cos(f)), 0] };
}

/** Where a flyby starts: `start` tidal radii out, and never inside 1.5 pericenters. */
export function flybyOrbit(q: number, beta: number, start: number, M = 1, R = 1, G = 1) {
  const rt = R * Math.cbrt(q);
  return parabolicFlyby(q, beta, Math.max(start * rt, (1.5 * rt) / beta), M, R, G);
}

/**
 * Two stars on a parabola about their center of mass, pericenter along +x at `b` sums of their
 * radii: positions and velocities at the start, `starts` sums of the radii apart (closer if
 * either would not fit in the box, never closer than touching or than pericenter).
 */
export function collisionOrbit(m1: number, m2: number, R1: number, R2: number, b: number, G: number, box: number, starts = 4) {
  const M = m1 + m2;
  const mu = G * M;
  const rp = b * (R1 + R2);
  const half = 0.45 * box;
  const fit = Math.min(((half - R1) * M) / m2, ((half - R2) * M) / m1);
  const d = Math.max(Math.min(starts * (R1 + R2), fit), Math.max(1.05 * (R1 + R2), 1.05 * rp));
  // Parabola r = 2 rp / (1 + cos f); on the way in, f < 0. Speed sqrt(2 mu / r), of which
  // h / r is across the line of centers, h = sqrt(2 mu rp).
  const f = -Math.acos(Math.min(1, Math.max(-1, (2 * rp) / d - 1)));
  const v = Math.sqrt((2 * mu) / d);
  const vt = Math.sqrt(2 * mu * rp) / d;
  const vr = -Math.sqrt(Math.max(v * v - vt * vt, 0));
  const [c, s] = [Math.cos(f), Math.sin(f)];
  // Star 2 relative to star 1.
  const rel = [d * c, d * s, 0];
  const vrel = [vr * c - vt * s, vr * s + vt * c, 0];
  return {
    d,
    rp,
    pos: [rel.map((x) => (-m2 / M) * x), rel.map((x) => (m1 / M) * x)],
    vel: [vrel.map((x) => (-m2 / M) * x), vrel.map((x) => (m1 / M) * x)],
  };
}

export interface Bodies {
  /** the setup's parts and the id of each */
  ids: string[];
  stars: BuiltStar[];
  /** the point masses, in the order of their parts, and each one's part id */
  masses: PointMass[];
  massIds: string[];
  /** the debris diagnostic a flyby asks for */
  debris?: { mass: number; de: number; tPeri: number; refMass: number; scalar: boolean };
  /** semi-major axis of the orbit joining two masses, by "idA+idB" */
  separations: Map<string, number>;
  /**
   * A circular binary: the frame turning with it about z through the box center (where its
   * center of mass is), and for each of its stars how far its L1 point is from its center, as
   * far as the star's settling may reach. Its stars are settled in the Roche potential.
   */
  binary?: BinaryFrame;
}

export interface BinaryFrame {
  /** angular velocity of the orbit about z */
  omega: number;
  /** the run is in the frame turning with it: the bodies start at rest */
  corotating: boolean;
  separation: number;
  /** each star's distance to L1, by its id */
  l1: Map<string, number>;
  /** the frame follows the orbit (OrbitPart.track) */
  track?: boolean;
}

export interface BodyContext {
  G: number;
  /** cell size the masses sit in, and the grid's dimensions */
  dx: number;
  ndim: 2 | 3;
  box: number;
  eos?: Eos;
}

const vec = (v: Vec3 | undefined): number[] => (v ? [...v] : [0, 0, 0]);

/** Build a setup's stars and point masses and place them by its orbits. */
export function buildBodies(spec: SetupSpec, ctx: BodyContext): Bodies {
  const parts = spec.parts ?? [];
  const ids = partIds(parts);
  const starParts = parts.flatMap((p, i) => (p.kind === 'star' ? [{ p, id: ids[i] }] : []));
  const stars: BuiltStar[] = starParts.map(({ p, id }, i) => ({
    ...buildStar(p, ctx.G, ctx.eos), id, part: p, center: vec(p.pos), vel: vec(p.vel),
    // The passive scalar tells the stars apart: spread over (0, 1].
    scalar: p.dye ?? 1 - i / starParts.length,
  }));
  const massParts = parts.flatMap((p, i) => (p.kind === 'mass' ? [{ p, id: ids[i] }] : []));
  const masses: PointMass[] = massParts.map(({ p }) => massOf(p, ctx.dx));
  const massIds = massParts.map((m) => m.id);
  const out: Bodies = { ids, stars, masses, massIds, separations: new Map() };

  const starOf = (id: string) => stars.find((s) => s.id === id);
  const massAt = (id: string) => massIds.indexOf(id);
  for (const o of parts) {
    if (o.kind !== 'orbit') continue;
    const [a, b] = o.of;
    if (o.type === 'binary') {
      if (starOf(a) || starOf(b)) placeStarBinary(o, out, ctx);
      else placeBinary(o, out, massAt(a), massAt(b), ctx);
    }
    else if (o.type === 'flyby') {
      const star = starOf(a);
      const i = massAt(b);
      if (!star || i < 0) throw new Error('a flyby orbit is of a star and a point mass');
      const q = o.q ?? masses[i].gm / (ctx.G * star.M);
      // Units made for G = 1 give it to within rounding: the orbit is then exactly that of G = 1.
      const G = Math.abs(ctx.G - 1) < 1e-12 ? 1 : ctx.G;
      const orb = flybyOrbit(q, o.beta ?? 1, o.start ?? 3, star.M, star.R, G);
      masses[i] = { ...masses[i], gm: ctx.G * star.M * q, pos: orb.pos, vel: orb.vel };
      if (o.debris) out.debris = { mass: i, de: (ctx.G * star.M * q * star.R) / (orb.rt * orb.rt), tPeri: orb.tPeri, refMass: star.M, scalar: true };
    } else {
      const s1 = starOf(a);
      const s2 = starOf(b);
      if (!s1 || !s2) throw new Error('a collision orbit is of two stars');
      const orb = collisionOrbit(s1.part.mass ?? s1.M, s2.part.mass ?? s2.M, s1.R, s2.R, o.impact ?? 0, ctx.G, ctx.box, o.starts ?? 4);
      s1.center = orb.pos[0];
      s1.vel = orb.vel[0];
      s2.center = orb.pos[1];
      s2.vel = orb.vel[1];
    }
  }
  // A star's core: a point mass where the star is and moving as it does, once the orbits have
  // placed the stars. It takes no gas in. After the parts' own masses, as "<star>.core".
  for (const s of stars) {
    if (!s.core) continue;
    masses.push({ gm: ctx.G * s.core.mass, pos: [...s.center], vel: [...s.vel], soft: s.core.soft, sinkRadius: 0 });
    massIds.push(`${s.id}.core`);
  }
  return out;
}


/** A point mass part as the solver takes it: softening and sink at least two cells across. */
function massOf(p: MassPart, dx: number): PointMass {
  const m: PointMass = { gm: p.gm, pos: vec(p.pos), vel: vec(p.vel) };
  if (p.soft !== undefined) m.soft = Math.max(p.soft, 2 * dx);
  if (p.sink !== undefined) m.sinkRadius = p.sink > 0 ? Math.max(p.sink, 2 * dx) : 0;
  if (p.sinkRate !== undefined) m.sinkRate = p.sinkRate;
  if (p.massless) m.massless = true;
  if (p.fixed) m.fixed = true;
  if (p.onOrbit) m.onOrbit = true;
  return m;
}

function placeBinary(o: OrbitPart, out: Bodies, i: number, j: number, ctx: BodyContext) {
  if (i < 0 || j < 0) throw new Error('a binary orbit is of two point masses');
  const [a, b] = [out.masses[i], out.masses[j]];
  const total = a.gm + b.gm;
  const q = o.q ?? b.gm / a.gm;
  const sep = o.separation ?? 1;
  const soft = a.soft ?? 2 * ctx.dx;
  const incl = ctx.ndim === 3 ? (o.incl ?? 0) : o.retrograde ? 180 : 0;
  // The unit orbit (G M = 1, a = 1), scaled to this one; nothing is scaled when it is the unit orbit.
  const unit = binaryMasses(q, o.ecc ?? 0, sep === 1 ? soft : soft / sep, incl);
  const vs = total === 1 && sep === 1 ? 1 : Math.sqrt(total / sep);
  const place = (m: PointMass, u: PointMass): PointMass => ({
    ...m,
    gm: total === 1 ? u.gm : total * u.gm,
    pos: sep === 1 ? u.pos : u.pos.map((x) => sep * x),
    vel: vs === 1 ? u.vel : u.vel.map((x) => vs * x),
    soft,
    // The sink is the softening's unless the part says otherwise.
    sinkRadius: m.sinkRadius ?? soft,
  });
  out.masses[i] = place(a, unit[0]);
  out.masses[j] = place(b, unit[1]);
  out.separations.set(`${o.of[0]}+${o.of[1]}`, sep);
  if (o.corotating && !(o.ecc ?? 0)) {
    // In the frame turning with the orbit the two masses start at rest.
    out.masses[i] = { ...out.masses[i], vel: [0, 0, 0] };
    out.masses[j] = { ...out.masses[j], vel: [0, 0, 0] };
    out.binary = { omega: binaryOmega(total, sep, soft), corotating: true, separation: sep, l1: new Map(), ...(o.track ? { track: true } : {}) };
  }
}

/**
 * A binary with a star in it: a star and a point mass, or two stars, about their center of mass
 * at the box center, at pericenter on the x axis, the first body at -x. Their masses are their
 * parts' (`q` sets a point mass's against the star's). `fill` puts the first body, a star,
 * against its Roche lobe. A circular one is tidally locked: its stars turn with the orbit, and
 * are settled in the Roche potential (BinaryFrame). An eccentric one is placed as two point
 * masses are, its stars each settled in its own potential.
 */
function placeStarBinary(o: OrbitPart, out: Bodies, ctx: BodyContext) {
  const body = (id: string) => {
    const star = out.stars.find((s) => s.id === id);
    const i = out.massIds.indexOf(id);
    if (!star && i < 0) throw new Error(`a binary orbit is of point masses and stars, and "${id}" is neither`);
    return { star, i };
  };
  const [A, B] = o.of.map(body);
  // Masses in code units (G M for a point mass over G).
  const G = ctx.G;
  if (o.q !== undefined && !(A.star && B.star)) {
    if (A.star) out.masses[B.i] = { ...out.masses[B.i], gm: G * A.star.M * o.q };
    else out.masses[A.i] = { ...out.masses[A.i], gm: (G * B.star!.M) / o.q };
  }
  const mOf = (b: typeof A) => (b.star ? b.star.M : out.masses[b.i].gm / G);
  const m1 = mOf(A);
  const m2 = mOf(B);
  const ecc = o.ecc ?? 0;
  let sep = o.separation ?? 1;
  if (o.fill !== undefined) {
    if (!A.star) throw new Error('a binary’s fill is of its first body, which must be a star');
    sep = fillSeparation(A.star.R, m1, m2, o.fill);
  }
  // A point mass's softening lowers the orbit's speed as the pull between the two is softened.
  const soft = [A, B].reduce((s, b) => (b.star ? s : Math.max(s, out.masses[b.i].soft ?? 2 * ctx.dx)), 0);
  const unit = binaryMasses(m2 / m1, ecc, soft / sep, 0);
  const vs = Math.sqrt((G * (m1 + m2)) / sep);
  const circular = ecc === 0;
  const corotating = circular && !!o.corotating;
  [A, B].forEach((b, k) => {
    const pos = unit[k].pos.map((x) => sep * x);
    const vel = corotating ? [0, 0, 0] : unit[k].vel.map((v) => vs * v);
    if (b.star) {
      b.star.center = pos;
      b.star.vel = vel;
    } else out.masses[b.i] = { ...out.masses[b.i], pos, vel };
  });
  out.separations.set(`${o.of[0]}+${o.of[1]}`, sep);
  if (!circular) return;
  const omega = binaryOmega(G * (m1 + m2), sep, soft);
  const d1 = l1Distance(m1, m2) * sep;
  const l1 = new Map<string, number>();
  if (A.star) l1.set(A.star.id, d1);
  if (B.star) l1.set(B.star.id, sep - d1);
  for (const b of [A, B]) if (b.star && !corotating) b.star.spin = omega;
  out.binary = { omega, corotating, separation: sep, l1, ...(corotating && o.track ? { track: true } : {}) };
}

/** A part's quantity by name, for units that follow one: "<id>.mass" or "<id>.radius". */
export function partQuantity(bodies: Bodies, G: number, name: string): number {
  const [id, what] = name.split('.');
  const star = bodies.stars.find((s) => s.id === id);
  const i = bodies.massIds.indexOf(id);
  if (star && what === 'mass') return star.M;
  if (star && what === 'radius') return star.R;
  if (i >= 0 && what === 'mass') return bodies.masses[i].gm / G;
  throw new Error(`units: no such quantity: ${name}`);
}
