// Bodies made here rather than read from a file (a `mesh` shape's `file` of "builtin:<name>"):
// public by construction, nothing to ship or fetch.
import { wholePart, type Mesh } from './mesh';

/**
 * A bluff car-like body after the Ahmed body's nominal proportions (Ahmed, Ramm & Faltin 1984,
 * SAE 840300): length 1.044, width 0.389, height 0.288 (meters), the front's four edges rounded
 * with radius 0.1, the roof's rear `slant` degrees down over its last 0.222 (along the slant).
 * No stilts. Built from rectangular sections along x (the front at x = 0, the floor at z = 0),
 * joined into a closed surface wound outward.
 */
export function ahmedBody(slant = 25, sections = 16): Mesh {
  const L = 1.044, W = 0.389, H = 0.288, R = 0.1, Ls = 0.222;
  const t = (slant * Math.PI) / 180;
  const xs = Ls * Math.cos(t);
  const stations: number[] = [];
  for (let i = 0; i <= sections; i++) stations.push(R * (1 - Math.cos((i / sections) * (Math.PI / 2))));
  stations.push(L - xs, L);
  /** half-width, top and bottom of the section at x */
  const section = (x: number): [number, number, number] => {
    const inset = x < R ? R - Math.sqrt(Math.max(0, R * R - (R - x) ** 2)) : 0;
    const drop = x > L - xs ? Math.tan(t) * (x - (L - xs)) : 0;
    return [W / 2 - inset, H - inset - drop, inset];
  };
  const pos: number[] = [];
  const idx: number[] = [];
  // Each section: four corners, counterclockwise seen from +x: (-y, bottom), (+y, bottom), (+y, top), (-y, top).
  for (const x of stations) {
    const [w, top, bot] = section(x);
    pos.push(x, -w, bot, x, w, bot, x, w, top, x, -w, top);
  }
  const n = stations.length;
  for (let s = 0; s + 1 < n; s++) {
    const a = 4 * s, b = 4 * (s + 1);
    for (let k = 0; k < 4; k++) {
      const k2 = (k + 1) % 4;
      // The side between corners k and k2 of sections s and s + 1, wound outward.
      idx.push(a + k, b + k2, b + k, a + k, a + k2, b + k2);
    }
  }
  // The front cap (facing -x) and the rear (facing +x).
  idx.push(0, 3, 2, 0, 2, 1);
  const r = 4 * (n - 1);
  idx.push(r, r + 1, r + 2, r, r + 2, r + 3);
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: [wholePart(idx.length / 3, 'ahmed')], meters: 1 };
}

/** A sphere of diameter 1: an icosahedron subdivided `levels` times, its vertices on the sphere. */
export function icosphere(levels = 4): Mesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let v: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = (p: number[]) => {
    const l = Math.hypot(p[0], p[1], p[2]);
    return p.map((x) => (0.5 * x) / l);
  };
  v = v.map(norm);
  for (let l = 0; l < levels; l++) {
    const mid = new Map<string, number>();
    const m = (a: number, b: number) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      if (!mid.has(k)) {
        v.push(norm(v[a].map((x, i) => (x + v[b][i]) / 2)));
        mid.set(k, v.length - 1);
      }
      return mid.get(k)!;
    };
    f = f.flatMap(([a, b, c]) => {
      const [ab, bc, ca] = [m(a, b), m(b, c), m(c, a)];
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }
  return { positions: new Float32Array(v.flat()), indices: new Uint32Array(f.flat()), parts: [wholePart(f.length, 'sphere')] };
}

/** Options of the built-in wind turbine rotor (turbineRotor). */
export interface TurbineOptions {
  /** tip-speed ratio the blades are pitched for, Omega R / U */
  tsr: number;
  /** angle of attack each section meets the relative wind at, at that ratio (degrees) */
  alpha: number;
  /** hub (spinner) radius and blade root, and the chord at the root and the tip, in R */
  hub: number;
  rootChord: number;
  tipChord: number;
  /** thickness over chord of the section */
  thickness: number;
  /** the thinnest the section may be anywhere, in R (a blunt trailing edge where it would be less) */
  minThickness: number;
  blades: number;
  /** with the spinner (false: the blades alone, to measure a blade's thinnest part) */
  spinner?: boolean;
}

export const TURBINE_DEFAULTS: TurbineOptions = { tsr: 1, alpha: 10, hub: 0.2, rootChord: 0.4, tipChord: 0.25, thickness: 0.2, minThickness: 0.06, blades: 3 };
/** A fast rotor: builtin:turbine-fast. */
export const TURBINE_FAST: TurbineOptions = { tsr: 4, alpha: 6, hub: 0.2, rootChord: 0.3, tipChord: 0.15, thickness: 0.2, minThickness: 0.03, blades: 3 };

/**
 * A wind turbine's rotor of radius 1 turning about +x (right-handed) in a wind along +x: a
 * spherical spinner and `blades` blades. Each blade's chord tapers linearly from root to tip;
 * its section is the NACA four-digit symmetric thickness (thickened to `minThickness` where it
 * would be thinner: a blunt trailing edge), set at `alpha` to the relative wind the section sees
 * at the design tip-speed ratio with the Betz induction (a third of the wind's speed taken by the
 * rotor): the inflow angle phi = atan((2/3) / (tsr r)), the twist following it from root to tip.
 * The leading edge leads in the direction of turning, and lift drives the rotor: the wind turns
 * it the way the frame turns (a turbine, not a fan).
 */
export function turbineRotor(o: TurbineOptions = TURBINE_DEFAULTS): Mesh {
  const pos: number[] = [];
  const idx: number[] = [];
  const add = (p: number[]) => (pos.push(p[0], p[1], p[2]), pos.length / 3 - 1);
  const nSpan = 56;
  const nHalf = 24; // points along each side of the section, leading edge to trailing edge
  // The blade starts well inside the spinner, narrowing there and keeping the spinner's pitch,
  // so that its root is buried (an exposed flat root cap, where the twist is fastest, was the
  // thinnest part: a ray along the span left it within a cell).
  const root = o.hub * 0.4;
  const al = (o.alpha * Math.PI) / 180;
  // NACA 00xx half-thickness at chord fraction x, for a chord of 1 and thickness ratio t.
  const yt = (x: number, t: number) => 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  for (let b = 0; b < o.blades; b++) {
    const az = (2 * Math.PI * b) / o.blades;
    const er = [0, Math.cos(az), Math.sin(az)];
    // The direction of turning there: x cross e_r.
    const et = [0, -er[2], er[1]];
    const first = pos.length / 3;
    for (let s = 0; s <= nSpan; s++) {
      const r = root + ((1 - root) * s) / nSpan;
      const frac = Math.max(0, (r - o.hub) / (1 - o.hub));
      const inside = Math.min(1, (r - root) / (o.hub - root));
      const c = (o.rootChord + (o.tipChord - o.rootChord) * frac) * (0.4 + 0.6 * inside);
      const rp = Math.max(r, o.hub);
      // The relative wind (axial (2/3) U, tangential -Omega r, with U = 1 and Omega = tsr) and the lift's direction.
      const wa = 2 / 3;
      const wt = -o.tsr * rp;
      const wl = Math.hypot(wa, wt);
      const w = [wa / wl, wt / wl]; // in (x, e_t)
      const n = [-wt / wl, wa / wl];
      // Chord from leading to trailing edge, and the section's normal, in (x, e_t).
      const d = [Math.cos(al) * w[0] - Math.sin(al) * n[0], Math.cos(al) * w[1] - Math.sin(al) * n[1]];
      const m = [Math.sin(al) * w[0] + Math.cos(al) * n[0], Math.sin(al) * w[1] + Math.cos(al) * n[1]];
      const at = (xi: number, eta: number) => {
        const u = (xi - 0.3) * c; // about the chord's 30% point
        const a = u * d[0] + eta * m[0];
        const t2 = u * d[1] + eta * m[1];
        return [a, r * er[1] + t2 * et[1], r * er[2] + t2 * et[2]];
      };
      // Around the section: upper side from the trailing edge to the leading, lower side back.
      for (let k = 0; k < 2 * nHalf; k++) {
        const upper = k < nHalf;
        const j = upper ? nHalf - 1 - k : k - nHalf;
        const xi = 0.5 * (1 - Math.cos((Math.PI * (j + 0.5)) / nHalf));
        const h = Math.max(yt(xi, o.thickness) * c, o.minThickness / 2);
        add(at(xi, upper ? h : -h));
      }
    }
    const ring = 2 * nHalf;
    for (let s = 0; s < nSpan; s++)
      for (let k = 0; k < ring; k++) {
        const a = first + s * ring + k, b2 = first + s * ring + ((k + 1) % ring);
        const a2 = a + ring, b3 = b2 + ring;
        idx.push(a, b2, b3, a, b3, a2);
      }
    // Caps: a fan about each end section's middle.
    for (const s of [0, nSpan]) {
      let cx = 0, cy = 0, cz = 0;
      for (let k = 0; k < ring; k++) {
        const v = first + s * ring + k;
        cx += pos[3 * v] / ring;
        cy += pos[3 * v + 1] / ring;
        cz += pos[3 * v + 2] / ring;
      }
      const cv = add([cx, cy, cz]);
      for (let k = 0; k < ring; k++) idx.push(cv, first + s * ring + ((k + 1) % ring), first + s * ring + k);
    }
  }
  if (o.spinner === false) return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: [wholePart(idx.length / 3, 'blades')], meters: 1 };
  // The spinner.
  const sph = icosphere(3);
  const base = pos.length / 3;
  for (let i = 0; i < sph.positions.length; i += 3) add([sph.positions[i] * 2 * o.hub, sph.positions[i + 1] * 2 * o.hub, sph.positions[i + 2] * 2 * o.hub]);
  for (const i of sph.indices) idx.push(base + i);
  // (The winding is made consistent and outward when the distance field is made: mesh.ts orient.)
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), parts: [wholePart(idx.length / 3, 'rotor')], meters: 1 };
}

export const BUILTIN_BODIES: Record<string, () => Mesh> = {
  // A three-bladed wind turbine rotor of radius 1 about +x, pitched for a tip-speed ratio of 1
  // (wide blades, steeply pitched: a grid of 2 cells through a blade still shows the pitch).
  turbine: () => turbineRotor(),
  // The same, pitched for a tip-speed ratio of 4 with slimmer blades (chord 0.3 R at the root to
  // 0.15 R at the tip, 6 degrees to the relative wind, 3 to 10 degrees from the rotor's plane
  // outboard): a fast rotor, which needs its blades resolved (plans/wind-tunnel.md 5.3).
  'turbine-fast': () => turbineRotor(TURBINE_FAST),
  ahmed: () => ahmedBody(25),
  'ahmed-35': () => ahmedBody(35),
  // A sphere of diameter 1 (5120 triangles): the mesh path against the built-in sphere shape.
  sphere: () => icosphere(4),
};

/** The body a "builtin:<name>" file names, or undefined. */
export function builtinBody(file: string): Mesh | undefined {
  const m = file.match(/^builtin:(.+)$/);
  return m ? BUILTIN_BODIES[m[1]]?.() : undefined;
}
