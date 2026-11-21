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

export const BUILTIN_BODIES: Record<string, () => Mesh> = {
  ahmed: () => ahmedBody(25),
  'ahmed-35': () => ahmedBody(35),
};

/** The body a "builtin:<name>" file names, or undefined. */
export function builtinBody(file: string): Mesh | undefined {
  const m = file.match(/^builtin:(.+)$/);
  return m ? BUILTIN_BODIES[m[1]]?.() : undefined;
}
