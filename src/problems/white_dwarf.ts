// Cold white dwarfs: hydrostatic equilibrium held up by fully degenerate electrons alone
// (Chandrasekhar 1939), for the equation of state of eos/degenerate.wgsl at zero temperature.
//
//   dm/dr = 4 pi r^2 rho,   drho/dr = -G m rho / (r^2 c^2),   c^2 = dP_deg/drho
//
// integrated outward from the center (RK4), with the central density found by bisection so
// that the star has the mass asked for. Everything is in code units, given the equation of
// state's two scales: rho_d (where p_F = m_e c, which carries Ye) and the pressure scale A.
import { degFG, type Eos } from '../eos/eos';
import { ASTRO } from '../units/constants';
import { gravityUnits } from '../units/units';
import type { Barotrope, Settling } from './settle';
import type { Prim } from './types';

export interface WhiteDwarf {
  M: number;
  /** radius: where the density reaches 1e-12 of the central density */
  R: number;
  rhoc: number;
  /** density at radius r (0 outside) */
  rho(r: number): number;
}

export interface DegenerateScales {
  degRho: number;
  degP: number;
}

/** Zero-temperature degenerate pressure at density rho. */
export const coldPressure = (s: DegenerateScales, rho: number): number => s.degP * degFG(Math.cbrt(rho / s.degRho))[0];

/** rho / c^2 of the cold gas, written so that it stays finite (and goes to zero) as rho does. */
function rhoOverC2(s: DegenerateScales, rho: number): number {
  if (!(rho > 0)) return 0;
  const x = Math.cbrt(rho / s.degRho);
  // c^2 = A 8 x^5 / (3 rho sqrt(1 + x^2)), with x^3 = rho / rho_d.
  return (3 * s.degRho * x * Math.sqrt(1 + x * x)) / (8 * s.degP);
}

interface Profile {
  r: Float64Array;
  rho: Float64Array;
  M: number;
  R: number;
}

/** Integrate a star of central density rhoc outward to its surface. */
function integrate(s: DegenerateScales, rhoc: number, G: number, steps = 4000): Profile {
  // Step size from the star's natural length sqrt(P_c / (G rho_c^2)): the radius is 1.6 of
  // these for a non-relativistic white dwarf (n = 3/2) and 3.9 near the Chandrasekhar mass (n = 3).
  const L = Math.sqrt(coldPressure(s, rhoc) / (G * rhoc * rhoc));
  const h = (8 * L) / steps;
  const f = (r: number, m: number, rho: number): [number, number] => {
    const d = Math.max(rho, 0);
    return [4 * Math.PI * r * r * d, r > 0 ? (-G * m * rhoOverC2(s, d)) / (r * r) : 0];
  };
  // Off the center with the series solution: m = 4/3 pi rhoc r^3, rho = rhoc - (2 pi / 3) G rhoc (rhoc / c^2) r^2.
  let r = 1e-3 * h;
  let m = ((4 * Math.PI) / 3) * rhoc * r ** 3;
  let rho = rhoc - ((2 * Math.PI) / 3) * G * rhoc * rhoOverC2(s, rhoc) * r * r;
  const rs = [0, r];
  const ds = [rhoc, rho];
  const stop = 1e-12 * rhoc;
  for (let i = 0; i < 20 * steps && rho > stop; i++) {
    const [a1, b1] = f(r, m, rho);
    const [a2, b2] = f(r + h / 2, m + (h / 2) * a1, rho + (h / 2) * b1);
    const [a3, b3] = f(r + h / 2, m + (h / 2) * a2, rho + (h / 2) * b2);
    const [a4, b4] = f(r + h, m + h * a3, rho + h * b3);
    const next = rho + (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    if (next <= stop) {
      // The surface lies within this step: end the profile there.
      const w = (rho - stop) / (rho - next);
      m += w * (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
      r += w * h;
      rho = stop;
      rs.push(r);
      ds.push(0);
      break;
    }
    m += (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
    rho = next;
    r += h;
    rs.push(r);
    ds.push(rho);
  }
  return { r: Float64Array.from(rs), rho: Float64Array.from(ds), M: m, R: r };
}

const cache = new Map<string, WhiteDwarf>();

/**
 * The cold white dwarf of mass M (below the Chandrasekhar mass of these scales). Cached:
 * every cell of the grid asks for it.
 */
export function whiteDwarf(M: number, s: DegenerateScales, G = 1): WhiteDwarf {
  const key = `${M}:${s.degRho}:${s.degP}:${G}`;
  const hit = cache.get(key);
  if (hit) return hit;
  // Mass grows with central density up to the Chandrasekhar limit: bracket, then bisect in log rho_c.
  let lo = Math.log(1e-6 * s.degRho);
  let hi = Math.log(1e6 * s.degRho);
  if (integrate(s, Math.exp(hi), G).M < M) throw new Error(`white dwarf of mass ${M} is above the Chandrasekhar mass`);
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (integrate(s, Math.exp(mid), G).M < M) lo = mid;
    else hi = mid;
  }
  const p = integrate(s, Math.exp(0.5 * (lo + hi)), G);
  const { r, rho } = p;
  const wd: WhiteDwarf = {
    M: p.M,
    R: p.R,
    rhoc: rho[0],
    rho(x: number) {
      if (x >= p.R) return 0;
      // Uniform steps after the first point: index directly, then interpolate.
      let i = x <= r[1] ? 0 : Math.min(r.length - 2, 1 + Math.floor((x - r[1]) / (r[2] - r[1])));
      while (i > 0 && r[i] > x) i--;
      while (i < r.length - 2 && r[i + 1] < x) i++;
      const t = (x - r[i]) / (r[i + 1] - r[i]);
      return Math.max(0, rho[i] + t * (rho[i + 1] - rho[i]));
    },
  };
  cache.set(key, wd);
  return wd;
}

/**
 * The cold degenerate gas as a barotrope, for settling (problems/settle.ts). Its enthalpy is
 * h = int dP / rho = (8 A / rho_d) (sqrt(1 + x^2) - 1), so x = sqrt(u (2 + u)) with
 * u = h rho_d / (8 A), written so that small h loses nothing.
 */
export function degenerateBarotrope(s: DegenerateScales): Barotrope {
  return {
    pressure: (rho) => coldPressure(s, rho),
    rhoOfH: (h) => {
      if (!(h > 0)) return 0;
      const u = (h * s.degRho) / (8 * s.degP);
      return s.degRho * Math.sqrt(u * (2 + u)) ** 3;
    },
  };
}

/** Code units for white dwarfs: 10^4 km, one solar mass, and the time that makes G = 1 (2.75 s). */
export const WD_UNITS = gravityUnits('white dwarf', 1e7, ASTRO.Msun);

/** Density below which gas is ambient around white dwarfs, and the ambient density (code units). */
export const WD_AMBIENT = { rhoCut: 1e-4, rhoAmb: 1e-6 };

/** A white dwarf placed in the box: its structure, center, velocity and the dye its gas carries. */
export interface PlacedDwarf {
  wd: WhiteDwarf;
  center: number[];
  vel: number[];
  scalar: number;
}

/** The initial state of white dwarfs in light ambient gas at rest, in pressure balance with their outermost layers. */
export function dwarfsInit(s: DegenerateScales, dwarfs: PlacedDwarf[]): (x: number, y: number, z: number) => Prim {
  const { rhoCut, rhoAmb } = WD_AMBIENT;
  const pAmb = coldPressure(s, rhoCut);
  return (x, y, z) => {
    for (const d of dwarfs) {
      const rho = d.wd.rho(Math.hypot(x - d.center[0], y - d.center[1], z - d.center[2]));
      if (rho > rhoCut) {
        // Cold: the degenerate pressure alone.
        const [vx, vy, vz] = d.vel;
        return { rho, vx, vy, vz, p: coldPressure(s, rho), s: d.scalar };
      }
    }
    return { rho: rhoAmb, vx: 0, vy: 0, vz: 0, p: pAmb, s: 0 };
  };
}

/** The same white dwarfs, to be settled into the grid's own balance (problems/settle.ts). */
export function dwarfsSettling(s: DegenerateScales, dwarfs: PlacedDwarf[], G: number): Settling {
  const barotrope = degenerateBarotrope(s);
  return {
    // A settled dwarf stays within a few cells of its analytic radius: 1.5 radii bounds the work.
    stars: dwarfs.map((d) => ({ barotrope, center: d.center, vel: d.vel, R: d.wd.R, reach: 1.5 * d.wd.R, scalar: d.scalar })),
    rhoCut: WD_AMBIENT.rhoCut,
    rhoAmb: WD_AMBIENT.rhoAmb,
    pAmb: coldPressure(s, WD_AMBIENT.rhoCut),
    G,
    // A 0.6 M_sun dwarf settled to 1e-3 (12 rounds) stays as still as one settled to 1e-5 (26):
    // its central density swings by about 1% either way, from what the settling cannot reach.
    tol: 1e-3,
    relax: 1.6,
  };
}

/** The degenerate scales of an equation of state's parameters. */
export const scalesOf = (eos: Eos): DegenerateScales => ({ degRho: eos.params.degRho, degP: eos.params.degP });
