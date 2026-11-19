// Planets: layered bodies of ANEOS materials (eos/aneos.ts) in hydrostatic balance, for giant
// impacts. A planet is given by its mass, its layers from the center out (a material and its
// share of the mass each) and the temperature of its surface. Each layer lies along its own
// adiabat, as a convecting mantle or core does and as the published impact calculations start
// theirs (Kegerreis et al. 2019; Ruiz-Bonilla et al. 2021): the adiabat of the tables as the
// hydro reads them, de = p / rho^2 drho, so a layer displaced whole stays on it. Where two
// layers meet, the pressure and the temperature are continuous and the density jumps.
//
// The structure is integrated inward from the surface, where the pressure is 1 GPa, for a
// trial radius; the radius is the one at which the mass runs out exactly at the center.
import { ASTRO, SI } from '../units/constants';
import { gravityUnits } from '../units/units';
import { ANEOS_RHO_MAX, aneosFromE, aneosFromT, type AneosTable } from '../eos/aneos';

/** Earth's mass (kg) and radius (m), the units a planetary setup is written in. */
export const EARTH = { mass: ASTRO.Mearth, radius: ASTRO.Rearth };
/** Code units of the planetary presets: Earth radii and Earth masses, with G = 1 (805.5 s, 7.91 km/s, 23.1 g/cm^3). */
export const PLANET_UNITS = gravityUnits('Earth', EARTH.radius, EARTH.mass);

/** A layer: which of the table's materials (its mass fraction of the second: 0 or 1), and its share of the planet's mass. */
export interface PlanetLayer {
  xb: number;
  fraction: number;
}

/** A planet's structure, cgs, from the center out. */
export interface Planet {
  /** radius (cm), mass (g) and central density (g/cm^3) */
  R: number;
  M: number;
  rhoc: number;
  /** samples from the center out: radius, density, pressure, temperature, and which layer */
  r: Float64Array;
  rho: Float64Array;
  p: Float64Array;
  T: Float64Array;
  layer: Int32Array;
  /** density, pressure and layer at a radius (zero density outside) */
  at(r: number): { rho: number; p: number; T: number; layer: number };
}

const G = SI.G * 1e3; // cm^3 / (g s^2)
/**
 * Pressure at which the integration starts, dyn/cm^2: 1 GPa, some 30 km down in the Earth. The
 * table's cell that holds a material's edge (from its vapor dome into the condensed phase) is
 * soft, the pressure rising from nothing across it, and a structure started there would be
 * integrating the table's interpolation.
 */
const P_SURFACE = 1e10;

/** The density at which a material has pressure p at temperature T: its condensed branch, by bisection. */
export function densityAt(t: AneosTable, xb: number, p: number, T: number, lo = 0.3, hi = 60): number {
  for (let k = 0; k < 70; k++) {
    const mid = Math.sqrt(lo * hi);
    if (aneosFromT(t, mid, T, xb).p > p) hi = mid;
    else lo = mid;
  }
  return Math.sqrt(lo * hi);
}

interface Trial {
  /** mass left at the center (positive: the radius is too small), or minus the radius at which it ran out */
  left: number;
  r: number[];
  rho: number[];
  p: number[];
  T: number[];
  layer: number[];
}

function integrate(t: AneosTable, layers: readonly PlanetLayer[], M: number, Ts: number, R: number, steps: number): Trial {
  const out: Trial = { left: 0, r: [], rho: [], p: [], T: [], layer: [] };
  // Mass inside each layer's outer edge, from the outermost in.
  const edges: number[] = [];
  let acc = 0;
  for (const l of layers) edges.push((acc += l.fraction * M));
  let li = layers.length - 1;
  let xb = layers[li].xb;
  let rho = densityAt(t, xb, P_SURFACE, Ts);
  let e = aneosFromT(t, rho, Ts, xb).e;
  let m = M;
  let r = R;
  const h = R / steps;
  // d(m, rho, e)/dr along the adiabat: drho/dr = (dp/dr) / (dp/drho)_s, de/dr = (p / rho^2) drho/dr.
  // The pressure's slope along the adiabat is taken from the table by differences along it, not
  // from the equation of state's sound speed: that is the thermodynamic one, which differs from
  // the slope of the table's own adiabat by a few percent where ANEOS is not consistent with a
  // free energy, and a structure integrated with it is not in balance with the pressures the
  // hydro reads (a Mars-mass body's center came out a quarter too slack).
  const rhs = (r_: number, m_: number, rho_: number, e_: number): [number, number, number] => {
    const s = aneosFromE(t, rho_, e_, xb);
    const h = 1e-4;
    const de = (s.p / rho_) * h;
    const slope = (aneosFromE(t, rho_ * (1 + h), e_ + de, xb).p - aneosFromE(t, rho_ * (1 - h), e_ - de, xb).p) / (2 * h * rho_);
    const dr = (-G * Math.max(m_, 0) * rho_) / (r_ * r_ * Math.max(slope, 1e8));
    return [4 * Math.PI * r_ * r_ * rho_, dr, (s.p / (rho_ * rho_)) * dr];
  };
  const push = () => {
    const s = aneosFromE(t, rho, e, xb);
    out.r.push(r);
    out.rho.push(rho);
    out.p.push(s.p);
    out.T.push(s.T);
    out.layer.push(li);
  };
  push();
  for (let k = 0; k < steps - 1; k++) {
    // RK4, inward.
    const k1 = rhs(r, m, rho, e);
    const k2 = rhs(r - h / 2, m - (h / 2) * k1[0], rho - (h / 2) * k1[1], e - (h / 2) * k1[2]);
    const k3 = rhs(r - h / 2, m - (h / 2) * k2[0], rho - (h / 2) * k2[1], e - (h / 2) * k2[2]);
    const k4 = rhs(r - h, m - h * k3[0], rho - h * k3[1], e - h * k3[2]);
    m -= (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    rho -= (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    e -= (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    r -= h;
    // Pressed past anything the tables hold: the radius is far too small, and the mass left at
    // this depth is being crushed by its own weight.
    if (!(rho < 0.5 * ANEOS_RHO_MAX)) {
      out.left = Math.max(m, 1);
      return out;
    }
    if (!(m > 0)) {
      out.left = -r;
      return out;
    }
    // Into the next layer down: the same pressure and temperature in its material.
    while (li > 0 && m < edges[li - 1]) {
      const s = aneosFromE(t, rho, e, xb);
      li--;
      xb = layers[li].xb;
      rho = densityAt(t, xb, s.p, s.T);
      e = aneosFromT(t, rho, s.T, xb).e;
    }
    push();
  }
  out.left = m;
  return out;
}

/**
 * The planet of mass M (g) with the given layers, from the center out, whose surface is at
 * temperature Ts (K), in the materials of table `t`.
 */
export function buildPlanet(t: AneosTable, layers: readonly PlanetLayer[], M: number, Ts: number, steps = 2000): Planet {
  const total = layers.reduce((s, l) => s + l.fraction, 0);
  const ls = layers.map((l) => ({ xb: l.xb, fraction: l.fraction / total }));
  // Between a body as dense throughout as its surface, and one thirty times denser.
  const rhoS = densityAt(t, ls[ls.length - 1].xb, P_SURFACE, Ts);
  let hi = Math.cbrt((3 * M) / (4 * Math.PI * rhoS));
  let lo = hi / Math.cbrt(30);
  let best: Trial | null = null;
  for (let k = 0; k < 50; k++) {
    const R = 0.5 * (lo + hi);
    const tr = integrate(t, ls, M, Ts, R, steps);
    if (tr.left > 0) {
      // Mass left over at the center: too small. The nearest such trial is the planet.
      lo = R;
      best = tr;
    } else hi = R;
    if (hi - lo < 1e-9 * hi) break;
  }
  if (!best) throw new Error('planet: no radius holds this mass');
  const n = best.r.length;
  const r = new Float64Array(n + 1);
  const rho = new Float64Array(n + 1);
  const p = new Float64Array(n + 1);
  const T = new Float64Array(n + 1);
  const layer = new Int32Array(n + 1);
  // From the center out; the center takes the innermost sample's values.
  for (let i = 0; i < n; i++) {
    const s = n - 1 - i;
    r[i + 1] = best.r[s];
    rho[i + 1] = best.rho[s];
    p[i + 1] = best.p[s];
    T[i + 1] = best.T[s];
    layer[i + 1] = best.layer[s];
  }
  rho[0] = rho[1];
  p[0] = p[1];
  T[0] = T[1];
  layer[0] = layer[1];
  const R = r[n];
  const at = (x: number) => {
    if (!(x < R)) return { rho: 0, p: 0, T: 0, layer: ls.length - 1 };
    // Samples are evenly spaced from r[1] to R.
    const f = Math.max(0, ((x - r[1]) / (R - r[1])) * (n - 1));
    const i = Math.min(Math.floor(f), n - 2) + 1;
    const w = f - (i - 1);
    // Across a layer's edge the density jumps: take the sample on this side of it.
    if (layer[i] !== layer[i + 1]) {
      const k = w < 0.5 ? i : i + 1;
      return { rho: rho[k], p: p[k], T: T[k], layer: layer[k] };
    }
    return { rho: rho[i] + w * (rho[i + 1] - rho[i]), p: p[i] + w * (p[i + 1] - p[i]), T: T[i] + w * (T[i + 1] - T[i]), layer: layer[i] };
  };
  return { R, M, rhoc: rho[0], r, rho, p, T, layer, at };
}
