// A first hydrostatic core, as protostellar collapse makes one (Larson 1969; Masunaga &
// Inutsuka 2000): a ball of molecular gas a few astronomical units across, all on one adiabat,
// that of gas which turned opaque at 1e-13 g/cm^3 and 10 K. Its structure is worked out here
// from the run's own equation of state (meant for hydrogen and helium, eos/hhe.ts): the
// adiabat by du = p / rho^2 drho, then hydrostatic balance outward from the central density.
//
// With that adiabat the cores grow to 0.040 solar masses, at a central density of 3e-9 g/cm^3
// and 1800 K. A denser center is hotter than 2000 K: H2 dissociates there, Gamma_1 falls to
// 1.1, below the 4/3 a ball needs to hold itself up, and the core is unstable. It collapses,
// the second collapse, until the hydrogen is atomic (above 1e-3 g/cm^3 and 1e4 K) and a
// second core, the protostar, forms at its center.
import { toCode } from '../../../units/units';
import type { FlowDef } from '../flow';

export interface CoreProfile {
  /** radius, density and pressure outward from the center, code units; the last is the surface */
  r: number[];
  rho: number[];
  p: number[];
  /** mass inside the surface */
  mass: number;
}

/**
 * The isentropic ball in hydrostatic balance with central density rhoC, on the adiabat through
 * density rhoA at energy per unit mass uA, out to where its density falls to rhoOut (all code
 * units). `pressure(rho, eint)` is the equation of state's.
 */
export function isentropicCore(pressure: (rho: number, eint: number) => number, G: number, rhoC: number, rhoA: number, uA: number, rhoOut: number): CoreProfile {
  // The adiabat, in ln rho from the less dense of the anchor and the surface to the center:
  // du/dln rho = p / rho.
  const lo = Math.log(Math.min(rhoA, rhoOut) * 0.5);
  const hi = Math.log(rhoC * 1.001);
  const N = 1500;
  const h = (hi - lo) / N;
  const f = (lr: number, u: number) => pressure(Math.exp(lr), Math.exp(lr) * u) / Math.exp(lr);
  const rk = (lr: number, u: number, s: number) => {
    const k1 = f(lr, u);
    const k2 = f(lr + s / 2, u + (s * k1) / 2);
    const k3 = f(lr + s / 2, u + (s * k2) / 2);
    const k4 = f(lr + s, u + s * k3);
    return u + (s * (k1 + 2 * k2 + 2 * k3 + k4)) / 6;
  };
  // From the anchor down to the grid's start in one stretch of small steps, then up the grid.
  let u = uA;
  const la = Math.log(rhoA);
  const back = Math.max(1, Math.ceil((la - lo) / h));
  for (let k = 0; k < back; k++) u = rk(la - (k * (la - lo)) / back, u, -(la - lo) / back);
  const lnP: number[] = [];
  const us: number[] = [];
  for (let k = 0; k <= N; k++) {
    us.push(u);
    lnP.push(Math.log(f(lo + k * h, u)) + lo + k * h);
    if (k < N) u = rk(lo + k * h, u, h);
  }
  // ln rho at pressure P (which rises with the density along an adiabat), by bisection on the grid.
  const lnRhoAt = (P: number): number => {
    const y = Math.log(P);
    let a = 0;
    let b = N;
    while (b - a > 1) {
      const m = (a + b) >> 1;
      if (lnP[m] > y) b = m;
      else a = m;
    }
    return lo + (a + Math.min(Math.max((y - lnP[a]) / (lnP[b] - lnP[a]), 0), 1)) * h;
  };
  const pAt = (lr: number): number => {
    const x = Math.min(Math.max((lr - lo) / h, 0), N - 1e-9);
    const k = Math.floor(x);
    return Math.exp(lnP[k] + (x - k) * (lnP[k + 1] - lnP[k]));
  };
  // Hydrostatic balance outward: dP/dr = -G m rho / r^2, dm/dr = 4 pi r^2 rho, by midpoint
  // steps of 1% in radius from a small uniform center.
  const pC = pAt(Math.log(rhoC));
  let r = 1e-3 * Math.sqrt(pC / (G * rhoC * rhoC));
  let m = (4 / 3) * Math.PI * r ** 3 * rhoC;
  let P = pC - (2 / 3) * Math.PI * G * rhoC * rhoC * r * r;
  let rho = rhoC;
  const out: CoreProfile = { r: [0, r], rho: [rhoC, rhoC], p: [pC, P], mass: m };
  for (let k = 0; k < 20000 && rho > rhoOut; k++) {
    const dr = 0.01 * r;
    const dP1 = (-G * m * rho) / (r * r);
    const rm = r + dr / 2;
    const Pm = P + (dP1 * dr) / 2;
    if (!(Pm > 0)) break;
    const rhoM = Math.exp(lnRhoAt(Pm));
    const mm = m + 2 * Math.PI * r * r * rho * dr;
    P += ((-G * mm * rhoM) / (rm * rm)) * dr;
    m += 4 * Math.PI * rm * rm * rhoM * dr;
    r += dr;
    if (!(P > 0)) break;
    rho = Math.exp(lnRhoAt(P));
    out.r.push(r);
    out.rho.push(rho);
    out.p.push(P);
    out.mass = m;
  }
  return out;
}

const firstCore: FlowDef<{ logrhoc: number; tad: number; deficit: number; logamb: number }> = {
  name: 'first-core',
  label: 'First core (protostellar)',
  tip: 'A first hydrostatic core of protostellar collapse at the middle of the box: molecular gas on one adiabat, in balance with its own gravity, in thin gas at its surface pressure. Past a central density of 3e-9 g/cm^3 its center is above 2000 K, where H2 dissociates, and it collapses. Needs physical units and an equation of state with a temperature (hydrogen and helium).',
  wholeBox: true,
  requires: [{ fact: 'units' }, { fact: 'eos.energy' }],
  fields: {
    logrhoc: { label: 'Central density (log₁₀ g/cm³)', default: -7.75, min: -11, max: -6, step: 0.25, tip: 'Central density of the core, log10 of g/cm^3. Up to -8.5 the core is stable (its mass rises with its central density, to 0.040 solar masses); denser, its center is dissociating and it collapses' },
    tad: { label: 'Adiabat (K at 10⁻¹³ g/cm³)', default: 10, min: 3, max: 100, step: 1, tip: 'Temperature of the core’s adiabat at 1e-13 g/cm^3, the density at which a collapsing cloud turns opaque: 10 K is the usual cloud' },
    deficit: { label: 'Pressure deficit', default: 0.05, min: 0, max: 0.5, step: 0.01, tip: 'Fraction of its pressure the core starts short of, which sets it falling inward' },
    logamb: { label: 'Density outside (log₁₀ g/cm³)', default: -13, min: -16, max: -11, step: 0.5, tip: 'Density of the gas around the core, log10 of g/cm^3; the core’s surface is where its own density falls to it' },
  },
  init: (ctx, q) => {
    const units = ctx.units;
    if (!units) throw new Error('the first core needs physical units (domain.units)');
    const eos = ctx.eos;
    if (!eos.hasEnergy) throw new Error('the first core needs an equation of state with an energy');
    const code = (logCgs: number) => toCode(units, 'density', 10 ** logCgs * 1e3);
    const rhoA = code(-13);
    const uA = eos.eintFromT(rhoA, q.tad / eos.params.helmT) / rhoA;
    const rhoAmb = code(q.logamb);
    const core = isentropicCore(eos.pressure, ctx.G, code(q.logrhoc), rhoA, uA, rhoAmb);
    const n = core.r.length;
    const R = core.r[n - 1];
    const pOut = core.p[n - 1];
    return (x, y, z) => {
      const r = Math.hypot(x, y, z);
      if (r >= R) return { rho: rhoAmb, vx: 0, vy: 0, vz: 0, p: pOut, s: 0 };
      // The profile's steps are 1% in radius: find the one r is in.
      let a = 0;
      let b = n - 1;
      while (b - a > 1) {
        const m = (a + b) >> 1;
        if (core.r[m] > r) b = m;
        else a = m;
      }
      const w = (r - core.r[a]) / (core.r[b] - core.r[a]);
      const mix = (v: number[]) => Math.exp(Math.log(v[a]) + w * (Math.log(v[b]) - Math.log(v[a])));
      return { rho: mix(core.rho), vx: 0, vy: 0, vz: 0, p: Math.max(mix(core.p) * (1 - q.deficit), pOut), s: 1 };
    };
  },
};

export const flows = [firstCore];
