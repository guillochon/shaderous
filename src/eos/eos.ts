// Float64 mirrors of the WGSL equation-of-state modules, used to build initial conditions,
// by the CPU reference solver, and by the tests.
import { meanMolecularWeight, nuclearAverages, COMPOSITIONS, type Composition } from '../units/composition';
import { SI } from '../units/constants';
import { eosConstants, kelvinPerCodeT, toCode, type UnitSystem } from '../units/units';

export type EosKind = 'ideal' | 'barotropic' | 'gas_radiation' | 'degenerate' | 'ionizing';

export interface EosParams {
  gamma: number;
  K: number; // barotropic constant (c_s^2 when gamma = 1)
  R: number; // gas constant in code units
  arad: number; // radiation constant in code units
  /** degenerate electrons (eos/degenerate.wgsl): the density at which p_F = m_e c, and the pressure scale A */
  degRho: number;
  degP: number;
  /**
   * Hydrogen ionization (eos/ionizing.wgsl): particles per atomic mass unit from hydrogen nuclei
   * and from the other species, both times the composition's mu; 13.6 eV / k in code
   * temperature; and the Saha density.
   */
  ionH: number;
  ionOther: number;
  ionT: number;
  ionRho: number;
}

export interface Eos {
  kind: EosKind;
  hasEnergy: boolean;
  params: EosParams;
  pressure(rho: number, eint: number): number;
  soundSpeed(rho: number, p: number): number;
  eintFromP(rho: number, p: number): number;
  temperature(rho: number, eint: number): number;
  eintFromT(rho: number, T: number): number;
}

export const EOS_MODULES: Record<EosKind, string> = {
  ideal: 'eos/ideal.wgsl',
  barotropic: 'eos/barotropic.wgsl',
  gas_radiation: 'eos/gas_radiation.wgsl',
  degenerate: 'eos/degenerate.wgsl',
  ionizing: 'eos/ionizing.wgsl',
};

export const EOS_LABELS: Record<EosKind, string> = {
  ideal: 'Ideal gas',
  barotropic: 'Barotropic',
  gas_radiation: 'Gas + radiation',
  degenerate: 'Gas + radiation + degenerate electrons',
  ionizing: 'Ionizing hydrogen (Saha)',
};

export const defaultEosParams = (): EosParams => ({
  gamma: 5 / 3, K: 1, R: 1, arad: 1, degRho: 1, degP: 1, ...ionParams(COMPOSITIONS.solar), ionT: 1, ionRho: 1,
});

/** The second vec4 of equation-of-state constants (Params.eos2 in hydro/common.wgsl). */
export function eosConstants2(kind: EosKind, p: EosParams): number[] {
  if (kind === 'degenerate') return [p.degRho, p.degP, 0, 0];
  if (kind === 'ionizing') return [p.ionH, p.ionOther, p.ionT, p.ionRho];
  return [0, 0, 0, 0];
}

/** Particles per atomic mass unit from hydrogen nuclei and from everything else, times mu. */
function ionParams(c: Composition): { ionH: number; ionOther: number } {
  const mu = meanMolecularWeight(c);
  const ih = c.species.findIndex((s) => s.name === 'h1');
  let other = 0;
  c.species.forEach((s, i) => {
    if (i === ih) return;
    const z = c.ionization === 'full' ? s.Z : c.ionization === 'neutral' ? 0 : c.ionization[i];
    other += (c.X[i] * (1 + z)) / s.A;
  });
  return { ionH: mu * (ih >= 0 ? c.X[ih] : 0), ionOther: mu * other };
}

/**
 * The constants of the degenerate and ionizing equations of state for a problem's composition
 * (default: solar) and, if it has them, its units (which also fix the radiation constant). A
 * scale-free problem keeps code-unit values of 1 for the physical scales, which the user can
 * change.
 */
export function physicalEosParams(units: UnitSystem | undefined, comp: Composition = COMPOSITIONS.solar): Partial<EosParams> {
  const out: Partial<EosParams> = ionParams(comp);
  if (!units) return out;
  Object.assign(out, eosConstants(units, comp)); // the gas constant (1) and the radiation constant
  const mu = meanMolecularWeight(comp);
  const { ye } = nuclearAverages(comp);
  const mec = SI.me * SI.c;
  out.degRho = toCode(units, 'density', ((SI.mu * 8 * Math.PI) / 3) * (mec / SI.h) ** 3 / ye);
  out.degP = toCode(units, 'pressure', (Math.PI * SI.me ** 4 * SI.c ** 5) / (3 * SI.h ** 3));
  out.ionT = SI.chiH / SI.kB / kelvinPerCodeT(units, mu);
  out.ionRho = toCode(units, 'density', mu * SI.mu * ((2 * Math.PI * SI.me * SI.chiH) / SI.h ** 2) ** 1.5);
  return out;
}

/** (f(x), g(x)) of the degenerate electron gas, split as deg_fg in eos/degenerate.wgsl. */
export function degFG(x: number): [number, number] {
  if (x < 0.5) {
    const x2 = x * x;
    let xp = x2 * x2 * x;
    let c = 1;
    let b = 0.5;
    let f = 0;
    let g = 0;
    for (let k = 0; k < 12; k++) {
      const n = 5 + 2 * k;
      f += (8 * c * xp) / n;
      g += (8 * b - (8 * c) / n) * xp;
      c *= -(2 * k + 1) / (2 * k + 2);
      b *= (0.5 - (k + 1)) / (k + 2);
      xp *= x2;
    }
    return [f, g];
  }
  const s = Math.sqrt(1 + x * x);
  const f = x * (2 * x * x - 3) * s + 3 * Math.asinh(x);
  return [f, 8 * x ** 3 * (s - 1) - f];
}

/** Gamma_1 of gas plus radiation from the gas share of the pressure (eos/quartic.wgsl). */
function gamma1GasRad(beta: number, gamma: number): number {
  const g1 = gamma - 1;
  const k = 4 - 3 * beta;
  return beta + (k * k * g1) / (beta + 12 * g1 * (1 - beta));
}

/** Root of A T^4 + B T = C; same iteration as eos_quartic in gas_radiation.wgsl. */
export function quartic(A: number, B: number, C: number, iters = 8): number {
  let T = C / B;
  if (A > 0) T = Math.min(T, Math.sqrt(Math.sqrt(C / A)));
  for (let i = 0; i < iters; i++) {
    const T3 = T * T * T;
    T = (3 * A * T3 * T + C) / (4 * A * T3 + B);
  }
  return T;
}

/**
 * The constants of an equation of state for a cell of a mixture with moments `k`
 * (species/mixture.ts), as the WGSL modules derive them with a varying composition: the gas
 * constant scaled by k[0], the degenerate electrons' density scale divided by k[1] (Ye
 * relative to the reference's), and the ionizing gas's hydrogen and other particles k[2], k[3].
 * The ideal gas's temperature also takes k[0]; makeEos() applies that.
 */
export function mixtureParams(kind: EosKind, params: EosParams, k: readonly number[]): EosParams {
  const out = { ...params };
  if (kind === 'gas_radiation' || kind === 'degenerate') out.R = params.R * k[0];
  if (kind === 'degenerate') out.degRho = params.degRho / k[1];
  if (kind === 'ionizing') {
    out.ionH = k[2];
    out.ionOther = k[3];
  }
  return out;
}

/**
 * An equation of state. With moments `k` (species/mixture.ts) it is that of a cell of a
 * mixture whose composition varies; without, the uniform one of `params`.
 */
export function makeEos(kind: EosKind, params: EosParams, k?: readonly number[]): Eos {
  if (k) {
    const eos = makeEos(kind, mixtureParams(kind, params, k));
    if (kind !== 'ideal') return eos;
    const x = k[0];
    return { ...eos, temperature: (rho, eint) => eos.temperature(rho, eint) / x, eintFromT: (rho, T) => eos.eintFromT(rho, T * x) };
  }
  const { gamma: g, K, R, arad: a } = params;
  switch (kind) {
    case 'ideal':
      return {
        kind,
        hasEnergy: true,
        params,
        pressure: (_rho, eint) => (g - 1) * eint,
        soundSpeed: (rho, p) => Math.sqrt((g * p) / rho),
        eintFromP: (_rho, p) => p / (g - 1),
        temperature: (rho, eint) => ((g - 1) * eint) / rho,
        eintFromT: (rho, T) => (rho * T) / (g - 1),
      };
    case 'barotropic': {
      const pressure = (rho: number) => (g === 1 ? K * rho : K * Math.pow(rho, g));
      return {
        kind,
        hasEnergy: false,
        params,
        pressure,
        soundSpeed: (rho, p) => Math.sqrt((g * p) / rho),
        eintFromP: (_rho, p) => (g < 1.001 ? 1.5 * p : p / (g - 1)),
        temperature: (rho) => pressure(rho) / rho,
        eintFromT: (rho) => (g < 1.001 ? 1.5 : 1 / (g - 1)) * pressure(rho),
      };
    }
    case 'gas_radiation':
      return {
        kind,
        hasEnergy: true,
        params,
        pressure: (rho, eint) => {
          const T = quartic(a, (R * rho) / (g - 1), eint);
          return R * rho * T + (a * T ** 4) / 3;
        },
        soundSpeed: (rho, p) => {
          const T = quartic(a / 3, R * rho, p);
          const beta = Math.min(1, Math.max(0, (R * rho * T) / p));
          return Math.sqrt((gamma1GasRad(beta, g) * p) / rho);
        },
        eintFromP: (rho, p) => {
          const T = quartic(a / 3, R * rho, p);
          return (R * rho * T) / (g - 1) + a * T ** 4;
        },
        temperature: (rho, eint) => quartic(a, (R * rho) / (g - 1), eint),
        eintFromT: (rho, T) => (R * rho * T) / (g - 1) + a * T ** 4,
      };
    case 'degenerate': {
      const xOf = (rho: number) => Math.cbrt(rho / params.degRho);
      const deg = (rho: number): [number, number] => {
        const [f, gx] = degFG(xOf(rho));
        return [params.degP * f, params.degP * gx];
      };
      const thermalE = (rho: number, T: number) => (R * rho * T) / (g - 1) + a * T ** 4;
      return {
        kind,
        hasEnergy: true,
        params,
        pressure: (rho, eint) => {
          const [pd, ed] = deg(rho);
          const T = quartic(a, (R * rho) / (g - 1), Math.max(eint - ed, 0));
          return pd + R * rho * T + (a * T ** 4) / 3;
        },
        soundSpeed: (rho, p) => {
          const x = xOf(rho);
          const pd = params.degP * degFG(x)[0];
          const cd2 = (params.degP * 8 * x ** 5) / (3 * rho * Math.sqrt(1 + x * x));
          const pth = Math.max(p - pd, 0);
          let ct2 = 0;
          if (pth > 0) {
            const T = quartic(a / 3, R * rho, pth);
            ct2 = (gamma1GasRad(Math.min(1, Math.max(0, (R * rho * T) / pth)), g) * pth) / rho;
          }
          return Math.sqrt(cd2 + ct2);
        },
        eintFromP: (rho, p) => {
          const [pd, ed] = deg(rho);
          return ed + thermalE(rho, quartic(a / 3, R * rho, Math.max(p - pd, 0)));
        },
        temperature: (rho, eint) => quartic(a, (R * rho) / (g - 1), Math.max(eint - deg(rho)[1], 0)),
        eintFromT: (rho, T) => deg(rho)[1] + thermalE(rho, T),
      };
    }
    case 'ionizing':
      return ionizingEos(params);
  }
}

/** Float64 mirror of eos/ionizing.wgsl: the same bracketed Newton, run to convergence. */
function ionizingEos(params: EosParams): Eos {
  const { gamma: g, ionH: h, ionOther: o, ionT: Ti, ionRho: rs } = params;
  const gm1 = g - 1;
  const ionX = (rho: number, T: number): [number, number] => {
    if (!(h > 0) || !(T > 0)) return [0, 0];
    const lnS = Math.log(rs / (h * rho)) + 1.5 * Math.log(T / Ti) - Ti / T;
    if (lnS < -80) return [0, 0];
    if (lnS > 80) return [1, 0];
    const x = 2 / (1 + Math.sqrt(1 + 4 * Math.exp(-lnS)));
    return [x, (x * (1 - x)) / (2 - x)];
  };
  const dxdT = (gx: number, T: number) => (gx > 0 ? (gx * (1.5 + Ti / T)) / T : 0);
  const N = (x: number) => h * (1 + x) + o;
  const solve = (rho: number, target: number, byP: boolean) => {
    let lo = byP ? target / (2 * h + o) : (Math.max(target - h * Ti, 0) * gm1) / (2 * h + o);
    let hi = byP ? target / (h + o) : (target * gm1) / (h + o);
    let T = hi;
    for (let i = 0; i < 200; i++) {
      const [x, gx] = ionX(rho, T);
      const n = N(x);
      const xT = dxdT(gx, T);
      const f = byP ? T * n - target : (T * n) / gm1 + h * x * Ti - target;
      const df = byP ? n + T * h * xT : n / gm1 + h * (T / gm1 + Ti) * xT;
      if (f > 0) hi = T;
      else lo = T;
      let next = T - f / df;
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (next === T) break;
      T = next;
    }
    return T;
  };
  const eintFromT = (rho: number, T: number) => {
    const x = ionX(rho, T)[0];
    return rho * ((T * N(x)) / gm1 + h * x * Ti);
  };
  return {
    kind: 'ionizing',
    hasEnergy: true,
    params,
    pressure: (rho, eint) => {
      const T = solve(rho, eint / rho, false);
      return rho * T * N(ionX(rho, T)[0]);
    },
    soundSpeed: (rho, p) => {
      const T = solve(rho, p / rho, true);
      const [x, gx] = ionX(rho, T);
      const n = N(x);
      const xT = dxdT(gx, T);
      const w = T / gm1 + Ti;
      const pRho = T * (n - h * gx);
      const pT = rho * (n + T * h * xT);
      const uT = n / gm1 + h * w * xT;
      const dT = (T * n + h * gx * w) / (rho * uT);
      return Math.sqrt(Math.max(pRho + pT * dT, 0));
    },
    eintFromP: (rho, p) => eintFromT(rho, solve(rho, p / rho, true)),
    temperature: (rho, eint) => solve(rho, eint / rho, false),
    eintFromT,
  };
}

/** Hydrogen's ionized fraction under the ionizing equation of state, at code temperature T. */
export function ionizedFraction(params: EosParams, rho: number, T: number): number {
  const { ionH: h, ionT: Ti, ionRho: rs } = params;
  if (!(h > 0) || !(T > 0)) return 0;
  const lnS = Math.log(rs / (h * rho)) + 1.5 * Math.log(T / Ti) - Ti / T;
  return lnS > 80 ? 1 : 2 / (1 + Math.sqrt(1 + 4 * Math.exp(-lnS)));
}
