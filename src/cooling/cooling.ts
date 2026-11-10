// Optically thin cooling: the settings, and float64 mirrors of the WGSL modules
// (cooling/*.wgsl, hydro/cooling.wgsl) for initial conditions and the tests.
import type { Eos } from '../eos/eos';

export type CoolingKind = 'off' | 'power' | 'ism';

export interface CoolingParams {
  /** power law: the coefficient Lambda0 of rho^2 Lambda0 T^alpha */
  rate: number;
  /** power law: the exponent alpha (1/2 is bremsstrahlung) */
  alpha: number;
  /** heating per unit mass (power law), or the heating rate at unit density (interstellar) */
  heat: number;
  /** gas at or below this temperature does not cool */
  tFloor: number;
  /** interstellar: kelvin per code unit of temperature (code density is atoms per cm^3) */
  tUnit: number;
}

export const COOLING_MODULES: Record<CoolingKind, string> = {
  off: 'cooling/off.wgsl',
  power: 'cooling/power.wgsl',
  ism: 'cooling/ism.wgsl',
};

export const COOLING_LABELS: Record<CoolingKind, string> = {
  off: 'Off',
  power: 'Power law',
  ism: 'Interstellar (two-phase)',
};

export const defaultCoolingParams = (): CoolingParams => ({ rate: 1, alpha: 0.5, heat: 0, tFloor: 0, tUnit: 1000 });

/** The four numbers of P.cool, in the order the selected module reads them. */
export function packCooling(kind: CoolingKind, p: CoolingParams): [number, number, number, number] {
  if (kind === 'ism') return [p.heat, p.tUnit, 0, p.tFloor];
  if (kind === 'power') return [p.rate, p.alpha, p.heat, p.tFloor];
  return [0, 0, 0, 0];
}

/** Koyama & Inutsuka's cooling function over the heating rate, for T in kelvin (cm^3). */
export const ismLambda = (tk: number): number => 1e7 * Math.exp(-118400 / (tk + 1000)) + 0.014 * Math.sqrt(tk) * Math.exp(-92 / tk);

/** Net heating rate per unit volume (negative: the gas cools). */
export function coolRate(kind: CoolingKind, p: CoolingParams, rho: number, T: number): number {
  const cools = T > p.tFloor && T > 0;
  if (kind === 'power') return rho * (p.heat - rho * (cools ? p.rate * Math.pow(T, p.alpha) : 0));
  if (kind === 'ism') return p.heat * rho * (1 - rho * (cools ? ismLambda(T * p.tUnit) : 0));
  return 0;
}

/** Density at which interstellar gas of temperature T (code units) neither heats nor cools. */
export const ismBalanceDensity = (p: CoolingParams, T: number): number => 1 / ismLambda(T * p.tUnit);

export const COOL_SUBSTEPS = 12;

/** Internal energy density after a step dt at fixed density: as apply_cooling in hydro/cooling.wgsl. */
export function integrateCooling(kind: CoolingKind, p: CoolingParams, eos: Eos, rho: number, eint: number, dt: number, pFloor = 0): number {
  if (kind === 'off' || !eos.hasEnergy) return eint;
  const emin = eos.eintFromP(rho, pFloor);
  const efloor = Math.max(emin, eos.eintFromT(rho, p.tFloor));
  const edot = (e: number) => coolRate(kind, p, rho, eos.temperature(rho, e));
  let e = eint;
  let left = dt;
  for (let i = 0; i < COOL_SUBSTEPS && left > 0; i++) {
    const k1 = edot(e);
    if (k1 === 0) break;
    const h = Math.min(left, (0.1 * e) / Math.abs(k1));
    const e1 = Math.max(e + h * k1, emin);
    const k2 = edot(e1);
    if (k1 * k2 < 0) {
      e += ((e1 - e) * k1) / (k1 - k2);
      continue;
    }
    let en = e + 0.5 * h * (k1 + k2);
    if (en < e) en = Math.max(en, Math.min(e, efloor));
    e = Math.max(en, emin);
    left -= h;
  }
  return e;
}
