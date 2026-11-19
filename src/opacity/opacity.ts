// Opacities for radiation transport (radiation/radiation.ts): the <opacity> slot of the
// radiation passes, and float64 mirrors of each module for initial conditions and the tests.
//
// Each module defines, per unit mass and in code units (area / mass):
//   kappa_planck(rho, T)     the absorption opacity, which couples the gas's temperature to E
//   kappa_rosseland(rho, T)  the total opacity (absorption and scattering), which drags on F
// T is the code temperature (p_gas / rho for the reference composition, units/units.ts).
// Their four constants are P.kap, in the order packOpacity() writes them.
import { SI } from '../units/constants';
import { kelvinPerCodeT, toCode, unitOf, type UnitSystem } from '../units/units';
import { meanMolecularWeight, type Composition } from '../units/composition';

export type OpacityKind = 'constant' | 'electron' | 'kramers';

export const OPACITY_KINDS: readonly OpacityKind[] = ['constant', 'electron', 'kramers'];

export const OPACITY_LABELS: Record<OpacityKind, string> = {
  constant: 'Constant',
  electron: 'Electron scattering',
  kramers: 'Kramers + electron scattering',
};

export const OPACITY_MODULES: Record<OpacityKind, string> = {
  constant: 'opacity/constant.wgsl',
  electron: 'opacity/electron.wgsl',
  kramers: 'opacity/kramers.wgsl',
};

/**
 * The constants of an opacity law, in code units. Which apply depends on the law:
 * - constant: `absorption` (kappa_P) and `scattering` (kappa_R = absorption + scattering);
 * - electron: `scattering` is the electron-scattering opacity, and `absorption` a small true
 *   absorption that still lets the gas and the radiation exchange heat;
 * - kramers: `kramers` is K0 in kappa_ff = K0 rho T^-3.5, which is both kappa_P and the
 *   absorption part of kappa_R; `scattering` the electron-scattering part, and `tFloor` the least
 *   temperature it is evaluated at (kappa_ff diverges as T goes to 0).
 */
export interface OpacityParams {
  absorption: number;
  scattering: number;
  kramers: number;
  tFloor: number;
}

export const defaultOpacityParams = (): OpacityParams => ({ absorption: 1, scattering: 0, kramers: 0, tFloor: 0 });

/** The four numbers of P.kap, in the order the selected module reads them. */
export function packOpacity(kind: OpacityKind, p: OpacityParams): [number, number, number, number] {
  if (kind === 'kramers') return [p.kramers, p.scattering, p.tFloor, 0];
  return [p.absorption, p.scattering, 0, 0];
}

/** Planck (absorption) and Rosseland (total) opacities, per unit mass: as the WGSL modules. */
export function opacity(kind: OpacityKind, p: OpacityParams, rho: number, T: number): { planck: number; rosseland: number } {
  if (kind === 'kramers') {
    const t = Math.max(T, p.tFloor, 1e-30);
    const ff = p.kramers * rho * t ** -3.5;
    return { planck: ff, rosseland: ff + p.scattering };
  }
  if (kind === 'electron') return { planck: p.absorption, rosseland: p.scattering + p.absorption };
  return { planck: p.absorption, rosseland: p.absorption + p.scattering };
}

/** Hydrogen's mass fraction in a composition (0 if it has none). */
const hydrogen = (c: Composition): number => {
  const i = c.species.findIndex((s) => s.name === 'h1');
  return i >= 0 ? c.X[i] : 0;
};

/**
 * Physical opacities in code units, for a problem with units and a composition: electron
 * scattering 0.2 (1 + X) cm^2/g, and Kramers' free-free opacity 3.7e22 (1 - Z)(1 + X) rho T^-3.5
 * cm^2/g (rho in g/cm^3, T in kelvin), from the composition's hydrogen (X) and metals (Z).
 */
export function physicalOpacity(units: UnitSystem, comp: Composition): Partial<OpacityParams> {
  const X = hydrogen(comp);
  const metals = comp.species.reduce((s, sp, i) => s + (sp.Z > 2 ? comp.X[i] : 0), 0);
  // cm^2/g in SI is 0.1 m^2/kg.
  const kappaUnit = unitOf(units, [-1, 2, 0]);
  const es = (0.2 * (1 + X) * 0.1) / kappaUnit;
  // kappa_ff in SI: the cgs coefficient times rho in g/cm^3 (1e-3 rho_SI), times 0.1 m^2/kg.
  const tk = kelvinPerCodeT(units, meanMolecularWeight(comp));
  const rhoCgs = unitOf(units, 'density') * 1e-3;
  const ff = (3.68e22 * (1 - metals) * (1 + X) * rhoCgs * tk ** -3.5 * 0.1) / kappaUnit;
  return { scattering: es, kramers: ff };
}

/** The speed of light in code units (1 without units: a scale-free problem sets its own). */
export const codeLightSpeed = (units: UnitSystem | undefined): number => (units ? toCode(units, 'velocity', SI.c) : 1);
