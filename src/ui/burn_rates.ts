// The rate burning changes each species' rho X at a cell, for the query readout's species
// equations (ui/terms.ts): float64 mirrors of what the burn passes do over a step.
import { burnIndices, isNetworkBurn, type AnyBurnSpec } from '../burn/burn';
import { NET_MAX_DT, NET_T_MAX, netStep, netUnits, watchesOnly } from '../burn/network';
import type { Composition } from '../units/composition';
import type { UnitSystem } from '../units/units';

export type BurnRates = (rho: number, T: number, X: number[]) => number[] | null;

/**
 * The burn rates of a problem's burning, or null where nothing burns (no burning, or a run that
 * only watches for it). Code units: rho and T as the state holds them, rates per unit volume and
 * time. `dt` is the run's step (code time): the network's rate is its change over one step.
 */
export function burnRates(spec: AnyBurnSpec | null, p: { species?: readonly string[]; units?: UnitSystem; composition?: Composition }, dt: () => number): BurnRates | null {
  if (!spec || !p.species?.length) return null;
  const n = p.species.length;
  if (!isNetworkBurn(spec)) {
    // One reaction, fuel to ash, at k exp(-Ea / T) per unit time (hydro/burn.wgsl).
    const [fuel, ash] = burnIndices(spec, p.species);
    return (rho, T, X) => {
      const w = T > 0 ? rho * Math.max(X[fuel], 0) * spec.k * Math.exp(-spec.ea / T) : 0;
      const out = new Array(n).fill(0);
      out[fuel] = -w;
      out[ash] = w;
      return out;
    };
  }
  if (watchesOnly(p.species)) return null;
  const u = netUnits(p.units, p.composition);
  const tMin = spec.tMin ?? 1e8;
  const rhoMin = spec.rhoMin ?? 1e4;
  return (rho, T, X) => {
    const rhoCgs = rho * u.rho;
    const tK = Math.min(T * u.T, NET_T_MAX);
    if (!(tK >= tMin) || !(rhoCgs >= rhoMin)) return new Array(n).fill(0);
    // The network's change over one step of the run (in seconds, within what it was trained on).
    const seconds = Math.min(Math.max(dt() * u.time, 1e-12), NET_MAX_DT);
    const D = netStep(Math.log10(rhoCgs), Math.log10(tK), Math.log10(seconds), X);
    const code = seconds / u.time;
    return D.map((d) => (rho * d) / code);
  };
}
