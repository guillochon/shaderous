// A carbon-oxygen detonation, to be burned by the trained network of the 13-isotope alpha chain
// (burn/network.ts): white dwarf matter (carbon and oxygen, degenerate electrons, ions and
// radiation: eos/helmholtz.wgsl) at rest in a tube, set alight at its closed end (the low-x
// edge of the box) by a layer held hotter than carbon can stand. The layer burns in
// microseconds, its pressure drives a shock into the fuel, and the shocked fuel burns behind
// it: a detonation, running down the tube at about 11,000 km/s at 1e7 g/cm^3. Carbon burns
// right behind the shock, oxygen over tens of kilometers, and silicon toward the iron group over
// far longer, how far depending on the density (Khokhlov 1989; Seitenzahl et al. 2009, Gamezo
// et al. 1999).
import { NET_SPECIES } from '../../../burn/network';
import { toCode } from '../../../units/units';
import type { FlowDef } from '../flow';

/** Mass fractions of the fuel, in the network's order. */
const fuelX = (c: number): number[] => NET_SPECIES.map((s) => (s === 'c12' ? c : s === 'o16' ? 1 - c : 0));

interface Params {
  logrho: number;
  carbon: number;
  tfuel: number;
  tign: number;
  width: number;
}

const ignition: FlowDef<Params> = {
  name: 'co-ignition',
  label: 'Carbon–oxygen fuel with an ignition layer',
  tip: 'Carbon and oxygen at rest at one density, with a hot layer at the low-x edge of the box that sets it alight. Needs physical units, an equation of state with a temperature (Helmholtz) and the species of the aprox13 network.',
  wholeBox: true,
  fields: {
    logrho: { label: 'Density (log₁₀ g/cm³)', default: 7, min: 6, max: 9.5, step: 0.25, tip: 'Density of the fuel, log10 of g/cm^3: white dwarf matter is 6 to 9.5. Above about 1e8 the ash reaches the iron group; below 1e7 burning stops at silicon or oxygen' },
    carbon: { label: 'Carbon fraction', default: 0.5, min: 0.2, max: 0.8, step: 0.05, tip: 'Mass fraction of carbon in the fuel, the rest oxygen' },
    tfuel: { label: 'Fuel temperature (10⁸ K)', default: 0.5, min: 0.1, max: 5, step: 0.1, tip: 'Temperature of the unburned fuel, in 1e8 K. Gas below 1e8 K does not burn' },
    tign: { label: 'Ignition temperature (10⁹ K)', default: 4, min: 2, max: 8, step: 0.25, tip: 'Temperature of the layer at the closed end that sets the fuel alight, in 1e9 K' },
    width: { label: 'Ignition layer (km)', default: 30, min: 5, max: 200, step: 5, tip: 'Thickness of the hot layer at the closed end, in km' },
  },
  init: (ctx, p) => {
    const units = ctx.units;
    if (!units) throw new Error('the carbon–oxygen ignition needs physical units (domain.units)');
    const eos = ctx.eos;
    const rho = toCode(units, 'density', 10 ** p.logrho * 1e3);
    // The fuel's pressure at a temperature in kelvin, by the run's equation of state.
    const pressure = (TK: number) => eos.pressure(rho, eos.eintFromT(rho, TK / eos.params.helmT));
    const X = fuelX(p.carbon);
    const cold = { rho, vx: 0, vy: 0, vz: 0, p: pressure(p.tfuel * 1e8), s: 0, X };
    const hot = { ...cold, p: pressure(p.tign * 1e9), s: 1 };
    const edge = -0.5 + (p.width * 1e3) / units.length;
    return (x) => (x < edge ? hot : cold);
  },
};

export const flows = [ignition];
