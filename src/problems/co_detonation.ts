import { OUTFLOW, REFLECT, type Problem } from './types';
import { NET_SPECIES } from '../burn/network';
import { COMPOSITIONS } from '../units/composition';
import { flowUnits, toCode } from '../units/units';

// A carbon-oxygen detonation, burned by the trained network of the 13-isotope alpha chain
// (burn/network.ts): white dwarf matter (equal parts C12 and O16 by mass, degenerate electrons,
// ions and radiation: eos/helmholtz.wgsl) at rest in a tube, set alight at its closed end by a
// layer held hotter than carbon can stand. The layer burns in microseconds, its pressure
// drives a shock into the fuel, and the shocked fuel burns behind it: a detonation, running
// down the tube at about 11,000 km/s at 1e7 g/cm^3. Carbon burns right behind the shock,
// oxygen over tens of kilometers, and silicon toward the iron group over far longer, how far
// depending on the density (Khokhlov 1989; Seitenzahl et al. 2009, Gamezo et al. 1999).
//
// The tube is 1000 km long, closed at x = 0 (a wall) and open at the far end; walls across.
// Units: 1000 km, 1e7 g/cm^3 and 1000 km/s, so that the time unit is one second.

export const CO_DETONATION_UNITS = flowUnits('C/O detonation', 1e6, 1e10, 1e6);
const COMP = COMPOSITIONS.carbonOxygen;
/** Mass fractions of the fuel, in the network's order. */
const fuelX = (c: number): number[] => NET_SPECIES.map((s) => (s === 'c12' ? c : s === 'o16' ? 1 - c : 0));

export const coDetonation: Problem = {
  id: 'detonation-co',
  label: 'Detonation: carbon–oxygen (network)',
  blurb:
    'A detonation in white dwarf matter, burned by a trained neural network of the 13-isotope alpha chain (aprox13): carbon and oxygen at rest in a 1000 km tube, set alight by a hot layer at its closed end. The shocked fuel burns behind the front through oxygen and silicon toward the iron group; how far it gets depends on the density.',
  eos: 'helmholtz',
  eosParams: { gamma: 5 / 3 },
  units: CO_DETONATION_UNITS,
  composition: COMP,
  species: NET_SPECIES,
  burn: (p) => ({ network: 'aprox13', inShocks: p.shockburn > 0.5, limiter: p.limiter }),
  // Closed at x = 0, open at the far end; walls across.
  bcLo: [REFLECT, REFLECT, REFLECT],
  bcHi: [OUTFLOW, REFLECT, REFLECT],
  gravity: 'off',
  G: 0,
  box: 1,
  // The front crosses most of the tube.
  tEnd: 0.08,
  tDyn: 0.01,
  supports2d: true,
  grid: { ndim: 2, n: 512, amr: 0 },
  view: { field: 1, log: true, mode: 'slice', axis: 2 },
  params: [
    { id: 'logrho', label: 'Density (log₁₀ g/cm³)', title: 'Density of the fuel, log10 of g/cm^3: white dwarf matter is 6 to 9.5. Above about 1e8 the ash reaches the iron group; below 1e7 burning stops at silicon or oxygen. Changing it starts the run again', value: 7, min: 6, max: 9.5, step: 0.25 },
    { id: 'carbon', label: 'Carbon fraction', title: 'Mass fraction of carbon in the fuel, the rest oxygen. Changing it starts the run again', value: 0.5, min: 0.2, max: 0.8, step: 0.05 },
    { id: 'tfuel', label: 'Fuel temperature (10⁸ K)', title: 'Temperature of the unburned fuel, in 1e8 K. Gas below 1e8 K does not burn. Changing it starts the run again', value: 0.5, min: 0.1, max: 5, step: 0.1 },
    { id: 'tign', label: 'Ignition temperature (10⁹ K)', title: 'Temperature of the layer at the closed end that sets the fuel alight, in 1e9 K. Changing it starts the run again', value: 4, min: 2, max: 8, step: 0.25 },
    { id: 'width', label: 'Ignition layer (km)', title: 'Thickness of the hot layer at the closed end, in km. Changing it starts the run again', value: 30, min: 5, max: 200, step: 5 },
    { id: 'shockburn', label: 'Burn inside shocks', title: 'Let cells the shock detector flags burn (1) or not (0). Off, as in most supernova codes: a captured shock is a few cells of numerical smearing, and burning there lets the scheme set the front. Changing it starts the run again', value: 0, min: 0, max: 1, step: 1 },
    { id: 'limiter', label: 'Burning limiter f', title: 'No cell takes in burning energy faster than f times its internal energy per sound crossing of the cell (Kushnir et al. 2013): on a grid far coarser than the burning, faster heating is numerically unstable and lights detonations too early. They use 0.1, but here anything under about 0.2 stops even resolved detonations, and on cells of hundreds of km it suppresses the collision detonation; 0 switches it off (the default). Changing it starts the run again', value: 0, min: 0, max: 1, step: 0.05 },
  ],
  init: (_n, eos, _ndim, p) => {
    const rho = toCode(CO_DETONATION_UNITS, 'density', 10 ** p.logrho * 1e3);
    // The fuel's pressure at a temperature in kelvin, by the run's equation of state.
    const pressure = (TK: number) => eos.pressure(rho, eos.eintFromT(rho, TK / eos.params.helmT));
    const X = fuelX(p.carbon);
    const cold = { rho, vx: 0, vy: 0, vz: 0, p: pressure(p.tfuel * 1e8), s: 0, X };
    const hot = { ...cold, p: pressure(p.tign * 1e9), s: 1 };
    const edge = -0.5 + (p.width * 1e3) / CO_DETONATION_UNITS.length;
    return (x) => (x < edge ? hot : cold);
  },
};
