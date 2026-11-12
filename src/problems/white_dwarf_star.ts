import { all, OUTFLOW, type Problem } from './types';
import { COMPOSITIONS } from '../units/composition';
import { codeG } from '../units/units';
import { dwarfsInit, dwarfsSettling, scalesOf, whiteDwarf, WD_AMBIENT, WD_UNITS, type PlacedDwarf } from './white_dwarf';
import type { Eos } from '../eos/eos';

// A single cold carbon-oxygen white dwarf at rest: hydrostatic equilibrium under the pressure of
// its degenerate electrons (problems/white_dwarf.ts), settled into the grid's own balance
// before the run (problems/settle.ts). It should simply sit there; how still it stays measures
// the scheme's hydrostatic balance with this equation of state. Units as for the white dwarf
// collision: 10^4 km, one solar mass, G = 1 (a 0.6 M_sun dwarf is 0.87 in radius, with a
// dynamical time sqrt(R^3 / G M) of 0.9, or 2.5 s).
export const WD_STAR = { box: 3 };

function placed(p: Record<string, number>, eos: Eos): PlacedDwarf[] {
  return [{ wd: whiteDwarf(p.m, scalesOf(eos), codeG(WD_UNITS)), center: [0, 0, 0], vel: [0, 0, 0], scalar: 1 }];
}

export const whiteDwarfStar: Problem = {
  id: 'white-dwarf',
  label: 'White dwarf (hydrostatic)',
  blurb: 'A cold carbon-oxygen white dwarf held up by degenerate electrons, in equilibrium. Mass in solar masses.',
  eos: 'degenerate',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'isolated',
  G: codeG(WD_UNITS),
  units: WD_UNITS,
  composition: COMPOSITIONS.carbonOxygen,
  box: WD_STAR.box,
  tEnd: Infinity,
  checkpointDt: 0.5,
  rhoFloor: 1e-10,
  pFloor: 1e-14,
  fluff: { rho: 10 * WD_AMBIENT.rhoAmb, u: 3, v: 10 },
  // Smooth inside, so held at the finest level outright above this density (2e6 g/cm^3).
  refineDense: 1,
  view: { field: 0, log: true, mode: 'slice', axis: 2 },
  params: [
    { id: 'm', label: 'Mass (M☉)', title: 'Mass of the white dwarf, in solar masses (carbon-oxygen white dwarfs are 0.5 to 1.3). Changing it starts the run again', value: 0.6, min: 0.5, max: 1.3, step: 0.05 },
  ],
  settle: (p, eos) => dwarfsSettling(scalesOf(eos), placed(p, eos), codeG(WD_UNITS)),
  init: (_n, eos, _ndim, p) => dwarfsInit(scalesOf(eos), placed(p, eos)),
};
