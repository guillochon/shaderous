import { all, OUTFLOW, type Problem, type ProblemParam } from '../../src/problems/types';
import { ASTRO } from '../../src/units/constants';
import { codeG, gravityUnits } from '../../src/units/units';

// An empty box: gas at the density and pressure floors everywhere, waiting (paused) for the
// editing tools to put something in it.
//
// Its code units are solar: one unit of length is a solar radius, one of mass a solar mass, and
// G = 1 (the time unit is 1593 s), so that what the tools place (a star, say) has a physical
// size and mass, and an equation of state with physical constants has them. The box's side is
// a parameter, in those units: ten solar radii unless it is set.
export const EMPTY = { rho: 1e-6, p: 1e-6 };
export const EMPTY_UNITS = gravityUnits('solar', ASTRO.Rsun, ASTRO.Msun);

const boxParam: ProblemParam = {
  id: 'box', label: 'Box size', value: 10, min: 1e-3, max: 1e9, factor: 2,
  title: 'Side of the box: in code units (one is a solar radius) or, with the readouts in physical units, in the unit picked beside it. Changing it builds the run again',
  unitChoices: [
    { label: 'R☉', value: 1 },
    { label: 'AU', value: ASTRO.au / ASTRO.Rsun },
    { label: 'pc', value: ASTRO.pc / ASTRO.Rsun },
  ],
};

export const empty: Problem = {
  id: 'empty',
  label: 'Empty box',
  blurb: 'Nothing but floor-density gas, at the floor temperature: add gas, heat, sources and obstacles with the tools.',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'off',
  G: codeG(EMPTY_UNITS),
  units: EMPTY_UNITS,
  box: boxParam.value,
  boxParam: 'box',
  params: [boxParam],
  tEnd: Infinity,
  checkpointDt: 0.1,
  rhoFloor: EMPTY.rho,
  pFloor: EMPTY.p,
  supports2d: true,
  startPaused: true,
  // The tools add gas of unit density and set speeds of order one, as in the other presets.
  toolScale: { rho: 1, v: 1 },
  view: { field: 0, log: true },
  init: () => () => ({ rho: EMPTY.rho, vx: 0, vy: 0, vz: 0, p: EMPTY.p, s: 0 }),
};

/**
 * The empty box threaded by a uniform magnetic field, along the axis drawn upward (y in 2D, z
 * in 3D): what the tools add moves in it.
 */
export const emptyMagnetic: Problem = {
  ...empty,
  id: 'empty-mhd',
  label: 'Empty box with a magnetic field',
  blurb: 'Floor-density gas threaded by a uniform magnetic field: what the tools add drags the field with it, and is held by it.',
  params: [
    boxParam,
    {
      id: 'b', label: 'Field strength B', value: 0.1, min: 0, max: 100, factor: 2,
      title: 'Strength of the uniform field, in units where the magnetic pressure is B² / 2 (code units): 0.1 has the pressure of gas of unit density at a temperature of 0.005. Changing it starts the run again',
    },
  ],
  mhd: { face: (p, ndim) => (axis) => (axis === (ndim === 2 ? 1 : 2) ? p.b : 0) },
};
