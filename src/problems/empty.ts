import { all, OUTFLOW, type Problem } from './types';

// An empty box: gas at the density and pressure floors everywhere, waiting (paused) for the
// editing tools to put something in it.
export const EMPTY = { rho: 1e-6, p: 1e-6 };

export const empty: Problem = {
  id: 'empty',
  label: 'Empty box',
  blurb: 'Nothing but floor-density gas, at the floor temperature: add gas, heat, sources and obstacles with the tools.',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: Infinity,
  rhoFloor: EMPTY.rho,
  pFloor: EMPTY.p,
  supports2d: true,
  startPaused: true,
  // The tools add gas of unit density and set speeds of order one, as in the other presets.
  toolScale: { rho: 1, v: 1 },
  view: { field: 0, log: true },
  init: () => () => ({ rho: EMPTY.rho, vx: 0, vy: 0, vz: 0, p: EMPTY.p, s: 0 }),
};
