import { OUTFLOW, PERIODIC, REFLECT, type Problem } from './types';

// Mach 3 wind tunnel with a forward-facing step (Emery 1968; Woodward & Colella 1984). Gas
// with rho = 1.4, p = 1 (sound speed 1) flows at speed 3 through a tunnel 3 long and 1 high,
// over a step 0.2 high that begins 0.6 from the inlet. A bow shock forms ahead of the step,
// reflects off the upper wall and then off the top of the step; by t = 4 the pattern is close
// to steady, with a Mach stem at the upper wall.
//
// The box here is square, so the tunnel is the middle third of it and the rest is solid. The
// inlet is a strip of cells held at the inflow state; the outlet is open (the flow leaves
// supersonically). No correction is applied at the corner of the step, so, as in most modern
// codes, a thin spurious boundary layer runs along the top of the step.
export const STEP = { rho: 1.4, p: 1, u: 3, start: 0.6, height: 0.2, inlet: 0.03 };

export const forwardStep: Problem = {
  id: 'forward-step',
  label: 'Mach 3 wind tunnel with a step',
  blurb: 'Mach 3 flow meets a step in a wind tunnel: a bow shock stands ahead of it and reflects off the walls.',
  eos: 'ideal',
  eosParams: { gamma: 1.4 },
  bcLo: [OUTFLOW, REFLECT, PERIODIC],
  bcHi: [OUTFLOW, REFLECT, PERIODIC],
  gravity: 'off',
  G: 0,
  box: 3,
  tEnd: 4,
  checkpointDt: 0.2,
  supports2d: true,
  view: { field: 0, log: false, mode: 'slice' },
  init: () => (x, y) => {
    // Tunnel coordinates: x from the inlet, y from the floor.
    const tx = x + 1.5;
    const ty = y + 0.5;
    const solid = ty < 0 || ty > 1 || (tx > STEP.start && ty < STEP.height);
    return { rho: STEP.rho, vx: solid ? 0 : STEP.u, vy: 0, vz: 0, p: STEP.p, s: 0, solid, fixed: !solid && tx < STEP.inlet };
  },
};
