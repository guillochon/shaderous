// Evrard (1988) collapse: cold gas sphere with rho = M / (2 pi R^2 r), specific internal
// energy 0.05 G M / R, G = M = R = 1. It collapses, bounces through a strong accretion
// shock and settles toward virial equilibrium.
import type { FlowDef } from '../flow';

const evrard: FlowDef<Record<string, never>> = {
  name: 'evrard',
  label: 'Evrard sphere',
  tip: 'A cold gas sphere of mass 1 and radius 1 whose density falls as 1/r, with specific internal energy 0.05, in thin gas (density 1e-4). Made for a box of side 4 with G = 1.',
  wholeBox: true,
  fields: {},
  init: (ctx) => {
    const n = ctx.n;
    const dx = 4 / n;
    const u = 0.05;
    const rhoAmb = 1e-4;
    return (x, y, z) => {
      const r = Math.max(Math.hypot(x, y, z), 0.5 * dx);
      const rho = r < 1 ? 1 / (2 * Math.PI * r) : rhoAmb;
      return { rho, vx: 0, vy: 0, vz: 0, p: (2 / 3) * rho * u, s: r < 1 ? 1 : 0 };
    };
  },
};

export const flows = [evrard];
