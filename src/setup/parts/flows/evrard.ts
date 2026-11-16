// Evrard (1988) collapse: cold gas sphere with rho = M / (2 pi R^2 r), specific internal
// energy 0.05 G M / R, G = M = R = 1. It collapses, bounces through a strong accretion
// shock and settles toward virial equilibrium.
import type { FlowDef } from '../flow';

const evrard: FlowDef<{ mass: number; radius: number; u: number; rhoAmb: number }> = {
  name: 'evrard',
  label: 'Evrard sphere',
  tip: 'A cold gas sphere whose density falls as 1/r, rho = M / (2 pi R^2 r), in thin gas at the same specific internal energy (for gamma = 5/3). With G = M = R = 1 it collapses in about a unit of time.',
  wholeBox: true,
  fields: {
    mass: { label: 'Mass', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Mass of the sphere' },
    radius: { label: 'Radius', default: 1, min: 0.05, max: 1e3, factor: 1.25, tip: 'Radius of the sphere, in code lengths: keep it well inside the box' },
    u: { label: 'Specific internal energy', default: 0.05, min: 0, max: 100, factor: 1.25, tip: 'Internal energy per unit mass of the gas, in units of G M / R when those are 1: 0.05 is cold, and the sphere collapses' },
    rhoAmb: { label: 'Density outside', default: 1e-4, min: 1e-12, max: 1, factor: 10, tip: 'Density of the thin gas around the sphere' },
  },
  init: (ctx, q) => {
    const dx = ctx.box / ctx.n;
    const { mass: M, radius: R, u, rhoAmb } = q;
    return (x, y, z) => {
      const r = Math.max(Math.hypot(x, y, z), 0.5 * dx);
      const rho = r < R ? M / (2 * Math.PI * R * R * r) : rhoAmb;
      return { rho, vx: 0, vy: 0, vz: 0, p: (2 / 3) * rho * u, s: r < R ? 1 : 0 };
    };
  },
};

export const flows = [evrard];
