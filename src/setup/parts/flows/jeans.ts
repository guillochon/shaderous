// Jeans instability: one unstable mode along x in a periodic box.
import type { FlowDef } from '../flow';

// With rho0 = 1, c_s = 1 and k = 2 pi, G is chosen so that 4 pi G rho0 = 2 k^2 c_s^2, giving a
// growth rate of k c_s.
export const JEANS = {
  amp: 1e-3,
  k: 2 * Math.PI,
  G: (2 * (2 * Math.PI) ** 2) / (4 * Math.PI),
  growth: 2 * Math.PI,
};

const jeans: FlowDef<Record<string, never>> = {
  name: 'jeans',
  label: 'Jeans-unstable wave',
  tip: 'A small adiabatic density wave along x about density 1 and sound speed 1, one wavelength across a unit box. With G = 2 pi it grows at the rate 2 pi.',
  wholeBox: true,
  fields: {},
  init: (ctx) => {
    const e = ctx.eos;
    return (x) => {
      const g = e.params.gamma;
      const d = JEANS.amp * Math.cos(JEANS.k * x);
      // Adiabatic perturbation about rho0 = 1, p0 = 1 / gamma (c_s = 1).
      return { rho: 1 + d, vx: 0, vy: 0, vz: 0, p: (1 + g * d) / g, s: 0.5 + 0.5 * Math.cos(JEANS.k * x) };
    };
  },
};

export const flows = [jeans];
