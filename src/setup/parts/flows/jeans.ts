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

const jeans: FlowDef<{ amp: number; modes: number }> = {
  name: 'jeans',
  label: 'Jeans-unstable wave',
  tip: 'A small adiabatic density wave along x about density 1 and sound speed 1. With one wavelength across a unit box and G = 2 pi it grows at the rate 2 pi; a shorter wave grows more slowly, or oscillates as sound once k^2 passes 4 pi G.',
  wholeBox: true,
  fields: {
    amp: { label: 'Amplitude', default: JEANS.amp, min: 1e-6, max: 0.5, factor: 2, tip: 'Relative amplitude of the density wave: small, for linear growth' },
    modes: { label: 'Wavelengths across', default: 1, min: 1, max: 32, step: 1, tip: 'Wavelengths of the wave along x in a length of 1' },
  },
  init: (ctx, q) => {
    const e = ctx.eos;
    const k = JEANS.k * Math.round(q.modes);
    return (x) => {
      const g = e.params.gamma;
      const d = q.amp * Math.cos(k * x);
      // Adiabatic perturbation about rho0 = 1, p0 = 1 / gamma (c_s = 1).
      return { rho: 1 + d, vx: 0, vy: 0, vz: 0, p: (1 + g * d) / g, s: 0.5 + 0.5 * Math.cos(k * x) };
    };
  },
};

export const flows = [jeans];
