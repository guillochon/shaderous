import { all, OUTFLOW, type Problem } from './types';

// Uniform-density sphere. As a static configuration it tests the Poisson solver against the
// analytic potential; left to run, it is a pressure-poor homologous collapse.
export const SPHERE = { R: 0.25, rho: 1, G: 1, rhoAmb: 1e-6 };

export function spherePotential(r: number): number {
  const M = (4 / 3) * Math.PI * SPHERE.R ** 3 * SPHERE.rho;
  return r < SPHERE.R
    ? (-SPHERE.G * M * (3 * SPHERE.R ** 2 - r * r)) / (2 * SPHERE.R ** 3)
    : (-SPHERE.G * M) / r;
}

export const sphere: Problem = {
  id: 'sphere',
  label: 'Uniform sphere collapse',
  blurb: 'Cold uniform sphere under its own gravity (isolated boundaries).',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(OUTFLOW),
  bcHi: all(OUTFLOW),
  gravity: 'isolated',
  G: SPHERE.G,
  box: 1,
  tEnd: Infinity,
  rhoFloor: 1e-8,
  pFloor: 1e-12,
  view: { field: 0, log: true, mode: 'volume' },
  init: (n) => {
    const dx = 1 / n;
    return (x, y, z) => {
      // Volume fraction inside the sphere, supersampled in cells that straddle the surface.
      const r = Math.hypot(x, y, z);
      let frac = r < SPHERE.R ? 1 : 0;
      if (Math.abs(r - SPHERE.R) < dx) {
        let hits = 0;
        for (let c = 0; c < 4; c++)
          for (let b = 0; b < 4; b++)
            for (let a = 0; a < 4; a++) {
              const o = (q: number) => ((q + 0.5) / 4 - 0.5) * dx;
              if (Math.hypot(x + o(a), y + o(b), z + o(c)) < SPHERE.R) hits++;
            }
        frac = hits / 64;
      }
      const rho = SPHERE.rhoAmb + frac * (SPHERE.rho - SPHERE.rhoAmb);
      return { rho, vx: 0, vy: 0, vz: 0, p: 1e-4, s: frac };
    };
  },
};
