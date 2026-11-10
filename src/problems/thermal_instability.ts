import { all, PERIODIC, type Problem } from './types';
import { ismLambda } from '../cooling/cooling';

// Thermal instability: interstellar gas in thermal balance at a temperature where that
// balance is unstable. Slightly denser gas cools faster than it is heated, is squeezed by its
// surroundings and cools further; the gas separates into cold clouds and a warm medium at
// nearly the same pressure.
//
// Units: density in hydrogen atoms per cm^3, temperature in units of 1000 K, box side 1. The
// heating rate sets the clock: the cooling time is about 2 / heating, against a sound-crossing
// time of the box of about 0.65, so structure forms on scales of a few percent of the box.
export const TI = { tUnit: 1000, modes: 48, kMax: 8 };

/** Temperature (code units) at which gas of density n neither heats nor cools, on the unstable branch. */
export function balanceTemperature(n: number): number {
  // The cooling function rises with temperature: bisect between 20 K and 20000 K.
  let lo = 20;
  let hi = 2e4;
  for (let i = 0; i < 60; i++) {
    const mid = Math.sqrt(lo * hi);
    if (n * ismLambda(mid) > 1) hi = mid;
    else lo = mid;
  }
  return Math.sqrt(lo * hi) / TI.tUnit;
}

// Small deterministic generator, so every run (and every resolution) starts from the same field.
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const thermalInstability: Problem = {
  id: 'thermal-instability',
  label: 'Thermal instability',
  blurb: 'Interstellar gas with cooling and heating, started in unstable balance: it separates into cold clouds and a warm medium.',
  eos: 'ideal',
  eosParams: { gamma: 5 / 3 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'off',
  G: 0,
  box: 1,
  tEnd: Infinity,
  supports2d: true,
  cooling: 'ism',
  coolParams: { heat: 50, tUnit: TI.tUnit, tFloor: 0.02 },
  view: { field: 0, log: true, mode: 'slice' },
  params: [
    { id: 'n0', label: 'Mean density (cm⁻³)', value: 2, min: 1, max: 8, step: 0.1 },
    { id: 'amp', label: 'Perturbation', value: 0.02, min: 0.001, max: 0.3, step: 0.001 },
  ],
  init: (_n, _eos, ndim, params) => {
    const T0 = balanceTemperature(params.n0);
    const p0 = params.n0 * T0;
    // A sum of plane waves with random directions and phases, periodic in the box.
    const rand = mulberry(20021);
    const waves: number[][] = [];
    while (waves.length < TI.modes) {
      const k = [0, 1, 2].map((a) => (a === 2 && ndim === 2 ? 0 : Math.round((2 * rand() - 1) * TI.kMax)));
      const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
      if (k2 === 0 || k2 > TI.kMax * TI.kMax) continue;
      waves.push([...k.map((q) => 2 * Math.PI * q), 2 * Math.PI * rand()]);
    }
    const norm = params.amp * Math.sqrt(2 / TI.modes);
    return (x, y, z) => {
      let d = 0;
      for (const w of waves) d += Math.cos(w[0] * x + w[1] * y + w[2] * z + w[3]);
      // Isobaric: a denser spot is a cooler one.
      const rho = params.n0 * (1 + norm * d);
      return { rho, vx: 0, vy: 0, vz: 0, p: p0, s: rho > params.n0 ? 1 : 0 };
    };
  },
};
