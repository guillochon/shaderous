import { all, PERIODIC, type Problem } from './types';

// Isothermal, self-gravitating periodic box seeded with a few long-wavelength velocity modes.
// About six Jeans lengths fit across the box, so the gas gathers into filaments and clumps.
// Isothermal collapse never stops on its own, so the Truelove pressure floor is on by default
// to hold the cores at the grid's Jeans scale; switch it off to watch the under-resolved-cell
// counter climb.
export const isothermalCollapse: Problem = {
  id: 'isothermal-collapse',
  label: 'Isothermal turbulent collapse',
  blurb: 'Barotropic (isothermal) gas with self-gravity fragmenting in a periodic box.',
  eos: 'barotropic',
  eosParams: { gamma: 1, K: 0.01 },
  bcLo: all(PERIODIC),
  bcHi: all(PERIODIC),
  gravity: 'periodic',
  G: 1,
  box: 1,
  tEnd: 0.9,
  rhoFloor: 1e-6,
  jeansFloor: true,
  view: { field: 0, log: true, mode: 'volume' },
  init: () => {
    // Deterministic pseudo-random phases.
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const modes: { k: number[]; a: number[]; ph: number }[] = [];
    for (let i = 0; i < 12; i++) {
      const k: number[] = [0, 0, 0].map(() => Math.floor(rnd() * 5) - 2);
      if (Math.hypot(...k) === 0) k[0] = 1;
      const kk = Math.hypot(...k);
      modes.push({ k, a: [0, 0, 0].map(() => (rnd() - 0.5) / kk ** 2), ph: rnd() * 2 * Math.PI });
    }
    const velocity = (x: number, y: number, z: number) => {
      const v = [0, 0, 0];
      for (const m of modes) {
        const s = Math.sin(2 * Math.PI * (m.k[0] * x + m.k[1] * y + m.k[2] * z) + m.ph);
        for (let d = 0; d < 3; d++) v[d] += m.a[d] * s;
      }
      return v;
    };
    // Normalise to an rms Mach number of 5 (c_s = 0.1).
    let sum = 0;
    for (let i = 0; i < 512; i++) {
      const v = velocity(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
      sum += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }
    const scale = 0.5 / Math.sqrt(sum / 512);
    return (x, y, z) => {
      const v = velocity(x, y, z);
      return { rho: 1, vx: scale * v[0], vy: scale * v[1], vz: scale * v[2], p: 0.01, s: 0.5 + 0.5 * Math.sin(2 * Math.PI * x) };
    };
  },
};
