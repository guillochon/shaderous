import { all, OUTFLOW, type Problem } from './types';
import type { EosKind } from '../eos/eos';

// Sedov-Taylor similarity constant: R(t) = xi0 (E t^2 / rho)^(1/5).
export const SEDOV_XI0: Record<string, number> = { '1.4': 1.033, '1.6667': 1.152 };

function blast(id: string, label: string, eos: EosKind, blurb: string): Problem {
  return {
    id,
    label,
    blurb,
    eos,
    eosParams: { gamma: 1.4, R: 1, arad: 30 },
    bcLo: all(OUTFLOW),
    bcHi: all(OUTFLOW),
    gravity: 'off',
    G: 0,
    box: 1,
    tEnd: 0.06,
    checkpointDt: 0.005,
    pFloor: 1e-12,
    view: { field: 0, log: false, mode: 'slice' },
    supports2d: true,
    init: (n, e, ndim) => {
      // Unit energy spread uniformly over the cells within 3.5 dx of the center. In 2D this is
      // energy per unit length along z: a cylindrical blast.
      const dx = 1 / n;
      const r0 = 3.5 * dx;
      let count = 0;
      for (let k = ndim === 3 ? -4 : 0; k < (ndim === 3 ? 4 : 1); k++)
        for (let j = -4; j < 4; j++)
          for (let i = -4; i < 4; i++)
            if (Math.hypot(i + 0.5, j + 0.5, ndim === 3 ? k + 0.5 : 0) * dx < r0) count++;
      const eDep = 1 / (count * dx ** ndim);
      const pHot = e.pressure(1, eDep);
      return (x, y, z) => {
        const hot = Math.hypot(x, y, z) < r0;
        return { rho: 1, vx: 0, vy: 0, vz: 0, p: hot ? pHot : 1e-5, s: hot ? 1 : 0 };
      };
    },
  };
}

export const sedov = blast('sedov', 'Sedov-Taylor blast', 'ideal', 'Point explosion in a cold uniform medium, gamma = 1.4.');
export const sedovRad = blast(
  'sedov-rad',
  'Radiation-dominated blast',
  'gas_radiation',
  'The same explosion with the gas + radiation equation of state; the hot bubble is radiation dominated.',
);
