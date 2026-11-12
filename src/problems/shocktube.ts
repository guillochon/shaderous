import { all, OUTFLOW, type Problem } from './types';

export interface TubeState {
  rho: number;
  u: number;
  p: number;
}

/** Riemann problem with the discontinuity through the box center, normal to `dir`. */
export function shockTube(
  id: string,
  label: string,
  dir: [number, number, number],
  left: TubeState,
  right: TubeState,
  tEnd: number,
  gamma = 1.4,
): Problem & { dir: [number, number, number]; left: TubeState; right: TubeState } {
  const norm = Math.hypot(...dir);
  const d = dir.map((c) => c / norm) as [number, number, number];
  return {
    id,
    label,
    blurb: `Shock tube, gamma = ${gamma}, normal (${dir.join(', ')}).`,
    eos: 'ideal',
    eosParams: { gamma },
    bcLo: all(OUTFLOW),
    bcHi: all(OUTFLOW),
    gravity: 'off',
    G: 0,
    box: 1,
    tEnd,
    checkpointDt: tEnd / 10,
    pFloor: 1e-12,
    supports2d: dir[2] === 0,
    view: { mode: 'slice', log: false },
    dir: d,
    left,
    right,
    init: () => (x, y, z) => {
      const s = x * d[0] + y * d[1] + z * d[2];
      const st = s < 0 ? left : right;
      return { rho: st.rho, vx: st.u * d[0], vy: st.u * d[1], vz: st.u * d[2], p: st.p, s: s < 0 ? 1 : 0 };
    },
  };
}

const SOD_L = { rho: 1, u: 0, p: 1 };
const SOD_R = { rho: 0.125, u: 0, p: 0.1 };

export const sodX = shockTube('sod-x', 'Sod shock tube (x)', [1, 0, 0], SOD_L, SOD_R, 0.2);
export const sodY = shockTube('sod-y', 'Sod shock tube (y)', [0, 1, 0], SOD_L, SOD_R, 0.2);
export const sodZ = shockTube('sod-z', 'Sod shock tube (z)', [0, 0, 1], SOD_L, SOD_R, 0.2);
// Shorter run so the outflow boundaries, which are not aligned with the flow, stay out of the solution.
export const sodDiag = shockTube('sod-diag', 'Sod shock tube (diagonal)', [1, 1, 1], SOD_L, SOD_R, 0.1);
// Pressure ratio 10^5 (Toro's test 3).
export const strongShock = shockTube(
  'strong-shock',
  'Strong shock tube (pressure ratio 1e5)',
  [1, 0, 0],
  { rho: 1, u: 0, p: 1000 },
  { rho: 1, u: 0, p: 0.01 },
  0.012,
);
