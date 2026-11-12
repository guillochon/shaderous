import { MASS_FRACTION_FIELD, OUTFLOW, REFLECT, type Problem } from './types';
import { cjSpeed, znd } from '../burn/burn';

// Detonations of the single-step Arrhenius model (burn/burn.ts), the standard tests of
// detonation numerics, in its standard units and parameters: fuel at rest with density and
// pressure 1, gamma 1.2, heat release q = 50 (Chapman-Jouguet speed 6.809), and the rate
// constant set so that half of the fuel burns within one unit of the shock in the steady
// (ZND) detonation. The box is measured in those half-reaction lengths.
//
// Each run starts from the exact ZND detonation moving toward +x and should keep it. The stable
// run is in the lab frame (fuel at rest, the shock a quarter of the way along x), where the
// validation measures the front's speed; the others run in the frame of the steady front, the
// fuel flowing in from +x at the detonation speed (supersonically, so an outflow boundary is a
// clean inflow), so that the front stays in the box for as long as its pulsations or cells take
// to develop:
//   stable     Ea = 25, just below the neutral stability of Ea = 25.26 (Lee & Stewart 1990): the
//              front runs at the CJ speed with the ZND profile behind it.
//   pulsating  Ea = 27: the lead shock's pressure oscillates about the ZND spike of 42.1 in a
//              limit cycle (Henrick, Aslam & Powers 2006 compute it to high accuracy).
//   cellular   2D, Ea = 27: a pocket of unburned gas behind the front breaks it into transverse
//              waves and triple points, which keep running along it and reflect off the walls
//              (at Ea = 20 the front goes back to planar).

const GAMMA = 1.2;
const Q = 50;
const D_CJ = cjSpeed(GAMMA, Q);

type Variant = 'stable' | 'pulsating' | 'cellular';

function detonation(variant: Variant): Problem {
  const ea = variant === 'stable' ? 25 : 27;
  const length = variant === 'cellular' ? 30 : 40;
  // In the front's frame: the shock three-quarters of the way along x, held there.
  const shockFrame = variant !== 'stable';
  const label = { stable: 'Stable (Ea = 25)', pulsating: 'Pulsating (Ea = 27)', cellular: 'Cellular front (2D, Ea = 27)' }[variant];
  return {
    id: `detonation-${variant}`,
    label: `Detonation: ${label}`,
    blurb:
      {
        stable: 'A steady detonation of the single-step Arrhenius model (gamma 1.2, q = 50, Ea = 25): it should run at the Chapman–Jouguet speed, 6.809, keeping the ZND profile behind its von Neumann spike. Lengths are in half-reaction lengths: n / box cells resolve each.',
        pulsating: 'The same detonation with Ea = 27, past the neutral stability of Ea = 25.26: the lead shock pulses, its pressure swinging above and below the ZND spike of 42.1. The pulsations need a well-resolved reaction zone, a few tens of cells per half-reaction length, or they damp out (Henrick, Aslam & Powers 2006).',
        cellular: 'A planar detonation (Ea = 27) perturbed by a pocket of unburned gas: the front breaks into transverse waves whose triple points ride along it and reflect off the walls, the cellular structure of real detonations.',
      }[variant],
    eos: 'ideal',
    eosParams: { gamma: GAMMA },
    species: ['fuel', 'ash'],
    burn: (params) => ({ fuel: 'fuel', ash: 'ash', q: Q, ea: params.ea, k: zndOf(params).K, inShocks: params.shockburn > 0.5 }),
    // Open along x (burned gas leaves on the left), walls across.
    bcLo: [OUTFLOW, REFLECT, REFLECT],
    bcHi: [OUTFLOW, REFLECT, REFLECT],
    gravity: 'off',
    G: 0,
    box: length,
    // Lab frame: the front crosses half the box at the CJ speed.
    tEnd: shockFrame ? (variant === 'cellular' ? 40 : 60) : Math.ceil((0.5 * length) / D_CJ),
    supports2d: true,
    view: { field: MASS_FRACTION_FIELD, log: false, mode: 'slice', axis: 2 },
    params: [
      { id: 'ea', label: 'Activation temperature Ea', title: 'Activation temperature of the reaction, in units of the unburned gas temperature. Above 25.26 a planar detonation is unstable and pulses. Changing it starts the run again', value: ea, min: 5, max: 50, step: 0.5 },
      { id: 'overdrive', label: 'Overdrive (D / D_CJ)²', title: 'Square of the detonation speed over the Chapman–Jouguet speed: 1 is a self-sustained detonation; above it a piston behind would drive it faster. Changing it starts the run again', value: 1, min: 1, max: 2, step: 0.05 },
      { id: 'shockburn', label: 'Burn inside shocks', title: 'Let cells the shock detector flags burn (1) or not (0). Suppressing it, as for fronts too thin to resolve, costs a resolved detonation speed: about 5% short of Chapman–Jouguet at 13 cells per half-reaction length, an error that halves with each doubling of resolution. Changing it starts the run again', value: 1, min: 0, max: 1, step: 1 },
    ],
    init: (_n, _eos, ndim, params) => {
      const z = zndOf(params);
      const shock = (shockFrame ? 0.25 : -0.25) * length;
      const frame = shockFrame ? Math.sqrt(params.overdrive) * D_CJ : 0;
      // The pocket of the cellular run: unburned gas just behind the front, off the axis.
      const pocket = variant === 'cellular' ? { x: shock - 1.5, y: 0.2 * length, r: 1 } : null;
      const ahead = { rho: 1, vx: -frame, vy: 0, vz: 0, p: 1, s: 0, X: [1, 0] };
      return (x, y, zc) => {
        if (x > shock) return ahead;
        const w = z.at(shock - x);
        let X = [w.f, 1 - w.f];
        if (pocket) {
          const d2 = (x - pocket.x) ** 2 + (y - pocket.y) ** 2 + (ndim === 3 ? zc * zc : 0);
          if (d2 < pocket.r ** 2) X = [1, 0];
        }
        return { rho: w.rho, vx: w.u - frame, vy: 0, vz: 0, p: w.p, s: x > shock - 1 ? 1 : 0, X };
      };
    },
  };
}

/** The steady detonation of a run's parameters (cached: init and burn both ask). */
const cache = new Map<string, ReturnType<typeof znd>>();
export function zndOf(params: Record<string, number>): ReturnType<typeof znd> {
  const key = `${params.ea}/${params.overdrive}`;
  let z = cache.get(key);
  if (!z) {
    z = znd(GAMMA, Q, params.ea, Math.sqrt(params.overdrive) * D_CJ);
    cache.set(key, z);
  }
  return z;
}

export const detonationStable = detonation('stable');
export const detonationPulsating = detonation('pulsating');
export const detonationCellular = detonation('cellular');
export { D_CJ as DETONATION_CJ_SPEED };
