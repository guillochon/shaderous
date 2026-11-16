// Detonations of the single-step Arrhenius model (burn/burn.ts), the standard tests of
// detonation numerics, in its standard units and parameters: fuel at rest with density and
// pressure 1, gamma 1.2, heat release q = 50 (Chapman-Jouguet speed 6.809), and the rate
// constant set so that half of the fuel burns within one unit of the shock in the steady
// (ZND) detonation. The box is measured in those half-reaction lengths.
//
// A run starts from the exact ZND detonation moving toward +x and should keep it. In the lab
// frame the fuel is at rest and the shock a quarter of the way along x: there the validation
// measures the front's speed. In the frame of the steady front the fuel flows in from +x at the
// detonation speed (supersonically, so an outflow boundary is a clean inflow), and the front
// stays in the box for as long as its pulsations or cells take to develop:
//   Ea = 25   just below the neutral stability of Ea = 25.26 (Lee & Stewart 1990): the front
//             runs at the CJ speed with the ZND profile behind it.
//   Ea = 27   the lead shock's pressure oscillates about the ZND spike of 42.1 in a limit cycle
//             (Henrick, Aslam & Powers 2006 compute it to high accuracy).
//   pocket    2D, Ea = 27: a pocket of unburned gas behind the front breaks it into transverse
//             waves and triple points, which keep running along it and reflect off the walls
//             (at Ea = 20 the front goes back to planar).
import { cjSpeed, znd } from '../../../burn/burn';
import type { FlowDef } from '../flow';

const GAMMA = 1.2;
const Q = 50;
const D_CJ = cjSpeed(GAMMA, Q);

/** The steady detonation of a run's parameters (cached: init and burn both ask). */
const cache = new Map<string, ReturnType<typeof znd>>();
export function zndOf(params: { ea: number; overdrive: number }): ReturnType<typeof znd> {
  const key = `${params.ea}/${params.overdrive}`;
  let z = cache.get(key);
  if (!z) {
    z = znd(GAMMA, Q, params.ea, Math.sqrt(params.overdrive) * D_CJ);
    cache.set(key, z);
  }
  return z;
}

interface Params {
  ea: number;
  overdrive: number;
  shockburn: boolean;
  frontFrame: boolean;
  pocket: boolean;
}

const detonation: FlowDef<Params> = {
  name: 'znd-detonation',
  label: 'ZND detonation',
  tip: 'The steady (ZND) detonation of the single-step Arrhenius model, moving toward +x into fuel of density 1 and pressure 1 (for gamma = 1.2; heat release 50), with its burning. Lengths are half-reaction lengths. Needs the species fuel and ash.',
  wholeBox: true,
  fields: {
    ea: { label: 'Activation temperature Ea', default: 25, min: 5, max: 50, step: 0.5, tip: 'Activation temperature of the reaction, in units of the unburned gas temperature. Above 25.26 a planar detonation is unstable and pulses' },
    overdrive: { label: 'Overdrive (D / D_CJ)²', default: 1, min: 1, max: 2, step: 0.05, tip: 'Square of the detonation speed over the Chapman–Jouguet speed: 1 is a self-sustained detonation; above it a piston behind would drive it faster' },
    shockburn: { label: 'Burn inside shocks', default: true, tip: 'Let cells the shock detector flags burn. Suppressing it, as for fronts too thin to resolve, costs a resolved detonation speed: about 5% short of Chapman–Jouguet at 13 cells per half-reaction length, an error that halves with each doubling of resolution' },
    frontFrame: { label: 'Frame of the front', default: false, tip: 'Run in the frame of the steady front, the shock held three-quarters of the way along x and the fuel flowing in from +x. Off: the lab frame, the fuel at rest and the shock a quarter of the way along x' },
    pocket: { label: 'Pocket of unburned gas', default: false, tip: 'A pocket of unburned gas of radius 1 just behind the front, off the axis: it breaks the front into the transverse waves of a cellular detonation' },
  },
  burn: (_ctx, p) => ({ fuel: 'fuel', ash: 'ash', q: Q, ea: p.ea, k: zndOf(p).K, inShocks: p.shockburn }),
  init: (ctx, p) => {
    const length = ctx.box;
    const ndim = ctx.ndim;
    const z = zndOf(p);
    // In the front's frame: the shock three-quarters of the way along x, held there.
    const shockFrame = p.frontFrame;
    const shock = (shockFrame ? 0.25 : -0.25) * length;
    const frame = shockFrame ? Math.sqrt(p.overdrive) * D_CJ : 0;
    // The pocket of the cellular run: unburned gas just behind the front, off the axis.
    const pocket = p.pocket ? { x: shock - 1.5, y: 0.2 * length, r: 1 } : null;
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

export const flows = [detonation];
export { D_CJ as DETONATION_CJ_SPEED };
