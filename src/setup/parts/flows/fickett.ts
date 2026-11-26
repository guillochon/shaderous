// Fickett's piston-driven detonation (src/reference/converging.ts fickettExact, ehepExact),
// in two flows with exact solutions.
//
// `mader-detonation`: the Mader problem, a slab of explosive along x detonated from a wall at
// rest at the box's lower x face (ExactPack's mader). The exact solution burns instantly at the
// Chapman-Jouguet state. Here the explosive burns by the single-step Arrhenius reaction
// (burn/burn.ts) with a rate fast enough that its reaction zone is far thinner than a cell;
// cold explosive (p = 0) never burns. The energy released is the q that makes D the CJ speed,
// D^2 / (2 (gamma^2 - 1)). The run starts at time t0 of the exact solution, the detonation
// already D t0 from the wall, so that nothing need start it; its clock reads t - t0.
//
// `ehep-escape`: the escape of the products (ExactPack's ehep), which a piston drove a
// detonation through, once all the explosive has burned (t0 at least L / D): the products
// alone, with no burning and no species, so that the piston can be a moving obstacle (moving
// bodies do not carry species). What it tests is what follows: the escape into the vacuum and
// the rarefaction that runs back to the piston and reflects from it.
import type { FlowDef } from '../flow';
import { cjState, ehepExact, fickettExact } from '../../../reference/converging';
import { gammaOf } from './converging';

interface Params {
  D: number;
  rho0: number;
  t0: number;
  ea: number;
  k: number;
  shockburn: boolean;
}

const mader: FlowDef<Params> = {
  name: 'mader-detonation',
  label: 'Mader detonation',
  tip: 'The Mader problem: a slab of explosive along x detonated from a wall at rest at the box’s lower x face (Fickett’s piston-driven detonation with the piston at rest). Behind the detonation, running at its Chapman–Jouguet speed, a Taylor wave brings the products to rest half way back to the wall. The explosive burns by a fast Arrhenius reaction releasing the energy that makes D the CJ speed. The run starts from the exact solution at a time t0, the detonation already under way. Needs the species fuel and ash',
  wholeBox: true,
  requires: [{ fact: 'species.fuelAsh' }],
  fields: {
    D: { label: 'Detonation speed', default: 0.8, min: 1e-3, max: 1e3, factor: 1.25, tip: 'The Chapman–Jouguet speed: the energy released is D^2 / (2 (gamma^2 - 1)) per unit mass' },
    rho0: { label: 'Explosive density', default: 1.875, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the unburned explosive, at rest and cold' },
    t0: { label: 'Start time', default: 0.25, min: 1e-3, max: 1e3, factor: 1.25, tip: 'The time of the exact solution at which the run starts: the detonation is then D t0 from the wall' },
    ea: { label: 'Activation temperature', default: 0.05, min: 0, max: 1e3, factor: 1.25, tip: 'Of the burn, in p / rho: well below the temperature behind the shock, so that the explosive burns as soon as it is shocked, and never while cold' },
    k: { label: 'Burn rate', default: 2000, min: 1e-3, max: 1e9, factor: 2, tip: 'The reaction’s rate constant: the reaction zone is about D / k long, which should be far less than a cell' },
    shockburn: { label: 'Burn inside shocks', default: true, tip: 'Let cells the shock detector flags burn. With a reaction zone far thinner than a cell the front is all shock; kept from burning there, the detonation ran 1.3% slow' },
  },
  burn: (ctx, p) => ({ fuel: 'fuel', ash: 'ash', q: cjState(gammaOf(ctx), p.D, p.rho0).q, ea: p.ea, k: p.k, inShocks: p.shockburn }),
  issues: (ctx) => ((ctx.spec.physics?.eos ?? 'ideal') !== 'ideal' ? 'the Mader detonation is for an ideal gas' : null),
  init: (ctx, p) => {
    const gamma = gammaOf(ctx);
    const wall = -ctx.box / 2;
    const q = { D: p.D, rho0: p.rho0, up: 0 };
    const cold = 1e-9 * p.rho0 * p.D * p.D;
    return (x) => {
      const w = fickettExact(gamma, q, Math.max(x - wall, 1e-12), p.t0);
      if (w.what === 'fuel') return { rho: p.rho0, vx: 0, vy: 0, vz: 0, p: cold, s: 0, X: [1, 0] };
      return { rho: w.rho, vx: w.u, vy: 0, vz: 0, p: w.p, s: 1, X: [0, 1] };
    };
  },
};

interface EscapeParams {
  D: number;
  rho0: number;
  up: number;
  at: number;
  length: number;
  start: number;
  vacuum: number;
}

const escape: FlowDef<EscapeParams> = {
  name: 'ehep-escape',
  label: 'Escape of detonation products (EHEP)',
  tip: 'The escape of detonation products (Fickett and Rivard 1974; gamma = 3): a piston drove a detonation through a slab of explosive with vacuum beyond it. The run starts from the exact solution once the explosive is all burned, at a time of at least length / D: the products escape into the vacuum, their edge moving at D, and a rarefaction runs back to the piston and reflects from it. The piston is an obstacle that slides at its speed, from its face’s position at the start',
  wholeBox: true,
  fields: {
    D: { label: 'Detonation speed', default: 0.85, min: 1e-3, max: 1e3, factor: 1.25, tip: 'The Chapman–Jouguet speed of the detonation that burned the explosive' },
    rho0: { label: 'Explosive density', default: 1.6, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Density of the explosive before it burned' },
    up: { label: 'Piston speed', default: 0.05, min: 0, max: 1e3, factor: 1.25, tip: 'Speed of the piston, below D / 4' },
    at: { label: 'Piston face', default: 0.2, min: 0, max: 1e3, step: 0.01, tip: 'Where the piston’s face was at time 0 of the solution, as a distance from the box’s lower x face (at the start it is the piston speed times the start time further on)' },
    length: { label: 'Explosive length', default: 1, min: 1e-3, max: 1e3, step: 0.01, tip: 'Length of the explosive, with vacuum beyond it' },
    start: { label: 'Start', default: 1.2, min: 1, max: 5, step: 0.05, tip: 'When in the solution the run starts, in units of the time the detonation takes to burn the explosive, length / D: at least 1' },
    vacuum: { label: 'Vacuum density', default: 1e-6, min: 1e-12, max: 1e-2, factor: 10, tip: 'The density of the void beyond the products, as a fraction of the explosive’s' },
  },
  issues: (ctx, p) => {
    if ((ctx.spec.physics?.eos ?? 'ideal') !== 'ideal' || Math.abs(gammaOf(ctx) - 3) > 1e-9) return 'the escape of the products has its exact solution for an ideal gas with gamma = 3 only';
    if (!(p.up < p.D / 4)) return 'the piston must be slower than the gas behind the detonation, D / 4';
    return null;
  },
  init: (ctx, p) => {
    const face = -ctx.box / 2 + p.at;
    const q = { D: p.D, rho0: p.rho0, up: p.up, L: p.length };
    const vac = { rho: p.vacuum * p.rho0, vx: 0, vy: 0, vz: 0, p: 1e-9 * p.vacuum * p.rho0 * p.D * p.D, s: 0 };
    return (x) => {
      const s = x - face;
      const t0 = (p.start * p.length) / p.D;
      const w = ehepExact(q, Math.max(s, p.up * t0 * (1 + 1e-9) + 1e-12), t0);
      if (w.what !== 'products' || w.rho < p.vacuum * p.rho0) return vac;
      return { rho: w.rho, vx: w.u, vy: 0, vz: 0, p: w.p, s: 1 };
    };
  },
};

export const flows = [mader, escape];
