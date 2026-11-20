// Slow flows, for the tests and presets of nearly incompressible flow (plans/incompressible.md):
// the Gresho vortex, the Taylor-Green vortex and the doubly periodic double shear layer. Each has
// flow speeds of order 1 and takes the sound speed from the equation of state: a gas with an
// energy is given the pressure that makes the fastest flow's Mach number `mach`; a gas without
// one (barotropic, liquid) has its own sound speed, and `mach` is not used.
import type { Eos } from '../../../eos/eos';
import type { FlowDef, FlowField } from '../flow';

const MACH: FlowField = {
  label: 'Mach number', default: 0.1, min: 1e-4, max: 1, factor: 1.25,
  tip: 'Mach number of the fastest flow (speed 1), for a gas with an energy: it sets the pressure the flow sits in. A barotropic gas or a liquid has its own sound speed, and this is not used',
};

/** The pressure at which gas of density 1 has sound speed c (bisection in the logarithm). */
export function pressureForSoundSpeed(eos: Eos, c: number): number {
  let lo = -60;
  let hi = 60;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (eos.soundSpeed(1, Math.exp(mid)) < c) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

/**
 * The background a slow flow sits in, and the state of a point of it from its velocity and the
 * pressure `dp` its balance asks for above the background's (at density 1). A gas with an
 * energy has density 1 and that pressure. A gas without one has the density whose enthalpy is
 * `dp` above the background's, int c^2 drho / rho = dp: for a steady flow that is its exact
 * balance, whatever the sound speed.
 */
function slowGas(eos: Eos, mach: number): (dp: number) => { rho: number; p: number } {
  if (eos.hasEnergy) {
    const p0 = pressureForSoundSpeed(eos, 1 / mach);
    return (dp) => ({ rho: 1, p: p0 + dp });
  }
  // The density of zero pressure difference: a liquid's reference density, else 1.
  const rho0 = eos.kind === 'liquid' ? eos.params.liqRho : 1;
  const slope = (rho: number) => rho / eos.soundSpeed(rho, eos.pressure(rho, 0)) ** 2;
  return (dh) => {
    // drho / dh = rho / c^2, by eight Runge-Kutta steps (dh is of order the Mach number squared).
    let rho = rho0;
    const h = dh / 8;
    for (let i = 0; i < 8; i++) {
      const k1 = slope(rho);
      const k2 = slope(rho + 0.5 * h * k1);
      const k3 = slope(rho + 0.5 * h * k2);
      const k4 = slope(rho + h * k3);
      rho += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    return { rho, p: eos.pressure(rho, 0) };
  };
}

/** The Gresho vortex's speed and pressure above the background at radius r (in a unit box). */
export function greshoProfile(r: number): { u: number; dp: number } {
  if (r < 0.2) return { u: 5 * r, dp: 12.5 * r * r };
  if (r < 0.4) return { u: 2 - 5 * r, dp: 12.5 * r * r + 4 * (1 - 5 * r - Math.log(0.2) + Math.log(r)) };
  return { u: 0, dp: -2 + 4 * Math.log(2) };
}

const gresho: FlowDef<{ mach: number }> = {
  name: 'gresho',
  label: 'Gresho vortex',
  tip: 'A steady vortex (Gresho & Chan 1990): the gas turns rigidly out to a fifth of the box, slows to rest by two fifths, and the pressure balances the centrifugal force. Nothing should change; what does is the solver’s own dissipation. In 3D it is a column along z.',
  wholeBox: true,
  fields: { mach: MACH },
  init: (ctx, q) => {
    const gas = slowGas(ctx.eos, q.mach);
    return (x, y) => {
      const r = Math.hypot(x, y) / ctx.box;
      const { u, dp } = greshoProfile(r);
      const c = r > 0 ? [-y / (r * ctx.box), x / (r * ctx.box)] : [0, 0];
      return { ...gas(dp), vx: u * c[0], vy: u * c[1], vz: 0, s: r < 0.2 ? 1 : 0 };
    };
  },
};

const taylorGreen: FlowDef<{ mach: number; modes: number }> = {
  name: 'taylor-green',
  label: 'Taylor-Green vortex',
  tip: 'An array of counter-rotating vortices. In 2D with viscosity it keeps its shape and decays as exp(-2 ν k² t), an exact solution of the incompressible equations; in 3D the same start, varying along z, breaks down into turbulence.',
  wholeBox: true,
  fields: {
    mach: MACH,
    modes: { label: 'Wavelengths across', default: 1, min: 1, max: 16, step: 1, tip: 'Wavelengths of the vortex array across the box' },
  },
  init: (ctx, q) => {
    const gas = slowGas(ctx.eos, q.mach);
    const k = (2 * Math.PI * q.modes) / ctx.box;
    return (x, y, z) => {
      const cz = ctx.ndim === 3 ? Math.cos(k * z) : 1;
      const dp = ctx.ndim === 3 ? ((Math.cos(2 * k * x) + Math.cos(2 * k * y)) * (Math.cos(2 * k * z) + 2)) / 16 : (Math.cos(2 * k * x) + Math.cos(2 * k * y)) / 4;
      return { ...gas(dp), vx: Math.sin(k * x) * Math.cos(k * y) * cz, vy: -Math.cos(k * x) * Math.sin(k * y) * cz, vz: 0, s: 0.5 + 0.5 * Math.sin(k * x) * Math.sin(k * y) };
    };
  },
};

const shearLayers: FlowDef<{ mach: number; sharp: number; kick: number }> = {
  name: 'shear-layers',
  label: 'Double shear layer',
  tip: 'Two shear layers in a doubly periodic box, the band between them moving at +1 and the rest at -1, with a small sideways kick (Bell, Colella & Glaz 1989): each layer rolls up into one vortex. Under-resolved, a scheme grows spurious vortices along the thin braids between them.',
  wholeBox: true,
  fields: {
    mach: MACH,
    sharp: { label: 'Layer sharpness', default: 30, min: 5, max: 400, factor: 1.25, tip: 'The layers’ inverse thickness, in box sides: 30 is the thick layer of the original test, 80 to 100 the thin one' },
    kick: { label: 'Kick amplitude', default: 0.05, min: 0, max: 0.5, step: 0.01, tip: 'Peak sideways velocity of the seeded mode, one wavelength across the box' },
  },
  init: (ctx, q) => {
    const gas = slowGas(ctx.eos, q.mach);
    return (x, y) => {
      const xs = x / ctx.box + 0.5;
      const ys = y / ctx.box + 0.5;
      const u = ys <= 0.5 ? Math.tanh(q.sharp * (ys - 0.25)) : Math.tanh(q.sharp * (0.75 - ys));
      return { ...gas(0), vx: u, vy: q.kick * Math.sin(2 * Math.PI * xs), vz: 0, s: 0.5 * (1 + u) };
    };
  },
};

export const flows = [gresho, taylorGreen, shearLayers];
