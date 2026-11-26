// Three of Coggeshall's similarity solutions of the Euler equations (Coggeshall 1991, as
// ExactPack has them: cog1, cog2, cog6; src/reference/converging.ts coggeshallExact), in planar,
// cylindrical or spherical symmetry, with their open faces held at the exact solution. The run
// starts at time t0 of the solution (the first two are singular at t = 0; Kidder's start is a
// fraction of its collapse time), so its clock reads the solution's time less t0. ExactPack's
// Grüneisen Gamma times its temperature is the pressure over the density.
import type { FlowContext, FlowDef } from '../flow';
import { coggeshallExact, kidderGamma, type CogCase } from '../../../reference/converging';
import { f32, gammaOf, heldFaces, holdField, holdIssue, radial } from './converging';

const symmetryField = {
  label: 'Symmetry',
  default: 3,
  min: 1,
  max: 3,
  step: 1,
  tip: 'What the flow is symmetric about: 3 a point (spherical), 2 a line (cylindrical), 1 a plane (planar), as for the Noh implosion',
};
const geometryOf = (s: number) => Math.min(3, Math.max(1, Math.round(s))) as 1 | 2 | 3;

interface CogValues {
  symmetry: number;
  rho0: number;
  b: number;
  hold: boolean;
  [k: string]: number | boolean;
}

/** When in the solution a run starts: t0, or Kidder's `start` times its collapse time. */
export const startOf = (q: Record<string, number | boolean>): number => (typeof q.start === 'number' ? q.start * (q.tau as number) : (q.t0 as number));

/** A Coggeshall flow: its state at its start everywhere, its open faces held at the solution. */
function cogFlow<P extends CogValues>(c: CogCase, def: Omit<FlowDef<P>, 'init' | 'damping' | 'wholeBox'>, wgsl: (ctx: FlowContext, q: P, g: 1 | 2 | 3, gamma: number) => string): FlowDef<P> {
  const params = (q: P) => ({ rho0: q.rho0, b: q.b, T0: q.T0 as number | undefined, Gamma: q.Gamma as number | undefined, tau: q.tau as number | undefined });
  return {
    ...def,
    wholeBox: true,
    init: (ctx, q) => {
      const at = radial(ctx, q.symmetry);
      const g = geometryOf(q.symmetry);
      const gamma = gammaOf(ctx);
      return (x, y, z) => {
        const { d, n } = at(x, y, z);
        const w = coggeshallExact(c, g, gamma, params(q), d, startOf(q));
        return { rho: w.rho, vx: w.u * n[0] + 0, vy: w.u * n[1] + 0, vz: w.u * n[2] + 0, p: w.p, s: 0 };
      };
    },
    damping: (ctx, q) => (q.hold ? heldFaces(ctx, q.symmetry, wgsl(ctx, q, geometryOf(q.symmetry), gammaOf(ctx))) : undefined),
  };
}

const common = {
  symmetry: symmetryField,
  rho0: { label: 'Density coefficient', default: 1.8, min: 1e-6, max: 1e6, factor: 1.25, tip: 'rho0: the density is rho0 r^b times a power of time' },
  b: { label: 'Density exponent', default: 1.2, min: -0.9, max: 4, step: 0.1, tip: 'b: the density goes as r^b' },
  hold: holdField,
};

const cog1 = cogFlow<CogValues & { T0: number; Gamma: number; t0: number }>(
  1,
  {
    name: 'coggeshall-1',
    label: 'Coggeshall 1: free expansion',
    tip: 'Gas expanding homologously, u = r / t, with its density a power of r and its pressure uniform, so that nothing pushes it: each element coasts along its adiabat (Coggeshall 1991, problem 1). The pressure falls as t^(-gamma k) for symmetry k',
    fields: {
      ...common,
      T0: { label: 'Temperature coefficient', default: 1.4, min: 1e-6, max: 1e6, factor: 1.25, tip: 'T0: with the Grüneisen parameter, the pressure is Gamma rho0 T0 t^(-gamma k)' },
      Gamma: { label: 'Grüneisen parameter', default: 40, min: 1e-6, max: 1e6, factor: 1.25, tip: 'Gamma: pressure over density times temperature' },
      t0: { label: 'Start time', default: 1, min: 1e-3, max: 1e3, factor: 1.25, tip: 'The time of the solution at which the run starts (it is singular at 0)' },
    },
    issues: (ctx, q) => holdIssue(ctx, q.hold),
  },
  (_ctx, q, g, gamma) => `fn held_radial(d: f32, t: f32) -> vec3<f32> {
  let T = t + ${f32(q.t0)};
  return vec3<f32>(${f32(q.rho0)} * pow(max(d, 1e-30), ${f32(q.b)}) * pow(T, ${f32(-q.b - g)}), d / T, ${f32(q.Gamma * q.rho0 * q.T0)} * pow(T, ${f32(-gamma * g)}));
}`,
);

const cog2 = cogFlow<CogValues & { t0: number }>(
  2,
  {
    name: 'coggeshall-2',
    label: 'Coggeshall 2: expansion against pressure',
    tip: 'Gas expanding homologously, u proportional to r / t, its density a power of r and its pressure rising outward as r^(b + 2), so that it slows as it goes (Coggeshall 1991, problem 2)',
    fields: { ...common, t0: { label: 'Start time', default: 1, min: 1e-3, max: 1e3, factor: 1.25, tip: 'The time of the solution at which the run starts (it is singular at 0)' } },
    issues: (ctx, q) => holdIssue(ctx, q.hold),
  },
  (_ctx, q, g, gamma) => {
    const c = 2 + (gamma - 1) * g;
    return `fn held_radial(d: f32, t: f32) -> vec3<f32> {
  let T = t + ${f32(q.t0)};
  let rho = ${f32(q.rho0)} * pow(max(d, 1e-30), ${f32(q.b)}) * pow(T, ${f32((-2 * (q.b + g)) / c)});
  let s = d / T;
  return vec3<f32>(rho, ${f32(2 / c)} * s, rho * ${f32((2 * (gamma - 1) * g) / ((q.b + 2) * c * c))} * s * s);
}`;
  },
);

const cog6 = cogFlow<CogValues & { tau: number; start: number }>(
  6,
  {
    name: 'coggeshall-6',
    label: 'Coggeshall 6: Kidder compression',
    tip: 'Kidder’s isentropic compression (Coggeshall 1991, problem 6): gas falling inward homologously, u = -r t / (tau^2 - t^2), squeezed without a shock so that all of it reaches the center at t = tau. Needs gamma = 5/3 in spherical symmetry, 2 in cylindrical and 3 in planar',
    fields: {
      ...common,
      tau: { label: 'Collapse time', default: 1.25, min: 1e-3, max: 1e3, factor: 1.25, tip: 'tau: when all the gas reaches the center' },
      start: { label: 'Start', default: 0, min: 0, max: 0.95, step: 0.05, tip: 'When in the solution the run starts, as a fraction of the collapse time: at 0 the gas starts at rest' },
    },
    issues: (ctx, q) => {
      const g = geometryOf(q.symmetry);
      const want = kidderGamma(g);
      const have = gammaOf(ctx);
      if (Math.abs(have - want) > 1e-6) return `Kidder’s compression in this symmetry needs gamma = ${+want.toFixed(4)}, not ${+have.toFixed(4)}`;
      return holdIssue(ctx, q.hold);
    },
  },
  (_ctx, q, g) => `fn held_radial(d: f32, t: f32) -> vec3<f32> {
  let T = t + ${f32(q.start * q.tau)};
  let x1 = max(${f32(q.tau * q.tau)} - T * T, 1e-12);
  let rho = ${f32(q.rho0)} * pow(max(d, 1e-30), ${f32(q.b)}) * pow(x1, ${f32(-(g + q.b) / 2)});
  let s = ${f32(q.tau)} * d / x1;
  return vec3<f32>(rho, -d * T / x1, rho * s * s / ${f32(q.b + 2)});
}`,
);

export const flows = [cog1, cog2, cog6];
