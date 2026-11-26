// Gas converging on a center, a line or a plane: the Noh implosion and the uniform collapse,
// two problems with exact solutions (src/reference/converging.ts; ExactPack's noh and noh2).
//
// `symmetry` says what the gas converges on: 3 a point (the origin), 2 a line, 1 a plane. In a
// Cartesian run the line is the z axis and the plane x = 0. In an axisymmetric (R, z) or a
// spherical (r, theta) run, whose setup coordinates are (R, z) with the axis along y, the line
// is that axis and the plane z = 0.
import type { FlowContext, FlowDef } from '../flow';

/** The direction away from what the gas converges on, and the distance to it. */
function radial(ctx: FlowContext, symmetry: number): (x: number, y: number, z: number) => { d: number; n: [number, number, number] } {
  const g = ctx.spec.domain?.geometry;
  const curved = g === 'cylindrical' || g === 'spherical';
  const k = Math.round(symmetry);
  const pick = (x: number, y: number, z: number): [number, number, number] =>
    k >= 3 ? [x, y, z] : k === 2 ? (curved ? [x, 0, z] : [x, y, 0]) : curved ? [0, y, 0] : [x, 0, 0];
  return (x, y, z) => {
    const v = pick(x, y, z);
    const d = Math.hypot(...v);
    // (+ 0: no negative zeros where the gas has no velocity)
    return { d, n: d > 0 ? (v.map((c) => c / d + 0) as [number, number, number]) : [0, 0, 0] };
  };
}

const symmetryField = {
  label: 'Symmetry',
  default: 3,
  min: 1,
  max: 3,
  step: 1,
  tip: 'What the gas converges on: 3 a point (spherical), 2 a line (cylindrical), 1 a plane (planar). In a Cartesian run the line is the z axis and the plane x = 0; in an axisymmetric or spherical run, the axis and the plane z = 0',
};

const noh: FlowDef<{ symmetry: number; speed: number; rho: number; p: number }> = {
  name: 'noh',
  label: 'Noh implosion',
  tip: 'Cold gas of uniform density falling at one speed toward a point, a line or a plane (Noh 1987). A shock leaves the center at a third of that speed (for gamma = 5/3), and the gas behind it is at rest, compressed 64 times in spherical symmetry (16 in cylindrical, 4 in planar). The exact solution is known everywhere, which makes it the standard test of wall heating at the center',
  wholeBox: true,
  fields: {
    symmetry: symmetryField,
    speed: { label: 'Speed', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'The speed at which the gas falls inward' },
    rho: { label: 'Density', default: 1, min: 1e-12, max: 1e12, factor: 1.25, tip: 'Density of the gas at the start' },
    p: { label: 'Pressure', default: 1e-6, min: 0, max: 1e6, factor: 10, tip: 'Pressure of the gas at the start; the exact solution has none, so keep it far below density times speed squared' },
  },
  init: (ctx, q) => {
    const at = radial(ctx, q.symmetry);
    return (x, y, z) => {
      const { n } = at(x, y, z);
      return { rho: q.rho, vx: -q.speed * n[0] + 0, vy: -q.speed * n[1] + 0, vz: -q.speed * n[2] + 0, p: q.p, s: 0 };
    };
  },
};

const uniformCollapse: FlowDef<{ symmetry: number; tCollapse: number; rho: number; e: number }> = {
  name: 'uniform-collapse',
  label: 'Uniform collapse',
  tip: 'Gas of uniform density and temperature falling inward at a speed proportional to the distance, u = -r / t_c, so all of it reaches the center at once at t = t_c (ExactPack’s noh2). It stays uniform as it is compressed along its adiabat, with no shock: the density grows as (1 - t / t_c)^-k, for k = 3, 2 or 1 by the symmetry',
  wholeBox: true,
  fields: {
    symmetry: symmetryField,
    tCollapse: { label: 'Collapse time', default: 1, min: 1e-6, max: 1e6, factor: 1.25, tip: 'When the gas reaches the center: the distance over the inward speed' },
    rho: { label: 'Density', default: 1, min: 1e-12, max: 1e12, factor: 1.25, tip: 'Density of the gas at the start' },
    e: { label: 'Specific internal energy', default: 1, min: 0, max: 1e6, factor: 1.25, tip: 'Internal energy per unit mass of the gas at the start' },
  },
  init: (ctx, q) => {
    const at = radial(ctx, q.symmetry);
    const p = ctx.eos.pressure(q.rho, q.rho * q.e);
    return (x, y, z) => {
      const { d, n } = at(x, y, z);
      const u = -d / q.tCollapse;
      return { rho: q.rho, vx: u * n[0] + 0, vy: u * n[1] + 0, vz: u * n[2] + 0, p, s: 0 };
    };
  },
};

export const flows = [noh, uniformCollapse];
